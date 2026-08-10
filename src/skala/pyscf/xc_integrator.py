# SPDX-License-Identifier: MIT

"""Tensor-level exchange-correlation integration."""

from collections.abc import Callable
from typing import NamedTuple, cast

import torch
from pyscf import gto
from pyscf.dft import numint as pyscf_numint
from torch import Tensor

from skala.features import Feature
from skala.functional.base import ExcFunctionalBase
from skala.pyscf import ao_evaluation, feature_math
from skala.pyscf.backend import Grid, check_gpu_imports_were_successful
from skala.pyscf.evaluation import EvaluationPolicy, FeatureSpec
from skala.pyscf.features import generate_features
from skala.pyscf.model_chunking import prepare_model_feature_chunks
from skala.pyscf.screening import (
    CPU_AO_SCREENING_BLOCK_SIZE,
    SpatialGridLayout,
    prepare_spatial_grid_layout,
    screened_feature_jvp,
)


def _should_screen_aos(mol: gto.Mole) -> bool:
    """Return whether PySCF's sparse-contraction crossover is exceeded."""
    # we use a smaller threshold because for MetaGGAs the AO evaluation is more expensive
    return 2 * mol.nao_nr() > pyscf_numint.SWITCH_SIZE


class XCResult(NamedTuple):
    """Tensor-valued result of exchange-correlation integration."""

    electron_count: Tensor
    energy: Tensor
    potential: Tensor


class XCIntegrator:
    """Evaluate XC energies, potentials, and potential responses in Torch."""

    def __init__(
        self,
        functional: ExcFunctionalBase,
        chunk_size: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.get_default_device()
        if self.device.type == "cuda":
            check_gpu_imports_were_successful()

        self.functional = functional.to(device=self.device)
        self.feature_spec = FeatureSpec(self.functional.features)
        self.evaluation_policy = EvaluationPolicy(ao_block_size=chunk_size)

    def density(
        self,
        mol: gto.Mole,
        dm: Tensor,
        grids: Grid,
        max_memory: int = 2000,
    ) -> Tensor:
        """Evaluate the total density on each grid point."""
        mol_features = generate_features(
            mol,
            dm,
            grids,
            features={Feature.DENSITY},
            chunk_size=self.evaluation_policy.ao_block_size,
            max_memory=max_memory,
            gpu=self.device.type == "cuda",
        )
        return mol_features[Feature.DENSITY].sum(0)

    def __call__(
        self,
        mol: gto.Mole,
        grids: Grid,
        dm: Tensor,
        max_memory: int = 2000,
    ) -> XCResult:
        """Evaluate electron count, XC energy, and XC potential."""
        self._validate_device(dm)
        if self.feature_spec.supports_spatial_decomposition and _should_screen_aos(mol):
            return self._integrate_screened(mol, grids, dm, max_memory)
        return self._integrate_dense(mol, grids, dm, max_memory)

    def gen_response(
        self,
        mol: gto.Mole,
        grids: Grid,
        dm0: Tensor,
        max_memory: int = 2000,
        safety_fraction: float | None = None,
    ) -> Callable[[Tensor], Tensor]:
        """Build an XC-only Hessian-vector product callable."""
        self._validate_device(dm0)
        if self.feature_spec.supports_spatial_decomposition and _should_screen_aos(mol):
            return self._gen_response_screened(
                mol,
                grids,
                dm0,
                max_memory=max_memory,
                safety_fraction=(
                    self.evaluation_policy.safety_fraction
                    if safety_fraction is None
                    else safety_fraction
                ),
            )
        return self._gen_response_dense(mol, grids, dm0, max_memory=max_memory)

    def _validate_device(self, dm: Tensor) -> None:
        if self.device != dm.device:
            raise ValueError(
                f"Density matrix device {dm.device} does not match functional device {self.device}"
            )

    def _get_spatial_grid_layout(
        self,
        mol: gto.Mole,
        grids: Grid,
    ) -> SpatialGridLayout:
        grid_state = vars(grids)
        spatial_grid_layout = cast(
            SpatialGridLayout | None,
            grid_state.get("_skala_spatial_grid_layout"),
        )
        if spatial_grid_layout is not None:
            return spatial_grid_layout

        if self.device.type == "cuda":
            check_gpu_imports_were_successful()
            from gpu4pyscf.dft import numint as dft_gpu_numint

            block_size = int(dft_gpu_numint.MIN_BLK_SIZE)
        else:
            block_size = CPU_AO_SCREENING_BLOCK_SIZE

        spatial_grid_layout = prepare_spatial_grid_layout(
            mol,
            grids,
            block_size,
            self.device,
        )
        grid_state["_skala_spatial_grid_layout"] = spatial_grid_layout
        return spatial_grid_layout

    def _integrate_screened(
        self,
        mol: gto.Mole,
        grids: Grid,
        dm: Tensor,
        max_memory: int,
    ) -> XCResult:
        dm = dm.detach().requires_grad_()
        dm_eval = dm.double()
        electron_count = torch.zeros(2, device=self.device, dtype=dm_eval.dtype)
        energy = torch.tensor(0.0, device=self.device, dtype=dm_eval.dtype)
        feature_function = feature_math.MGGAFeatureFunction(self.feature_spec)
        spatial_grid_layout = self._get_spatial_grid_layout(mol, grids)
        sorted_raw_features = cast(
            Tensor,
            ao_evaluation.ChunkEvalForward.apply(  # type: ignore[no-untyped-call]
                dm_eval,
                mol,
                spatial_grid_layout.sorted_grids,
                feature_function,
                spatial_grid_layout.block_size,
                False,
            ),
        )
        atom_major_raw_features = sorted_raw_features.index_select(
            -1, spatial_grid_layout.inverse_permutation
        )
        model_chunks = prepare_model_feature_chunks(
            mol,
            dm,
            grids,
            atom_major_raw_features=atom_major_raw_features,
            feature_function=feature_function,
            deriv_order=1,
            max_memory_in_mb=max_memory if dm.device.type == "cpu" else None,
            safety_fraction=self.evaluation_policy.safety_fraction,
        )
        atom_major_cotangent = torch.zeros_like(atom_major_raw_features)
        for chunk in model_chunks:
            local_raw_features = chunk.raw_features
            mol_features = chunk.model_features
            energy_chunk = self.functional.get_exc(mol_features)
            (local_cotangent,) = torch.autograd.grad(
                energy_chunk,
                local_raw_features,
                torch.ones_like(energy_chunk),
            )
            atom_major_cotangent.index_copy_(
                -1, chunk.grid_indices, local_cotangent.detach()
            )
            electron_count += (
                (mol_features[Feature.DENSITY] * mol_features[Feature.GRID_WEIGHTS])
                .sum(dim=-1)
                .detach()
            )
            energy += energy_chunk.detach()
            del energy_chunk, local_cotangent, local_raw_features, mol_features

        sorted_cotangent = atom_major_cotangent.index_select(
            -1, spatial_grid_layout.forward_permutation
        )
        (potential,) = torch.autograd.grad(
            sorted_raw_features,
            dm,
            sorted_cotangent,
        )
        return XCResult(electron_count, energy, potential)

    def _integrate_dense(
        self,
        mol: gto.Mole,
        grids: Grid,
        dm: Tensor,
        max_memory: int,
        *,
        create_graph: bool = False,
    ) -> XCResult:
        dm = dm.requires_grad_()
        mol_features = generate_features(
            mol,
            dm,
            grids,
            set(self.feature_spec.names) | {Feature.DENSITY, Feature.GRID_WEIGHTS},
            chunk_size=self.evaluation_policy.ao_block_size,
            max_memory=max_memory,
            gpu=self.device.type == "cuda",
        )
        energy = self.functional.get_exc(mol_features)
        (potential,) = torch.autograd.grad(
            energy,
            dm,
            torch.ones_like(energy),
            retain_graph=create_graph,
            create_graph=create_graph,
        )
        electron_count = (
            mol_features[Feature.DENSITY] * mol_features[Feature.GRID_WEIGHTS]
        ).sum(dim=-1)
        return XCResult(electron_count, energy, potential)

    def _gen_response_screened(
        self,
        mol: gto.Mole,
        grids: Grid,
        dm0: Tensor,
        *,
        max_memory: int,
        safety_fraction: float,
    ) -> Callable[[Tensor], Tensor]:
        dm0 = dm0.requires_grad_()
        feature_function = feature_math.MGGAFeatureFunction(self.feature_spec)
        spatial_grid_layout = self._get_spatial_grid_layout(mol, grids)
        sorted_raw_features = cast(
            Tensor,
            ao_evaluation.ChunkEvalForward.apply(  # type: ignore[no-untyped-call]
                dm0.double(),
                mol,
                spatial_grid_layout.sorted_grids,
                feature_function,
                spatial_grid_layout.block_size,
                False,
            ),
        )
        atom_major_raw_features = sorted_raw_features.index_select(
            -1, spatial_grid_layout.inverse_permutation
        )
        model_chunks = prepare_model_feature_chunks(
            mol,
            dm0,
            grids,
            atom_major_raw_features=atom_major_raw_features,
            feature_function=feature_function,
            deriv_order=2,
            max_memory_in_mb=max_memory if dm0.device.type == "cpu" else None,
            safety_fraction=safety_fraction,
        )

        def hessian_vector_product(dm1: Tensor) -> Tensor:
            atom_major_tangent = screened_feature_jvp(
                dm1,
                mol,
                spatial_grid_layout,
                feature_function,
            )
            atom_major_hessian_action = torch.zeros_like(atom_major_raw_features)
            for chunk in model_chunks:
                local_raw_features = chunk.raw_features
                mol_features = chunk.model_features
                energy_chunk = self.functional.get_exc(mol_features)
                (local_gradient,) = torch.autograd.grad(
                    energy_chunk,
                    local_raw_features,
                    torch.ones_like(energy_chunk),
                    create_graph=True,
                )
                if local_gradient.requires_grad:
                    (local_hessian_action,) = torch.autograd.grad(
                        local_gradient,
                        local_raw_features,
                        atom_major_tangent.index_select(-1, chunk.grid_indices),
                    )
                else:
                    local_hessian_action = torch.zeros_like(local_raw_features)
                atom_major_hessian_action.index_copy_(
                    -1, chunk.grid_indices, local_hessian_action.detach()
                )
                del (
                    energy_chunk,
                    local_gradient,
                    local_hessian_action,
                    local_raw_features,
                    mol_features,
                )

            sorted_hessian_action = atom_major_hessian_action.index_select(
                -1, spatial_grid_layout.forward_permutation
            )
            (hvp_total,) = torch.autograd.grad(
                sorted_raw_features,
                dm0,
                sorted_hessian_action,
                retain_graph=True,
            )
            return hvp_total

        return hessian_vector_product

    def _gen_response_dense(
        self,
        mol: gto.Mole,
        grids: Grid,
        dm0: Tensor,
        *,
        max_memory: int,
    ) -> Callable[[Tensor], Tensor]:
        dm0 = dm0.requires_grad_()
        potential = self._integrate_dense(
            mol,
            grids,
            dm0,
            max_memory,
            create_graph=True,
        ).potential

        def hessian_vector_product(dm1: Tensor) -> Tensor:
            return torch.autograd.grad(
                potential,
                dm0,
                dm1,
                retain_graph=True,
            )[0]

        return hessian_vector_product
