# SPDX-License-Identifier: MIT

"""Tensor-level exchange-correlation integration."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, TypeGuard

import torch
from pyscf.dft import numint as pyscf_numint
from torch import Tensor

from pyscf import gto
from skala.features import Feature
from skala.functional.base import ExcFunctionalBase
from skala.pyscf import ao_evaluation, feature_math
from skala.pyscf.backend import Grid, check_gpu_imports_were_successful
from skala.pyscf.evaluation import EvaluationPolicy, FeatureSpec
from skala.pyscf.grids import SkalaGrids as PySCFSkalaGrids
from skala.pyscf.model_chunking import ModelFeatureChunker, ModelFeaturePlan
from skala.pyscf.spatial_grid_layout import SpatialGridLayout

if TYPE_CHECKING:
    from skala.gpu4pyscf.grids import SkalaGrids as GPU4PySCFSkalaGrids

    _SkalaGrid: TypeAlias = PySCFSkalaGrids | GPU4PySCFSkalaGrids


def _should_screen_aos(mol: gto.Mole) -> bool:
    """Return whether PySCF's sparse-contraction crossover is exceeded."""
    # we use a smaller threshold because for MetaGGAs the AO evaluation is more expensive
    result = (2 * mol.nao_nr()) > pyscf_numint.SWITCH_SIZE
    assert isinstance(result, bool)
    return result


def _assert_skala_grid(grids: Grid, device: torch.device) -> TypeGuard["_SkalaGrid"]:
    if device.type == "cuda":
        check_gpu_imports_were_successful()
        from skala.gpu4pyscf.grids import SkalaGrids as GPU4PySCFSkalaGrids

        expected_type = GPU4PySCFSkalaGrids
    else:
        expected_type = PySCFSkalaGrids

    if not isinstance(grids, expected_type):
        raise TypeError(
            f"{device.type.upper()} Skala XC evaluation requires "
            f"{expected_type.__module__}.{expected_type.__name__}, got "
            f"{type(grids).__module__}.{type(grids).__name__}"
        )
    return True


@dataclass(frozen=True)
class XCResult:
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
        self._validate_device(dm)
        _assert_skala_grid(grids, self.device)
        feature_function = feature_math.MGGAFeatureFunction({Feature.DENSITY})
        evaluate_raw_features = self._raw_feature_evaluator(
            mol,
            grids,
            feature_function,
            max_memory,
            screened=_should_screen_aos(mol),
        )
        return feature_function.to_dict(evaluate_raw_features(dm))[Feature.DENSITY].sum(
            0
        )

    def __call__(
        self,
        mol: gto.Mole,
        grids: Grid,
        dm: Tensor,
        max_memory: int = 2000,
    ) -> XCResult:
        """Evaluate electron count, XC energy, and XC potential."""
        self._validate_device(dm)
        _assert_skala_grid(grids, self.device)
        dm = dm.detach().requires_grad_()
        dm_eval = dm.double()
        electron_count = torch.zeros(2, device=self.device, dtype=dm_eval.dtype)
        energy = torch.tensor(0.0, device=self.device, dtype=dm_eval.dtype)
        evaluation_feature_spec = FeatureSpec(
            self.feature_spec
            | {
                Feature.DENSITY,
                Feature.GRID_WEIGHTS,
                Feature.ATOMIC_GRID_SIZES,
            }
        )
        model_feature_spec = self.feature_spec | {
            Feature.DENSITY,
            Feature.GRID_WEIGHTS,
        }
        feature_plan = ModelFeaturePlan(evaluation_feature_spec, model_feature_spec)
        feature_function = feature_math.MGGAFeatureFunction(
            evaluation_feature_spec.ao_features
        )
        evaluate_raw_features = self._raw_feature_evaluator(
            mol,
            grids,
            feature_function,
            max_memory,
            screened=_should_screen_aos(mol),
        )
        raw_features = evaluate_raw_features(dm_eval)
        model_chunks = ModelFeatureChunker(
            mol,
            dm,
            grids,
            atom_major_raw_features=raw_features,
            feature_plan=feature_plan,
            deriv_order=1,
            max_memory_in_mb=max_memory if dm.device.type == "cpu" else None,
            safety_fraction=self.evaluation_policy.safety_fraction,
        )
        raw_cotangent = torch.zeros_like(raw_features)
        for chunk in model_chunks:
            local_raw_features = chunk.raw_features
            mol_features = chunk.model_features
            energy_chunk = self.functional.get_exc(mol_features)
            (local_cotangent,) = torch.autograd.grad(
                energy_chunk,
                local_raw_features,
                torch.ones_like(energy_chunk),
            )
            raw_cotangent.index_copy_(-1, chunk.grid_indices, local_cotangent.detach())
            electron_count += (
                (mol_features[Feature.DENSITY] * mol_features[Feature.GRID_WEIGHTS])
                .sum(dim=-1)
                .detach()
            )
            energy += energy_chunk.detach()
            del energy_chunk, local_cotangent, local_raw_features, mol_features

        (potential,) = torch.autograd.grad(raw_features, dm, raw_cotangent)
        return XCResult(electron_count, energy, potential)

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
        _assert_skala_grid(grids, self.device)
        response_safety_fraction = (
            self.evaluation_policy.safety_fraction
            if safety_fraction is None
            else safety_fraction
        )
        dm0 = dm0.requires_grad_()
        evaluation_feature_spec = self.feature_spec | {Feature.ATOMIC_GRID_SIZES}
        feature_plan = ModelFeaturePlan(evaluation_feature_spec, self.feature_spec)
        feature_function = feature_math.MGGAFeatureFunction(
            evaluation_feature_spec.ao_features
        )
        evaluate_raw_features = self._raw_feature_evaluator(
            mol,
            grids,
            feature_function,
            max_memory,
            screened=_should_screen_aos(mol),
        )
        raw_features = evaluate_raw_features(dm0)
        model_chunks = ModelFeatureChunker(
            mol,
            dm0,
            grids,
            atom_major_raw_features=raw_features,
            feature_plan=feature_plan,
            deriv_order=2,
            max_memory_in_mb=max_memory if dm0.device.type == "cpu" else None,
            safety_fraction=response_safety_fraction,
        )

        def hessian_vector_product(dm1: Tensor) -> Tensor:
            raw_tangent = evaluate_raw_features(dm1).detach()
            raw_hessian_action = torch.zeros_like(raw_features)
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
                # A gradient independent of the raw features is constant, so its
                # directional derivative (the local Hessian action) is zero.
                if not local_gradient.requires_grad:
                    local_hessian_action = torch.zeros_like(local_raw_features)
                else:
                    (local_hessian_action,) = torch.autograd.grad(
                        local_gradient,
                        local_raw_features,
                        raw_tangent.index_select(-1, chunk.grid_indices),
                    )

                raw_hessian_action.index_copy_(
                    -1, chunk.grid_indices, local_hessian_action.detach()
                )
                del (
                    energy_chunk,
                    local_gradient,
                    local_hessian_action,
                    local_raw_features,
                    mol_features,
                )

            (hvp_total,) = torch.autograd.grad(
                raw_features,
                dm0,
                raw_hessian_action,
                retain_graph=True,
            )
            return hvp_total

        return hessian_vector_product

    def _validate_device(self, dm: Tensor) -> None:
        if self.device != dm.device:
            raise ValueError(
                f"Density matrix device {dm.device} does not match functional device {self.device}"
            )

    def _get_spatial_grid_layout(self, mol: gto.Mole, grids: Grid) -> SpatialGridLayout:
        if _assert_skala_grid(grids, self.device):
            return grids.prepare_spatial_grid_layout(mol, self.device)
        raise AssertionError("unreachable")

    def _raw_feature_evaluator(
        self,
        mol: gto.Mole,
        grids: Grid,
        feature_function: feature_math.MGGAFeatureFunction,
        max_memory: int,
        *,
        screened: bool,
    ) -> Callable[[Tensor], Tensor]:
        """Build an evaluator returning raw features in atom-major grid order."""
        if screened:
            spatial_grid_layout = self._get_spatial_grid_layout(mol, grids)

            def evaluate_screened(value: Tensor) -> Tensor:
                sorted_features = ao_evaluation.evaluate_ao_features_blockwise(
                    value.double(),
                    mol,
                    spatial_grid_layout.sorted_grids,
                    feature_function,
                    spatial_grid_layout.block_size,
                )
                return sorted_features.index_select(
                    -1, spatial_grid_layout.inverse_permutation
                )

            return evaluate_screened

        def evaluate_dense(value: Tensor) -> Tensor:
            return ao_evaluation.evaluate_raw_features_auto_chunk(
                value,
                mol,
                grids,
                feature_function,
                block_size=self.evaluation_policy.ao_block_size,
                max_memory=max_memory,
                gpu=self.device.type == "cuda",
            )

        return evaluate_dense
