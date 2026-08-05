# SPDX-License-Identifier: MIT

from collections.abc import Callable
from typing import Any, Generic, Protocol, cast, overload

import torch
from pyscf import gto
from pyscf.dft import numint as pyscf_numint
from torch import Tensor

from skala.functional.base import ExcFunctionalBase
from skala.pyscf import ao_evaluation, feature_math
from skala.pyscf.backend import (
    KS,
    Array,
    Grid,
    check_gpu_imports_were_successful,
    from_numpy_or_cupy,
    to_cupy,
    to_numpy,
)
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
    """Determine whether a molecule is large enough for AO screening.

    Args:
        mol: Molecule whose AO count is compared with PySCF's sparse-contraction
            crossover.

    Returns:
        Whether the molecule has more AOs than PySCF's screening threshold.
    """
    # Keep the compatibility fallback here, not at call sites. PySCF uses this
    # crossover before selecting sparse density/Vxc contractions.
    switch_size = pyscf_numint.SWITCH_SIZE
    return mol.nao_nr() > switch_size


class LibXCSpec(Protocol):
    __version__: str | None
    __references__: str | None

    @staticmethod
    def is_hybrid_xc(xc: str) -> bool: ...

    @staticmethod
    def is_nlc(xc: str) -> bool: ...


class PySCFNumInt(Protocol, Generic[Array]):
    """Interface for PySCF-compatible numint functionals."""

    libxc: LibXCSpec

    def get_rho(
        self,
        mol: gto.Mole,
        dm: Array,
        grids: Grid,
        max_memory: int = 2000,
    ) -> Array: ...

    def nr_rks(
        self,
        mol: gto.Mole,
        grids: Grid,
        xc_code: str | None,
        dm: Array,
        max_memory: int = 2000,
    ) -> tuple[float, float, Array]:
        """Restricted Kohn-Sham method, applicable if both spin-densities as equal."""
        ...

    def nr_uks(
        self,
        mol: gto.Mole,
        grids: Grid,
        xc_code: str | None,
        dm: Array,
        max_memory: int = 2000,
    ) -> tuple[Array, float, Array]:
        """Unrestricted Kohn-Sham method, spin densities can be different."""
        ...

    def rsh_and_hybrid_coeff(self) -> tuple[float, float, float]:
        return 0, 0, 0

    def gen_response(
        self,
        mo_coeff: Array | None,
        mo_occ: Array | None,
        *,
        ks: KS,
        **kwargs: Any,
    ) -> Callable[[Array], Array]:
        """Generates the response function for the functional."""
        ...

    def reset(self) -> "PySCFNumInt[Array]":
        """GPU4PySCF-specific method to reset the internal cache of the functional, if any."""
        return self


class SkalaNumInt(PySCFNumInt[Array]):
    """PySCF-compatible reimplementation of `pyscf.dft.numint.NumInt`.

    Evaluation of atomic orbitals and one-electron integrals on a grid
    is cached for speed.

    Example
    -------
    >>> from pyscf import gto, dft
    >>> from skala.functional import load_functional
    >>> from skala.pyscf.numint import SkalaNumInt
    >>>
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="def2-svp", verbose=0)
    >>> ks = dft.KS(mol)
    >>> ks._numint = SkalaNumInt(load_functional("skala-1.1"))
    >>> ks.grids.build(mol, sort_grids=False)  # DOCTEST: Ellipsis
    <pyscf.dft.gen_grid.Grids object at 0x...>
    >>> energy = ks.kernel()
    >>> print(energy)  # DOCTEST: Ellipsis
    -1.1425799...
    """

    device: torch.device

    def __init__(
        self,
        functional: ExcFunctionalBase,
        chunk_size: int | None = None,
        device: torch.device | None = None,
    ):
        self.device = device or torch.get_default_device()

        if self.device.type == "cuda":
            check_gpu_imports_were_successful()

        self.func = functional.to(device=self.device)
        self.feature_spec = FeatureSpec(self.func.features)
        self.evaluation_policy = EvaluationPolicy(ao_block_size=chunk_size)

    def reset(self) -> "SkalaNumInt[Array]":
        """Return this integrator; spatial layouts are owned by grid objects."""
        return self

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

    def from_backend(
        self,
        x: Array,
        device: torch.device | None = None,
        transpose: bool = False,
    ) -> Tensor:
        return from_numpy_or_cupy(x, device=device or self.device, transpose=transpose)

    @overload
    def to_backend(self, x: Tensor) -> Array: ...

    @overload
    def to_backend(self, x: list[Tensor]) -> list[Array]: ...

    def to_backend(self, x: Tensor | list[Tensor]) -> Array | list[Array]:
        if isinstance(x, list):
            return [self.to_backend(y) for y in x]

        if self.device.type == "cuda":
            return to_cupy(x)
        else:
            return to_numpy(x)

    def get_rho(
        self,
        mol: gto.Mole,
        dm: Array,
        grids: Grid,
        max_memory: int = 2000,
        verbose: int = 0,
    ) -> Array:
        mol_features = generate_features(
            mol,
            self.from_backend(dm),
            grids,
            features={"density"},
            chunk_size=self.evaluation_policy.ao_block_size,
            max_memory=max_memory,
            gpu=self.device.type == "cuda",
        )
        return self.to_backend(mol_features["density"].sum(0))

    def __call__(
        self,
        mol: gto.Mole,
        grids: Grid,
        xc_code: str | None,
        dm: Tensor,
        second_order: bool = False,
        max_memory: int = 2000,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate the XC functional for a molecule and density matrix."""
        if second_order:
            raise NotImplementedError(
                "Direct second-order evaluation is not supported; use gen_response()."
            )

        if self.device != dm.device:
            raise ValueError(
                f"Density matrix device {dm.device} does not match functional device {self.device}"
            )

        if self.feature_spec.supports_screened_evaluation and _should_screen_aos(mol):
            return self._call_screened(mol, grids, dm, max_memory)
        return self._call_dense(mol, grids, dm, max_memory)

    def _call_screened(
        self,
        mol: gto.Mole,
        grids: Grid,
        dm: Tensor,
        max_memory: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dm = dm.detach().requires_grad_()
        tot_dens = torch.tensor((0.0, 0.0), device=self.device, dtype=dm.dtype)
        E_xc = torch.tensor(0.0, device=self.device, dtype=dm.dtype)
        feature_function = feature_math.MGGAFeatureFunction(self.feature_spec)
        spatial_grid_layout = self._get_spatial_grid_layout(mol, grids)
        sorted_raw_features = cast(
            Tensor,
            ao_evaluation.ChunkEvalForward.apply(  # type: ignore[no-untyped-call]
                dm.double(),
                mol,
                spatial_grid_layout.sorted_grids,
                feature_function,
                spatial_grid_layout.block_size,
                False,
                dm.device.type == "cuda",
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
            func_deriv=1,
            max_memory_in_mb=max_memory if dm.device.type == "cpu" else None,
            safety_fraction=self.evaluation_policy.safety_fraction,
        )
        # Store only full-grid feature cotangents; model activations remain chunk-local.
        atom_major_cotangent = torch.zeros_like(atom_major_raw_features)
        for chunk in model_chunks:
            local_raw_features = chunk.raw_features
            mol_features = chunk.model_features
            E_xc_chunk = self.func.get_exc(mol_features)
            (local_cotangent,) = torch.autograd.grad(
                E_xc_chunk,
                local_raw_features,
                torch.ones_like(E_xc_chunk),
            )
            atom_major_cotangent[..., chunk.grid_slice] = local_cotangent.detach()
            tot_dens += (
                (mol_features["density"] * mol_features["grid_weights"])
                .sum(dim=-1)
                .detach()
            )
            E_xc += E_xc_chunk.detach()
            del E_xc_chunk, local_cotangent, local_raw_features, mol_features

        # Reorder detached cotangents explicitly instead of backpropagating through it.
        sorted_cotangent = atom_major_cotangent.index_select(
            -1, spatial_grid_layout.forward_permutation
        )
        # The custom VJP reevaluates AO blocks sequentially without a full-grid AO graph.
        (V_xc,) = torch.autograd.grad(
            sorted_raw_features,
            dm,
            sorted_cotangent,
        )
        return tot_dens, E_xc, V_xc

    def _call_dense(
        self,
        mol: gto.Mole,
        grids: Grid,
        dm: Tensor,
        max_memory: int,
        *,
        create_graph: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:

        dm = dm.requires_grad_()
        mol_features = generate_features(
            mol,
            dm,
            grids,
            set(self.feature_spec.names),
            chunk_size=self.evaluation_policy.ao_block_size,
            max_memory=max_memory,
            gpu=self.device.type == "cuda",
        )
        E_xc = self.func.get_exc(mol_features)
        (V_xc,) = torch.autograd.grad(
            E_xc,
            dm,
            torch.ones_like(E_xc),
            retain_graph=create_graph,
            create_graph=create_graph,
        )

        rho = mol_features["density"]
        grid_weights = mol_features.get(
            "grid_weights", self.from_backend(grids.weights)
        )
        tot_dens = (rho * grid_weights).sum(dim=-1)
        return tot_dens, E_xc, V_xc

    def nr_rks(
        self,
        mol: gto.Mole,
        grids: Grid,
        xc_code: str | None,
        dm: Array,
        max_memory: int = 2000,
    ) -> tuple[float, float, Array]:
        """Restricted Kohn-Sham method, applicable if both spin-densities as equal."""
        assert len(dm.shape) == 2
        N, E_xc, V_xc = self(
            mol, grids, xc_code, self.from_backend(dm), max_memory=max_memory
        )
        return N.sum().item(), E_xc.item(), self.to_backend(V_xc)

    def nr_uks(
        self,
        mol: gto.Mole,
        grids: Grid,
        xc_code: str | None,
        dm: Array,
        max_memory: int = 2000,
    ) -> tuple[Array, float, Array]:
        """Unrestricted Kohn-Sham method, spin densities can be different."""
        assert len(dm.shape) == 3 and dm.shape[0] == 2
        N, E_xc, V_xc = self(
            mol, grids, xc_code, self.from_backend(dm), max_memory=max_memory
        )
        return self.to_backend(N), E_xc.item(), self.to_backend(V_xc)

    class libxc:
        __version__ = None
        __reference__ = None

        @staticmethod
        def is_hybrid_xc(xc: str) -> bool:
            return False

        @staticmethod
        def is_nlc(xc: str) -> bool:
            return False

    # Overrides PySCF's base with a wider Array type for mo_coeff/mo_occ.
    def gen_response(
        self,
        mo_coeff: Array | None,
        mo_occ: Array | None,
        *,
        ks: KS,
        **kwargs: Any,
    ) -> Callable[[Array], Array]:
        assert mo_coeff is not None
        assert mo_occ is not None
        if kwargs is not None:
            # check if kwargs are valid
            # this response function only works for KS DFT with meta GGA
            if "hermi" in kwargs:
                assert kwargs["hermi"] == 1
            if "singlet" in kwargs:
                assert kwargs["singlet"] is None
            if "with_j" in kwargs:
                assert kwargs["with_j"]

        dm0 = self.from_backend(ks.make_rdm1(mo_coeff, mo_occ))

        if self.feature_spec.supports_screened_evaluation and _should_screen_aos(
            ks.mol
        ):
            return self._gen_response_screened(
                ks,
                dm0,
                safety_fraction=kwargs.get(
                    "safety_fraction", self.evaluation_policy.safety_fraction
                ),
            )
        return self._gen_response_dense(ks, dm0)

    def _gen_response_screened(
        self,
        ks: KS,
        dm0: Tensor,
        *,
        safety_fraction: float,
    ) -> Callable[[Array], Array]:
        dm0 = dm0.requires_grad_()
        feature_function = feature_math.MGGAFeatureFunction(self.feature_spec)
        spatial_grid_layout = self._get_spatial_grid_layout(ks.mol, ks.grids)
        sorted_raw_features = cast(
            Tensor,
            ao_evaluation.ChunkEvalForward.apply(  # type: ignore[no-untyped-call]
                dm0.double(),
                ks.mol,
                spatial_grid_layout.sorted_grids,
                feature_function,
                spatial_grid_layout.block_size,
                False,
                dm0.device.type == "cuda",
            ),
        )
        atom_major_raw_features = sorted_raw_features.index_select(
            -1, spatial_grid_layout.inverse_permutation
        )
        model_chunks = prepare_model_feature_chunks(
            ks.mol,
            dm0,
            ks.grids,
            atom_major_raw_features=atom_major_raw_features,
            feature_function=feature_function,
            func_deriv=2,
            max_memory_in_mb=ks.max_memory if dm0.device.type == "cpu" else None,
            safety_fraction=safety_fraction,
        )

        def hessian_vector_product_atom_chunked(dm1: Array) -> Array:
            dm1_tensor = self.from_backend(dm1)
            atom_major_tangent = screened_feature_jvp(
                dm0,
                dm1_tensor,
                ks.mol,
                spatial_grid_layout,
                feature_function,
            )
            # Store the full-grid model Hessian action, not per-chunk model graphs.
            atom_major_hessian_action = torch.zeros_like(atom_major_raw_features)
            for chunk in model_chunks:
                local_raw_features = chunk.raw_features
                mol_features = chunk.model_features
                E_xc_chunk = self.func.get_exc(mol_features)
                (local_gradient,) = torch.autograd.grad(
                    E_xc_chunk,
                    local_raw_features,
                    torch.ones_like(E_xc_chunk),
                    create_graph=True,
                )
                if local_gradient.requires_grad:
                    (local_hessian_action,) = torch.autograd.grad(
                        local_gradient,
                        local_raw_features,
                        atom_major_tangent[..., chunk.grid_slice],
                    )
                else:
                    local_hessian_action = torch.zeros_like(local_raw_features)
                atom_major_hessian_action[..., chunk.grid_slice] = (
                    local_hessian_action.detach()
                )
                del (
                    E_xc_chunk,
                    local_gradient,
                    local_hessian_action,
                    local_raw_features,
                    mol_features,
                )

            # Restore block order after all chunk-local Hessian actions are detached.
            sorted_hessian_action = atom_major_hessian_action.index_select(
                -1, spatial_grid_layout.forward_permutation
            )
            # The custom VJP traverses AO blocks sequentially and retains no AO graph.
            (hvp_total,) = torch.autograd.grad(
                sorted_raw_features,
                dm0,
                sorted_hessian_action,
                retain_graph=True,
            )

            v1 = self.to_backend(hvp_total)
            vj = ks.get_j(ks.mol, dm1, hermi=1)
            if ks.mol.spin == 0:
                v1 += vj
            else:
                v1 += vj[0] + vj[1]
            return v1

        return hessian_vector_product_atom_chunked

    def _gen_response_dense(
        self,
        ks: KS,
        dm0: Tensor,
    ) -> Callable[[Array], Array]:
        dm0 = dm0.requires_grad_()
        _, _, V_xc = self._call_dense(
            ks.mol,
            ks.grids,
            dm0,
            ks.max_memory,
            create_graph=True,
        )

        def hessian_vector_product(dm1: Array) -> Array:
            v1 = self.to_backend(
                torch.autograd.grad(
                    V_xc, dm0, self.from_backend(dm1), retain_graph=True
                )[0]
            )
            vj = ks.get_j(ks.mol, dm1, hermi=1)

            if ks.mol.spin == 0:
                v1 += vj
            else:
                v1 += vj[0] + vj[1]

            return v1

        return hessian_vector_product
