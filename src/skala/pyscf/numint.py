# SPDX-License-Identifier: MIT

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import torch
from pyscf import dft, gto
from torch import Tensor

from skala.functional.base import ExcFunctionalBase
from skala.pyscf.backend import (
    KS,
    Array,
    Grid,
    check_gpu_imports_were_successful,
    from_numpy_or_cupy,
    to_cupy,
    to_numpy,
)
from skala.pyscf.features import chunked_features, generate_features


class LibXCSpec(Protocol):
    __version__: str | None
    __references__: str | None

    @staticmethod
    def is_hybrid_xc(xc: str) -> bool: ...

    @staticmethod
    def is_nlc(xc: str) -> bool: ...


class PySCFNumInt(
    Protocol[Array],
):
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
    ) -> Callable[[np.ndarray], np.ndarray]:
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
        if device is None:
            self.device = torch.get_default_device()
        else:
            self.device = device

        if self.device.type == "cuda":
            check_gpu_imports_were_successful()

        self.func = functional.to(device=self.device)
        self.chunk_size = chunk_size

    def from_backend(
        self,
        x: Array,
        device: torch.device | None = None,
        transpose: bool = False,
    ) -> Tensor:
        return from_numpy_or_cupy(x, device=device or self.device, transpose=transpose)

    def to_backend(self, x: Tensor | list[Tensor]) -> Array | list[Array]:
        if isinstance(x, list):
            return [self.to_backend(y) for y in x]  # type: ignore

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
            chunk_size=self.chunk_size,
            max_memory=max_memory,
            gpu=self.device.type == "cuda",
        )
        return self.to_backend(mol_features["density"].sum(0))  # type: ignore

    def __call__(
        self,
        mol: gto.Mole,
        grids: dft.Grids,
        xc_code: str | None,
        dm: Tensor,
        second_order: bool = False,
        max_memory: int = 2000,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate the XC functional for the given molecule and density matrix.
        Input:
            mol: The molecule.
            grids: The grid.
            xc_code: The XC code (not used in the reimplementation).
            dm: The density matrix.
            second_order: Whether to compute second-order derivatives.
            max_memory: The maximum memory to use for each chunk in megabytes (MB). If None, the maximum memory is determined automatically.

        Returns:
            A tuple of the total integrated density, the XC energy, and the XC potential.
        """

        if self.device != dm.device:
            raise ValueError(
                f"Density matrix device {dm.device} does not match functional device {self.device}"
            )

        if self._functional_supports_atom_chunking():
            dm = dm.detach().requires_grad_()
            tot_dens = torch.tensor((0.0, 0.0), device=self.device, dtype=dm.dtype)
            E_xc = torch.tensor(0.0, device=self.device, dtype=dm.dtype)
            V_xc = torch.zeros_like(dm)
            for mol_features in chunked_features(
                mol,
                dm,
                grids,
                features=set(self.func.features),
                func_deriv=1,
                max_memory_in_mb=max_memory if dm.device.type == "cpu" else None,
                safety_fraction=0.8,  # tends to be faster for large chunks
            ):
                E_xc_chunk = self.func.get_exc(mol_features)
                (V_xc_chunk,) = torch.autograd.grad(
                    E_xc_chunk,
                    dm,
                    torch.ones_like(E_xc_chunk),
                )
                tot_dens += (
                    (mol_features["density"] * mol_features["grid_weights"])
                    .sum(dim=-1)
                    .detach()
                )
                E_xc += E_xc_chunk.detach()
                V_xc += V_xc_chunk.detach()
                del E_xc_chunk, V_xc_chunk, mol_features

            return tot_dens, E_xc, V_xc
        else:
            dm = dm.requires_grad_()
            mol_features = generate_features(
                mol,
                dm,
                grids,
                set(self.func.features),
                chunk_size=self.chunk_size,
                max_memory=max_memory,
                gpu=self.device.type == "cuda",
            )
            E_xc = self.func.get_exc(mol_features)
            (V_xc,) = torch.autograd.grad(
                E_xc,
                dm,
                torch.ones_like(E_xc),
                retain_graph=second_order,
                create_graph=second_order,
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
        return N.sum().item(), E_xc.item(), self.to_backend(V_xc)  # type: ignore

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
        return self.to_backend(N), E_xc.item(), self.to_backend(V_xc)  # type: ignore

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

        if self._functional_supports_atom_chunking():
            dm0 = dm0.requires_grad_()

            def hessian_vector_product_atom_chunked(dm1: Array) -> Array:
                dm1_tensor = self.from_backend(dm1)
                hvp_total = torch.zeros_like(dm0)
                for mol_features in chunked_features(
                    ks.mol,
                    dm0,
                    ks.grids,
                    features=set(self.func.features),
                    func_deriv=2,
                    max_memory_in_mb=ks.max_memory
                    if dm0.device.type == "cpu"
                    else None,
                    safety_fraction=kwargs.get(
                        "safety_fraction", 0.0
                    ),  # Force small chunks (single atoms) because it's empirically fastest.
                ):
                    E_xc_chunk = self.func.get_exc(mol_features)
                    (V_xc_chunk,) = torch.autograd.grad(
                        E_xc_chunk,
                        dm0,
                        torch.ones_like(E_xc_chunk),
                        retain_graph=True,
                        create_graph=True,
                    )
                    (hvp_chunk,) = torch.autograd.grad(
                        V_xc_chunk,
                        dm0,
                        dm1_tensor,
                        retain_graph=True,
                    )
                    hvp_total += hvp_chunk
                    del E_xc_chunk, V_xc_chunk, hvp_chunk, mol_features

                v1 = self.to_backend(hvp_total)
                vj = ks.get_j(ks.mol, dm1, hermi=1)
                if ks.mol.spin == 0:
                    v1 += vj
                else:
                    v1 += vj[0] + vj[1]
                return v1

            return hessian_vector_product_atom_chunked

        else:
            # caching V_xc saves a forward pass in each iteration
            dm0 = dm0.requires_grad_()
            V_xc = self(ks.mol, ks.grids, None, dm0, second_order=True)[2]

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

    def _functional_supports_atom_chunking(self) -> bool:
        return "atomic_grid_sizes" in self.func.features
