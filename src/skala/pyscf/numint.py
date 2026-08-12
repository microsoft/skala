# SPDX-License-Identifier: MIT

from collections.abc import Callable
from typing import Any, Generic, Protocol, overload

import torch
from pyscf import gto
from torch import Tensor

from skala.functional.base import ExcFunctionalBase
from skala.pyscf.backend import (
    KS,
    Array,
    Grid,
    from_numpy_or_cupy,
    to_cupy,
    to_numpy,
)
from skala.pyscf.xc_integrator import XCIntegrator


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
    >>> from skala.pyscf.grids import SkalaGrids
    >>> from skala.pyscf.numint import SkalaNumInt
    >>>
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="def2-svp", verbose=0)
    >>> ks = dft.KS(mol)
    >>> ks._numint = SkalaNumInt(load_functional("skala-1.1"))
    >>> ks.grids = SkalaGrids(mol)
    >>> ks.grids.build(mol)  # DOCTEST: Ellipsis
    <skala.pyscf.grids.SkalaGrids object at 0x...>
    >>> energy = ks.kernel()
    >>> print(energy)  # DOCTEST: Ellipsis
    -1.1425799...
    """

    def __init__(
        self,
        functional: ExcFunctionalBase,
        chunk_size: int | None = None,
        device: torch.device | None = None,
    ):
        self.integrator = XCIntegrator(functional, chunk_size=chunk_size, device=device)

    @property
    def device(self) -> torch.device:
        """Torch device used by the XC integrator."""
        return self.integrator.device

    @property
    def func(self) -> ExcFunctionalBase:
        """Functional retained for gradient-adapter compatibility."""
        return self.integrator.functional

    def _from_backend(self, x: Array) -> Tensor:
        return from_numpy_or_cupy(x, device=self.device)

    @overload
    def _to_backend(self, x: Tensor) -> Array: ...
    @overload
    def _to_backend(self, x: list[Tensor]) -> list[Array]: ...
    def _to_backend(self, x: Tensor | list[Tensor]) -> Array | list[Array]:
        if isinstance(x, list):
            return [self._to_backend(y) for y in x]

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
        density = self.integrator.density(
            mol,
            self._from_backend(dm),
            grids,
            max_memory=max_memory,
        )
        return self._to_backend(density)

    def __call__(
        self,
        mol: gto.Mole,
        grids: Grid,
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
            second_order: Unsupported; use ``gen_response`` for response evaluation.
            max_memory: The maximum memory to use for each chunk in megabytes (MB).

        Returns:
            A tuple of the total integrated density, the XC energy, and the XC potential.
        """
        if second_order:
            raise NotImplementedError(
                "Direct second-order evaluation is not supported; use gen_response()."
            )

        result = self.integrator(mol, grids, dm, max_memory=max_memory)
        return result.electron_count, result.energy, result.potential

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
            mol, grids, xc_code, self._from_backend(dm), max_memory=max_memory
        )
        return N.sum().item(), E_xc.item(), self._to_backend(V_xc)

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
            mol, grids, xc_code, self._from_backend(dm), max_memory=max_memory
        )
        return self._to_backend(N), E_xc.item(), self._to_backend(V_xc)

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
        """Generates the response function for the functional."""
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

        dm0 = self._from_backend(ks.make_rdm1(mo_coeff, mo_occ))
        xc_response = self.integrator.gen_response(
            ks.mol,
            ks.grids,
            dm0,
            max_memory=ks.max_memory,
            safety_fraction=kwargs.get("safety_fraction"),
        )

        def hessian_vector_product(dm1: Array) -> Array:
            v1 = self._to_backend(xc_response(self._from_backend(dm1)))
            vj = ks.get_j(ks.mol, dm1, hermi=1)

            if ks.mol.spin == 0:
                v1 += vj
            else:
                v1 += vj[0] + vj[1]

            return v1

        return hessian_vector_product
