# SPDX-License-Identifier: MIT

from collections.abc import Callable
from typing import Any, ClassVar, Generic, cast

import numpy as np
import torch
from torch import Tensor

from pyscf import gto
from skala.functional.base import ExcFunctionalBase
from skala.pyscf.backend import (
    KS,
    ArrayF64,
    Grid,
    from_numpy_or_cupy,
    to_cupy,
    to_numpy,
)
from skala.pyscf.xc_integrator import XCIntegrator, XCResult


class SkalaNumInt(Generic[ArrayF64]):
    """Skala implementation of the ``pyscf.dft.numint.NumInt`` interface.

    This class mimics the methods that PySCF and GPU4PySCF use from
    :class:`pyscf.dft.numint.NumInt` without inheriting from it.

    Evaluation of atomic orbitals and one-electron integrals on a grid
    is cached for speed.

    Response functions returned by :meth:`gen_response` retain the current
    KS/grid state and XC autograd graph. They are short-lived calculation
    objects and must not be stored for later reuse.

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

    def reset(self) -> "SkalaNumInt[ArrayF64]":
        """Retain the GPU4PySCF cache-reset interface."""
        return self

    def _from_backend(self, x: ArrayF64) -> Tensor:
        return from_numpy_or_cupy(x, device=self.device)

    def _to_backend(self, x: Tensor) -> ArrayF64:
        assert x.dtype == torch.float64
        if self.device.type == "cuda":
            result = to_cupy(x)
        else:
            result = to_numpy(x)
        assert result.dtype == np.dtype(np.float64)
        return cast(ArrayF64, result)

    def get_rho(
        self,
        mol: gto.Mole,
        dm: ArrayF64,
        grids: Grid,
        max_memory: int = 2000,
        verbose: int = 0,
    ) -> ArrayF64:
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
    ) -> XCResult:
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
            The total integrated density, XC energy, and XC potential.
        """
        if second_order:
            raise NotImplementedError(
                "Direct second-order evaluation is not supported; use gen_response()."
            )

        return self.integrator(mol, grids, dm, max_memory=max_memory)

    def nr_rks(
        self,
        mol: gto.Mole,
        grids: Grid,
        xc_code: str | None,
        dm: ArrayF64,
        max_memory: int = 2000,
    ) -> tuple[float, float, ArrayF64]:
        """Restricted Kohn-Sham method, applicable if both spin-densities as equal."""
        assert len(dm.shape) == 2
        result = self(
            mol, grids, xc_code, self._from_backend(dm), max_memory=max_memory
        )
        return (
            result.electron_count.sum().item(),
            result.energy.item(),
            self._to_backend(result.potential),
        )

    def nr_uks(
        self,
        mol: gto.Mole,
        grids: Grid,
        xc_code: str | None,
        dm: ArrayF64,
        max_memory: int = 2000,
    ) -> tuple[ArrayF64, float, ArrayF64]:
        """Unrestricted Kohn-Sham method, spin densities can be different."""
        assert len(dm.shape) == 3 and dm.shape[0] == 2
        result = self(
            mol, grids, xc_code, self._from_backend(dm), max_memory=max_memory
        )
        return (
            self._to_backend(result.electron_count),
            result.energy.item(),
            self._to_backend(result.potential),
        )

    def rsh_and_hybrid_coeff(
        self, xc_code: str | None = None, spin: int = 0
    ) -> tuple[float, float, float]:
        """Return zero range-separation and hybrid coefficients for Skala."""
        return 0, 0, 0

    class libxc:
        __version__: ClassVar[str | None] = None
        __reference__: ClassVar[str | None] = None

        @staticmethod
        def is_hybrid_xc(xc: str) -> bool:
            return False

        @staticmethod
        def is_nlc(xc: str) -> bool:
            return False

    # Overrides PySCF's base with a wider array type for mo_coeff/mo_occ.
    def gen_response(
        self,
        mo_coeff: ArrayF64 | None,
        mo_occ: ArrayF64 | None,
        *,
        ks: KS,
        **kwargs: Any,
    ) -> Callable[[ArrayF64], ArrayF64]:
        """Generate a short-lived response function for the current KS state.

        The returned closure retains the current KS/grid state and XC autograd
        graph. Use it only for the immediate response calculation, then discard
        it; do not store or reuse it across state changes.

        Returns:
            A Hessian-vector product closure for immediate response evaluation.
        """
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

        dm0 = self._from_backend(cast(ArrayF64, ks.make_rdm1(mo_coeff, mo_occ)))
        xc_response = self.integrator.gen_response(
            ks.mol,
            ks.grids,
            dm0,
            max_memory=ks.max_memory,
            safety_fraction=kwargs.get("safety_fraction"),
        )

        def hessian_vector_product(dm1: ArrayF64) -> ArrayF64:
            v1: ArrayF64 = self._to_backend(xc_response(self._from_backend(dm1)))
            vj: ArrayF64 = cast(ArrayF64, ks.get_j(ks.mol, dm1, hermi=1))

            if ks.mol.spin == 0:
                v1[...] += vj
            else:
                v1[...] += vj[0] + vj[1]

            return v1

        return hessian_vector_product
