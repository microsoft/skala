# SPDX-License-Identifier: MIT

"""
Extension of GPU4PySCF's Kohn-Sham calculators to support custom functionals.
This module provides a restricted and unrestricted Kohn-Sham method, which extend the
GPU4PySCF Kohn-Sham classes by providing a custom numerical integration method which
mimics the behavior of GPU4PySCF's ``numint`` module.

Examples
--------
>>> from pyscf import gto
>>> from skala.functional import load_functional
>>> from skala.gpu4pyscf import dft
>>> import torch
>>>
>>> mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="def2-svp", verbose=0)
>>> # Create restricted KS calculator
>>> rks = dft.SkalaRKS(mol, xc=load_functional("skala", device=torch.device("cuda:0")))
>>> energy = rks.kernel()
>>> print(energy)  # DOCTEST: Ellipsis
-1.142654...
>>> # Create unrestricted KS calculator
>>> uks = dft.SkalaUKS(mol, xc=load_functional("skala", device=torch.device("cuda:0")))
>>> energy = uks.kernel()
>>> print(energy)  # DOCTEST: Ellipsis
-1.142654...

The `SkalaRKS` and `SkalaUKS` classes can be used in the same way as (GPU4)PySCF's
`dft.rks.RKS <https://pyscf.org/pyscf_api_docs/pyscf.dft.html#pyscf.dft.rks.RKS>`__ and
`dft.uks.UKS <https://pyscf.org/pyscf_api_docs/pyscf.dft.html#pyscf.dft.uks.UKS>`__ classes.
The provided classes support the same transformations and methods as the original (GPU4)PySCF ones:

>>> from pyscf import gto
>>> from skala.functional import load_functional
>>> from skala.gpu4pyscf import dft
>>> import torch
>>>
>>> mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="def2-svp")
>>> ks = dft.SkalaRKS(mol, xc=load_functional("skala", device=torch.device("cuda:0")))
>>> # Apply density fitting
>>> ks = ks.density_fit(auxbasis="def2-svp-jkfit")
>>> ks  # DOCTEST: Ellipsis
<gpu4pyscf.df.df_jk.DFSkalaRKS object at ...>
>>> # Create gradient calculator
>>> ks_grad = ks.nuc_grad_method()
>>> ks_grad  # DOCTEST: Ellipsis
<skala.gpu4pyscf.gradients.SkalaRKSGradient object at ...>
>>> # Create energy scanner
>>> ks_scanner = ks.as_scanner()
>>> ks_scanner  # DOCTEST: Ellipsis
<pyscf.scf.hf.DFSkalaRKS_Scanner object at ...>
"""

import warnings
from collections.abc import Callable
from typing import Any, cast

import cupy as cp
import torch
from dftd3.pyscf import DFTD3Dispersion
from gpu4pyscf import dft
from gpu4pyscf.df import df_jk
from pyscf import __version__ as pyscf_version
from pyscf import gto

# Set the default CuPy memory allocator to avoid memory leak issues
cp.cuda.set_allocator(cp.get_default_memory_pool().malloc)

from skala.functional.base import ExcFunctionalBase
from skala.gpu4pyscf.gradients import SkalaRKSGradient, SkalaUKSGradient
from skala.pyscf.numint import SkalaNumInt
from skala.pyscf.utils import pyscf_version_newer_than_2_10


class SkalaRKS(dft.rks.RKS):  # type: ignore[misc]
    """Restricted Kohn-Sham method with support for Skala functional."""

    with_dftd3: DFTD3Dispersion | None = None
    """DFT-D3 dispersion correction."""

    def __init__(self, mol: gto.Mole, xc: ExcFunctionalBase):
        super().__init__(mol, xc="custom")
        self._keys.add("with_dftd3")
        self._numint = SkalaNumInt(xc, device=torch.device("cuda:0"))

        d3 = xc.get_d3_settings()
        self.with_dftd3 = DFTD3Dispersion(mol, d3) if d3 is not None else None

    def energy_nuc(self) -> float:
        enuc = float(super().energy_nuc())
        if self.with_dftd3:
            edisp = self.with_dftd3.kernel()[0]
            self.scf_summary["dispersion"] = edisp
            enuc += edisp
        return enuc

    def Gradients(self) -> SkalaRKSGradient:
        return SkalaRKSGradient(self)

    def nuc_grad_method(self) -> SkalaRKSGradient:
        return self.Gradients()

    def gen_response(
        self,
        mo_coeff: cp.ndarray | None = None,
        mo_occ: cp.ndarray | None = None,
        **kwargs: dict[str, Any],
    ) -> Callable[[cp.ndarray], cp.ndarray]:
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ

        return self._numint.gen_response(mo_coeff, mo_occ, **kwargs, ks=self)

    def density_fit(
        self,
        auxbasis: str | None = None,
        with_df: bool | None = None,
        only_dfj: bool | None = True,
    ) -> "SkalaRKS":
        ks = df_jk.density_fit(self, auxbasis, with_df, only_dfj)
        ks.Gradients = lambda: SkalaRKSGradient(ks)
        ks.nuc_grad_method = ks.Gradients
        return cast(SkalaRKS, ks)


class SkalaUKS(dft.uks.UKS):  # type: ignore[misc]
    """Unrestricted Kohn-Sham method with support for Skala functional."""

    with_dftd3: DFTD3Dispersion | None = None
    """DFT-D3 dispersion correction."""

    def __init__(self, mol: gto.Mole, xc: ExcFunctionalBase):
        super().__init__(mol, xc="custom")
        self._keys.add("with_dftd3")
        self._numint = SkalaNumInt(xc, device=torch.device("cuda:0"))

        d3 = xc.get_d3_settings()
        self.with_dftd3 = DFTD3Dispersion(mol, d3) if d3 is not None else None

    def energy_nuc(self) -> float:
        enuc = float(super().energy_nuc())
        if self.with_dftd3:
            edisp = self.with_dftd3.kernel()[0]
            self.scf_summary["dispersion"] = edisp
            enuc += edisp
        return enuc

    def Gradients(self) -> SkalaUKSGradient:
        return SkalaUKSGradient(self)

    def nuc_grad_method(self) -> SkalaUKSGradient:
        return self.Gradients()

    def gen_response(
        self,
        mo_coeff: cp.ndarray | None = None,
        mo_occ: cp.ndarray | None = None,
        **kwargs: dict[str, Any],
    ) -> Callable[[cp.ndarray], cp.ndarray]:
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ

        return self._numint.gen_response(mo_coeff, mo_occ, **kwargs, ks=self)

    def density_fit(
        self,
        auxbasis: str | None = None,
        with_df: bool | None = None,
        only_dfj: bool | None = True,
    ) -> "SkalaUKS":
        if pyscf_version_newer_than_2_10() and auxbasis is None:
            warnings.warn(
                "Using density_fit without specifying auxbasis will lead to different behavior in PySCF >= 2.10.0 compared to PySCF 2.9.0, which was used for benchmarking skala. To reproduce benchmarks, please specify an auxbasis (def2-universal-jkfit for (ma-)def2 basis sets).",
            )
        ks = df_jk.density_fit(self, auxbasis, with_df, only_dfj)
        ks.Gradients = lambda: SkalaUKSGradient(ks)
        ks.nuc_grad_method = ks.Gradients
        return cast(SkalaUKS, ks)
