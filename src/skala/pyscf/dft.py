# SPDX-License-Identifier: MIT

"""
Extension of PySCF's Kohn-Sham calculators to support custom functionals.
This module provides a restricted and unrestricted Kohn-Sham method, which extend the
PySCF Kohn-Sham classes by providing a custom numerical integration method which
mimics the behavior of PySCF's ``numint`` module.

Examples
--------
>>> from pyscf import gto
>>> from skala.functional import load_functional
>>> from skala.pyscf import dft
>>>
>>> mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="def2-svp", verbose=0)
>>> func = load_functional("skala")
>>> # Create restricted KS calculator
>>> rks = dft.SkalaRKS(mol, xc=func)
>>> energy = rks.kernel()
>>> print(energy)  # DOCTEST: Ellipsis
-1.142654...
>>> # Create unrestricted KS calculator
>>> uks = dft.SkalaUKS(mol, xc=func)
>>> energy = uks.kernel()
>>> print(energy)  # DOCTEST: Ellipsis
-1.142654...

The `SkalaRKS` and `SkalaUKS` classes can be used in the same way as PySCF's
`dft.rks.RKS <https://pyscf.org/pyscf_api_docs/pyscf.dft.html#pyscf.dft.rks.RKS>`__ and
`dft.uks.UKS <https://pyscf.org/pyscf_api_docs/pyscf.dft.html#pyscf.dft.uks.UKS>`__ classes.
The provided classes support the same transformations and methods as the original PySCF ones:

>>> from pyscf import gto
>>> from skala.functional import load_functional
>>> from skala.pyscf import dft
>>>
>>> mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="def2-svp")
>>> ks = dft.SkalaRKS(mol, xc=load_functional("skala"))
>>> # Apply density fitting
>>> ks = ks.density_fit()
>>> ks  # DOCTEST: Ellipsis
<pyscf.df.df_jk.DFSkalaRKS object at ...>
>>> # Create gradient calculator
>>> ks_grad = ks.nuc_grad_method()
>>> ks_grad  # DOCTEST: Ellipsis
<skala.pyscf.gradients.SkalaRKSGradient object at ...>
>>> # Create energy scanner
>>> ks_scanner = ks.as_scanner()
>>> ks_scanner  # DOCTEST: Ellipsis
<pyscf.scf.hf.DFSkalaRKS_Scanner object at ...>
"""

from collections.abc import Callable

import numpy as np
from dftd3.pyscf import DFTD3Dispersion
from pyscf import dft, gto

from skala.functional.base import ExcFunctionalBase
from skala.pyscf.gradients import SkalaRKSGradient, SkalaUKSGradient
from skala.pyscf.numint import SkalaNumInt


class SkalaRKS(dft.rks.RKS):
    """Restricted Kohn-Sham method with support for Skala functional."""

    with_dftd3: DFTD3Dispersion | None = None
    """DFT-D3 dispersion correction."""

    def __init__(self, mol: gto.Mole, xc: ExcFunctionalBase):
        super().__init__(mol, xc="custom")
        self._keys.add("with_dftd3")
        self._numint = SkalaNumInt(xc)

        d3 = xc.get_d3_settings()
        self.with_dftd3 = DFTD3Dispersion(mol, d3) if d3 is not None else None

    def energy_nuc(self) -> float:
        enuc = super().energy_nuc()
        if self.with_dftd3:
            edisp = self.with_dftd3.kernel()[0]
            self.scf_summary["dispersion"] = edisp
            enuc += edisp
        return enuc

    def Gradients(self) -> SkalaRKSGradient:
        return SkalaRKSGradient(self)

    def nuc_grad_method(self) -> SkalaRKSGradient:
        return self.Gradients()

    def gen_response(self, *args, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
        if hasattr(self, "_numint") and hasattr(self._numint, "gen_response"):
            return self._numint.gen_response(*args, **kwargs, ks=self)
        else:
            return super().gen_response(*args, **kwargs)

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=True):
        import pyscf.df.df_jk

        ks = pyscf.df.df_jk.density_fit(self, auxbasis, with_df, only_dfj)
        ks.Gradients = lambda: SkalaRKSGradient(ks)
        ks.nuc_grad_method = ks.Gradients
        return ks


class SkalaUKS(dft.uks.UKS):
    """Unrestricted Kohn-Sham method with support for Skala functional."""

    with_dftd3: DFTD3Dispersion | None = None
    """DFT-D3 dispersion correction."""

    def __init__(self, mol: gto.Mole, xc: ExcFunctionalBase):
        super().__init__(mol, xc="custom")
        self._keys.add("with_dftd3")
        self._numint = SkalaNumInt(xc)

        d3 = xc.get_d3_settings()
        self.with_dftd3 = DFTD3Dispersion(mol, d3) if d3 is not None else None

    def energy_nuc(self) -> float:
        enuc = super().energy_nuc()
        if self.with_dftd3:
            edisp = self.with_dftd3.kernel()[0]
            self.scf_summary["dispersion"] = edisp
            enuc += edisp
        return enuc

    def Gradients(self) -> SkalaUKSGradient:
        return SkalaUKSGradient(self)

    def nuc_grad_method(self) -> SkalaUKSGradient:
        return self.Gradients()

    def gen_response(self, *args, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
        if hasattr(self, "_numint") and hasattr(self._numint, "gen_response"):
            return self._numint.gen_response(*args, **kwargs, ks=self)
        else:
            return super().gen_response(*args, **kwargs)

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=True):
        import pyscf.df.df_jk

        ks = pyscf.df.df_jk.density_fit(self, auxbasis, with_df, only_dfj)
        ks.Gradients = lambda: SkalaUKSGradient(ks)
        ks.nuc_grad_method = ks.Gradients
        return ks
