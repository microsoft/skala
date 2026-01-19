# SPDX-License-Identifier: MIT

"""
PySCF integration for *Skala* functional.

This module provides seamless integration between Skala exchange-correlation
functionals and the PySCF quantum chemistry package, enabling DFT calculations
with neural network-based functionals.
"""

try:
    import pyscf  # noqa: F401
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "PySCF is not installed. Please install it with `pip install pyscf` or `conda install pyscf`."
    ) from e

from typing import Any

from pyscf import gto

from skala.functional import ExcFunctionalBase, load_functional
from skala.pyscf import dft


def SkalaKS(
    mol: gto.Mole,
    xc: ExcFunctionalBase | str,
    *,
    with_density_fit: bool = False,
    with_newton: bool = False,
    with_dftd3: bool = True,
    auxbasis: str | None = None,
    ks_config: dict[str, Any] | None = None,
    soscf_config: dict[str, Any] | None = None,
) -> dft.SkalaRKS | dft.SkalaUKS:
    """
    Create a Kohn-Sham calculator for the Skala functional.

    Parameters
    ----------
    mol : gto.Mole
        The PySCF molecule object.
    xc : ExcFunctionalBase or str
        The exchange-correlation functional to use. Can be a string (name of the functional) or an instance of `ExcFunctionalBase`.
    with_density_fit : bool, optional
        Whether to use density fitting. Default is False.
    with_newton : bool, optional
        Whether to use Newton's method for convergence. Default is False.
    with_dftd3 : bool, optional
        Whether to apply DFT-D3 dispersion correction. Default is True.
    auxbasis : str, optional
        Auxiliary basis set to use for density fitting. Default is None.
    ks_config : dict, optional
        Additional configuration options for the Kohn-Sham calculator. Default is None.
    soscf_config : dict, optional
        Additional configuration options for the second-order SCF (SOSCF) method. Default is None.

    Returns
    -------
    dft.SkalaRKS or dft.SkalaUKS
        The Kohn-Sham calculator object.

    Example
    -------
    >>> from pyscf import gto
    >>> from skala.functional import load_functional
    >>> from skala.pyscf import SkalaKS
    >>>
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="def2-svp")
    >>> ks = SkalaKS(mol, xc=load_functional("skala"))
    >>> ks = ks.density_fit(auxbasis="def2-svp-jkfit")  # Optional: use density fitting
    >>> ks = ks.set(verbose=0)
    >>> energy = ks.kernel()
    >>> print(energy)  # DOCTEST: Ellipsis
    -1.142773...
    >>> ks = ks.nuc_grad_method()
    >>> gradient = ks.kernel()
    >>> print(abs(gradient).mean())  # DOCTEST: Ellipsis
    0.029477...
    """
    if isinstance(xc, str):
        xc = load_functional(xc)
    if mol.spin == 0:
        return SkalaRKS(
            mol,
            xc,
            with_density_fit=with_density_fit,
            with_newton=with_newton,
            with_dftd3=with_dftd3,
            auxbasis=auxbasis,
            ks_config=ks_config,
            soscf_config=soscf_config,
        )
    else:
        return SkalaUKS(
            mol,
            xc,
            with_density_fit=with_density_fit,
            with_newton=with_newton,
            with_dftd3=with_dftd3,
            auxbasis=auxbasis,
            ks_config=ks_config,
            soscf_config=soscf_config,
        )


def SkalaRKS(
    mol: gto.Mole,
    xc: ExcFunctionalBase | str,
    *,
    with_density_fit: bool = False,
    with_newton: bool = False,
    with_dftd3: bool = True,
    auxbasis: str | None = None,
    ks_config: dict[str, Any] | None = None,
    soscf_config: dict[str, Any] | None = None,
) -> dft.SkalaRKS:
    """
    Create a restricted Kohn-Sham calculator for the Skala functional.

    Parameters
    ----------
    mol : gto.Mole
        The PySCF molecule object.
    xc : ExcFunctionalBase or str
        The exchange-correlation functional to use. Can be a string (name of the functional) or an instance of `ExcFunctionalBase`.
    with_density_fit : bool, optional
        Whether to use density fitting. Default is False.
    with_newton : bool, optional
        Whether to use Newton's method for convergence. Default is False.
    with_dftd3 : bool, optional
        Whether to apply DFT-D3 dispersion correction. Default is True.
    auxbasis : str, optional
        Auxiliary basis set to use for density fitting. Default is None.
    ks_config : dict, optional
        Additional configuration options for the Kohn-Sham calculator. Default is None.
    soscf_config : dict, optional
        Additional configuration options for the second-order SCF (SOSCF) method. Default is None.

    Returns
    -------
    dft.SkalaRKS
        The Kohn-Sham calculator object.

    Example
    -------
    >>> from pyscf import gto
    >>> from skala.pyscf import SkalaRKS
    >>>
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="def2-svp")
    >>> ks = SkalaRKS(mol, xc="skala", with_density_fit=True, auxbasis="def2-svp-jkfit")(verbose=0)
    >>> ks  # DOCTEST: Ellipsis
    <pyscf.df.df_jk.DFSkalaRKS object at ...>
    >>> energy = ks.kernel()
    >>> print(energy)  # DOCTEST: Ellipsis
    -1.142773...
    """
    if isinstance(xc, str):
        xc = load_functional(xc)
    ks = dft.SkalaRKS(mol, xc)

    if ks_config is not None:
        ks = ks(**ks_config)

    if not with_dftd3:
        ks.with_dftd3 = None

    if with_density_fit:
        ks = ks.density_fit(auxbasis=auxbasis)
    else:
        if auxbasis is not None:
            raise ValueError(
                "Auxiliary basis can only be set when density fitting is enabled."
            )

    if with_newton:
        ks = ks.newton()
        if soscf_config is not None:
            ks.__dict__.update(soscf_config)

    return ks


def SkalaUKS(
    mol: gto.Mole,
    xc: ExcFunctionalBase | str,
    *,
    with_density_fit: bool = False,
    with_newton: bool = False,
    with_dftd3: bool = True,
    auxbasis: str | None = None,
    ks_config: dict[str, Any] | None = None,
    soscf_config: dict[str, Any] | None = None,
) -> dft.SkalaUKS:
    """
    Create an unrestricted Kohn-Sham calculator for the Skala functional.

    Parameters
    ----------
    mol : gto.Mole
        The PySCF molecule object.
    xc : ExcFunctionalBase or str
        The exchange-correlation functional to use. Can be a string (name of the functional) or an instance of `ExcFunctionalBase`.
    with_density_fit : bool, optional
        Whether to use density fitting. Default is False.
    with_newton : bool, optional
        Whether to use Newton's method for convergence. Default is False.
    with_dftd3 : bool, optional
        Whether to apply DFT-D3 dispersion correction. Default is True.
    auxbasis : str, optional
        Auxiliary basis set to use for density fitting. Default is None.
    ks_config : dict, optional
        Additional configuration options for the Kohn-Sham calculator. Default is None.
    soscf_config : dict, optional
        Additional configuration options for the second-order SCF (SOSCF) method. Default is None.

    Returns
    -------
    dft.SkalaUKS
        The Kohn-Sham calculator object.

    Example
    -------
    >>> from pyscf import gto
    >>> from skala.pyscf import SkalaUKS
    >>>
    >>> mol = gto.M(atom="H", basis="def2-svp", spin=1)
    >>> ks = SkalaUKS(mol, xc="skala", with_density_fit=True, auxbasis="def2-svp-jkfit")(verbose=0)
    >>> ks  # DOCTEST: Ellipsis
    <pyscf.df.df_jk.DFSkalaUKS object at ...>
    >>> energy = ks.kernel()
    >>> print(energy)  # DOCTEST: Ellipsis
    -0.499031...
    """
    if isinstance(xc, str):
        xc = load_functional(xc)
    ks = dft.SkalaUKS(mol, xc)

    if ks_config is not None:
        ks = ks(**ks_config)

    if not with_dftd3:
        ks.with_dftd3 = None

    if with_density_fit:
        ks = ks.density_fit(auxbasis=auxbasis)
    else:
        if auxbasis is not None:
            raise ValueError(
                "Auxiliary basis can only be set when density fitting is enabled."
            )

    if with_newton:
        ks = ks.newton()
        if soscf_config is not None:
            ks.__dict__.update(soscf_config)

    return ks
