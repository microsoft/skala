# SPDX-License-Identifier: MIT

"""
GPU4PySCF integration for *Skala* functional.

This module provides seamless integration between Skala exchange-correlation
functionals and the GPU4PySCF quantum chemistry package, enabling DFT calculations
with neural network-based functionals.
"""

from importlib.util import find_spec
from typing import Any

import torch

if not torch.cuda.is_available() and find_spec("pytest") is not None:
    import pytest

    pytest.skip(
        "Skipping gpu4pyscf doctests, because CUDA is not available.",
        allow_module_level=True,
    )

try:
    import cupy  # noqa: F401
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "CuPy is not installed. Please install it with `pip install cupy` or `conda install cupy`."
    ) from e

try:
    import gpu4pyscf  # noqa: F401
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "GPU4PySCF is not installed. Please install it with `pip install gpu4pyscf`."
    ) from e

# Reset the default CuPy memory allocator to avoid memory leak issues
# that seem to arise when combining the custom allocator of gpu4pyscf with DLPack usage.
cupy.cuda.set_allocator(cupy.get_default_memory_pool().malloc)

from pyscf import gto

from skala.functional import ExcFunctionalBase, load_functional
from skala.gpu4pyscf import dft


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
    >>> from skala.gpu4pyscf import SkalaKS
    >>>
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="def2-svp")
    >>> ks = SkalaKS(mol, xc="skala")
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
        xc = load_functional(xc, device=torch.device("cuda:0"))
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
    xc: ExcFunctionalBase,
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
    >>> from skala.functional import load_functional
    >>> from skala.gpu4pyscf import SkalaRKS
    >>> import torch
    >>>
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="def2-svp")
    >>> ks = SkalaRKS(mol, xc=load_functional("skala", device=torch.device("cuda:0")), with_density_fit=True)(verbose=0)
    >>> ks  # DOCTEST: Ellipsis
    <gpu4pyscf.df.df_jk.DFSkalaRKS object at ...>
    >>> energy = ks.kernel()
    >>> print(energy)  # DOCTEST: Ellipsis
    -1.142773...
    """
    if isinstance(xc, str):
        xc = load_functional(xc, device=torch.device("cuda:0"))
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
    xc: ExcFunctionalBase,
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
    >>> from skala.functional import load_functional
    >>> from skala.gpu4pyscf import SkalaUKS
    >>> import torch
    >>>
    >>> mol = gto.M(atom="H", basis="def2-svp", spin=1)
    >>> ks = SkalaUKS(mol, xc=load_functional("skala", device=torch.device("cuda:0")), with_density_fit=True, auxbasis="def2-svp-jkfit")(verbose=0)
    >>> ks  # DOCTEST: Ellipsis
    <gpu4pyscf.df.df_jk.DFSkalaUKS object at ...>
    >>> energy = ks.kernel()
    >>> print(energy)  # DOCTEST: Ellipsis
    -0.499031...
    """
    if isinstance(xc, str):
        xc = load_functional(xc, device=torch.device("cuda:0"))
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
