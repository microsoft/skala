from collections.abc import Callable

import pytest
from pyscf import dft, gto

from skala.functional.base import ExcFunctionalBase
from skala.pyscf import SkalaKS
from skala.pyscf.dft import SkalaRKS, SkalaUKS
from skala.pyscf.gradients import SkalaRKSGradient, SkalaUKSGradient
from skala.pyscf.grids import SkalaGrids


@pytest.fixture(params=["skala-1.0", "skala-1.1"])
def skala_xc(
    request: pytest.FixtureRequest,
    load_functional_cached: Callable[..., ExcFunctionalBase | str],
) -> ExcFunctionalBase:
    """Load the Skala functional under test."""
    func = load_functional_cached(request.param)
    assert isinstance(func, ExcFunctionalBase)
    return func


@pytest.fixture(params=["H", "H2"])
def mol(request: pytest.FixtureRequest) -> gto.Mole:
    if request.param == "H":
        return gto.M(atom="H", basis="sto-3g", spin=1)
    if request.param == "H2":
        return gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    raise ValueError(f"Unknown molecule: {request.param}")


@pytest.fixture(params=["dfj", "no df"])
def with_density_fit(request: pytest.FixtureRequest) -> bool:
    return request.param == "dfj"


@pytest.fixture(params=["soscf", "scf"])
def with_newton(request: pytest.FixtureRequest) -> bool:
    return request.param == "soscf"


@pytest.fixture(params=["d3", "no d3"])
def with_dftd3(request: pytest.FixtureRequest) -> bool:
    return request.param == "d3"


def test_skala_class(
    mol: gto.Mole,
    skala_xc: ExcFunctionalBase,
    with_density_fit: bool,
    with_newton: bool,
    with_dftd3: bool,
) -> None:
    """Test whether classes get correctly preserved."""
    ks = SkalaKS(
        mol,
        xc=skala_xc,
        with_density_fit=with_density_fit,
        auxbasis="def2-universal-jkfit" if with_density_fit else None,
        with_newton=with_newton,
        with_dftd3=with_dftd3,
    )
    assert ks.xc == "custom"
    assert isinstance(ks, SkalaRKS if mol.spin == 0 else SkalaUKS)
    assert ks.with_dftd3 is not None if with_dftd3 else ks.with_dftd3 is None
    assert isinstance(ks.grids, SkalaGrids)

    ks_scanner = ks.as_scanner()
    assert isinstance(ks_scanner, SkalaRKS if mol.spin == 0 else SkalaUKS)
    assert (
        ks_scanner.with_dftd3 is not None
        if with_dftd3
        else ks_scanner.with_dftd3 is None
    )

    grad = ks.nuc_grad_method()
    assert isinstance(grad, SkalaRKSGradient if mol.spin == 0 else SkalaUKSGradient)
    assert grad.with_dftd3 is not None if with_dftd3 else grad.with_dftd3 is None
    assert isinstance(grad.grids, SkalaGrids)

    grad = ks.Gradients()
    assert isinstance(grad, SkalaRKSGradient if mol.spin == 0 else SkalaUKSGradient)
    assert grad.with_dftd3 is not None if with_dftd3 else grad.with_dftd3 is None
    assert isinstance(grad.grids, SkalaGrids)

    ks = grad.base
    assert isinstance(ks, SkalaRKS if mol.spin == 0 else SkalaUKS)
    assert ks.with_dftd3 is not None if with_dftd3 else ks.with_dftd3 is None
    assert isinstance(ks.grids, SkalaGrids)


def test_skala_class_with_dftd3_and_native_functional_raises() -> None:
    """Test that using DFT-D3 with a native PySCF functional raises an error."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    with pytest.raises(
        ValueError, match="DFT-D3 dispersion correction is not supported"
    ):
        SkalaKS(mol, xc="b3lyp", with_dftd3=True)


def test_skala_class_with_native_functional_and_no_dftd3_is_allowed() -> None:
    """Native PySCF functionals should be allowed when DFT-D3 is disabled."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    ks = SkalaKS(mol, xc="b3lyp", with_dftd3=False)
    assert ks.xc == "b3lyp"
    assert not isinstance(ks, (SkalaRKS, SkalaUKS))


def test_initialize_grids_rejects_non_skala_grids(
    skala_xc: ExcFunctionalBase,
) -> None:
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    ks = SkalaRKS(mol, xc=skala_xc)
    ks.grids = dft.gen_grid.Grids(mol)

    with pytest.raises(TypeError, match="SkalaRKS requires .*SkalaGrids"):
        ks.initialize_grids()


def test_skala_grids_require_unit_alignment() -> None:
    mol = gto.M(atom="H", basis="sto-3g", spin=1, verbose=0)
    grids = SkalaGrids(mol)

    assert grids.alignment == 1
    grids.alignment = 1
    with pytest.raises(ValueError, match="alignment must be 1"):
        grids.alignment = 8
