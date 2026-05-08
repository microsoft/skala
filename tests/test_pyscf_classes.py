import pytest
from pyscf import gto

from skala.functional import load_functional
from skala.functional.base import ExcFunctionalBase
from skala.pyscf import SkalaKS
from skala.pyscf.dft import SkalaRKS, SkalaUKS
from skala.pyscf.gradients import SkalaRKSGradient, SkalaUKSGradient


@pytest.fixture(params=["skala-1.0", "skala-1.1"])
def skala_xc(request: pytest.FixtureRequest) -> ExcFunctionalBase:
    """Load the Skala functional under test."""
    func = load_functional(request.param)
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

    grad = ks.Gradients()
    assert isinstance(grad, SkalaRKSGradient if mol.spin == 0 else SkalaUKSGradient)
    assert grad.with_dftd3 is not None if with_dftd3 else grad.with_dftd3 is None

    ks = grad.base
    assert isinstance(ks, SkalaRKS if mol.spin == 0 else SkalaUKS)
    assert ks.with_dftd3 is not None if with_dftd3 else ks.with_dftd3 is None


def test_grid_alignment_mismatch_raises() -> None:
    """generate_features raises ValueError when grid has alignment padding."""
    from unittest.mock import patch

    import torch

    from skala.pyscf.features import generate_features

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    func = load_functional("skala-1.1")

    def _build_grids_keep_padding(grids: gto.Mole, mol: gto.Mole) -> gto.Mole:
        """Build grids WITHOUT disabling alignment, so padding is preserved."""
        grids.build(mol, sort_grids=False)
        return grids

    with patch("skala.pyscf.dft._build_grids_unsorted", _build_grids_keep_padding):
        ks = SkalaKS(mol, xc=func, with_dftd3=False)

    # The default PySCF alignment is 8, so grids may have padding.
    # Force alignment to something large to guarantee a mismatch.
    ks.grids.alignment = 128
    ks.grids.build(mol, sort_grids=False)

    dm = torch.from_numpy(ks.get_init_guess())

    with pytest.raises(ValueError, match="Grid size mismatch"):
        generate_features(mol, dm, ks.grids, set(func.features))
