import pytest
from pyscf import gto

from skala.pyscf import SkalaKS
from skala.pyscf.dft import SkalaRKS, SkalaUKS
from skala.pyscf.gradients import SkalaRKSGradient, SkalaUKSGradient


@pytest.fixture(params=["H", "H2"])
def mol(request) -> gto.Mole:
    if request.param == "H":
        return gto.M(atom="H", basis="sto-3g", spin=1)
    if request.param == "H2":
        return gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    raise ValueError(f"Unknown molecule: {request.param}")


@pytest.fixture(params=["dfj", "no df"])
def with_density_fit(request) -> bool:
    return request.param == "dfj"


@pytest.fixture(params=["soscf", "scf"])
def with_newton(request) -> bool:
    return request.param == "soscf"


@pytest.fixture(params=["d3", "no d3"])
def with_dftd3(request) -> bool:
    return request.param == "d3"


def test_skala_class(
    mol: gto.Mole, with_density_fit: bool, with_newton: bool, with_dftd3: bool
):
    """Test whether classes get correctly preserved."""
    ks = SkalaKS(
        mol,
        xc="skala",
        with_density_fit=with_density_fit,
        with_newton=with_newton,
        with_dftd3=with_dftd3,
    )
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
