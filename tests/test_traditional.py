# SPDX-License-Identifier: MIT

import pytest
from pyscf import dft, gto
from pytest import approx

from skala.functional import ExcFunctionalBase, load_functional
from skala.pyscf import SkalaKS


@pytest.fixture(params=["HF", "B", "H"])
def mol(request: pytest.FixtureRequest) -> gto.Mole:
    if request.param == "HF":
        return gto.M(atom="H 0 0 0; F 0 0 1.1", basis="cc-pvdz")
    elif request.param == "B":
        return gto.M(atom="B 0 0 0", basis="cc-pvdz", spin=1)
    elif request.param == "H":
        return gto.M(atom="H 0 0 0", basis="cc-pvdz", spin=1)
    raise AssertionError()


@pytest.fixture(params=["lda", "spw92", "pbe", "tpss"])
def xc(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def xc_fun(xc: str) -> ExcFunctionalBase:
    """Fixture to load the functional."""
    func = load_functional(xc)
    assert isinstance(func, ExcFunctionalBase)
    return func


@pytest.fixture
def xc_str(xc: str) -> str:
    """Fixture to return the functional name as a string."""
    return {
        "lda": "lda,",
        "spw92": "lda,pw",
    }.get(xc, xc)


def test_scf(mol: gto.Mole, xc_str: str, xc_fun: ExcFunctionalBase) -> None:
    ks = dft.KS(mol, xc=xc_str)
    ene_ref = ks.kernel()
    ks = SkalaKS(mol, xc=xc_fun, with_dftd3=False)
    ene = ks.kernel()
    assert ene == approx(ene_ref), ene
