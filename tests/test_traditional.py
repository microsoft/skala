# SPDX-License-Identifier: MIT

import pytest
from pyscf import dft, gto
from pytest import approx
from torch import nn

from skala.functional import ExcFunctionalBase, load_functional
from skala.pyscf import SkalaKS


@pytest.fixture(params=["HF", "Ar", "H"])
def mol(request: pytest.FixtureRequest) -> gto.Mole:
    if request.param == "HF":
        return gto.M(atom="H 0 0 0; F 0 0 1.1", basis="cc-pvdz")
    elif request.param == "Ar":
        return gto.M(atom="Ar 0 0 0", basis="def2-svp")
    elif request.param == "H":
        return gto.M(atom="H 0 0 0", basis="cc-pvdz", spin=1)
    raise AssertionError()


@pytest.fixture(params=["lda", "spw92", "pbe", "tpss", "scan", "rscan", "r2scan"])
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


@pytest.mark.parametrize("xc_name", ["pbe", "tpss", "scan", "rscan", "r2scan"])
def test_parameters(xc_name: str) -> None:
    xc_fun = load_functional(xc_name)
    assert isinstance(xc_fun, ExcFunctionalBase)

    expected_num_params = {
        "pbe": 2,
        "tpss": 6,
        "scan": 22,
        "rscan": 22,
        "r2scan": 22,
    }[xc_name]

    params = list(xc_fun.parameters())
    assert params
    assert len(params) == expected_num_params, (
        f"Expected {expected_num_params} parameters, got {len(params)}"
    )
    assert all(isinstance(param, nn.Parameter) for param in params)
    assert all(not param.requires_grad for param in params)


@pytest.mark.parametrize("xc_name", ["scan", "rscan", "r2scan"])
def test_scan_constants(xc_name: str) -> None:
    xc_fun = load_functional(xc_name)
    assert isinstance(xc_fun, ExcFunctionalBase)

    assert approx(xc_fun.ax) == -0.7385587663820224, (
        f"Expected ax to be -0.7385587663820224, got {xc_fun.ax}"
    )
    assert approx(xc_fun.b1, abs=1e-6) == 0.156632, (
        f"Expected b1 to be 0.156632, got {xc_fun.b1}"
    )
    assert approx(xc_fun.b2, abs=1e-5) == 0.12083, (
        f"Expected b2 to be 0.12083, got {xc_fun.b2}"
    )
    assert approx(xc_fun.b3) == 0.5, f"Expected b3 to be 0.5, got {xc_fun.b3}"
