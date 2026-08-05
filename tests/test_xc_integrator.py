from typing import Any, cast

import numpy as np
import pytest
import torch
from pyscf import dft, gto

from skala.functional.base import ExcFunctionalBase
from skala.pyscf import xc_integrator as xc_integrator_module
from skala.pyscf.numint import SkalaNumInt
from skala.pyscf.xc_integrator import XCIntegrator, XCResult


class QuadraticDensityFunctional(ExcFunctionalBase):
    def __init__(self) -> None:
        super().__init__()
        self.features = ["density"]

    def get_exc(self, mol: dict[str, torch.Tensor]) -> torch.Tensor:
        return (mol["density"].square() * mol["grid_weights"]).sum()


def test_xc_integrator_returns_tensors_and_xc_only_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    grids = dft.Grids(mol)

    def fake_generate_features(
        mol: gto.Mole,
        dm: torch.Tensor,
        grids: object,
        features: set[str],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        assert features == {"density", "grid_weights"}
        return {
            "density": dm.sum().reshape(1),
            "grid_weights": torch.tensor([2.0], dtype=dm.dtype),
        }

    monkeypatch.setattr(
        xc_integrator_module,
        "generate_features",
        fake_generate_features,
    )
    integrator = XCIntegrator(QuadraticDensityFunctional())
    dm = torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float64)

    result = integrator(mol, grids, dm)
    response = integrator.gen_response(mol, grids, dm.detach().clone())

    assert isinstance(result, XCResult)
    torch.testing.assert_close(result.electron_count, dm.new_tensor(16.0))
    torch.testing.assert_close(result.energy, dm.new_tensor(128.0))
    torch.testing.assert_close(result.potential, torch.full_like(dm, 32.0))
    torch.testing.assert_close(response(torch.ones_like(dm)), torch.full_like(dm, 16.0))


class FakeKS:
    def __init__(self, mol: gto.Mole) -> None:
        self.mol = mol
        self.grids = dft.Grids(mol)
        self.max_memory = 123

    def make_rdm1(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        return np.eye(self.mol.nao_nr())

    def get_j(self, mol: gto.Mole, dm: np.ndarray, hermi: int) -> np.ndarray:
        assert hermi == 1
        return np.full_like(dm, 3.0)


class FakeXCIntegrator:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.calls: list[tuple[int, float | None]] = []

    def gen_response(
        self,
        mol: gto.Mole,
        grids: object,
        dm0: torch.Tensor,
        max_memory: int,
        safety_fraction: float | None,
    ) -> Any:
        self.calls.append((max_memory, safety_fraction))
        return lambda dm1: 2 * dm1


def test_numint_response_adds_coulomb_to_xc_response() -> None:
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    ks = FakeKS(mol)
    numint = SkalaNumInt(QuadraticDensityFunctional())
    fake_integrator = FakeXCIntegrator()
    numint.integrator = cast(XCIntegrator, fake_integrator)

    response = numint.gen_response(
        np.eye(mol.nao_nr()),
        np.ones(mol.nao_nr()),
        ks=cast(Any, ks),
        safety_fraction=0.6,
    )

    np.testing.assert_allclose(response(np.ones((mol.nao_nr(), mol.nao_nr()))), 5.0)
    assert fake_integrator.calls == [(123, 0.6)]
