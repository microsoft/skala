import pytest
import torch
from pyscf import dft, gto
from utils import QuadraticFunctional

from skala.features import Feature, FeatureMap
from skala.pyscf import xc_integrator as xc_integrator_module
from skala.pyscf.xc_integrator import XCIntegrator, XCResult


def test_xc_integrator_returns_tensors_and_xc_only_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the tensor-level integrator contract independently of PySCF NumInt.

    The synthetic features give closed-form electron count, energy, potential, and
    Hessian action values. Checking them here verifies that ``XCIntegrator`` returns
    tensors and that ``gen_response`` contains only the XC Hessian action, without
    the Coulomb response that the higher-level NumInt wrapper adds.
    """
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    grids = dft.Grids(mol)

    def fake_generate_features(
        mol: gto.Mole,
        dm: torch.Tensor,
        grids: object,
        features: set[Feature],
        **kwargs: object,
    ) -> FeatureMap:
        assert features == {Feature.DENSITY, Feature.GRID_WEIGHTS}
        return {
            Feature.DENSITY: dm.sum().reshape(1),
            Feature.GRID_WEIGHTS: torch.tensor([2.0], dtype=dm.dtype),
        }

    monkeypatch.setattr(
        xc_integrator_module,
        "generate_features",
        fake_generate_features,
    )
    integrator = XCIntegrator(QuadraticFunctional([Feature.DENSITY]))
    dm = torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float64)

    result = integrator(mol, grids, dm)
    response = integrator.gen_response(mol, grids, dm.detach().clone())

    assert isinstance(result, XCResult)
    torch.testing.assert_close(result.electron_count, dm.new_tensor(16.0))
    torch.testing.assert_close(result.energy, dm.new_tensor(128.0))
    torch.testing.assert_close(result.potential, torch.full_like(dm, 32.0))
    torch.testing.assert_close(response(torch.ones_like(dm)), torch.full_like(dm, 16.0))
