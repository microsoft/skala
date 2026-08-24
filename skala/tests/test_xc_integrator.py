import pytest
import torch
from pyscf import dft, gto
from skala.features import Feature, FeatureMap
from skala.pyscf import xc_integrator as xc_integrator_module
from skala.pyscf.grids import SkalaGrids
from skala.pyscf.xc_integrator import XCIntegrator, XCResult

from tests.utils import QuadraticFunctional, force_ao_screening


def test_screened_xc_derivatives_match_finite_differences() -> None:
    """Validate the screened first- and second-order XC derivatives numerically.

    The symmetric density matrix is varied along one symmetric direction as
    ``D(t) = D + t P``. The centered energy slope is compared with the analytic
    directional derivative ``<Vxc(D), P>``, checking that the potential returned by
    ``XCIntegrator`` is the derivative of the XC energy with respect to the density
    matrix.

    The same two perturbed integrations also give a centered derivative of Vxc. Its
    ``(0, 0)`` element is compared with the corresponding element of the analytic
    Hessian action ``H(D) P`` returned by ``gen_response``. Checking one component
    exercises the second-order path without constructing or finite-differencing the
    full density-matrix Hessian.

    AO screening is forced so both comparisons cover the custom linear AO autograd
    operators. Density, gradient, and kinetic features exercise the meta-GGA paths.
    Because those raw features are linear in D and ``QuadraticFunctional`` is
    quadratic in the features, the energy is quadratic in D and Vxc is linear; the
    centered differences are therefore exact apart from floating-point roundoff.
    """
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    grids = SkalaGrids(mol)
    grids.level = 0
    grids.alignment = 1
    grids.build(sort_grids=False)
    functional = QuadraticFunctional(
        [
            Feature.ATOMIC_GRID_SIZES,
            Feature.DENSITY,
            Feature.GRAD,
            Feature.KIN,
            Feature.GRID_WEIGHTS,
        ]
    )
    integrator = XCIntegrator(functional)
    dm = torch.tensor([[1.0, 0.2], [0.2, 0.8]], dtype=torch.float64)
    direction = torch.tensor([[0.3, -0.2], [-0.2, 0.1]], dtype=dm.dtype)
    step = 1e-4

    with force_ao_screening(True):
        result = integrator(mol, grids, dm)
        response = integrator.gen_response(mol, grids, dm.clone())
        plus = integrator(mol, grids, dm + step * direction)
        minus = integrator(mol, grids, dm - step * direction)
        hessian_action = response(direction)

    energy_slope = (plus.energy - minus.energy) / (2 * step)
    potential_directional_derivative = torch.sum(result.potential * direction)
    torch.testing.assert_close(
        energy_slope,
        potential_directional_derivative,
        rtol=1e-9,
        atol=1e-9,
    )

    potential_slope = (plus.potential[0, 0] - minus.potential[0, 0]) / (2 * step)
    torch.testing.assert_close(
        potential_slope,
        hessian_action[0, 0],
        rtol=1e-9,
        atol=1e-9,
    )


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
    grids = SkalaGrids(mol)

    def fake_generate_features(
        mol: gto.Mole,
        dm: torch.Tensor,
        grids: object,
        features: set[Feature],
        **kwargs: object,
    ) -> FeatureMap:
        return {
            Feature.DENSITY: dm.sum().reshape(1),
            Feature.GRID_WEIGHTS: torch.tensor([2.0], dtype=dm.dtype),
        }

    monkeypatch.setattr(
        xc_integrator_module,
        "generate_features",
        fake_generate_features,
    )
    integrator = XCIntegrator(QuadraticFunctional())
    dm = torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float64)

    result = integrator(mol, grids, dm)
    response = integrator.gen_response(mol, grids, dm.detach().clone())

    assert isinstance(result, XCResult)
    torch.testing.assert_close(result.electron_count, dm.new_tensor(16.0))
    torch.testing.assert_close(result.energy, dm.new_tensor(128.0))
    torch.testing.assert_close(result.potential, torch.full_like(dm, 32.0))
    torch.testing.assert_close(response(torch.ones_like(dm)), torch.full_like(dm, 16.0))


def test_xc_integrator_requires_skala_grids() -> None:
    """Check that the integrator rejects non-Skala grids."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    grids = dft.Grids(mol)
    integrator = XCIntegrator(QuadraticFunctional())
    dm = torch.eye(mol.nao_nr(), dtype=torch.float64)

    with pytest.raises(TypeError, match=r"XC evaluation requires .*\.SkalaGrids"):
        integrator(mol, grids, dm)
    with pytest.raises(TypeError, match=r"XC evaluation requires .*\.SkalaGrids"):
        integrator.gen_response(mol, grids, dm)
