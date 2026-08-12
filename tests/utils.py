"""Shared functional and route-control helpers for tests."""

from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

from skala.features import Feature, FeatureMap
from skala.functional.base import ExcFunctionalBase
from skala.pyscf import xc_integrator as xc_integrator_module


def require_gpu() -> ModuleType:
    """Require CUDA and import CuPy, skipping when either is unavailable.

    Returns:
        The imported CuPy module.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.", allow_module_level=True)
    return pytest.importorskip("cupy", reason="CuPy is not available.")


class QuadraticFunctional(ExcFunctionalBase):
    """Functional whose energy is a weighted sum of squared AO-derived features."""

    def __init__(
        self,
        features: Iterable[Feature] = (
            Feature.ATOMIC_GRID_SIZES,
            Feature.DENSITY,
            Feature.GRID_WEIGHTS,
        ),
    ) -> None:
        """Initialize the functional with its required model features.

        AO-derived entries contribute quadratic energy terms. Other entries declare
        metadata needed by the evaluation route but do not contribute to the energy.

        Args:
            features: Features required from the model evaluation.

        Raises:
            ValueError: If no AO-derived feature is selected.
        """
        super().__init__()
        self.features = list(features)
        self._quadratic_features = tuple(
            feature
            for feature in self.features
            if feature
            in {
                Feature.DENSITY,
                Feature.GRAD,
                Feature.KIN,
                Feature.LAPL,
            }
        )
        if not self._quadratic_features:
            raise ValueError("At least one AO-derived feature must be selected")

    def get_exc(self, mol: FeatureMap) -> torch.Tensor:
        """Return the grid-integrated quadratic feature energy.

        The vector components of the density gradient are summed before combining
        them with scalar density, kinetic, or Laplacian terms.

        Args:
            mol: Model features keyed by their feature identifiers.

        Returns:
            Scalar exchange-correlation energy.
        """
        grid_weights = mol[Feature.GRID_WEIGHTS]
        quadratic_terms = [
            (
                mol[feature].square().sum(dim=-2)
                if feature is Feature.GRAD
                else mol[feature].square()
            )
            for feature in self._quadratic_features
        ]
        energy_density = torch.stack(quadratic_terms).sum(dim=0)
        return (energy_density * grid_weights).sum()


@contextmanager
def force_ao_screening(
    enabled: bool,
    module: ModuleType = xc_integrator_module,
) -> Generator[None]:
    """Temporarily force the AO-screening route decision.

    Args:
        enabled: Whether calls should select screened AO evaluation.
        module: Module whose ``_should_screen_aos`` decision function is patched.

    Yields:
        Control while the forced decision is active. The previous function is
        restored when the context exits.
    """
    with patch.object(
        module,
        "_should_screen_aos",
        return_value=enabled,
    ):
        yield


FULL_GRAD_REF: Mapping[str, torch.Tensor] = {
    "HF:pbe": torch.tensor(
        [[0.0, 0.0, -1.0283181338840031e-01], [0.0, 0.0, 1.0283181338840475e-01]],
        dtype=torch.float64,
    ),
    "H2O:pbe": torch.tensor(
        [
            [7.3868922411540083e-02, 0.0, 0.0],
            [-3.6934461205758495e-02, 0.0, -1.3005275018782658e-01],
            [-3.6934461205764268e-02, 0.0, 1.3005275018783147e-01],
        ],
        dtype=torch.float64,
    ),
    "H2O+:pbe": torch.tensor(
        [
            [1.3766133501961964e-01, 0.0, 0.0],
            [-6.8830667509800936e-02, 0.0, -1.6302458647600626e-01],
            [-6.8830667509806709e-02, 0.0, 1.6302458647600737e-01],
        ],
        dtype=torch.float64,
    ),
    "HF:skala-1.0": torch.tensor(
        [
            [0.0, 0.0, -0.11766455110756313],
            [0.0, 0.0, 0.11766455110756091],
        ],
        dtype=torch.float64,
    ),
    "H2O:skala-1.0": torch.tensor(
        [
            [0.04761426020567949, 0.0, 0.0],
            [-0.023807130986786884, 0.0, -0.12656276817486223],
            [-0.023807129218868184, 0.0, 0.126562766972401],
        ],
        dtype=torch.float64,
    ),
    "H2O+:skala-1.0": torch.tensor(
        [
            [0.11016447311737299, 0.0, 0.0],
            [-0.055082237334041384, 0.0, -0.15564537931499212],
            [-0.05508223578332139, 0.0, 0.15564537861887162],
        ],
        dtype=torch.float64,
    ),
    "HF:skala-1.1": torch.tensor(
        [
            [0.0, 0.0, -0.11922130029704636],
            [0.0, 0.0, 0.11922130029705125],
        ],
        dtype=torch.float64,
    ),
    "H2O:skala-1.1": torch.tensor(
        [
            [0.05518685428627901, 0.0, 0.0],
            [-0.027593427632945478, 0.0, -0.12591870031741337],
            [-0.027593426653364173, 0.0, 0.12591869974465286],
        ],
        dtype=torch.float64,
    ),
    "H2O+:skala-1.1": torch.tensor(
        [
            [0.11201511304824052, 0.0, 0.0],
            [-0.05600755684684611, 0.0, -0.15729176960843216],
            [-0.056007556201392195, 0.0, 0.15729176919222287],
        ],
        dtype=torch.float64,
    ),
}
