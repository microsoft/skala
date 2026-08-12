"""Shared functional and route-control helpers for tests."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import patch

import torch

from skala.features import Feature, FeatureMap
from skala.functional.base import ExcFunctionalBase
from skala.pyscf import xc_integrator as xc_integrator_module


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
def patch_ao_screening(
    enabled: bool,
    module: ModuleType = xc_integrator_module,
) -> Iterator[None]:
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
