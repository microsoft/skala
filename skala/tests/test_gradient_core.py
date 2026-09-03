# SPDX-License-Identifier: MIT

"""Tests for backend-independent nuclear-gradient operations."""

from typing import cast

import pytest
import torch
from skala.features import Feature, FeatureMap
from skala.functional.base import ExcFunctionalBase
from skala.pyscf import gradient_core
from skala.pyscf.backend import Grid
from skala.pyscf.gradient_core import (
    contract_ao_derivative_block,
    feature_derivatives,
    grid_derivative_block,
)

from pyscf import gto


def test_nuclear_feature_derivatives_propagate_memory_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_budgets: list[int | None] = []
    density = torch.ones(1, dtype=torch.float64)

    class TestFunctional(ExcFunctionalBase):
        features = [Feature.DENSITY]

        def get_exc(self, mol: FeatureMap) -> torch.Tensor:
            return mol[Feature.DENSITY].sum()

    def fake_evaluate_model_features(
        mol: gto.Mole,
        dm: torch.Tensor,
        grid: Grid,
        feature_spec: object,
        max_memory_in_mb: int = 2000,
    ) -> FeatureMap:
        observed_budgets.append(max_memory_in_mb)
        return {
            Feature.DENSITY: density,
            Feature.ATOMIC_GRID_SIZES: torch.ones(1, dtype=torch.long),
        }

    def fake_evaluate_chunked_feature_gradients(
        functional: ExcFunctionalBase,
        dm: torch.Tensor,
        model_features: FeatureMap,
        differentiable_features: set[Feature],
        max_memory_in_mb: int | None = None,
    ) -> FeatureMap:
        observed_budgets.append(max_memory_in_mb)
        return {Feature.DENSITY: density}

    monkeypatch.setattr(
        gradient_core, "evaluate_model_features", fake_evaluate_model_features
    )
    monkeypatch.setattr(
        gradient_core,
        "evaluate_chunked_feature_gradients",
        fake_evaluate_chunked_feature_gradients,
    )

    gradient_core.evaluate_nuclear_feature_derivatives(
        TestFunctional(),
        cast(gto.Mole, object()),
        cast(Grid, object()),
        torch.eye(1, dtype=torch.float64),
        max_memory_in_mb=321,
    )

    assert observed_budgets == [321, 321]


def test_feature_derivatives() -> None:
    density = torch.tensor([1.0, 2.0], dtype=torch.float64)
    weights = torch.tensor([3.0, 4.0], dtype=torch.float64)

    derivatives = feature_derivatives(
        lambda features: (
            features[Feature.DENSITY].square() * features[Feature.GRID_WEIGHTS]
        ).sum(),
        {Feature.DENSITY: density, Feature.GRID_WEIGHTS: weights},
    )

    torch.testing.assert_close(derivatives[Feature.DENSITY], 2 * density * weights)
    torch.testing.assert_close(derivatives[Feature.GRID_WEIGHTS], density.square())
    assert feature_derivatives(lambda _: torch.tensor(0.0), {}) == {}


def test_feature_derivatives_rejects_disconnected_feature() -> None:
    used = torch.tensor(2.0)
    unused = torch.tensor(3.0)

    with pytest.raises(
        RuntimeError,
        match="XC energy is disconnected from requested features: grid_weights",
    ):
        feature_derivatives(
            lambda features: features[Feature.DENSITY].square(),
            {Feature.DENSITY: used, Feature.GRID_WEIGHTS: unused},
        )


def test_feature_derivatives_rejects_constant_energy() -> None:
    feature = torch.tensor(2.0)

    with pytest.raises(
        RuntimeError,
        match="XC energy is disconnected from requested features: density",
    ):
        feature_derivatives(lambda _: torch.tensor(1.0), {Feature.DENSITY: feature})


def test_grid_derivative_block_slices_each_grid_dimension() -> None:
    derivatives = {
        Feature.DENSITY: torch.arange(8).reshape(2, 4),
        Feature.GRAD: torch.arange(24).reshape(2, 3, 4),
        Feature.GRID_COORDS: torch.arange(12).reshape(4, 3),
        Feature.GRID_WEIGHTS: torch.arange(4),
        Feature.COARSE_0_ATOMIC_COORDS: torch.arange(6).reshape(2, 3),
    }

    block = grid_derivative_block(derivatives, grid_start=1, grid_end=3)

    assert set(block) == {
        Feature.DENSITY,
        Feature.GRAD,
        Feature.GRID_COORDS,
        Feature.GRID_WEIGHTS,
    }
    torch.testing.assert_close(
        block[Feature.DENSITY], derivatives[Feature.DENSITY][..., 1:3]
    )
    torch.testing.assert_close(block[Feature.GRAD], derivatives[Feature.GRAD][..., 1:3])
    torch.testing.assert_close(
        block[Feature.GRID_COORDS], derivatives[Feature.GRID_COORDS][1:3]
    )
    torch.testing.assert_close(
        block[Feature.GRID_WEIGHTS], derivatives[Feature.GRID_WEIGHTS][1:3]
    )


def test_contract_ao_derivative_block_matches_reference() -> None:
    ao = torch.tensor(
        [2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0],
        dtype=torch.float64,
    ).reshape(10, 1, 1)
    derivatives = {
        Feature.DENSITY: torch.tensor([[0.0, 31.0], [0.0, 37.0]], dtype=torch.float64),
        Feature.GRAD: torch.tensor(
            [
                [[0.0, 41.0], [0.0, 43.0], [0.0, 47.0]],
                [[0.0, 53.0], [0.0, 59.0], [0.0, 61.0]],
            ],
            dtype=torch.float64,
        ),
        Feature.KIN: torch.tensor([[0.0, 67.0], [0.0, 71.0]], dtype=torch.float64),
        Feature.GRID_COORDS: torch.ones((2, 3), dtype=torch.float64),
        Feature.GRID_WEIGHTS: torch.ones(2, dtype=torch.float64),
    }
    actual = contract_ao_derivative_block(ao, grid_derivative_block(derivatives, 1, 2))
    expected = torch.tensor(
        [
            [[[13074.5]], [[18389.5]], [[23562.5]]],
            [[[15342.5]], [[21673.5]], [[27838.5]]],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == ao.dtype
    assert actual.device == ao.device
