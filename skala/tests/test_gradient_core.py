# SPDX-License-Identifier: MIT

"""Tests for backend-independent nuclear-gradient operations."""

import torch
from skala.features import Feature
from skala.pyscf.gradient_core import (
    contract_veff_block,
    feature_derivatives,
    grid_derivative_block,
)


def test_feature_derivatives() -> None:
    density = torch.tensor([1.0, 2.0], dtype=torch.float64)
    weights = torch.tensor([3.0, 4.0], dtype=torch.float64)

    derivatives = feature_derivatives(
        lambda rho, grid_weights: (rho.square() * grid_weights).sum(),
        [density, weights],
    )

    torch.testing.assert_close(derivatives[0], 2 * density * weights)
    torch.testing.assert_close(derivatives[1], density.square())
    assert feature_derivatives(lambda: torch.tensor(0.0), []) == ()


def test_feature_derivatives_returns_zero_for_disconnected_feature() -> None:
    used = torch.tensor(2.0)
    unused = torch.tensor(3.0)

    derivatives = feature_derivatives(lambda value, _: value.square(), [used, unused])

    torch.testing.assert_close(derivatives[0], 2 * used)
    torch.testing.assert_close(derivatives[1], torch.zeros_like(unused))


def test_feature_derivatives_returns_zero_for_constant_energy() -> None:
    feature = torch.tensor(2.0)

    (derivative,) = feature_derivatives(lambda _: torch.tensor(1.0), [feature])

    torch.testing.assert_close(derivative, torch.zeros_like(feature))


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


def test_contract_veff_block_matches_reference() -> None:
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
    }
    actual = contract_veff_block(ao, grid_derivative_block(derivatives, 1, 2))
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
