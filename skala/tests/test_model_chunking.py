# SPDX-License-Identifier: MIT

from typing import cast

import pytest
import torch
from skala.features import Feature, FeatureMap
from skala.functional.base import ExcFunctionalBase
from skala.pyscf import model_chunking
from skala.pyscf.backend import Grid
from skala.pyscf.evaluation import FeatureSpec
from skala.pyscf.feature_math import feature_derivatives

from pyscf import gto


def test_model_feature_plan_rejects_features_missing_from_evaluation() -> None:
    with pytest.raises(
        ValueError, match="Model features missing from evaluation: grid_weights"
    ):
        model_chunking.ModelFeaturePlan(
            evaluation_feature_spec=FeatureSpec([Feature.DENSITY]),
            model_feature_spec=FeatureSpec([Feature.DENSITY, Feature.GRID_WEIGHTS]),
        )


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


def test_model_feature_chunker_sorts_complete_atomic_grids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atomic_grid_sizes = torch.tensor([3, 1, 2, 1])
    point_ids = torch.arange(7, dtype=torch.float64)
    atom_ids = torch.arange(4, dtype=torch.float64)
    grid_features: FeatureMap = {
        Feature.GRID_COORDS: point_ids[:, None].expand(-1, 3),
        Feature.GRID_WEIGHTS: point_ids + 10,
        Feature.ATOMIC_GRID_WEIGHTS: point_ids + 20,
        Feature.COARSE_0_ATOMIC_COORDS: atom_ids[:, None].expand(-1, 3),
        Feature.ATOMIC_GRID_SIZES: atomic_grid_sizes,
        Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE: torch.zeros(3, 0, dtype=torch.long),
    }

    def fake_get_grid_features(*args: object, **kwargs: object) -> FeatureMap:
        return grid_features

    def fake_estimate_max_model_atoms_per_chunk(
        **kwargs: object,
    ) -> dict[int, int]:
        return {1: 2, 2: 2, 3: 1}

    monkeypatch.setattr(model_chunking, "get_grid_features", fake_get_grid_features)
    monkeypatch.setattr(
        model_chunking,
        "estimate_max_model_atoms_per_chunk",
        fake_estimate_max_model_atoms_per_chunk,
    )

    features = {
        Feature.DENSITY,
        Feature.GRID_COORDS,
        Feature.GRID_WEIGHTS,
        Feature.ATOMIC_GRID_WEIGHTS,
        Feature.COARSE_0_ATOMIC_COORDS,
        Feature.ATOMIC_GRID_SIZES,
        Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE,
    }
    evaluation_feature_spec = FeatureSpec(features)
    feature_spec = FeatureSpec(features)
    raw_features = point_ids.reshape(1, -1)
    chunker = model_chunking.ModelFeatureChunker(
        mol=cast(gto.Mole, object()),
        dm=torch.eye(1, dtype=torch.float64),
        grids=cast(Grid, object()),
        atom_major_raw_features=raw_features,
        feature_plan=model_chunking.ModelFeaturePlan(
            evaluation_feature_spec=evaluation_feature_spec,
            model_feature_spec=feature_spec,
        ),
        deriv_order=1,
    )

    chunks = list(chunker)
    assert len(chunks) == 3
    assert torch.equal(chunks[0].grid_indices, torch.tensor([3, 6]))
    assert torch.equal(chunks[0].raw_features.flatten(), torch.tensor([3.0, 6.0]))
    assert torch.equal(
        chunks[0].model_features[Feature.ATOMIC_GRID_SIZES], torch.tensor([1, 1])
    )
    assert torch.equal(
        chunks[0].model_features[Feature.COARSE_0_ATOMIC_COORDS][:, 0],
        torch.tensor([1.0, 3.0]),
    )
    assert torch.equal(chunks[1].grid_indices, torch.tensor([4, 5]))
    assert torch.equal(chunks[2].grid_indices, torch.tensor([0, 1, 2]))


def test_model_feature_chunker_builds_bound_shape_from_internal_grid_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atomic_grid_sizes = torch.tensor([3, 1, 2, 1])

    monkeypatch.setattr(
        model_chunking,
        "get_grid_features",
        lambda *args, **kwargs: {Feature.ATOMIC_GRID_SIZES: atomic_grid_sizes},
    )

    evaluation_feature_spec = FeatureSpec(
        {
            Feature.DENSITY,
            Feature.ATOMIC_GRID_SIZES,
            Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE,
        }
    )
    chunker = model_chunking.ModelFeatureChunker(
        mol=cast(gto.Mole, type("FakeMole", (), {"natm": 4})()),
        dm=torch.eye(1, dtype=torch.float64),
        grids=cast(Grid, object()),
        atom_major_raw_features=torch.arange(7, dtype=torch.float64).reshape(1, -1),
        feature_plan=model_chunking.ModelFeaturePlan(
            evaluation_feature_spec=evaluation_feature_spec,
            model_feature_spec=FeatureSpec({Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE}),
        ),
        deriv_order=1,
    )

    (chunk,) = tuple(chunker)
    assert set(chunk.model_features) == {Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE}
    assert chunk.model_features[Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE].shape == (3, 0)


def test_chunked_feature_gradients_match_unchunked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atomic_grid_sizes = torch.tensor([3, 1, 2, 1])
    point_ids = torch.arange(7, dtype=torch.float64)
    atom_ids = torch.arange(4, dtype=torch.float64)
    differentiable_features: FeatureMap = {
        Feature.DENSITY: point_ids.reshape(1, -1).requires_grad_(),
        Feature.GRID_WEIGHTS: (point_ids + 10).requires_grad_(),
        Feature.COARSE_0_ATOMIC_COORDS: atom_ids[:, None]
        .expand(-1, 3)
        .clone()
        .requires_grad_(),
    }
    model_features: FeatureMap = {
        **differentiable_features,
        Feature.ATOMIC_GRID_SIZES: atomic_grid_sizes,
    }

    class TestFunctional(ExcFunctionalBase):
        features = list(model_features)

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def get_exc(self, mol: FeatureMap) -> torch.Tensor:
            self.calls += 1
            return (
                mol[Feature.DENSITY].square() * mol[Feature.GRID_WEIGHTS]
            ).sum() + mol[Feature.COARSE_0_ATOMIC_COORDS].square().sum()

    monkeypatch.setattr(
        model_chunking,
        "estimate_max_model_atoms_per_chunk",
        lambda **kwargs: {1: 2, 2: 1, 3: 1},
    )
    functional = TestFunctional()
    reference = torch.autograd.grad(
        functional.get_exc(model_features), tuple(differentiable_features.values())
    )

    actual = model_chunking.evaluate_chunked_feature_gradients(
        functional,
        dm=torch.eye(1, dtype=torch.float64),
        model_features=model_features,
        differentiable_features=set(differentiable_features),
        max_memory_in_mb=100,
    )

    for feature_name, reference_gradient in zip(
        differentiable_features, reference, strict=True
    ):
        torch.testing.assert_close(actual[feature_name], reference_gradient)
    assert functional.calls == 4


@pytest.mark.parametrize("supports_spatial_decomposition", [False, True])
@pytest.mark.parametrize(
    ("constant_energy", "disconnected_features"),
    [(False, "grid_weights"), (True, "density, grid_weights")],
)
def test_chunked_feature_gradients_reject_disconnected_features(
    monkeypatch: pytest.MonkeyPatch,
    supports_spatial_decomposition: bool,
    constant_energy: bool,
    disconnected_features: str,
) -> None:
    density = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    grid_weights = torch.tensor([4.0, 5.0], dtype=torch.float64)
    model_features: FeatureMap = {
        Feature.DENSITY: density,
        Feature.GRID_WEIGHTS: grid_weights,
        Feature.ATOMIC_GRID_SIZES: torch.tensor([2]),
    }

    class TestFunctional(ExcFunctionalBase):
        features = [Feature.DENSITY, Feature.GRID_WEIGHTS]
        if supports_spatial_decomposition:
            features.append(Feature.ATOMIC_GRID_SIZES)

        def get_exc(self, mol: FeatureMap) -> torch.Tensor:
            if constant_energy:
                return mol[Feature.DENSITY].new_tensor(1.0)
            return mol[Feature.DENSITY].square().sum()

    monkeypatch.setattr(
        model_chunking,
        "estimate_max_model_atoms_per_chunk",
        lambda **kwargs: {2: 1},
    )

    with pytest.raises(
        RuntimeError,
        match=f"XC energy is disconnected from requested features: {disconnected_features}",
    ):
        model_chunking.evaluate_chunked_feature_gradients(
            TestFunctional(),
            dm=torch.eye(1, dtype=torch.float64),
            model_features=model_features,
            differentiable_features={Feature.DENSITY, Feature.GRID_WEIGHTS},
        )


def test_atom_grid_chunks_pack_equal_sizes_up_to_cap() -> None:
    chunks = model_chunking._make_atom_grid_chunks(
        torch.tensor([2, 2, 2, 2, 2]), max_atoms_per_grid_size={2: 2}
    )

    assert chunks == [
        model_chunking.AtomGridChunk(slice(0, 2), slice(0, 4)),
        model_chunking.AtomGridChunk(slice(2, 4), slice(4, 8)),
        model_chunking.AtomGridChunk(slice(4, 5), slice(8, 10)),
    ]


def test_atom_grid_chunks_apply_limits_per_grid_size() -> None:
    chunks = model_chunking._make_atom_grid_chunks(
        torch.tensor([1, 1, 1, 2, 2, 2]),
        max_atoms_per_grid_size={1: 3, 2: 1},
    )

    assert chunks == [
        model_chunking.AtomGridChunk(slice(0, 3), slice(0, 3)),
        model_chunking.AtomGridChunk(slice(3, 4), slice(3, 5)),
        model_chunking.AtomGridChunk(slice(4, 5), slice(5, 7)),
        model_chunking.AtomGridChunk(slice(5, 6), slice(7, 9)),
    ]
