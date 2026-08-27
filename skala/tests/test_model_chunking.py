# SPDX-License-Identifier: MIT

from typing import cast

import pytest
import torch
from skala.features import Feature, FeatureMap
from skala.pyscf import model_chunking
from skala.pyscf.backend import Grid
from skala.pyscf.evaluation import FeatureSpec
from skala.pyscf.feature_math import MGGAFeatureFunction

from pyscf import gto


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

    feature_spec = FeatureSpec(
        {
            Feature.DENSITY,
            Feature.GRID_COORDS,
            Feature.GRID_WEIGHTS,
            Feature.ATOMIC_GRID_WEIGHTS,
            Feature.COARSE_0_ATOMIC_COORDS,
            Feature.ATOMIC_GRID_SIZES,
            Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE,
        }
    )
    raw_features = point_ids.reshape(1, -1)
    chunker = model_chunking.ModelFeatureChunker(
        mol=cast(gto.Mole, object()),
        dm=torch.eye(1, dtype=torch.float64),
        grids=cast(Grid, object()),
        atom_major_raw_features=raw_features,
        feature_function=MGGAFeatureFunction(feature_spec),
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
