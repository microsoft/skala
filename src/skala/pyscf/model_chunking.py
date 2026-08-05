# SPDX-License-Identifier: MIT

"""Build atom-aligned model feature chunks from globally evaluated raw features.

This module controls how many complete atomic grids are fed through the functional
model at once. Atomic grids are never split because model features may depend on
atom-local shapes and coordinates.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from pyscf import gto
from torch import Tensor

from skala.pyscf import feature_math
from skala.pyscf.backend import Grid
from skala.pyscf.features import get_grid_features
from skala.pyscf.memory_estimators import (
    estimate_global_raw_feature_buffer_memory,
    estimate_max_model_grid_points,
)

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class AtomGridChunk:
    """Matching atom and grid slices for one model evaluation chunk."""

    atom_slice: slice
    grid_slice: slice


def _make_atom_grid_chunks(
    atomic_grid_sizes: Tensor, max_model_grid_points: int
) -> list[AtomGridChunk]:
    """Build atom-aligned slices up to the requested model grid chunk size."""
    if max_model_grid_points < atomic_grid_sizes.max().item():
        raise ValueError(
            "max_model_grid_points must be at least the maximum atomic grid size"
        )

    chunks: list[AtomGridChunk] = []
    atom_start = 0
    grid_start = 0
    chunk_size = 0

    for atom_index, atom_grid_size in enumerate(atomic_grid_sizes):
        chunk_size += atom_grid_size.item()
        if chunk_size > max_model_grid_points:
            chunks.append(
                AtomGridChunk(
                    atom_slice=slice(atom_start, atom_index),
                    grid_slice=slice(
                        grid_start, grid_start + chunk_size - atom_grid_size.item()
                    ),
                )
            )
            atom_start = atom_index
            grid_start += chunk_size - atom_grid_size.item()
            chunk_size = atom_grid_size.item()

    if chunk_size > 0:
        chunks.append(
            AtomGridChunk(
                atom_slice=slice(atom_start, len(atomic_grid_sizes)),
                grid_slice=slice(grid_start, grid_start + chunk_size),
            )
        )

    LOG.debug(
        "Generated %d model chunks of grid sizes: %s",
        len(chunks),
        [chunk.grid_slice.stop - chunk.grid_slice.start for chunk in chunks],
    )
    return chunks


@dataclass(frozen=True)
class ModelFeatureChunk:
    """Chunk-local raw features and the corresponding model input dictionary."""

    grid_slice: slice
    raw_features: Tensor
    model_features: dict[str, Tensor]


@dataclass(frozen=True)
class ModelFeatureChunker:
    """Reusable atom-aligned partition of raw and model features."""

    atom_major_raw_features: Tensor
    grid_features: dict[str, Tensor]
    feature_function: feature_math.MGGAFeatureFunction
    chunk_layouts: list[AtomGridChunk]
    with_spin: bool

    def __iter__(self) -> Iterator[ModelFeatureChunk]:
        """Yield detached raw features paired with atom-aligned model inputs."""
        feature_spec = self.feature_function.feature_spec
        for layout in self.chunk_layouts:
            raw_features = (
                self.atom_major_raw_features[..., layout.grid_slice]
                .detach()
                .requires_grad_()
            )
            model_features: dict[str, Tensor] = {}
            for feature_name in (
                "grid_coords",
                "grid_weights",
                "atomic_grid_weights",
            ):
                if feature_spec.requests(feature_name):
                    model_features[feature_name] = self.grid_features[feature_name][
                        layout.grid_slice
                    ]

            for feature_name in ("coarse_0_atomic_coords", "atomic_grid_sizes"):
                if feature_spec.requests(feature_name):
                    model_features[feature_name] = self.grid_features[feature_name][
                        layout.atom_slice
                    ]

            if feature_spec.requests("atomic_grid_size_bound_shape"):
                max_size = int(model_features["atomic_grid_sizes"].max().item())
                model_features["atomic_grid_size_bound_shape"] = torch.zeros(
                    max_size,
                    0,
                    dtype=torch.long,
                    device=raw_features.device,
                )

            for feature_name, feature in self.feature_function.to_dict(
                raw_features
            ).items():
                model_features[feature_name] = feature_math.maybe_expand_and_divide(
                    feature, not self.with_spin, 2
                )
            yield ModelFeatureChunk(
                grid_slice=layout.grid_slice,
                raw_features=raw_features,
                model_features=model_features,
            )


def prepare_model_feature_chunks(
    mol: gto.Mole,
    dm: Tensor,
    grids: Grid,
    atom_major_raw_features: Tensor,
    feature_function: feature_math.MGGAFeatureFunction,
    func_deriv: int,
    max_memory_in_mb: int | None = None,
    safety_fraction: float = 0.8,
) -> ModelFeatureChunker:
    """Prepare memory-sized, atom-aligned chunks for functional model evaluation."""
    feature_spec = feature_function.feature_spec
    if not feature_spec.supports_screened_evaluation:
        raise ValueError("Atom-aligned model chunking requires 'atomic_grid_sizes'.")

    grid_features = get_grid_features(mol, dm, grids, feature_spec)
    max_model_grid_points = estimate_max_model_grid_points(
        dm=dm,
        deriv=feature_function.deriv,
        max_memory_in_mb=max_memory_in_mb,
        safety_fraction=safety_fraction,
        func_deriv=func_deriv,
        reserved_memory_in_bytes=estimate_global_raw_feature_buffer_memory(
            dm,
            feature_function.nfeats,
            atom_major_raw_features.shape[-1],
            func_deriv,
        ),
    )
    max_atom_grid = int(grid_features["atomic_grid_sizes"].max().item())
    if max_model_grid_points < max_atom_grid:
        LOG.warning(
            "Adjusted model chunk size %d to match the largest atomic grid %d. "
            "Hope for no OOM.",
            max_model_grid_points,
            max_atom_grid,
        )
        max_model_grid_points = max_atom_grid

    return ModelFeatureChunker(
        atom_major_raw_features=atom_major_raw_features,
        grid_features=grid_features,
        feature_function=feature_function,
        chunk_layouts=_make_atom_grid_chunks(
            grid_features["atomic_grid_sizes"], max_model_grid_points
        ),
        with_spin=dm.ndim == 3,
    )
