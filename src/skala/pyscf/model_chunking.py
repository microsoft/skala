# SPDX-License-Identifier: MIT

"""Build atom-aligned model feature chunks from globally evaluated raw features.

This module controls how many complete atomic grids are fed through the functional
model at once. Atomic grids are never split because model features may depend on
atom-local shapes and coordinates.
"""

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
from pyscf import gto
from torch import Tensor

from skala.features import Feature, FeatureMap
from skala.pyscf import feature_math
from skala.pyscf.backend import Grid
from skala.pyscf.features import get_grid_features
from skala.pyscf.memory_estimators import (
    estimate_global_raw_feature_buffer_memory,
    estimate_max_gridpoint_chunk_size,
)

LOG = logging.getLogger(__name__)


class AtomGridChunk(NamedTuple):
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


class ModelFeatureChunk(NamedTuple):
    """Chunk-local raw features and the corresponding model input dictionary."""

    grid_slice: slice
    raw_features: Tensor
    model_features: FeatureMap


@dataclass(frozen=True)
class ModelFeatureChunker:
    """Reusable atom-aligned partition of raw and model features."""

    atom_major_raw_features: Tensor
    grid_features: Mapping[Feature, Tensor]
    feature_function: feature_math.MGGAFeatureFunction
    chunk_layouts: Sequence[AtomGridChunk]
    is_spin_polarized: bool

    def __iter__(self) -> Iterator[ModelFeatureChunk]:
        """Yield detached raw features paired with atom-aligned model inputs."""
        feature_spec = self.feature_function.feature_spec
        for layout in self.chunk_layouts:
            raw_features = (
                self.atom_major_raw_features[..., layout.grid_slice]
                .detach()
                .requires_grad_()
            )
            model_features: FeatureMap = {}
            for feature_name in (
                Feature.GRID_COORDS,
                Feature.GRID_WEIGHTS,
                Feature.ATOMIC_GRID_WEIGHTS,
            ):
                if feature_spec.requests(feature_name):
                    model_features[feature_name] = self.grid_features[feature_name][
                        layout.grid_slice
                    ]

            for feature_name in (
                Feature.COARSE_0_ATOMIC_COORDS,
                Feature.ATOMIC_GRID_SIZES,
            ):
                if feature_spec.requests(feature_name):
                    model_features[feature_name] = self.grid_features[feature_name][
                        layout.atom_slice
                    ]

            if feature_spec.requests(Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE):
                max_size = int(model_features[Feature.ATOMIC_GRID_SIZES].max().item())
                model_features[Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE] = torch.zeros(
                    max_size,
                    0,
                    dtype=torch.long,
                    device=raw_features.device,
                )

            for feature_name, feature in self.feature_function.to_dict(
                raw_features
            ).items():
                model_features[feature_name] = feature_math.maybe_expand_and_divide(
                    feature, not self.is_spin_polarized, 2
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
    deriv_order: int,
    max_memory_in_mb: int | None = None,
    safety_fraction: float = 0.8,
) -> ModelFeatureChunker:
    """Prepare memory-sized, atom-aligned chunks for functional model evaluation."""
    feature_spec = feature_function.feature_spec
    if not feature_spec.supports_screened_evaluation:
        raise ValueError(
            f"Atom-aligned model chunking requires {Feature.ATOMIC_GRID_SIZES.value!r}."
        )

    grid_features = get_grid_features(mol, dm, grids, feature_spec)
    max_model_grid_points = estimate_max_gridpoint_chunk_size(
        dm=dm,
        deriv=feature_function.deriv,
        max_memory_in_mb=max_memory_in_mb,
        safety_fraction=safety_fraction,
        func_deriv=deriv_order,
        reserved_memory_in_bytes=estimate_global_raw_feature_buffer_memory(
            dm,
            feature_function.nfeats,
            atom_major_raw_features.shape[-1],
            deriv_order,
        ),
    )
    max_atom_grid = int(grid_features[Feature.ATOMIC_GRID_SIZES].max().item())
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
            grid_features[Feature.ATOMIC_GRID_SIZES], max_model_grid_points
        ),
        is_spin_polarized=dm.ndim == 3,
    )
