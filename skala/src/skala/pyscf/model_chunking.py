# SPDX-License-Identifier: MIT

"""Build atom-aligned model feature chunks from globally evaluated raw features.

This module controls how many complete atomic grids are fed through the functional
model at once. Atomic grids are never split because model features may depend on
atom-local shapes and coordinates.
"""

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from pyscf import gto
from skala.features import Feature, FeatureMap
from skala.pyscf import feature_math
from skala.pyscf.backend import Grid
from skala.pyscf.features import get_grid_features
from skala.pyscf.memory_estimators import (
    estimate_max_model_atoms_per_chunk,
)

LOG = logging.getLogger(__name__)

_GRID_POINT_FEATURES = (
    Feature.GRID_COORDS,
    Feature.GRID_WEIGHTS,
    Feature.ATOMIC_GRID_WEIGHTS,
)
_ATOM_FEATURES = (
    Feature.COARSE_0_ATOMIC_COORDS,
    Feature.ATOMIC_GRID_SIZES,
)


@dataclass(frozen=True)
class AtomGridChunk:
    """Matching atom and grid slices for one model evaluation chunk."""

    atom_slice: slice
    grid_slice: slice


def _make_atom_grid_chunks(
    atomic_grid_sizes: Tensor, max_atoms_per_grid_size: Mapping[int, int]
) -> list[AtomGridChunk]:
    """Pack equal-sized atomic grids up to each size group's atom limit."""
    if any(max_atoms < 1 for max_atoms in max_atoms_per_grid_size.values()):
        raise ValueError("max_atoms_per_grid_size values must be positive")

    chunks: list[AtomGridChunk] = []
    grid_sizes = [int(size) for size in atomic_grid_sizes.tolist()]
    atom_start = 0
    grid_start = 0
    while atom_start < len(grid_sizes):
        atom_grid_size = grid_sizes[atom_start]
        group_stop = atom_start + 1
        while group_stop < len(grid_sizes) and grid_sizes[group_stop] == atom_grid_size:
            group_stop += 1

        atoms_per_chunk = max_atoms_per_grid_size[atom_grid_size]
        for chunk_atom_start in range(atom_start, group_stop, atoms_per_chunk):
            chunk_atom_stop = min(chunk_atom_start + atoms_per_chunk, group_stop)
            chunk_grid_size = (chunk_atom_stop - chunk_atom_start) * atom_grid_size
            chunks.append(
                AtomGridChunk(
                    atom_slice=slice(chunk_atom_start, chunk_atom_stop),
                    grid_slice=slice(grid_start, grid_start + chunk_grid_size),
                )
            )
            grid_start += chunk_grid_size
        atom_start = group_stop

    LOG.debug(
        "Generated %d homogeneous model chunks of grid sizes: %s",
        len(chunks),
        [chunk.grid_slice.stop - chunk.grid_slice.start for chunk in chunks],
    )
    return chunks


@dataclass(frozen=True)
class AtomGridOrder:
    """Atom and grid-point indices in ascending atomic-grid-size order."""

    atom_indices: Tensor
    grid_indices: Tensor


def _make_atom_grid_order(atomic_grid_sizes: Tensor) -> AtomGridOrder:
    """Build a stable atom ordering and its matching complete grid-block ordering."""
    atom_indices = torch.argsort(atomic_grid_sizes, stable=True)
    sorted_sizes = atomic_grid_sizes.index_select(0, atom_indices)
    total_grid_points = int(atomic_grid_sizes.sum().item())

    original_starts = atomic_grid_sizes.cumsum(0) - atomic_grid_sizes
    sorted_starts = sorted_sizes.cumsum(0) - sorted_sizes
    point_atom_indices = torch.repeat_interleave(
        atom_indices, sorted_sizes, output_size=total_grid_points
    )
    point_sorted_starts = torch.repeat_interleave(
        sorted_starts, sorted_sizes, output_size=total_grid_points
    )
    grid_indices = (
        original_starts.index_select(0, point_atom_indices)
        + torch.arange(total_grid_points, device=atomic_grid_sizes.device)
        - point_sorted_starts
    )
    return AtomGridOrder(atom_indices=atom_indices, grid_indices=grid_indices)


@dataclass(frozen=True)
class ModelFeatureChunk:
    """Chunk-local raw features and the corresponding model input dictionary."""

    grid_indices: Tensor
    raw_features: Tensor
    model_features: FeatureMap


class ModelFeatureChunker:
    """Prepared atom-aligned partition of raw and model features.

    The chunker is a snapshot of the supplied density matrix, grid, and raw
    features. It can be iterated repeatedly while those inputs represent the
    same calculation, but must not be reused after their state changes.
    """

    def __init__(
        self,
        mol: gto.Mole,
        dm: Tensor,
        grids: Grid,
        atom_major_raw_features: Tensor,
        feature_function: feature_math.MGGAFeatureFunction,
        deriv_order: int,
        max_memory_in_mb: int | None = None,
        safety_fraction: float = 0.8,
    ) -> None:
        feature_spec = feature_function.feature_spec
        if not feature_spec.supports_spatial_decomposition:
            raise ValueError(
                f"Atom-aligned model chunking requires "
                f"{Feature.ATOMIC_GRID_SIZES.value!r}."
            )

        grid_features = get_grid_features(mol, dm, grids, feature_spec)
        atomic_grid_sizes = grid_features[Feature.ATOMIC_GRID_SIZES]
        atom_grid_order = _make_atom_grid_order(atomic_grid_sizes)
        sorted_atomic_grid_sizes = atomic_grid_sizes.index_select(
            0, atom_grid_order.atom_indices
        )

        max_atoms_per_grid_size = estimate_max_model_atoms_per_chunk(
            dm=dm,
            atomic_grid_sizes=sorted_atomic_grid_sizes,
            nfeatures=feature_function.nfeats,
            max_memory_in_mb=max_memory_in_mb,
            safety_fraction=safety_fraction,
            func_deriv=deriv_order,
        )
        for grid_size, max_atoms in max_atoms_per_grid_size.items():
            if max_atoms < 1:
                LOG.warning(
                    "Adjusted model chunk capacity for atomic grid size %d from %d "
                    "to one atom. Hope for no OOM.",
                    grid_size,
                    max_atoms,
                )
                max_atoms_per_grid_size[grid_size] = 1

        self._atom_major_raw_features = atom_major_raw_features
        self._grid_features = grid_features
        self._feature_function = feature_function
        self._chunk_layouts = _make_atom_grid_chunks(
            sorted_atomic_grid_sizes, max_atoms_per_grid_size
        )
        self._atom_order = atom_grid_order.atom_indices
        self._grid_order = atom_grid_order.grid_indices
        self._is_spin_polarized = dm.ndim == 3

    def __iter__(self) -> Iterator[ModelFeatureChunk]:
        """Yield detached raw features paired with atom-aligned model inputs."""
        feature_spec = self._feature_function.feature_spec
        for layout in self._chunk_layouts:
            atom_indices = self._atom_order[layout.atom_slice]
            grid_indices = self._grid_order[layout.grid_slice]
            raw_features = (
                self._atom_major_raw_features.index_select(-1, grid_indices)
                .detach()
                .requires_grad_()
            )
            model_features: FeatureMap = {}
            for feature_name in _GRID_POINT_FEATURES:
                if feature_spec.requests(feature_name):
                    model_features[feature_name] = self._grid_features[
                        feature_name
                    ].index_select(0, grid_indices)

            for feature_name in _ATOM_FEATURES:
                if feature_spec.requests(feature_name):
                    model_features[feature_name] = self._grid_features[
                        feature_name
                    ].index_select(0, atom_indices)

            if feature_spec.requests(Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE):
                max_size = int(model_features[Feature.ATOMIC_GRID_SIZES].max().item())
                model_features[Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE] = torch.zeros(
                    max_size,
                    0,
                    dtype=torch.long,
                    device=raw_features.device,
                )

            for feature_name, feature in self._feature_function.to_dict(
                raw_features
            ).items():
                model_features[feature_name] = feature_math.maybe_expand_and_divide(
                    feature, not self._is_spin_polarized, 2
                )
            yield ModelFeatureChunk(
                grid_indices=grid_indices,
                raw_features=raw_features,
                model_features=model_features,
            )
