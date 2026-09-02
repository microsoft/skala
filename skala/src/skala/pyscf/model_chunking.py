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
from skala.functional.base import ExcFunctionalBase
from skala.pyscf import ao_evaluation, feature_math
from skala.pyscf.backend import Grid
from skala.pyscf.evaluation import FeatureSpec
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
_AO_DERIVED_FEATURES = (
    Feature.DENSITY,
    Feature.GRAD,
    Feature.KIN,
    Feature.LAPL,
)


def evaluate_model_features(
    mol: gto.Mole,
    dm: Tensor,
    grids: Grid,
    feature_spec: FeatureSpec,
    max_memory_in_mb: int = 2000,
) -> FeatureMap:
    """Evaluate a model's named features without running the model."""
    model_features = get_grid_features(mol, dm, grids, feature_spec)
    if not feature_spec.requires_ao_evaluation:
        return model_features

    feature_function = feature_math.MGGAFeatureFunction(feature_spec)
    raw_features = ao_evaluation.evaluate_raw_features_auto_chunk(
        dm,
        mol,
        grids,
        feature_function,
        block_size=None,
        max_memory=max_memory_in_mb,
        gpu=dm.device.type == "cuda",
    )
    is_spin_polarized = dm.ndim == 3
    for feature_name, feature_value in feature_function.to_dict(raw_features).items():
        model_features[feature_name] = feature_math.maybe_expand_and_divide(
            feature_value, not is_spin_polarized, 2
        )
    return model_features


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
class ModelChunkIndices:
    """Original atom and grid-point indices for one model evaluation."""

    atom_indices: Tensor
    grid_indices: Tensor


def _make_model_chunk_indices(
    dm: Tensor,
    atomic_grid_sizes: Tensor,
    nfeatures: int,
    deriv_order: int,
    max_memory_in_mb: int | None,
    safety_fraction: float,
) -> list[ModelChunkIndices]:
    """Build memory-sized homogeneous chunks in a stable atom ordering."""
    atom_grid_order = _make_atom_grid_order(atomic_grid_sizes)
    sorted_atomic_grid_sizes = atomic_grid_sizes.index_select(
        0, atom_grid_order.atom_indices
    )
    max_atoms_per_grid_size = estimate_max_model_atoms_per_chunk(
        dm=dm,
        atomic_grid_sizes=sorted_atomic_grid_sizes,
        nfeatures=nfeatures,
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

    return [
        ModelChunkIndices(
            atom_indices=atom_grid_order.atom_indices[layout.atom_slice],
            grid_indices=atom_grid_order.grid_indices[layout.grid_slice],
        )
        for layout in _make_atom_grid_chunks(
            sorted_atomic_grid_sizes, max_atoms_per_grid_size
        )
    ]


@dataclass(frozen=True)
class ModelFeatureChunk:
    """Chunk-local raw features and the corresponding model input dictionary."""

    grid_indices: Tensor
    raw_features: Tensor
    model_features: FeatureMap


@dataclass(frozen=True)
class ModelFeaturePlan:
    """Describe how packed AO features become functional model inputs.

    The raw feature function defines the channel layout of the packed tensor
    produced by AO evaluation and decodes those channels into named features.
    The model feature specification defines the smaller public feature surface
    passed to the functional. These specifications may differ because integration
    can request bookkeeping features, such as atomic grid sizes for chunk layout,
    that the functional itself must not receive.

    Attributes:
        raw_feature_function: Decoder matching the packed AO feature tensor.
        model_feature_spec: Features exposed to each functional invocation.
    """

    raw_feature_function: feature_math.MGGAFeatureFunction
    model_feature_spec: FeatureSpec


class ModelFeatureChunker:
    """Prepare atom-aligned model inputs from globally evaluated AO features.

    ``atom_major_raw_features`` must use the packed channel layout described by
    ``feature_plan.raw_feature_function`` and must arrange complete atomic grids
    contiguously in molecular atom order. The chunker groups equal-sized atomic
    grids, chooses a memory-limited number of atoms per model invocation, decodes
    each packed slice, and exposes only ``feature_plan.model_feature_spec`` to the
    functional. Atomic grids are never split.

    Iteration yields detached chunk-local raw tensors requiring gradients so model
    cotangents can be assembled and propagated through the global AO evaluation.
    The chunker is a snapshot and may only be reused while its density matrix,
    grid, and raw features still describe the same calculation. When spatial
    decomposition is unsupported, it yields one full-grid chunk to preserve
    non-additive model semantics.
    """

    def __init__(
        self,
        mol: gto.Mole,
        dm: Tensor,
        grids: Grid,
        atom_major_raw_features: Tensor,
        feature_plan: ModelFeaturePlan,
        deriv_order: int,
        max_memory_in_mb: int | None = None,
        safety_fraction: float = 0.8,
    ) -> None:
        feature_function = feature_plan.raw_feature_function
        feature_spec = feature_function.feature_spec
        grid_features = get_grid_features(mol, dm, grids, feature_spec)
        atomic_grid_sizes = grid_features[Feature.ATOMIC_GRID_SIZES]
        if feature_plan.model_feature_spec.supports_spatial_decomposition:
            self._chunk_indices = _make_model_chunk_indices(
                dm,
                atomic_grid_sizes,
                nfeatures=feature_function.nfeats,
                deriv_order=deriv_order,
                max_memory_in_mb=max_memory_in_mb,
                safety_fraction=safety_fraction,
            )
        else:
            self._chunk_indices = [
                ModelChunkIndices(
                    atom_indices=torch.arange(mol.natm, device=dm.device),
                    grid_indices=torch.arange(
                        int(atomic_grid_sizes.sum().item()), device=dm.device
                    ),
                )
            ]

        self._atom_major_raw_features = atom_major_raw_features
        self._grid_features = grid_features
        self._feature_function = feature_function
        self._model_feature_spec = feature_plan.model_feature_spec
        self._is_spin_polarized = dm.ndim == 3

    def __iter__(self) -> Iterator[ModelFeatureChunk]:
        """Yield detached raw features paired with atom-aligned model inputs."""
        feature_spec = self._model_feature_spec
        for chunk_indices in self._chunk_indices:
            atom_indices = chunk_indices.atom_indices
            grid_indices = chunk_indices.grid_indices
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
                if feature_spec.requests(feature_name):
                    model_features[feature_name] = feature_math.maybe_expand_and_divide(
                        feature, not self._is_spin_polarized, 2
                    )
            yield ModelFeatureChunk(
                grid_indices=grid_indices,
                raw_features=raw_features,
                model_features=model_features,
            )


def evaluate_chunked_feature_gradients(
    functional: ExcFunctionalBase,
    dm: Tensor,
    model_features: FeatureMap,
    differentiable_features: set[Feature],
    max_memory_in_mb: int | None = None,
    safety_fraction: float = 0.8,
) -> FeatureMap:
    """Evaluate and assemble model gradients from precomputed feature chunks."""
    atomic_grid_sizes = model_features[Feature.ATOMIC_GRID_SIZES]
    feature_spec = FeatureSpec(functional.features)
    if not feature_spec.supports_spatial_decomposition:
        full_inputs = {
            feature_name: model_features[feature_name].detach().requires_grad_()
            for feature_name in differentiable_features
        }
        if not full_inputs:
            return {}
        full_features = {
            feature_name: feature_value
            for feature_name, feature_value in model_features.items()
            if feature_spec.requests(feature_name)
        } | full_inputs
        local_gradients = torch.autograd.grad(
            functional.get_exc(full_features), tuple(full_inputs.values())
        )
        return {
            feature_name: gradient.detach()
            for feature_name, gradient in zip(full_inputs, local_gradients, strict=True)
        }

    chunk_indices = _make_model_chunk_indices(
        dm,
        atomic_grid_sizes,
        nfeatures=feature_spec.mgga_feature_count,
        deriv_order=1,
        max_memory_in_mb=max_memory_in_mb,
        safety_fraction=safety_fraction,
    )
    gradients = {
        feature_name: torch.zeros_like(model_features[feature_name])
        for feature_name in differentiable_features
    }
    if not differentiable_features:
        return gradients

    for indices in chunk_indices:
        chunk_features: FeatureMap = {}
        for feature_name, feature_value in model_features.items():
            if not feature_spec.requests(feature_name):
                continue
            if feature_name in _AO_DERIVED_FEATURES:
                chunk_features[feature_name] = feature_value.index_select(
                    -1, indices.grid_indices
                )
            elif feature_name in _GRID_POINT_FEATURES:
                chunk_features[feature_name] = feature_value.index_select(
                    0, indices.grid_indices
                )
            elif feature_name in _ATOM_FEATURES:
                chunk_features[feature_name] = feature_value.index_select(
                    0, indices.atom_indices
                )
            elif feature_name is Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE:
                max_grid_size = int(
                    atomic_grid_sizes.index_select(0, indices.atom_indices).max().item()
                )
                chunk_features[feature_name] = torch.zeros(
                    max_grid_size,
                    0,
                    dtype=torch.long,
                    device=feature_value.device,
                )
            else:
                raise ValueError(f"Unsupported model feature: {feature_name}")

        chunk_inputs = []
        for feature_name in differentiable_features:
            local_input = chunk_features[feature_name].detach().requires_grad_()
            chunk_features[feature_name] = local_input
            chunk_inputs.append(local_input)

        energy_chunk = functional.get_exc(chunk_features)
        local_gradients = torch.autograd.grad(energy_chunk, tuple(chunk_inputs))
        for feature_name, local_gradient in zip(
            differentiable_features, local_gradients, strict=True
        ):
            if feature_name in _AO_DERIVED_FEATURES:
                gradients[feature_name].index_copy_(
                    -1, indices.grid_indices, local_gradient.detach()
                )
            elif feature_name in _GRID_POINT_FEATURES:
                gradients[feature_name].index_copy_(
                    0, indices.grid_indices, local_gradient.detach()
                )
            elif feature_name in _ATOM_FEATURES:
                gradients[feature_name].index_copy_(
                    0, indices.atom_indices, local_gradient.detach()
                )
            else:
                raise ValueError(
                    f"Unsupported differentiable model feature: {feature_name}"
                )

    return gradients
