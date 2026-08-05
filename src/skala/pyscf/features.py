# SPDX-License-Identifier: MIT

"""
Methods for generating and manipulating density features.
"""

import logging
from copy import copy
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch
from pyscf import dft, gto
from torch import Tensor

from skala.pyscf import ao_evaluation, feature_math
from skala.pyscf.backend import (
    Grid,
    check_gpu_imports_were_successful,
    dft_gpu,
    from_numpy_or_cupy,
)
from skala.pyscf.evaluation import EvaluationPolicy, FeatureSpec
from skala.pyscf.memory_estimators import (
    estimate_global_raw_feature_buffer_memory,
    estimate_max_grid_chunk_size,
)

LOG = logging.getLogger(__name__)

DEFAULT_FEATURES = ["density", "kin", "grad", "grid_coords", "grid_weights"]
DEFAULT_FEATURES_SET = set(DEFAULT_FEATURES)
CPU_AO_SCREENING_BLOCK_SIZE = 9 * dft.gen_grid.BLKSIZE

_Float64Coordinates: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]
_Int64Permutation: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int64]]
_SPATIAL_GRID_CACHE_ATTRIBUTE = "_skala_spatial_grid_cache"


@dataclass(frozen=True)
class _SpatialGridCache:
    mol: gto.Mole
    source_coords: object
    source_weights: object
    block_size: int
    gpu: bool
    sorted_grids: Grid
    forward: _Int64Permutation
    inverse: _Int64Permutation

    def matches(
        self,
        mol: gto.Mole,
        grids: Grid,
        block_size: int,
        gpu: bool,
    ) -> bool:
        """Return whether this entry belongs to the current built grid."""
        return (
            self.mol is mol
            and self.source_coords is grids.coords
            and self.source_weights is grids.weights
            and self.block_size == block_size
            and self.gpu is gpu
        )


def _spatial_grid_permutations(
    coords: _Float64Coordinates, block_size: int
) -> tuple[_Int64Permutation, _Int64Permutation]:
    """Order a molecular grid into exact-size spatial blocks.

    Recursively partitions points along the longest Cartesian extent. Every left
    subtree contains a whole number of evaluator blocks, so all output blocks have
    ``block_size`` points except for a possible final remainder.

    Args:
        coords: Molecular grid coordinates with shape ``(ngrids, 3)``.
        block_size: Fixed number of points consumed by each backend block.

    Returns:
        The forward permutation from atom-major to spatial order and its inverse.

    Raises:
        ValueError: If the coordinates or block size are invalid.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (ngrids, 3)")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    def partition(indices: _Int64Permutation) -> list[_Int64Permutation]:
        if indices.size <= block_size:
            return [indices]

        block_count = (indices.size + block_size - 1) // block_size
        left_size = (block_count // 2) * block_size
        extents = np.ptp(coords[indices], axis=0)
        split_axis = int(np.argmax(extents))
        positions = np.lexsort((indices, coords[indices, split_axis]))
        ordered_indices = indices[positions]
        return partition(ordered_indices[:left_size]) + partition(
            ordered_indices[left_size:]
        )

    ngrids = coords.shape[0]
    if ngrids == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()

    forward = np.concatenate(partition(np.arange(ngrids, dtype=np.int64)))
    inverse = np.empty_like(forward)
    inverse[forward] = np.arange(ngrids, dtype=np.int64)
    return forward, inverse


def _prepare_spatially_sorted_grids(
    mol: gto.Mole,
    grids: Grid,
    block_size: int,
    gpu: bool,
) -> tuple[Grid, _Int64Permutation, _Int64Permutation]:
    """Copy and spatially order a grid for backend AO screening.

    Preparation is cached on the source grid and reused while its coordinate and
    weight arrays, molecule, backend, and evaluator block size remain unchanged.

    Args:
        mol: Molecule used to rebuild CPU shell-screening data.
        grids: Built CPU or GPU integration grid in atom-major order.
        block_size: Fixed number of points consumed by each backend block.
        gpu: Whether ``grids`` belongs to GPU4PySCF.

    Returns:
        The sorted grid copy, atom-major-to-spatial permutation, and inverse.
    """
    if grids.coords is None or grids.weights is None:
        raise ValueError("Grids must be built before spatial sorting.")

    cache = getattr(grids, _SPATIAL_GRID_CACHE_ATTRIBUTE, None)
    if isinstance(cache, _SpatialGridCache) and cache.matches(
        mol, grids, block_size, gpu
    ):
        return cache.sorted_grids, cache.forward, cache.inverse

    if gpu:
        check_gpu_imports_were_successful()
        import cupy

        host_coords = cupy.asnumpy(grids.coords)
    else:
        host_coords = grids.coords

    forward, inverse = _spatial_grid_permutations(host_coords, block_size)
    sorted_grids = copy(grids)
    vars(sorted_grids).pop(_SPATIAL_GRID_CACHE_ATTRIBUTE, None)
    if gpu:
        backend_forward = cupy.asarray(forward)
        sorted_grids.coords = grids.coords[backend_forward]
        sorted_grids.weights = grids.weights[backend_forward]
        sorted_grids._non0ao_idx = None
    else:
        sorted_grids.coords = grids.coords[forward]
        sorted_grids.weights = grids.weights[forward]
        sorted_grids.non0tab = dft.gen_grid.make_screen_index(
            mol,
            sorted_grids.coords,
            cutoff=sorted_grids.cutoff,
        )
    setattr(
        grids,
        _SPATIAL_GRID_CACHE_ATTRIBUTE,
        _SpatialGridCache(
            mol=mol,
            source_coords=grids.coords,
            source_weights=grids.weights,
            block_size=block_size,
            gpu=gpu,
            sorted_grids=sorted_grids,
            forward=forward,
            inverse=inverse,
        ),
    )
    return sorted_grids, forward, inverse


def make_chunks(
    atomic_grid_sizes: Tensor, max_grid_chunk_size: int
) -> list[tuple[slice, slice]]:
    """
    Generate chunks of atomic and grid indices based on the maximum grid chunk size.
    Input:
        atomic_grid_sizes: A tensor of atomic grid sizes.
        max_grid_chunk_size: The maximum size of each grid chunk.
    Returns:
        A list of tuples, where each tuple contains a slice for the atomic indices and a slice for the grid indices.
    """

    if max_grid_chunk_size < atomic_grid_sizes.max().item():
        raise ValueError(
            "max_grid_chunk_size must be at least the maximum atomic grid size"
        )

    atom_and_grid_slices = []
    atom_start = 0
    grid_start = 0
    chunk_size = 0

    for i, atom_grid_size in enumerate(atomic_grid_sizes):
        chunk_size += atom_grid_size.item()
        if chunk_size > max_grid_chunk_size:
            atom_and_grid_slices.append(
                (
                    slice(atom_start, i),
                    slice(grid_start, grid_start + chunk_size - atom_grid_size.item()),
                )
            )
            atom_start = i
            grid_start += chunk_size - atom_grid_size.item()
            chunk_size = atom_grid_size.item()

    if chunk_size > 0:
        atom_and_grid_slices.append(
            (
                slice(atom_start, len(atomic_grid_sizes)),
                slice(grid_start, grid_start + chunk_size),
            )
        )

    LOG.debug(
        f"Generated {len(atom_and_grid_slices)} chunks of grid sizes: {[g.stop - g.start for _, g in atom_and_grid_slices]}"
    )

    return atom_and_grid_slices


def generate_features(
    mol: gto.Mole,
    dm: Tensor,
    grids: Grid,
    features: set[str] | None = None,
    chunk_size: int | None = None,
    max_memory: int = 2000,
    gpu: bool = False,
) -> dict[str, Tensor]:
    """Generate density features for a given molecule. The density features are stored in a dictionary
    with the keys matching the requested features.

    Parameters
    ----------
    mol: gto.Mole
      the molecule
    dm: Tensor
      the density matrix
    grids: Grid
      the grid
    features: set[str] | None
      the requested features
    chunk_size: int | None
        a manually specified chunk size for processing the grids, if None the chunk size is determined automatically
    max_memory: int
      the maximum memory to use for calculating the features
    gpu: bool
        whether to use the GPU(4pyscf) for calculations

    Returns
    -------
    dict[str, Tensor]
        A dictionary containing the requested features. The keys are the feature names,
        and the values are the corresponding tensors.
    """
    feature_spec = FeatureSpec(DEFAULT_FEATURES_SET if features is None else features)
    evaluation_policy = EvaluationPolicy(ao_block_size=chunk_size)

    # if dm is a 3D tensor, then we have a spin-polarized system
    with_spin = len(dm.shape) == 3

    if gpu and dm.device.type != "cuda":
        raise ValueError("Density matrix must be on the GPU when gpu=True.")

    mol_features = get_grid_features(mol, dm, grids, feature_spec)

    if feature_spec.requires_mgga:
        mgga_features = ao_evaluation.auto_chunk(
            dm,
            mol,
            grids,
            feature_math.MGGAFeatureFunction(feature_spec),
            block_size=evaluation_policy.ao_block_size,
            max_memory=max_memory,
            gpu=gpu,
        )

        for feature in mgga_features:
            mol_features[feature] = feature_math.maybe_expand_and_divide(
                mgga_features[feature], not with_spin, 2
            )

    return mol_features


def get_grid_features(
    mol: gto.Mole,
    dm: Tensor,
    grids: Grid,
    feature_spec: FeatureSpec,
) -> dict[str, Tensor]:
    grid_features = {}

    if feature_spec.requests("grid_coords"):
        grid_features["grid_coords"] = from_numpy_or_cupy(
            grids.coords, device=dm.device, dtype=dm.dtype
        )

    if feature_spec.requests("grid_weights"):
        grid_features["grid_weights"] = from_numpy_or_cupy(
            grids.weights, device=dm.device, dtype=dm.dtype
        )

    if feature_spec.requests("coarse_0_atomic_coords"):
        grid_features["coarse_0_atomic_coords"] = from_numpy_or_cupy(
            mol.atom_coords(), device=dm.device, dtype=dm.dtype
        )

    if feature_spec.requires_atomic_layout:
        atom_grids_tab = grids.gen_atomic_grids(
            mol, grids.atom_grid, grids.radi_method, grids.level, grids.prune
        )
        sizes = [len(atom_grids_tab[mol.atom_symbol(ia)][1]) for ia in range(mol.natm)]

        n_atomic = sum(sizes)
        n_grid = grids.weights.shape[0]
        if n_atomic != n_grid:
            raise ValueError(
                f"Grid size mismatch: sum of atomic grid sizes ({n_atomic}) does not match "
                f"total grid points ({n_grid}). This is likely caused by grid alignment padding "
                f"(grids.alignment={getattr(grids, 'alignment', '?')}). "
                f"Set grids.alignment = 1 before building grids to disable padding."
            )

        if feature_spec.requests("atomic_grid_sizes"):
            grid_features["atomic_grid_sizes"] = torch.tensor(
                sizes, dtype=torch.long, device=dm.device
            )

        if feature_spec.requests("atomic_grid_size_bound_shape"):
            max_size = max(sizes)
            grid_features["atomic_grid_size_bound_shape"] = torch.zeros(
                max_size, 0, dtype=torch.long, device=dm.device
            )

        if feature_spec.requests("atomic_grid_weights"):
            raw_weights = np.concatenate(
                [atom_grids_tab[mol.atom_symbol(ia)][1] for ia in range(mol.natm)]
            )
            grid_features["atomic_grid_weights"] = from_numpy_or_cupy(
                raw_weights, device=dm.device, dtype=dm.dtype
            )

    return grid_features


@dataclass
class _GlobalScreenedFeatures:
    dm: Tensor
    mol: gto.Mole
    sorted_grids: Grid
    sorted_raw_features: Tensor
    atom_major_raw_features: Tensor
    forward_permutation: Tensor
    inverse_permutation: Tensor
    feature_function: feature_math.MGGAFeatureFunction
    block_size: int
    compile_feature_function: bool
    gpu: bool
    grid_features: dict[str, Tensor]
    feature_spec: FeatureSpec
    chunks: list[tuple[slice, slice]]
    with_spin: bool

    def atom_major_jvp(self, dm_tangent: Tensor) -> Tensor:
        """Apply the global raw-feature Jacobian and restore atom-major order."""
        sorted_tangent = ao_evaluation.ChunkEvalForward.apply(
            self.dm,
            self.mol,
            self.sorted_grids,
            self.feature_function,
            self.block_size,
            self.compile_feature_function,
            self.gpu,
            dm_tangent,
        )
        return sorted_tangent.index_select(-1, self.inverse_permutation).detach()

    def build_model_chunk(
        self,
        raw_features: Tensor,
        atom_slice: slice,
        grid_slice: slice,
    ) -> dict[str, Tensor]:
        """Build one atom-aligned model dictionary from raw feature values."""
        feature_chunk: dict[str, Tensor] = {}
        for feature_name in ("grid_coords", "grid_weights", "atomic_grid_weights"):
            if self.feature_spec.requests(feature_name):
                feature_chunk[feature_name] = self.grid_features[feature_name][
                    grid_slice
                ]

        for feature_name in ("coarse_0_atomic_coords", "atomic_grid_sizes"):
            if self.feature_spec.requests(feature_name):
                feature_chunk[feature_name] = self.grid_features[feature_name][
                    atom_slice
                ]

        if self.feature_spec.requests("atomic_grid_size_bound_shape"):
            max_size = int(feature_chunk["atomic_grid_sizes"].max().item())
            feature_chunk["atomic_grid_size_bound_shape"] = torch.zeros(
                max_size,
                0,
                dtype=torch.long,
                device=raw_features.device,
            )

        for feature_name, feature in self.feature_function.to_dict(
            raw_features
        ).items():
            feature_chunk[feature_name] = feature_math.maybe_expand_and_divide(
                feature, not self.with_spin, 2
            )
        return feature_chunk


def _global_screened_features(
    mol: gto.Mole,
    dm: Tensor,
    grids: Grid,
    features: FeatureSpec | set[str],
    func_deriv: int,
    max_memory_in_mb: int | None = None,
    safety_fraction: float = 0.8,
    compile_feature_function: bool = False,
) -> _GlobalScreenedFeatures:
    """Evaluate raw AO features once on a spatially ordered molecular grid."""
    feature_spec = (
        features if isinstance(features, FeatureSpec) else FeatureSpec(features)
    )
    if not feature_spec.supports_screened_evaluation:
        raise ValueError(
            "Global screened features require 'atomic_grid_sizes' for model chunks."
        )
    if grids.coords is None or grids.weights is None:
        raise ValueError("Grids must be built before generating screened features.")

    feature_function = feature_math.MGGAFeatureFunction(feature_spec)
    grid_features = get_grid_features(mol, dm, grids, feature_spec)
    max_grid_chunk_size = estimate_max_grid_chunk_size(
        dm=dm,
        deriv=feature_function.deriv,
        max_memory_in_mb=max_memory_in_mb,
        safety_fraction=safety_fraction,
        func_deriv=func_deriv,
        reserved_memory_in_bytes=estimate_global_raw_feature_buffer_memory(
            dm,
            feature_function.nfeats,
            grids.weights.size,
            func_deriv,
        ),
    )
    max_atom_grid = int(grid_features["atomic_grid_sizes"].max().item())
    if max_grid_chunk_size < max_atom_grid:
        LOG.warning(
            f"Adjusted chunk size {max_grid_chunk_size} to match the largest atomic grid "
            f"{max_atom_grid}. Hope for no OOM."
        )
        max_grid_chunk_size = max_atom_grid

    gpu = dm.device.type == "cuda"
    if gpu:
        check_gpu_imports_were_successful()
        block_size = int(dft_gpu.numint.MIN_BLK_SIZE)
    else:
        block_size = CPU_AO_SCREENING_BLOCK_SIZE
    sorted_grids, forward, inverse = _prepare_spatially_sorted_grids(
        mol, grids, block_size, gpu
    )
    sorted_raw_features = ao_evaluation.ChunkEvalForward.apply(
        dm.double(),
        mol,
        sorted_grids,
        feature_function,
        block_size,
        compile_feature_function,
        gpu,
    )
    forward_permutation = torch.as_tensor(forward, device=dm.device)
    inverse_permutation = torch.as_tensor(inverse, device=dm.device)
    atom_major_raw_features = sorted_raw_features.index_select(-1, inverse_permutation)
    chunks = make_chunks(grid_features["atomic_grid_sizes"], max_grid_chunk_size)
    return _GlobalScreenedFeatures(
        dm=dm,
        mol=mol,
        sorted_grids=sorted_grids,
        sorted_raw_features=sorted_raw_features,
        atom_major_raw_features=atom_major_raw_features,
        forward_permutation=forward_permutation,
        inverse_permutation=inverse_permutation,
        feature_function=feature_function,
        block_size=block_size,
        compile_feature_function=compile_feature_function,
        gpu=gpu,
        grid_features=grid_features,
        feature_spec=feature_spec,
        chunks=chunks,
        with_spin=dm.ndim == 3,
    )
