# SPDX-License-Identifier: MIT

"""Extend PySCF and GPU4PySCF grids for screened Skala evaluation.

Skala's AO evaluator benefits from spatially local grid blocks, while PySCF and
GPU4PySCF provide integration grids in atom-major order with backend-specific AO
screening metadata. This module is the extension layer interposed between those
backend-owned grid objects and Skala's feature evaluation. It deliberately avoids
subclassing either grid implementation so the same screened path can serve both.

Grid preparation attaches a :class:`SpatialGridLayout` cache to the source grid.
The source coordinate and weight arrays retain their order, but the extra cache
attribute modifies the grid object. A shallow grid copy receives spatially reordered
coordinates and weights. For PySCF, its ``non0tab`` shell-screening table is rebuilt;
for GPU4PySCF, ``_non0ao_idx`` is cleared so the backend can rebuild it for the new
order. The cached forward and inverse permutations bridge spatial AO evaluation and
the atom-major layout expected by model features.

:func:`prepare_screened_feature_buffer` orchestrates this grid extension and one
global raw-feature evaluation. The returned :class:`ScreenedFeatureBuffer` retains
the spatial ordering needed for Jacobian and adjoint AO passes. Atom-aligned model
batching is a separate process owned by :mod:`skala.pyscf.model_chunking`.
"""

from copy import copy
from dataclasses import dataclass
from typing import Generic, TypeAlias, cast

import numpy as np
import torch
from pyscf import dft, gto
from torch import Tensor

from skala.pyscf import ao_evaluation, feature_math
from skala.pyscf.backend import (
    Array,
    Grid,
    check_gpu_imports_were_successful,
)
from skala.pyscf.evaluation import FeatureSpec

CPU_AO_SCREENING_BLOCK_SIZE = 9 * dft.gen_grid.BLKSIZE

_Float64Coordinates: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]
_Int64Permutation: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int64]]
_SPATIAL_GRID_CACHE_ATTRIBUTE = "_skala_spatial_grid_cache"


@dataclass(frozen=True)
class SpatialGridLayout(Generic[Array]):
    """Cached spatial ordering derived from an atom-major integration grid."""

    mol: gto.Mole
    source_coords: Array
    source_weights: Array
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
        """Return whether this layout belongs to the current built grid."""
        return (
            self.mol is mol
            and self.source_coords is grids.coords
            and self.source_weights is grids.weights
            and self.block_size == block_size
            and self.gpu is gpu
        )


def _decompose_grid_into_spatial_blocks(
    coords: _Float64Coordinates, block_size: int
) -> tuple[_Int64Permutation, _Int64Permutation]:
    """Decompose a molecular grid into spatial blocks and return its permutations.

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
    if isinstance(cache, SpatialGridLayout) and cache.matches(
        mol, grids, block_size, gpu
    ):
        return cache.sorted_grids, cache.forward, cache.inverse

    if gpu:
        check_gpu_imports_were_successful()
        import cupy

        host_coords = cast(_Float64Coordinates, cupy.asnumpy(grids.coords))
    else:
        host_coords = cast(_Float64Coordinates, grids.coords)

    forward, inverse = _decompose_grid_into_spatial_blocks(host_coords, block_size)
    sorted_grids = copy(grids)
    vars(sorted_grids).pop(_SPATIAL_GRID_CACHE_ATTRIBUTE, None)
    if gpu:
        import cupy

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
        SpatialGridLayout(
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


@dataclass
class ScreenedFeatureBuffer:
    """Globally screened raw features and transformations between grid orders."""

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

    def atom_major_jvp(self, dm_tangent: Tensor) -> Tensor:
        """Apply the global raw-feature Jacobian and restore atom-major order."""
        sorted_tangent = cast(
            Tensor,
            ao_evaluation.ChunkEvalForward.apply(
                self.dm,
                self.mol,
                self.sorted_grids,
                self.feature_function,
                self.block_size,
                self.compile_feature_function,
                self.gpu,
                dm_tangent,
            ),
        )
        return sorted_tangent.index_select(-1, self.inverse_permutation).detach()


def prepare_screened_feature_buffer(
    mol: gto.Mole,
    dm: Tensor,
    grids: Grid,
    features: FeatureSpec | set[str],
    compile_feature_function: bool = False,
) -> ScreenedFeatureBuffer:
    """Prepare the backend grid extension and its screened feature buffer.

    The source grid receives a cached :class:`SpatialGridLayout`; its coordinate and
    weight arrays remain atom-major. Feature evaluation runs once on a spatially
    reordered grid copy, and the resulting buffer restores atom-major order for model
    chunk construction.
    """
    feature_spec = (
        features if isinstance(features, FeatureSpec) else FeatureSpec(features)
    )
    if grids.coords is None or grids.weights is None:
        raise ValueError("Grids must be built before generating screened features.")

    feature_function = feature_math.MGGAFeatureFunction(feature_spec)
    gpu = dm.device.type == "cuda"
    if gpu:
        check_gpu_imports_were_successful()
        from gpu4pyscf.dft import numint as dft_gpu_numint

        block_size = int(dft_gpu_numint.MIN_BLK_SIZE)
    else:
        block_size = CPU_AO_SCREENING_BLOCK_SIZE
    sorted_grids, forward, inverse = _prepare_spatially_sorted_grids(
        mol, grids, block_size, gpu
    )
    sorted_raw_features = cast(
        Tensor,
        ao_evaluation.ChunkEvalForward.apply(
            dm.double(),
            mol,
            sorted_grids,
            feature_function,
            block_size,
            compile_feature_function,
            gpu,
        ),
    )
    forward_permutation = torch.as_tensor(forward, device=dm.device)
    inverse_permutation = torch.as_tensor(inverse, device=dm.device)
    atom_major_raw_features = sorted_raw_features.index_select(-1, inverse_permutation)
    return ScreenedFeatureBuffer(
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
    )
