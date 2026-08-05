# SPDX-License-Identifier: MIT

"""Extend PySCF and GPU4PySCF grids for screened Skala evaluation.

Skala's AO evaluator benefits from spatially local grid blocks, while PySCF and
GPU4PySCF provide integration grids in atom-major order with backend-specific AO
screening metadata. This module is the extension layer interposed between those
backend-owned grid objects and Skala's feature evaluation. It deliberately avoids
subclassing either grid implementation so the same screened path can serve both.

Grid preparation produces a :class:`SpatialGridLayout`, which is cached on the source
grid for later evaluations. The source grid's integration data remains unchanged. A
shallow grid copy receives spatially reordered coordinates and weights. For PySCF,
its ``non0tab`` shell-screening table is rebuilt; for GPU4PySCF, ``_non0ao_idx`` is
cleared so the backend can rebuild it for the new order. The cached forward and
inverse permutations bridge spatial AO evaluation and the atom-major layout expected
by model features.

:func:`prepare_spatial_grid_layout` owns this reusable grid extension. The integrator
attaches it to the source grid and owns density-dependent feature evaluation.
Atom-aligned model batching is a separate process owned by
:mod:`skala.pyscf.model_chunking`.
"""

from copy import copy
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
import torch
from pyscf import dft, gto
from torch import Tensor

from skala.pyscf import ao_evaluation, feature_math
from skala.pyscf.backend import Grid, check_gpu_imports_were_successful

CPU_AO_SCREENING_BLOCK_SIZE = 9 * dft.gen_grid.BLKSIZE

_Float64Coordinates: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]
_Int64Permutation: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int64]]


@dataclass(frozen=True)
class SpatialGridLayout:
    """Evaluation-ready spatial ordering derived from an atom-major grid."""

    block_size: int
    sorted_grids: Grid
    forward_permutation: Tensor
    inverse_permutation: Tensor


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


def prepare_spatial_grid_layout(
    mol: gto.Mole,
    grids: Grid,
    block_size: int,
    device: torch.device,
) -> SpatialGridLayout:
    """Build a spatially ordered grid layout for backend AO screening.

    Args:
        mol: Molecule used to rebuild CPU shell-screening data.
        grids: Built CPU or GPU integration grid in atom-major order.
        block_size: Fixed number of points consumed by each backend block.
        device: Torch device used for permutation tensors.

    Returns:
        An evaluation-ready layout containing the sorted grid and both permutations.
    """
    if grids.coords is None or grids.weights is None:
        raise ValueError("Grids must be built before spatial sorting.")

    gpu = device.type == "cuda"
    if gpu:
        check_gpu_imports_were_successful()
        import cupy

        host_coords = cast(_Float64Coordinates, cupy.asnumpy(grids.coords))
    else:
        host_coords = cast(_Float64Coordinates, grids.coords)

    forward, inverse = _decompose_grid_into_spatial_blocks(host_coords, block_size)
    sorted_grids = copy(grids)
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
    return SpatialGridLayout(
        block_size=block_size,
        sorted_grids=sorted_grids,
        forward_permutation=torch.as_tensor(forward, device=device),
        inverse_permutation=torch.as_tensor(inverse, device=device),
    )


def screened_feature_jvp(
    dm: Tensor,
    dm_tangent: Tensor,
    mol: gto.Mole,
    spatial_grid_layout: SpatialGridLayout,
    feature_function: feature_math.MGGAFeatureFunction,
    compile_feature_function: bool = False,
) -> Tensor:
    """Apply the raw-feature Jacobian and restore atom-major grid order."""
    sorted_tangent = cast(
        Tensor,
        ao_evaluation.ChunkEvalForward.apply(
            dm,
            mol,
            spatial_grid_layout.sorted_grids,
            feature_function,
            spatial_grid_layout.block_size,
            compile_feature_function,
            dm.device.type == "cuda",
            dm_tangent,
        ),
    )
    return sorted_tangent.index_select(
        -1, spatial_grid_layout.inverse_permutation
    ).detach()
