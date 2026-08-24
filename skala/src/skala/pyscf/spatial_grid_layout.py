# SPDX-License-Identifier: MIT

"""Prepare spatially ordered PySCF and GPU4PySCF integration grids.

Skala's AO evaluator benefits from spatially local grid blocks, while PySCF and
GPU4PySCF provide integration grids in atom-major order with backend-specific AO
screening metadata. This module derives a reusable :class:`SpatialGridLayout`
without modifying the backend-owned source grid.

A shallow grid copy receives spatially reordered coordinates and weights. For
PySCF, its ``non0tab`` shell-screening table is rebuilt; for GPU4PySCF,
``_non0ao_idx`` is cleared so the backend can rebuild it for the new order. The
cached forward and inverse permutations bridge spatial AO evaluation and the
atom-major layout expected by model features.
"""

from copy import copy
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from pyscf import dft, gto
from torch import Tensor

from skala.pyscf.backend import Grid, check_gpu_imports_were_successful
from skala.typing import D1, D2, F64, I64

CPU_AO_SCREENING_BLOCK_SIZE = 9 * dft.gen_grid.BLKSIZE


@dataclass(frozen=True)
class SpatialGridLayout:
    """Evaluation-ready spatial ordering derived from an atom-major grid."""

    block_size: int
    sorted_grids: Grid
    forward_permutation: Tensor
    inverse_permutation: Tensor


def _decompose_grid_into_spatial_blocks(
    coords: np.ndarray[D2, F64], block_size: int
) -> tuple[np.ndarray[D1, I64], np.ndarray[D1, I64]]:
    """Decompose a molecular grid into spatial blocks and return its permutations.

    Recursively partitions points along their principal spatial direction. Every
    left subtree contains a whole number of evaluator blocks, so all output blocks
    have ``block_size`` points except for a possible final remainder. Degenerate
    principal directions fall back to the longest Cartesian extent.

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

    def split_projections(indices: np.ndarray[D1, I64]) -> np.ndarray:
        point_coords = coords[indices]
        centered_coords = point_coords - point_coords.mean(axis=0)
        scatter = centered_coords.T @ centered_coords
        eigenvalues, eigenvectors = np.linalg.eigh(scatter)
        eigenvalue_scale = max(abs(eigenvalues[-1]), abs(eigenvalues[-2]))
        if np.isclose(
            eigenvalues[-1],
            eigenvalues[-2],
            rtol=1e-12,
            atol=np.finfo(np.float64).eps * eigenvalue_scale,
        ):
            split_axis = int(np.argmax(np.ptp(point_coords, axis=0)))
            return point_coords[:, split_axis]

        principal_direction = eigenvectors[:, -1]
        largest_component = int(np.argmax(np.abs(principal_direction)))
        if principal_direction[largest_component] < 0:
            principal_direction = -principal_direction
        result = centered_coords @ principal_direction
        assert isinstance(result, np.ndarray)
        return result

    def partition(indices: np.ndarray[D1, I64]) -> list[np.ndarray[D1, I64]]:
        if indices.size <= block_size:
            return [indices]

        block_count = (indices.size + block_size - 1) // block_size
        left_size = (block_count // 2) * block_size
        positions = np.lexsort((indices, split_projections(indices)))
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

        host_coords = cast(np.ndarray[D2, F64], cupy.asnumpy(grids.coords))
    else:
        host_coords = cast(np.ndarray[D2, F64], grids.coords)

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
