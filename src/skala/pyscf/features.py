# SPDX-License-Identifier: MIT

"""
Methods for generating and manipulating density features.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from copy import copy
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import torch
from pyscf import dft, gto
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import FunctionCtx

from skala.pyscf.backend import (
    Array,
    Grid,
    check_gpu_imports_were_successful,
    dft_gpu,
    from_numpy_or_cupy,
)
from skala.pyscf.memory_estimators import (
    estimate_global_raw_feature_buffer_memory,
    estimate_max_grid_chunk_size,
)

LOG = logging.getLogger(__name__)

DEFAULT_FEATURES = ["density", "kin", "grad", "grid_coords", "grid_weights"]
DEFAULT_FEATURES_SET = set(DEFAULT_FEATURES)
CPU_AO_SCREENING_BLOCK_SIZE = 9 * dft.gen_grid.BLKSIZE

# Features that require per-atom grid decomposition.
_ATOMIC_GRID_FEATURES = {
    "atomic_grid_weights",
    "atomic_grid_sizes",
    "atomic_grid_size_bound_shape",
}

_Float64Coordinates: TypeAlias = np.ndarray[
    tuple[int, Literal[3]], np.dtype[np.float64]
]
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


def _active_cpu_aos(mol: gto.Mole, screen_index: np.ndarray) -> np.ndarray:
    """Expand a PySCF shell-screening mask into active AO indices.

    Args:
        mol: Molecule defining the shell-to-AO ranges.
        screen_index: Screening rows whose columns correspond to molecular shells.

    Returns:
        Sorted indices of AOs belonging to a shell active in any screening row.
    """
    active_shells = np.any(screen_index, axis=0)
    ao_loc = mol.ao_loc_nr()
    return np.flatnonzero(np.repeat(active_shells, np.diff(ao_loc)))


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


def _spatially_group_atom_grids(
    mol: gto.Mole, coords: np.ndarray, atomic_grid_sizes: Tensor
) -> np.ndarray:
    """Build a spatial grid permutation independently within each atom.

    Makes screening much more effective as points are spatially grouped
    within each atom, while preserving the original atom order and atom boundaries.

    Args:
        mol: Molecule used by PySCF to define the spatial grouping boxes.
        coords: Atom-major grid coordinates to group.
        atomic_grid_sizes: Number of consecutive grid points owned by each atom.

    Returns:
        A permutation that spatially groups each atom's points while preserving
        atom order and atom boundaries.
    """
    sort_indices = []
    start = 0
    for size in atomic_grid_sizes.tolist():
        stop = start + size
        atom_sort_indices = dft.gen_grid.arg_group_grids(mol, coords[start:stop])
        sort_indices.append(start + atom_sort_indices)
        start = stop
    return np.concatenate(sort_indices)


def maybe_expand_and_divide(
    feature: torch.Tensor, expand: bool, divisor: float
) -> torch.Tensor:
    """
    Expand feature along spin channels and divide its value by divisor if expand is True.
    """
    if expand:
        return torch.stack([feature / divisor, feature / divisor], dim=0)
    else:
        return feature


def chunked_features(
    mol: gto.Mole,
    dm: Tensor,
    grids: Grid,
    features: set[str],
    func_deriv: int,
    max_memory_in_mb: int | None = None,
    safety_fraction: float = 0.8,
    compile_feature_function: bool = False,
    screen_aos: bool = False,
) -> Iterator[dict[str, Tensor]]:
    """
    Chunked feature generation for a given molecule. The density features are generated in chunks to avoid memory issues.

    Input:
        mol: The molecule for which to generate features.
        dm: The density matrix.
        grids: The grid points.
        features: The set of features to generate.
        func_deriv: The order of the functional derivative.
        max_memory_in_mb: The maximum memory to use for each chunk in megabytes (MB). If None, the maximum memory is determined automatically.
        safety_fraction: The fraction of the available memory to use for each chunk.
        compile_feature_function: Whether to compile the feature function.
        screen_aos: Whether to evaluate each atom chunk through backend AO screening.

    Yields:
        A dictionary of features for each chunk.
    """

    features = features or DEFAULT_FEATURES_SET
    if "atomic_grid_sizes" not in features:
        raise ValueError(
            "The current implementation of chunked_features requires 'atomic_grid_sizes' to be in the requested features."
        )
    if grids.coords is None or grids.weights is None:
        raise ValueError("Grids must be built before generating chunked features.")

    # if dm is a 3D tensor, then we have a spin-polarized system
    with_spin = True if len(dm.shape) == 3 else False

    grid_features = get_grid_features(mol, dm, grids, features)
    with_mgga_feature = (
        "density" in features
        or "grad" in features
        or "kin" in features
        or "lapl" in features
    )

    # Build the feature function once; it is reused for every chunk.
    ff = None
    if with_mgga_feature:
        ff = MGGAFeatureFunction(
            with_density="density" in features,
            with_grad="grad" in features,
            with_kin="kin" in features,
            with_lapl="lapl" in features,
        )

    # Determine the chunk size automatically when not explicitly provided.
    if ff is not None:
        max_grid_chunk_size = estimate_max_grid_chunk_size(
            dm=dm,
            deriv=ff.deriv,
            max_memory_in_mb=max_memory_in_mb,
            safety_fraction=safety_fraction,
            func_deriv=func_deriv,
        )
        if max_grid_chunk_size < (
            max_atom_grid := int(grid_features["atomic_grid_sizes"].max().item())
        ):
            LOG.warning(
                f"Adjusted chunk size {max_grid_chunk_size} to match the largest atomic grid {max_atom_grid}. Hope for no OOM."
            )
            max_grid_chunk_size = max_atom_grid
    else:  # no feature function is available, use the full grid.
        max_grid_chunk_size = grid_features["grid_weights"].shape[0]

    for atom_slice, grid_slice in make_chunks(
        grid_features["atomic_grid_sizes"], max_grid_chunk_size
    ):
        feature_chunk = {}
        for feat_name in ["grid_coords", "grid_weights", "atomic_grid_weights"]:
            if feat_name in features:
                feature_chunk[feat_name] = grid_features[feat_name][grid_slice]

        for feat_name in ["coarse_0_atomic_coords", "atomic_grid_sizes"]:
            if feat_name in features:
                feature_chunk[feat_name] = grid_features[feat_name][atom_slice]

        if "atomic_grid_size_bound_shape" in features:
            max_size = int(feature_chunk["atomic_grid_sizes"].max().item())
            feature_chunk["atomic_grid_size_bound_shape"] = torch.zeros(
                max_size, 0, dtype=torch.long, device=dm.device
            )

        if with_mgga_feature:
            assert ff is not None
            gpu = dm.device.type == "cuda"
            if screen_aos:
                chunk_grids = copy(grids)
                chunk_grids.coords = grids.coords[grid_slice]
                chunk_grids.weights = grids.weights[grid_slice]
                if gpu:
                    chunk_grids._non0ao_idx = None
                else:
                    grid_sort_indices = _spatially_group_atom_grids(
                        mol,
                        chunk_grids.coords,
                        feature_chunk["atomic_grid_sizes"],
                    )
                    chunk_grids.coords = chunk_grids.coords[grid_sort_indices]
                    chunk_grids.weights = chunk_grids.weights[grid_sort_indices]
                    grid_sort_indices_t = torch.as_tensor(
                        grid_sort_indices, device=dm.device
                    )
                    for feat_name in (
                        "grid_coords",
                        "grid_weights",
                        "atomic_grid_weights",
                    ):
                        if feat_name in feature_chunk:
                            feature_chunk[feat_name] = feature_chunk[feat_name][
                                grid_sort_indices_t
                            ]
                    chunk_grids.non0tab = dft.gen_grid.make_screen_index(
                        mol,
                        chunk_grids.coords,
                        cutoff=chunk_grids.cutoff,
                    )
                feat_tensor = ChunkEvalForward.apply(
                    dm.double(),
                    mol,
                    chunk_grids,
                    ff,
                    None if gpu else CPU_AO_SCREENING_BLOCK_SIZE,
                    compile_feature_function,
                    gpu,
                )
                mgga_features = ff.to_dict(feat_tensor)
            else:
                feat_tensor = non_chunk(
                    dm.double(),
                    mol,
                    grids.coords[grid_slice],
                    ff,
                    compile_feature_function=compile_feature_function,
                    gpu=gpu,
                )
                mgga_features = ff.to_dict(feat_tensor)

            for k, v in mgga_features.items():
                feature_chunk[k] = maybe_expand_and_divide(v, not with_spin, 2)

        yield feature_chunk


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
    features = features or DEFAULT_FEATURES_SET

    # if dm is a 3D tensor, then we have a spin-polarized system
    with_spin = True if len(dm.shape) == 3 else False

    if gpu and dm.device.type != "cuda":
        raise ValueError("Density matrix must be on the GPU when gpu=True.")

    mol_features = get_grid_features(mol, dm, grids, features)

    with_mgga_feature = (
        "density" in features
        or "grad" in features
        or "kin" in features
        or "lapl" in features
    )
    if with_mgga_feature:
        mgga_features = auto_chunk(
            dm,
            mol,
            grids,
            MGGAFeatureFunction(
                with_density="density" in features,
                with_grad="grad" in features,
                with_kin="kin" in features,
                with_lapl="lapl" in features,
            ),
            block_size=chunk_size,
            max_memory=max_memory,
            fix_block_size=chunk_size is None,
            gpu=gpu,
        )

        for feature in mgga_features:
            mol_features[feature] = maybe_expand_and_divide(
                mgga_features[feature], not with_spin, 2
            )

    return mol_features


def get_grid_features(
    mol: gto.Mole,
    dm: Tensor,
    grids: Grid,
    requested_features: set[str],
) -> dict[str, Tensor]:
    grid_features = {}

    if "grid_coords" in requested_features:
        grid_features["grid_coords"] = from_numpy_or_cupy(
            grids.coords, device=dm.device, dtype=dm.dtype
        )

    if "grid_weights" in requested_features:
        grid_features["grid_weights"] = from_numpy_or_cupy(
            grids.weights, device=dm.device, dtype=dm.dtype
        )

    if "coarse_0_atomic_coords" in requested_features:
        grid_features["coarse_0_atomic_coords"] = from_numpy_or_cupy(
            mol.atom_coords(), device=dm.device, dtype=dm.dtype
        )

    if requested_features & _ATOMIC_GRID_FEATURES:
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

        if "atomic_grid_sizes" in requested_features:
            grid_features["atomic_grid_sizes"] = torch.tensor(
                sizes, dtype=torch.long, device=dm.device
            )

        if "atomic_grid_size_bound_shape" in requested_features:
            max_size = max(sizes)
            grid_features["atomic_grid_size_bound_shape"] = torch.zeros(
                max_size, 0, dtype=torch.long, device=dm.device
            )

        if "atomic_grid_weights" in requested_features:
            raw_weights = np.concatenate(
                [atom_grids_tab[mol.atom_symbol(ia)][1] for ia in range(mol.natm)]
            )
            grid_features["atomic_grid_weights"] = from_numpy_or_cupy(
                raw_weights, device=dm.device, dtype=dm.dtype
            )

    return grid_features


def is_density_feature(feature: str) -> bool:
    return feature in {"density", "grad", "kin"}


def partial_feature_function_over_aos(
    feature_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ao: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a function that computes the feature function with the given ao,
    but not the dm already passed to the function.

    Purpose is to allow for chaining of derivatives.
    """

    def partial_feature_function(dm: torch.Tensor) -> torch.Tensor:
        return feature_function(dm, ao)

    return partial_feature_function


def partial_jvp_function_over_tangents(
    func: Callable[[torch.Tensor], torch.Tensor],
    tangents: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a function that computes the jvp of the given function with tangents,
    but not primals already passed to the function.

    Purpose is to allow for chaining of derivatives over primals."""

    def reduced_jvp(primals: torch.Tensor) -> torch.Tensor:
        _, tangent = torch.func.jvp(func, (primals,), (tangents,))
        return tangent

    return reduced_jvp


def partial_vjp_function_over_tangents(
    func: Callable[[torch.Tensor], torch.Tensor],
    tangents: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a function that computes the vjp of the given function with tangents,
    but not primals already passed to the function.

    Purpose is to allow for chaining of derivatives over primals."""

    def reduced_vjp(primals: torch.Tensor) -> torch.Tensor:
        return torch.func.vjp(func, primals)[1](tangents)[0]

    return reduced_vjp


class FeatureFunction(nn.Module, ABC):
    deriv: int
    nfeats: int
    only_linear_feats: bool

    @abstractmethod
    def forward(self, dm: torch.Tensor, ao: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def to_dict(self, features: torch.Tensor) -> dict[str, torch.Tensor]: ...


class MGGAFeatureFunction(FeatureFunction):
    with_density: bool
    with_grad: bool
    with_kin: bool
    with_lapl: bool
    with_ked_var: bool
    with_ked_det: bool

    def __init__(
        self,
        with_density: bool = True,
        with_grad: bool = True,
        with_kin: bool = True,
        with_lapl: bool = False,
        with_ked_var: bool = False,
        with_ked_det: bool = False,
    ):
        super().__init__()

        self.with_density = with_density
        self.with_grad = with_grad
        self.with_kin = with_kin
        self.with_lapl = with_lapl
        self.with_ked_var = with_ked_var
        self.with_ked_det = with_ked_det

        self.deriv = 0
        if with_grad or with_kin or with_ked_var or with_ked_det:
            self.deriv = 1
        if with_lapl:
            self.deriv = 2

        self.nfeats = (
            with_density
            + with_grad * 3
            + with_kin
            + with_lapl
            + with_ked_var
            + with_ked_det
        )

        if self.nfeats == 0:
            raise ValueError("At least one feature must be selected.")

        self.only_linear_feats = not (with_ked_var or with_ked_det)

    def to_dict(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert the features to a dictionary with the keys being the feature names."""
        feature_index = 0
        feature_dict: dict[str, torch.Tensor] = {}
        if self.with_density:
            feature_dict["density"] = features[..., feature_index, :]
            feature_index += 1
        if self.with_grad:
            feature_dict["grad"] = features[..., feature_index : feature_index + 3, :]
            feature_index += 3
        if self.with_kin:
            feature_dict["kin"] = features[..., feature_index, :]
            feature_index += 1
        if self.with_lapl:
            feature_dict["lapl"] = features[..., feature_index, :]
            feature_index += 1
        if self.with_ked_var:
            feature_dict["ked_var"] = features[..., feature_index, :]
            feature_index += 1
        if self.with_ked_det:
            feature_dict["ked_det"] = features[..., feature_index, :]
            feature_index += 1
        return feature_dict

    def forward(self, dm: torch.Tensor, ao: torch.Tensor) -> torch.Tensor:
        with_Q: bool = self.with_ked_var or self.with_ked_det

        # Flatten all but the last two dimensions
        # then restore the original shape at the end
        dm_view = dm.view(-1, dm.shape[-2], dm.shape[-1])
        # Explicit symmetrization for autodiff
        dm_view = 0.5 * (dm_view + dm_view.transpose(-1, -2))

        features = torch.zeros(
            (dm_view.shape[0], self.nfeats, ao.shape[-1]),
            device=dm.device,
            dtype=dm.dtype,
        )

        # Handle the density only case, where ao has one dim less
        if self.deriv == 0:
            c0 = dm_view @ ao
            features[..., 0, :] = torch.sum(c0 * ao[None, :, :], dim=-2)
            if len(dm.shape) == 2:
                return features.reshape((self.nfeats, -1))
            else:
                return features.reshape((*dm.shape[:-2], self.nfeats, -1))

        c0 = dm_view @ ao[0]

        feat_idx = 0
        if self.with_density:
            features[..., feat_idx, :] = torch.sum(c0 * ao[0][None, :, :], dim=-2)
            feat_idx += 1

        if self.with_grad:
            for i in range(3):
                features[..., feat_idx, :] = 2 * torch.sum(
                    c0 * ao[i + 1][None, :, :], dim=-2
                )
                feat_idx += 1

        if (self.with_kin or self.with_lapl) and not with_Q:
            for i in range(3):
                ci = dm_view @ ao[i + 1]
                features[..., feat_idx, :] += 0.5 * torch.sum(
                    ci * ao[i + 1][None, :, :], dim=-2
                )

            if self.with_kin:
                feat_idx += 1
                if self.with_lapl:
                    features[..., feat_idx, :] = 4 * features[..., feat_idx - 1, :]
            else:
                # Multiply times four for the laplacian
                features[..., feat_idx, :] *= 4.0

            if self.with_lapl:
                # 0 is without derivative
                # 1 2 3 are x y z derivatives
                # 4 5 6 are xx xy xz derivatives
                # 7 8 9 are yy yz zz derivatives
                for i in (4, 7, 9):
                    features[..., feat_idx, :] += 2 * torch.sum(
                        c0 * ao[i][None, :, :], dim=-2
                    )

        if with_Q:
            Q = torch.zeros(
                (dm_view.shape[0], ao.shape[-1], 3, 3), device=dm.device, dtype=dm.dtype
            )

            for i in range(3):
                ci = dm_view @ ao[i + 1]
                for j in range(i, 3):
                    Q = torch.sum(ci * ao[j + 1][None, :, :], dim=-2)

            if self.with_kin:
                features[..., feat_idx, :] = 0.5 * torch.einsum("...ii->...", Q)
                feat_idx += 1

            if self.with_lapl:
                features[..., feat_idx, :] = 2 * torch.einsum("...ii->...", Q)
                # 0 is without derivative
                # 1 2 3 are x y z derivatives
                # 4 5 6 are xx xy xz derivatives
                # 7 8 9 are yy yz zz derivatives
                for i in (4, 7, 9):
                    features[..., feat_idx, :] += 2 * torch.sum(
                        c0 * ao[i][None, :, :], dim=-2
                    )
                feat_idx += 1

            if self.with_ked_var:
                if not self.with_kin:
                    trace = torch.einsum("...ii->...", Q)
                else:
                    trace = 2 * features[:, feat_idx - 1, :]
                features[..., feat_idx, :] = 0.5 * torch.sum(
                    (
                        trace[:, None, None]
                        * torch.eye(3, device=dm.device, dtype=dm.dtype)[None, :, :]
                        - Q
                    )
                    ** 2,
                    dim=(-2, -1),
                )
                feat_idx += 1

            if self.with_ked_det:
                features[..., feat_idx, :] = torch.det(Q)
                feat_idx += 1
        if len(dm.shape) == 2:
            return features.reshape((self.nfeats, -1))
        else:
            return features.reshape((*dm.shape[:-2], self.nfeats, -1))


@dataclass
class _GlobalScreenedFeatures:
    dm: Tensor
    mol: gto.Mole
    sorted_grids: Grid
    sorted_raw_features: Tensor
    atom_major_raw_features: Tensor
    forward_permutation: Tensor
    inverse_permutation: Tensor
    feature_function: MGGAFeatureFunction
    block_size: int
    compile_feature_function: bool
    gpu: bool
    grid_features: dict[str, Tensor]
    feature_names: set[str]
    chunks: list[tuple[slice, slice]]
    with_spin: bool

    def atom_major_jvp(self, dm_tangent: Tensor) -> Tensor:
        """Apply the global raw-feature Jacobian and restore atom-major order."""
        if not self.feature_function.only_linear_feats:
            raise NotImplementedError(
                "Global screened response requires raw features linear in the density "
                "matrix."
            )
        sorted_tangent = ChunkEvalForward.apply(
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
            if feature_name in self.feature_names:
                feature_chunk[feature_name] = self.grid_features[feature_name][
                    grid_slice
                ]

        for feature_name in ("coarse_0_atomic_coords", "atomic_grid_sizes"):
            if feature_name in self.feature_names:
                feature_chunk[feature_name] = self.grid_features[feature_name][
                    atom_slice
                ]

        if "atomic_grid_size_bound_shape" in self.feature_names:
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
            feature_chunk[feature_name] = maybe_expand_and_divide(
                feature, not self.with_spin, 2
            )
        return feature_chunk


def _global_screened_features(
    mol: gto.Mole,
    dm: Tensor,
    grids: Grid,
    features: set[str],
    func_deriv: int,
    max_memory_in_mb: int | None = None,
    safety_fraction: float = 0.8,
    compile_feature_function: bool = False,
) -> _GlobalScreenedFeatures:
    """Evaluate raw AO features once on a spatially ordered molecular grid."""
    if "atomic_grid_sizes" not in features:
        raise ValueError(
            "Global screened features require 'atomic_grid_sizes' for model chunks."
        )
    if grids.coords is None or grids.weights is None:
        raise ValueError("Grids must be built before generating screened features.")

    feature_function = MGGAFeatureFunction(
        with_density="density" in features,
        with_grad="grad" in features,
        with_kin="kin" in features,
        with_lapl="lapl" in features,
    )
    grid_features = get_grid_features(mol, dm, grids, features)
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
    sorted_raw_features = ChunkEvalForward.apply(
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
        feature_names=features,
        chunks=chunks,
        with_spin=dm.ndim == 3,
    )


class ChunkEvalForward(Function):
    @staticmethod
    def setup_context(
        ctx: FunctionCtx,
        inputs: tuple[
            torch.Tensor,
            gto.Mole,
            Grid,
            FeatureFunction,
            int | None,
            int,
            bool,
            bool,
            torch.Tensor,
        ],
        output: torch.Tensor,
    ) -> None:
        (
            ctx.dm,
            ctx.mol,
            ctx.grids,
            ctx.feature_function,
            ctx.blksize,
            ctx.compile_feature_function,
            ctx.gpu,
            *ctx.vectors_jvp,
        ) = inputs
        ctx.save_for_backward(ctx.dm)

    @staticmethod
    def forward(
        dm: torch.Tensor,
        mol: gto.Mole,
        grids: Grid,
        feature_function: FeatureFunction,
        blksize: int | None,
        compile_feature_function: bool,
        gpu: bool,
        *vectors_jvp: torch.Tensor,
    ) -> torch.Tensor:
        ngrids = grids.weights.size
        block_loop_args = (mol, grids, mol.nao)
        block_loop_kwargs = {
            "deriv": feature_function.deriv,
            "blksize": blksize,
            "non0tab": None if gpu else getattr(grids, "non0tab", None),
        }
        if gpu:
            check_gpu_imports_were_successful()
            ni = dft_gpu.numint.NumInt().build(mol, grids.coords)
            ni.grid_blksize = blksize
            sort_idx = ni.gdftopt._ao_idx
        else:
            ni = dft.numint.NumInt()
            sort_idx = np.arange(mol.nao_nr())

        features = torch.zeros(
            *dm.shape[:-2],
            feature_function.nfeats,
            ngrids,
            device=dm.device,
            dtype=dm.dtype,
        )
        if len(vectors_jvp) > 1 and feature_function.only_linear_feats:
            return features

        # Pre-sort DM and JVP vectors once (sort_idx is constant across blocks)
        sort_idx_t = torch.as_tensor(sort_idx, device=dm.device)
        dm_sorted = dm[..., sort_idx_t, :][..., sort_idx_t]
        vectors_jvp_sorted = [
            v[..., sort_idx_t, :][..., sort_idx_t] for v in vectors_jvp
        ]

        end = 0
        for ao_block, mask, weights, _ in ni.block_loop(
            *block_loop_args, **block_loop_kwargs
        ):
            start, end = end, end + weights.size
            ao = from_numpy_or_cupy(
                ao_block, device=dm.device, dtype=dm.dtype, transpose=not gpu
            )
            if gpu and mask is not None:
                mask = from_numpy_or_cupy(mask, device=dm.device, dtype=torch.long)
            elif not gpu and mask is not None:
                num_screen_rows = (
                    weights.size + dft.gen_grid.BLKSIZE - 1
                ) // dft.gen_grid.BLKSIZE
                mask = mask[:num_screen_rows]
                mask = torch.as_tensor(
                    _active_cpu_aos(mol, mask), device=dm.device, dtype=torch.long
                )
                ao = ao[..., mask, :]
            if mask is not None and mask.numel() == 0:
                continue
            masked_dm = (
                dm_sorted
                if mask is None
                else dm_sorted[..., mask[:, None], mask[None, :]]
            )

            # Apply chain rule for this particular block
            partial_func = partial_feature_function_over_aos(
                feature_function,
                ao,
            )
            for v_sorted in vectors_jvp_sorted:
                partial_func = partial_jvp_function_over_tangents(
                    partial_func,
                    (
                        v_sorted
                        if mask is None
                        else v_sorted[..., mask[:, None], mask[None, :]]
                    ),
                )

            # Compute feature (or its jvp) for this block with masked dm
            if compile_feature_function:
                temp_feature = torch.compile(partial_func)(masked_dm)
            else:
                temp_feature = partial_func(masked_dm)

            features[..., start:end] = temp_feature
        return features

    @staticmethod
    def jvp(ctx: FunctionCtx, *grad_inputs: torch.Tensor) -> torch.Tensor:
        # Chain rule for the jvp
        return ChunkEvalForward.apply(
            ctx.dm,
            ctx.mol,
            ctx.grids,
            ctx.feature_function,
            ctx.blksize,
            ctx.compile_feature_function,
            ctx.gpu,
            *ctx.vectors_jvp,
            grad_inputs[0],
        )

    @staticmethod
    def backward(
        ctx: FunctionCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        # After one vjp (backward) the signature of the function changes from dm.shape -> (*dm.shape[:-2], nfeats, ngrid) to dm.shape -> dm.shape
        # therefore we move to a different function that does essentially the same thing, but with the new signature

        # Derivative to dm
        grads = [
            ChunkEvalBackward.apply(
                ctx.dm,
                ctx.mol,
                ctx.grids,
                ctx.feature_function,
                ["jvp"] * len(ctx.vectors_jvp) + ["first_vjp"],
                ctx.blksize,
                ctx.compile_feature_function,
                ctx.gpu,
                *ctx.vectors_jvp,
                *grad_outputs,
            )
        ]

        # We need to provide None for the gradients of the non-differentiable inputs
        # these are mol (1), grids (2), feature_function (3), blksize (4),
        # compile_feature_function (5), gpu (6)
        num_non_differentiable_inputs = 6

        grads += [None] * num_non_differentiable_inputs

        # Gradients of earlier tangents
        for i in range(len(ctx.vectors_jvp)):
            derivative_types = ["jvp"] * len(ctx.vectors_jvp)
            derivative_types[i] = "first_vjp"
            grads.append(
                ChunkEvalBackward.apply(
                    ctx.dm,
                    ctx.mol,
                    ctx.grids,
                    ctx.feature_function,
                    derivative_types,
                    ctx.blksize,
                    ctx.compile_feature_function,
                    ctx.gpu,
                    *ctx.vectors_jvp[:i],
                    *grad_outputs,
                    *ctx.vectors_jvp[i + 1 :],
                )
            )

        return tuple(grads)


class ChunkEvalBackward(Function):
    @staticmethod
    def setup_context(
        ctx: FunctionCtx,
        inputs: tuple[
            torch.Tensor,
            gto.Mole,
            Grid,
            FeatureFunction,
            list[str],
            int | None,
            bool,
            bool,
            torch.Tensor,
        ],
        output: tuple[torch.Tensor, ...],
    ) -> None:
        (
            ctx.dm,
            ctx.mol,
            ctx.grids,
            ctx.feature_function,
            ctx.derivative_types,
            ctx.blksize,
            ctx.compile_feature_function,
            ctx.gpu,
            *ctx.vectors,
        ) = inputs
        ctx.save_for_backward(ctx.dm)

    @staticmethod
    def forward(
        dm: torch.Tensor,
        mol: gto.Mole,
        grids: Grid,
        feature_function: FeatureFunction,
        derivative_types: list[str],
        blksize: int | None,
        compile_feature_function: bool,
        gpu: bool,
        *vectors: torch.Tensor,
    ) -> torch.Tensor:
        block_loop_args = (mol, grids, mol.nao)
        block_loop_kwargs = {
            "deriv": feature_function.deriv,
            "blksize": blksize if not gpu else None,
            "non0tab": None if gpu else getattr(grids, "non0tab", None),
        }
        if gpu:
            check_gpu_imports_were_successful()
            ni = dft_gpu.numint.NumInt().build(mol, grids.coords)
            ni.grid_blksize = blksize
            sort_idx = ni.gdftopt._ao_idx
        else:
            ni = dft.numint.NumInt()
            sort_idx = np.arange(mol.nao_nr())

        end: int = 0
        out = torch.zeros_like(dm)
        if len(vectors) > 1 and feature_function.only_linear_feats:
            return out

        # Pre-sort DM and derivative vectors once (sort_idx is constant across blocks)
        sort_idx_t = torch.as_tensor(sort_idx, device=dm.device)
        unsort_idx = torch.argsort(sort_idx_t)
        dm_sorted = dm[..., sort_idx_t, :][..., sort_idx_t]
        vectors_sorted = [
            v[..., sort_idx_t, :][..., sort_idx_t] if dt in ("jvp", "vjp") else v
            for dt, v in zip(derivative_types, vectors, strict=True)
        ]

        for ao_block, mask, weights, _ in ni.block_loop(
            *block_loop_args,
            **block_loop_kwargs,
        ):
            start, end = end, end + weights.size

            ao = from_numpy_or_cupy(
                ao_block, device=dm.device, dtype=dm.dtype, transpose=not gpu
            )
            if gpu and mask is not None:
                mask = from_numpy_or_cupy(mask, device=dm.device, dtype=torch.long)
            elif not gpu and mask is not None:
                num_screen_rows = (
                    weights.size + dft.gen_grid.BLKSIZE - 1
                ) // dft.gen_grid.BLKSIZE
                mask = mask[:num_screen_rows]
                mask = torch.as_tensor(
                    _active_cpu_aos(mol, mask), device=dm.device, dtype=torch.long
                )
                ao = ao[..., mask, :]
            if mask is not None and mask.numel() == 0:
                continue

            # Apply chain rule for this particular block
            # but be careful with signature change upon first vjp
            partial_func = partial_feature_function_over_aos(
                feature_function,
                ao,
            )
            for derivative_type, vector, v_sorted in zip(
                derivative_types, vectors, vectors_sorted, strict=True
            ):
                if derivative_type == "jvp":
                    partial_func = partial_jvp_function_over_tangents(
                        partial_func,
                        (
                            v_sorted
                            if mask is None
                            else v_sorted[..., mask[:, None], mask[None, :]]
                        ),
                    )
                elif derivative_type == "vjp":
                    partial_func = partial_vjp_function_over_tangents(
                        partial_func,
                        (
                            v_sorted
                            if mask is None
                            else v_sorted[..., mask[:, None], mask[None, :]]
                        ),
                    )
                elif derivative_type == "first_vjp":
                    partial_func = partial_vjp_function_over_tangents(
                        partial_func, vector[..., start:end]
                    )
                else:
                    raise ValueError(
                        f"Unknown derivative {derivative_type} (must be one of 'jvp', 'vjp', 'first_vjp')"
                    )
            masked_dm = (
                dm_sorted
                if mask is None
                else dm_sorted[..., mask[:, None], mask[None, :]]
            )
            if compile_feature_function:
                block_result = torch.compile(partial_func)(masked_dm)
            else:
                block_result = partial_func(masked_dm)
            if mask is None:
                out += block_result
            else:
                out[..., mask[:, None], mask[None, :]] += block_result
        return out[..., unsort_idx, :][..., unsort_idx]

    @staticmethod
    def jvp(ctx: FunctionCtx, *grad_input: torch.Tensor) -> torch.Tensor:
        # Chain rule for the jvp
        return ChunkEvalBackward.apply(
            ctx.dm,
            ctx.mol,
            ctx.grids,
            ctx.feature_function,
            ctx.derivative_types + ["jvp"],
            ctx.blksize,
            ctx.compile_feature_function,
            ctx.gpu,
            *ctx.vectors,
            grad_input,
        )

    @staticmethod
    def backward(
        ctx: FunctionCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        # Chain rule for the vjp

        # Gradient corresponding to dm
        grads = [
            ChunkEvalBackward.apply(
                ctx.dm,
                ctx.mol,
                ctx.grids,
                ctx.feature_function,
                ctx.derivative_types + ["vjp"],
                ctx.blksize,
                ctx.compile_feature_function,
                ctx.gpu,
                *ctx.vectors,
                *grad_outputs,
            )
        ]
        # We need to provide None for the gradients of the non-differentiable inputs
        # these are mol (1), grids (2), feature_function (3), derivative_types (4), blksize (5),
        # compile_feature_function (6), gpu (7)
        num_non_differentiable_inputs = 7

        grads += [None] * num_non_differentiable_inputs
        # Gradients of gradients
        for i, derivative_type in enumerate(ctx.derivative_types):
            derivative_types = copy(ctx.derivative_types)
            if derivative_type == "jvp" or derivative_type == "vjp":
                derivative_types[i] = "vjp"
                grads.append(
                    ChunkEvalBackward.apply(
                        ctx.dm,
                        ctx.mol,
                        ctx.grids,
                        ctx.feature_function,
                        derivative_types,
                        ctx.blksize,
                        ctx.compile_feature_function,
                        ctx.gpu,
                        *ctx.vectors[:i],
                        *grad_outputs,
                        *ctx.vectors[i + 1 :],
                    )
                )
            elif derivative_type == "first_vjp":
                grads.append(
                    ChunkEvalForward.apply(
                        ctx.dm,
                        ctx.mol,
                        ctx.grids,
                        ctx.feature_function,
                        ctx.blksize,
                        ctx.compile_feature_function,
                        ctx.gpu,
                        *ctx.vectors[:i],
                        *grad_outputs,
                        *ctx.vectors[i + 1 :],
                    )
                )
            else:
                raise ValueError(
                    f"Unknown derivative {derivative_type} (must be one of 'jvp', 'vjp', 'first_vjp')"
                )
        return tuple(grads)


def non_chunk(
    dm: torch.Tensor,
    mol: gto.Mole,
    coords: Array,
    feature_function: FeatureFunction,
    compile_feature_function: bool = False,
    gpu: bool = False,
) -> torch.Tensor:
    if gpu:
        check_gpu_imports_were_successful()
        ni = dft_gpu.numint.NumInt().build(mol, coords)
    else:
        ni = dft.numint.NumInt()
    ao = from_numpy_or_cupy(
        ni.eval_ao(mol, coords, deriv=feature_function.deriv, non0tab=None),
        device=dm.device,
        dtype=dm.dtype,
        transpose=True,
    )
    if compile_feature_function:
        return torch.compile(feature_function.forward)(dm, ao)
    else:
        return feature_function.forward(dm, ao)


def auto_chunk(
    dm: torch.Tensor,
    mol: gto.Mole,
    grids: Grid,
    feature_function: FeatureFunction,
    block_size: int | None = None,
    max_memory: int = 2000,
    fix_block_size: bool = True,
    compile_feature_function: bool = False,
    gpu: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Automatically splits feature evaluation into smaller chunks if needed.

    This function determines the appropriate chunk size for evaluating a feature
    function on molecular grids, based on available memory and number of basis
    functions. If the computed chunk size is larger than the size of the grid, or
    if a fixed block size was provided, it uses a non-chunked approach.

    Parameters
    ----------
    dm: torch.Tensor
        Density matrix or set of density matrices used for
        evaluating the feature function.
    mol: gto.Mole
        PySCF molecule object representing the system of interest.
    grids: Grid
        Grids object defining the points in space on which
        the feature function is evaluated.
    feature_function: FeatureFunction
        The object representing the feature function to evaluate. The number of derivatives (deriv) determines
        how many components to compute.
    gpu: bool, optional
        Whether to use GPU for computation. Defaults to False.
    block_size: int | None, optional
        Manually specified block size for chunking. (CPU only)
        Defaults to None.
    max_memory: int, optional
        Maximum memory in MB to use for chunking (CPU only)
    fix_block_size: bool, optional
        Whether to fix the block size or compute it
        automatically based on system resources. Defaults to True. (CPU only)
    compile_feature_function: bool, optional
        If True, compiles the feature function for efficiency. Defaults to False.

    Returns
    -------
    dict[str, torch.Tensor]:
        The evaluated feature function on the specified grids, either
        computed in smaller chunks or in a single pass, depending on the block size.
    """

    if gpu:
        check_gpu_imports_were_successful()
        if dm.device.type != "cuda":
            raise ValueError("Density matrix must be on the GPU when gpu=True.")

    blksize: int | None

    if gpu and block_size is not None:
        raise ValueError("Setting custom block size is not supported on GPU.")

    if block_size is None and fix_block_size and not gpu:
        nao = mol.nao_nr()
        comp = (
            (feature_function.deriv + 1)
            * (feature_function.deriv + 2)
            * (feature_function.deriv + 3)
            // 6
        )
        BLKSIZE = dft.gen_grid.BLKSIZE
        blksize = int(max_memory * 1e6 / ((comp + 1) * nao * 8 * BLKSIZE))
        blksize = max(4, min(blksize, 1200)) * BLKSIZE
    else:
        blksize = block_size

    if blksize is not None and not gpu:
        blksize = blksize - blksize % dft.gen_grid.BLKSIZE

    if blksize is not None and blksize >= grids.weights.shape[0]:
        features = non_chunk(
            dm.double(),
            mol,
            grids.coords,
            feature_function,
            compile_feature_function=compile_feature_function,
            gpu=gpu,
        )
    else:
        features = ChunkEvalForward.apply(
            dm.double(),
            mol,
            grids,
            feature_function,
            blksize,
            compile_feature_function,
            gpu,
        )
    return feature_function.to_dict(features)
