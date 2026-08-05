# SPDX-License-Identifier: MIT

"""
Methods for generating and manipulating density features.
"""

import logging
from collections.abc import Callable, Iterator
from copy import copy
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch
from pyscf import dft, gto
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx

from skala.pyscf import feature_math
from skala.pyscf.backend import (
    Array,
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
        mgga_features = auto_chunk(
            dm,
            mol,
            grids,
            feature_math.MGGAFeatureFunction(feature_spec),
            block_size=evaluation_policy.ao_block_size,
            max_memory=max_memory,
            fix_block_size=evaluation_policy.ao_block_size is None,
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


def partial_feature_function_over_aos(
    feature_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ao: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a function that computes the feature function with the given ao,
    but not the dm already passed to the function.

    Purpose is to allow evaluating a block-local VJP.
    """

    def partial_feature_function(dm: torch.Tensor) -> torch.Tensor:
        return feature_function(dm, ao)

    return partial_feature_function


def partial_vjp_function_over_tangents(
    func: Callable[[torch.Tensor], torch.Tensor],
    tangents: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a function that computes the vjp of the given function with tangents,
    but not primals already passed to the function.

    Purpose is to evaluate the feature-space adjoint for one AO block."""

    def reduced_vjp(primals: torch.Tensor) -> torch.Tensor:
        return torch.func.vjp(func, primals)[1](tangents)[0]

    return reduced_vjp


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
        feature_spec=feature_spec,
        chunks=chunks,
        with_spin=dm.ndim == 3,
    )


@dataclass(frozen=True)
class _AOBlock:
    ao: Tensor
    active_aos: Tensor | None
    grid_slice: slice

    def select_aos(self, matrix: Tensor) -> Tensor:
        if self.active_aos is None:
            return matrix
        return matrix[..., self.active_aos[:, None], self.active_aos[None, :]]

    def add_to(self, matrix: Tensor, block_result: Tensor) -> None:
        if self.active_aos is None:
            matrix += block_result
        else:
            matrix[..., self.active_aos[:, None], self.active_aos[None, :]] += (
                block_result
            )


def _evaluate_feature_block(
    feature_function: feature_math.FeatureFunction,
    block: _AOBlock,
    active_dm: Tensor,
    compile_feature_function: bool,
    feature_cotangent: Tensor | None = None,
) -> Tensor:
    """Evaluate one active-AO feature block or its feature-space VJP."""
    partial_func = partial_feature_function_over_aos(feature_function, block.ao)
    if feature_cotangent is not None:
        partial_func = partial_vjp_function_over_tangents(
            partial_func, feature_cotangent[..., block.grid_slice]
        )

    if compile_feature_function:
        return torch.compile(partial_func)(active_dm)
    return partial_func(active_dm)


class _AOBlockLoop:
    def __init__(
        self,
        dm: Tensor,
        mol: gto.Mole,
        grids: Grid,
        feature_function: feature_math.FeatureFunction,
        blksize: int | None,
        gpu: bool,
    ) -> None:
        self.dm = dm
        self.mol = mol
        self.grids = grids
        self.feature_function = feature_function
        self.blksize = blksize
        self.gpu = gpu
        self.sort_idx: Tensor | None
        self.unsort_idx: Tensor | None

        if gpu:
            check_gpu_imports_were_successful()
            self.numint = dft_gpu.numint.NumInt().build(mol, grids.coords)
            self.numint.grid_blksize = blksize
            self.sort_idx = torch.as_tensor(
                self.numint.gdftopt._ao_idx, device=dm.device
            )
            self.unsort_idx = torch.argsort(self.sort_idx)
        else:
            self.numint = dft.numint.NumInt()
            self.sort_idx = None
            self.unsort_idx = None

    def order_aos(self, matrix: Tensor) -> Tensor:
        if self.sort_idx is None:
            return matrix
        return matrix[..., self.sort_idx, :][..., self.sort_idx]

    def restore_ao_order(self, matrix: Tensor) -> Tensor:
        if self.unsort_idx is None:
            return matrix
        return matrix[..., self.unsort_idx, :][..., self.unsort_idx]

    def __iter__(self) -> Iterator[_AOBlock]:
        block_loop_options: dict[str, bool] = {}
        if self.gpu:
            # GPU4PySCF otherwise omits zero-AO blocks, shifting all later grid slices.
            block_loop_options["strict_grid_order"] = True

        end = 0
        for ao_block, mask, weights, _ in self.numint.block_loop(
            mol=self.mol,
            grids=self.grids,
            nao=self.mol.nao,
            deriv=self.feature_function.deriv,
            blksize=self.blksize,
            non0tab=(None if self.gpu else getattr(self.grids, "non0tab", None)),
            **block_loop_options,
        ):
            start, end = end, end + weights.size
            ao = from_numpy_or_cupy(
                ao_block,
                device=self.dm.device,
                dtype=self.dm.dtype,
                transpose=not self.gpu,
            )
            active_aos: Tensor | None
            if mask is None:
                active_aos = None
            elif self.gpu:
                active_aos = from_numpy_or_cupy(
                    mask, device=self.dm.device, dtype=torch.long
                )
            else:
                num_screen_rows = (
                    weights.size + dft.gen_grid.BLKSIZE - 1
                ) // dft.gen_grid.BLKSIZE
                active_aos = torch.as_tensor(
                    _active_cpu_aos(self.mol, mask[:num_screen_rows]),
                    device=self.dm.device,
                    dtype=torch.long,
                )
                ao = ao[..., active_aos, :]
            if active_aos is not None and active_aos.numel() == 0:
                continue
            yield _AOBlock(ao, active_aos, slice(start, end))


class ChunkEvalForward(Function):
    @staticmethod
    def setup_context(
        ctx: FunctionCtx,
        inputs: tuple[
            torch.Tensor,
            gto.Mole,
            Grid,
            feature_math.FeatureFunction,
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
        feature_function: feature_math.FeatureFunction,
        blksize: int | None,
        compile_feature_function: bool,
        gpu: bool,
        *vectors_jvp: torch.Tensor,
    ) -> torch.Tensor:
        ngrids = grids.weights.size
        block_loop = _AOBlockLoop(dm, mol, grids, feature_function, blksize, gpu)

        features = torch.zeros(
            *dm.shape[:-2],
            feature_function.nfeats,
            ngrids,
            device=dm.device,
            dtype=dm.dtype,
        )
        # Raw AO features are linear in dm, so derivatives above first order vanish.
        if len(vectors_jvp) > 1:
            return features

        # Since the raw feature map is linear, its JVP is direct evaluation on
        # the tangent density matrix.
        evaluation_dm = vectors_jvp[0] if vectors_jvp else dm
        evaluation_dm_ordered = block_loop.order_aos(evaluation_dm)
        for block in block_loop:
            active_dm = block.select_aos(evaluation_dm_ordered)
            temp_feature = _evaluate_feature_block(
                feature_function,
                block,
                active_dm,
                compile_feature_function,
            )

            features[..., block.grid_slice] = temp_feature
        return features

    @staticmethod
    def jvp(ctx: FunctionCtx, *grad_inputs: torch.Tensor | None) -> torch.Tensor:
        if len(ctx.vectors_jvp) > 1:
            return torch.zeros(
                *ctx.dm.shape[:-2],
                ctx.feature_function.nfeats,
                ctx.grids.weights.size,
                device=ctx.dm.device,
                dtype=ctx.dm.dtype,
            )
        vector_tangent = grad_inputs[7] if ctx.vectors_jvp else grad_inputs[0]
        if vector_tangent is None:
            return torch.zeros(
                *ctx.dm.shape[:-2],
                ctx.feature_function.nfeats,
                ctx.grids.weights.size,
                device=ctx.dm.device,
                dtype=ctx.dm.dtype,
            )
        return ChunkEvalForward.apply(
            ctx.dm,
            ctx.mol,
            ctx.grids,
            ctx.feature_function,
            ctx.blksize,
            ctx.compile_feature_function,
            ctx.gpu,
            vector_tangent,
        )

    @staticmethod
    def backward(
        ctx: FunctionCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        feature_cotangent = grad_outputs[0]
        if ctx.vectors_jvp:
            dm_grad = ctx.dm * 0
        else:
            dm_grad = ChunkEvalBackward.apply(
                ctx.dm,
                ctx.mol,
                ctx.grids,
                ctx.feature_function,
                ctx.blksize,
                ctx.compile_feature_function,
                ctx.gpu,
                feature_cotangent,
            )
        grads = [dm_grad]

        # We need to provide None for the gradients of the non-differentiable inputs
        # these are mol (1), grids (2), feature_function (3), blksize (4),
        # compile_feature_function (5), gpu (6)
        num_non_differentiable_inputs = 6

        grads += [None] * num_non_differentiable_inputs

        # A first JVP is linear in its tangent; higher JVPs are identically zero.
        for vector in ctx.vectors_jvp:
            if len(ctx.vectors_jvp) == 1:
                vector_grad = ChunkEvalBackward.apply(
                    ctx.dm,
                    ctx.mol,
                    ctx.grids,
                    ctx.feature_function,
                    ctx.blksize,
                    ctx.compile_feature_function,
                    ctx.gpu,
                    feature_cotangent,
                )
            else:
                vector_grad = vector * 0
            grads.append(vector_grad)

        return tuple(grads)


class ChunkEvalBackward(Function):
    @staticmethod
    def setup_context(
        ctx: FunctionCtx,
        inputs: tuple[
            torch.Tensor,
            gto.Mole,
            Grid,
            feature_math.FeatureFunction,
            int | None,
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
            ctx.feature_cotangent,
        ) = inputs
        ctx.save_for_backward(ctx.dm)

    @staticmethod
    def forward(
        dm: torch.Tensor,
        mol: gto.Mole,
        grids: Grid,
        feature_function: feature_math.FeatureFunction,
        blksize: int | None,
        compile_feature_function: bool,
        gpu: bool,
        feature_cotangent: torch.Tensor,
    ) -> torch.Tensor:
        block_loop = _AOBlockLoop(dm, mol, grids, feature_function, blksize, gpu)
        dm_ordered = block_loop.order_aos(dm)

        out = torch.zeros_like(dm)
        for block in block_loop:
            active_dm = block.select_aos(dm_ordered)
            block_result = _evaluate_feature_block(
                feature_function,
                block,
                active_dm,
                compile_feature_function,
                feature_cotangent,
            )
            block.add_to(out, block_result)
        return block_loop.restore_ao_order(out)

    @staticmethod
    def jvp(ctx: FunctionCtx, *grad_inputs: torch.Tensor | None) -> torch.Tensor:
        feature_cotangent_tangent = grad_inputs[7]
        if feature_cotangent_tangent is None:
            return torch.zeros_like(ctx.dm)
        return ChunkEvalBackward.apply(
            ctx.dm,
            ctx.mol,
            ctx.grids,
            ctx.feature_function,
            ctx.blksize,
            ctx.compile_feature_function,
            ctx.gpu,
            feature_cotangent_tangent,
        )

    @staticmethod
    def backward(
        ctx: FunctionCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        # The raw feature Jacobian is constant in dm. The only nonzero gradient
        # propagates through the feature-space cotangent.
        grads = [ctx.dm * 0]
        # We need to provide None for the gradients of the non-differentiable inputs
        # these are mol (1), grids (2), feature_function (3), blksize (4),
        # compile_feature_function (5), gpu (6)
        num_non_differentiable_inputs = 6

        grads += [None] * num_non_differentiable_inputs
        grads.append(
            ChunkEvalForward.apply(
                ctx.dm,
                ctx.mol,
                ctx.grids,
                ctx.feature_function,
                ctx.blksize,
                ctx.compile_feature_function,
                ctx.gpu,
                grad_outputs[0],
            )
        )
        return tuple(grads)


def non_chunk(
    dm: torch.Tensor,
    mol: gto.Mole,
    coords: Array,
    feature_function: feature_math.FeatureFunction,
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
    feature_function: feature_math.FeatureFunction,
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
