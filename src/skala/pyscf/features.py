# SPDX-License-Identifier: MIT

"""
Methods for generating and manipulating density features.
"""

import logging
from collections.abc import Callable, Iterator
from copy import copy

import numpy as np
import torch
from pyscf import dft, gto
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx

from skala.features import Feature, FeatureMap
from skala.pyscf import feature_math
from skala.pyscf.backend import (
    Array,
    Grid,
    check_gpu_imports_were_successful,
    dft_gpu,
    from_numpy_or_cupy,
)
from skala.pyscf.evaluation import FeatureSpec
from skala.pyscf.memory_estimators import estimate_max_grid_chunk_size

LOG = logging.getLogger(__name__)

DEFAULT_FEATURES = [
    Feature.DENSITY,
    Feature.KIN,
    Feature.GRAD,
    Feature.GRID_COORDS,
    Feature.GRID_WEIGHTS,
]
DEFAULT_FEATURES_SET = set(DEFAULT_FEATURES)

# Features that require per-atom grid decomposition.
_ATOMIC_GRID_FEATURES = {
    Feature.ATOMIC_GRID_WEIGHTS,
    Feature.ATOMIC_GRID_SIZES,
    Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE,
}


def chunked_features(
    mol: gto.Mole,
    dm: Tensor,
    grids: Grid,
    features: set[Feature],
    func_deriv: int,
    max_memory_in_mb: int | None = None,
    safety_fraction: float = 0.8,
    compile_feature_function: bool = False,
) -> Iterator[FeatureMap]:
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

    Yields:
        A dictionary of features for each chunk.
    """

    features = features or DEFAULT_FEATURES_SET
    if "atomic_grid_sizes" not in features:
        raise ValueError(
            "The current implementation of chunked_features requires 'atomic_grid_sizes' to be in the requested features."
        )

    # if dm is a 3D tensor, then we have a spin-polarized system
    with_spin = True if len(dm.shape) == 3 else False

    grid_features = get_grid_features(mol, dm, grids, features)
    feature_spec = FeatureSpec(features)

    # Build the feature function once; it is reused for every chunk.
    ff = None
    if feature_spec.requires_ao_evaluation:
        ff = feature_math.MGGAFeatureFunction(feature_spec)

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

        if feature_spec.requires_ao_evaluation:
            assert ff is not None
            feat_tensor = non_chunk(
                dm.double(),
                mol,
                grids.coords[grid_slice],
                ff,
                compile_feature_function=compile_feature_function,
                gpu=dm.device.type == "cuda",
            )

            for k, v in ff.to_dict(feat_tensor).items():
                feature_chunk[k] = feature_math.maybe_expand_and_divide(
                    v, not with_spin, 2
                )

        yield {Feature(name): value for name, value in feature_chunk.items()}


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
    features: set[Feature] | None = None,
    chunk_size: int | None = None,
    max_memory: int = 2000,
    gpu: bool = False,
) -> FeatureMap:
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

    feature_spec = FeatureSpec(features)
    if feature_spec.requires_ao_evaluation:
        mgga_features = auto_chunk(
            dm,
            mol,
            grids,
            feature_math.MGGAFeatureFunction(feature_spec),
            block_size=chunk_size,
            max_memory=max_memory,
            fix_block_size=chunk_size is None,
            gpu=gpu,
        )

        for feature in mgga_features:
            mol_features[feature] = feature_math.maybe_expand_and_divide(
                mgga_features[feature], not with_spin, 2
            )

    return {Feature(name): value for name, value in mol_features.items()}


def get_grid_features(
    mol: gto.Mole,
    dm: Tensor,
    grids: Grid,
    requested_features: set[Feature],
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


class ChunkEvalForward(Function):
    @staticmethod
    def setup_context(
        ctx: FunctionCtx,
        inputs: tuple[
            torch.Tensor,
            gto.Mole,
            Grid,
            feature_math.LinearFeature,
            int,
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
        feature_function: feature_math.LinearFeature,
        blksize: int,
        compile_feature_function: bool,
        gpu: bool,
        *vectors_jvp: torch.Tensor,
    ) -> torch.Tensor:
        ngrids = grids.weights.size
        block_loop_args = (mol, grids, mol.nao)
        block_loop_kwargs = {
            "deriv": feature_function.deriv,
            "blksize": blksize if not gpu else None,
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
        if len(vectors_jvp) > 1:
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
            # Mask dm to only include the relevant AOs
            if mask is None or not gpu:
                mask = torch.arange(mol.nao_nr(), device=dm.device)
            else:
                mask = torch.from_dlpack(mask)
            masked_dm = dm_sorted[..., mask[:, None], mask[None, :]]

            # Apply chain rule for this particular block
            partial_func = partial_feature_function_over_aos(
                feature_function,
                from_numpy_or_cupy(
                    ao_block, device=dm.device, dtype=dm.dtype, transpose=not gpu
                ),
            )
            for v_sorted in vectors_jvp_sorted:
                partial_func = partial_jvp_function_over_tangents(
                    partial_func,
                    v_sorted[..., mask[:, None], mask[None, :]],
                )

            # Compute feature (or its jvp) for this block with masked dm
            if compile_feature_function:
                temp_feature = torch.compile(partial_func)(masked_dm)
            else:
                temp_feature = partial_func(masked_dm)

            features[..., start:end] = temp_feature
        return features

    @staticmethod
    def jvp(ctx: FunctionCtx, grad_input: torch.Tensor) -> torch.Tensor:
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
            grad_input,
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
            feature_math.LinearFeature,
            list[str],
            int,
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
        feature_function: feature_math.LinearFeature,
        derivative_types: list[str],
        blksize: int,
        compile_feature_function: bool,
        gpu: bool,
        *vectors: torch.Tensor,
    ) -> torch.Tensor:
        block_loop_args = (mol, grids, mol.nao)
        block_loop_kwargs = {
            "deriv": feature_function.deriv,
            "blksize": blksize if not gpu else None,
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
        if len(vectors) > 1:
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

            # Mask to only include the relevant AOs
            if mask is None or not gpu:
                mask = torch.arange(mol.nao_nr(), device=dm.device)
            else:
                mask = from_numpy_or_cupy(mask, device=dm.device, dtype=torch.long)

            # Apply chain rule for this particular block
            # but be careful with signature change upon first vjp
            partial_func = partial_feature_function_over_aos(
                feature_function,
                from_numpy_or_cupy(
                    ao_block, device=dm.device, dtype=dm.dtype, transpose=not gpu
                ),
            )
            for derivative_type, vector, v_sorted in zip(
                derivative_types, vectors, vectors_sorted, strict=True
            ):
                if derivative_type == "jvp":
                    partial_func = partial_jvp_function_over_tangents(
                        partial_func,
                        v_sorted[..., mask[:, None], mask[None, :]],
                    )
                elif derivative_type == "vjp":
                    partial_func = partial_vjp_function_over_tangents(
                        partial_func,
                        v_sorted[..., mask[:, None], mask[None, :]],
                    )
                elif derivative_type == "first_vjp":
                    partial_func = partial_vjp_function_over_tangents(
                        partial_func, vector[..., start:end]
                    )
                else:
                    raise ValueError(
                        f"Unknown derivative {derivative_type} (must be one of 'jvp', 'vjp', 'first_vjp')"
                    )
            if compile_feature_function:
                out[..., mask[:, None], mask[None, :]] += torch.compile(partial_func)(
                    dm_sorted[..., mask[:, None], mask[None, :]]
                )
            else:
                out[..., mask[:, None], mask[None, :]] += partial_func(
                    dm_sorted[..., mask[:, None], mask[None, :]]
                )
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
    feature_function: feature_math.LinearFeature,
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
    feature_function: feature_math.LinearFeature,
    block_size: int | None = None,
    max_memory: int = 2000,
    fix_block_size: bool = True,
    compile_feature_function: bool = False,
    gpu: bool = False,
) -> FeatureMap:
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
    feature_function: feature_math.LinearFeature
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
    FeatureMap:
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
