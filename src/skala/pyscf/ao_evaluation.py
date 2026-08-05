# SPDX-License-Identifier: MIT

"""Blockwise atomic-orbital feature evaluation and custom autograd."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import torch
from pyscf import dft, gto
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from typing_extensions import Unpack

from skala.pyscf import feature_math
from skala.pyscf.backend import (
    Array,
    Grid,
    check_gpu_imports_were_successful,
    dft_gpu,
    from_numpy_or_cupy,
)


class _ChunkEvalForwardContext(Protocol):
    dm: Tensor
    mol: gto.Mole
    grids: Grid
    feature_function: feature_math.FeatureFunction
    blksize: int | None
    compile_feature_function: bool
    gpu: bool
    vectors_jvp: tuple[Tensor, ...]


class _ChunkEvalBackwardContext(Protocol):
    dm: Tensor
    mol: gto.Mole
    grids: Grid
    feature_function: feature_math.FeatureFunction
    blksize: int | None
    compile_feature_function: bool
    gpu: bool


def _active_cpu_aos(mol: gto.Mole, screen_index: np.ndarray) -> np.ndarray:
    """Expand a PySCF shell-screening mask into active AO indices."""
    active_shells = np.any(screen_index, axis=0)
    ao_loc = mol.ao_loc_nr()
    return np.flatnonzero(np.repeat(active_shells, np.diff(ao_loc)))


def partial_feature_function_over_aos(
    feature_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ao: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Bind an AO block to a feature function for block-local evaluation."""

    def partial_feature_function(dm: torch.Tensor) -> torch.Tensor:
        return feature_function(dm, ao)

    return partial_feature_function


def partial_vjp_function_over_tangents(
    func: Callable[[torch.Tensor], torch.Tensor],
    tangents: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Bind feature cotangents to a function for block-local VJP evaluation."""

    def reduced_vjp(primals: torch.Tensor) -> torch.Tensor:
        return torch.func.vjp(func, primals)[1](tangents)[0]

    return reduced_vjp


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
            Tensor,
            gto.Mole,
            Grid,
            feature_math.FeatureFunction,
            int | None,
            bool,
            bool,
            # The starred spelling requires Python 3.11.
            Unpack[tuple[Tensor, ...]],  # noqa: UP044
        ],
        output: torch.Tensor,
    ) -> None:
        if len(inputs) < 7:
            raise ValueError("ChunkEvalForward requires seven fixed inputs.")
        context = cast(_ChunkEvalForwardContext, ctx)
        (
            context.dm,
            context.mol,
            context.grids,
            context.feature_function,
            context.blksize,
            context.compile_feature_function,
            context.gpu,
            *vectors_jvp,
        ) = inputs
        context.vectors_jvp = tuple(vectors_jvp)
        ctx.save_for_backward(context.dm)

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
    def jvp(
        ctx: _ChunkEvalForwardContext, *grad_inputs: torch.Tensor | None
    ) -> torch.Tensor:
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
        ctx: _ChunkEvalForwardContext, *grad_outputs: torch.Tensor
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
        grads: list[Tensor | None] = [dm_grad]
        grads += [None] * 6

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
        context = cast(_ChunkEvalBackwardContext, ctx)
        (
            context.dm,
            context.mol,
            context.grids,
            context.feature_function,
            context.blksize,
            context.compile_feature_function,
            context.gpu,
            _feature_cotangent,
        ) = inputs
        ctx.save_for_backward(context.dm)

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
    def jvp(
        ctx: _ChunkEvalBackwardContext, *grad_inputs: torch.Tensor | None
    ) -> torch.Tensor:
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
        ctx: _ChunkEvalBackwardContext, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        grads: list[Tensor | None] = [ctx.dm * 0]
        grads += [None] * 6
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
    """Evaluate raw features over the full grid without block chunking."""
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
    return feature_function.forward(dm, ao)


def _resolve_ao_block_size(
    mol: gto.Mole,
    feature_function: feature_math.FeatureFunction,
    block_size: int | None,
    max_memory: int,
    gpu: bool,
) -> int | None:
    """Resolve an aligned CPU block size or delegate GPU sizing to its backend."""
    if gpu:
        if block_size is not None:
            raise ValueError("Setting custom block size is not supported on GPU.")
        return None

    if block_size is None:
        nao = mol.nao_nr()
        comp = (
            (feature_function.deriv + 1)
            * (feature_function.deriv + 2)
            * (feature_function.deriv + 3)
            // 6
        )
        backend_block_size = dft.gen_grid.BLKSIZE
        block_size = int(max_memory * 1e6 / ((comp + 1) * nao * 8 * backend_block_size))
        block_size = max(4, min(block_size, 1200)) * backend_block_size

    return block_size - block_size % dft.gen_grid.BLKSIZE


def auto_chunk(
    dm: torch.Tensor,
    mol: gto.Mole,
    grids: Grid,
    feature_function: feature_math.FeatureFunction,
    block_size: int | None = None,
    max_memory: int = 2000,
    gpu: bool = False,
) -> dict[str, torch.Tensor]:
    """Evaluate raw features with a memory-derived or explicit AO block size."""
    if gpu:
        check_gpu_imports_were_successful()
        if dm.device.type != "cuda":
            raise ValueError("Density matrix must be on the GPU when gpu=True.")

    blksize = _resolve_ao_block_size(mol, feature_function, block_size, max_memory, gpu)

    if blksize is not None and blksize >= grids.weights.shape[0]:
        features = non_chunk(
            dm.double(),
            mol,
            grids.coords,
            feature_function,
        )
    else:
        features = ChunkEvalForward.apply(
            dm.double(),
            mol,
            grids,
            feature_function,
            blksize,
            False,
            gpu,
        )
    return feature_function.to_dict(features)
