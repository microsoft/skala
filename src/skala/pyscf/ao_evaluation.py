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


def _active_cpu_ao_indices(mol: gto.Mole, screen_index: np.ndarray) -> np.ndarray:
    """Expand active shells in a PySCF screen-index slice to AO indices.

    A shell is active for the grid block if it is nonzero in any of the
    ``BLKSIZE``-point rows covered by that block. ``ao_loc_nr`` maps each shell
    to its contiguous range in PySCF's AO ordering.
    """
    active_shells = np.any(screen_index, axis=0)
    ao_loc = mol.ao_loc_nr()
    return np.flatnonzero(np.repeat(active_shells, np.diff(ao_loc)))


def partial_feature_function_over_ao_values(
    feature_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ao_values: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Bind evaluated AO values to a feature function for one grid block."""

    def partial_feature_function(dm: torch.Tensor) -> torch.Tensor:
        return feature_function(dm, ao_values)

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
    """Evaluated AO data and index metadata for one contiguous grid block.

    ``ao_values`` contains only the active AO rows when screening is enabled.
    ``active_ao_indices`` identifies those rows in the backend's current AO
    ordering; ``None`` means that ``ao_values`` contains every AO. The CPU
    backend uses PySCF's native AO order, while the GPU backend uses
    GPU4PySCF's sorted AO order until the completed matrix is restored.
    """

    ao_values: Tensor
    active_ao_indices: Tensor | None
    grid_slice: slice

    def select_active_ao_submatrix(self, matrix: Tensor) -> Tensor:
        """Gather the square matrix corresponding to this block's AO values."""
        if self.active_ao_indices is None:
            return matrix
        return matrix[
            ..., self.active_ao_indices[:, None], self.active_ao_indices[None, :]
        ]

    def add_active_ao_submatrix(self, matrix: Tensor, block_result: Tensor) -> None:
        """Add a block result into its active rows and columns in ``matrix``."""
        if self.active_ao_indices is None:
            matrix += block_result
        else:
            matrix[
                ..., self.active_ao_indices[:, None], self.active_ao_indices[None, :]
            ] += block_result


def _evaluate_feature_block(
    feature_function: feature_math.FeatureFunction,
    block: _AOBlock,
    active_dm_submatrix: Tensor,
    compile_feature_function: bool,
    feature_cotangent: Tensor | None = None,
) -> Tensor:
    """Evaluate one active-AO feature block or its feature-space VJP."""
    partial_func = partial_feature_function_over_ao_values(
        feature_function, block.ao_values
    )
    if feature_cotangent is not None:
        partial_func = partial_vjp_function_over_tangents(
            partial_func, feature_cotangent[..., block.grid_slice]
        )

    if compile_feature_function:
        return torch.compile(partial_func)(active_dm_submatrix)
    return partial_func(active_dm_submatrix)


class _CPUAOBlockLoop:
    """Yield CPU AO values screened with the exact PySCF screen-index table.

    PySCF evaluates AOs with ``grids.non0tab``, whose rows each describe one
    ``dft.gen_grid.BLKSIZE``-point range and whose columns describe shells. The
    loop converts the rows covered by each yielded grid block into AO indices,
    slices the evaluated AO tensor, and records those indices for density-matrix
    gathering and result scattering. If every shell is active for a particular
    block, the loop keeps the full AO tensor and records ``None`` instead of an
    identity index. Whether a block is dense can therefore vary across the
    rows of one ``non0tab`` table.

    The second item yielded by ``NumInt.block_loop`` is intentionally ignored.
    Despite being called ``mask`` by PySCF, it is not the authoritative
    screening table for that block. After AO evaluation, PySCF may replace it
    with ``None`` to request dense downstream contractions. That policy depends
    on the total grid's ``ALIGNMENT_UNIT`` divisibility and PySCF's sparsity
    heuristic, not on whether shells were screened during AO evaluation. Using
    that yielded value would therefore make Skala's active AO set depend on
    contraction policy and grid alignment. Reading the exact rows from
    ``grids.non0tab`` preserves the screening information actually used for AO
    evaluation.
    """

    def __init__(
        self,
        dm: Tensor,
        mol: gto.Mole,
        grids: Grid,
        feature_function: feature_math.FeatureFunction,
        blksize: int | None,
    ) -> None:
        self.dm = dm
        self.mol = mol
        assert isinstance(grids, dft.Grids)
        self.grids = grids
        self.feature_function = feature_function
        self.blksize = blksize
        self.numint = dft.numint.NumInt()

    def order_aos(self, matrix: Tensor) -> Tensor:
        return matrix

    def restore_ao_order(self, matrix: Tensor) -> Tensor:
        return matrix

    def _active_ao_indices(
        self,
        non0tab: np.ndarray,
        grid_start: int,
        grid_end: int,
    ) -> Tensor | None:
        """Create active AO indices for the exact rows covering a grid block.

        ``NumInt.block_loop`` requires CPU block sizes to be integer multiples
        of ``dft.gen_grid.BLKSIZE``. Consequently every non-final block starts
        and ends on screen-index row boundaries; the ceiling for ``grid_end``
        also includes the final partial row. All shells active in any covered
        row are included because one AO tensor is shared by the whole grid
        block.

        Returns ``None`` when the covered rows activate every AO. ``_AOBlock``
        uses that value as its dense sentinel, avoiding identity indexing of AO
        values and density matrices. An empty tensor means that no AO is active
        and the caller can omit the block entirely.

        This method requires the authoritative screen-index table and must not
        consume the mask yielded by ``NumInt.block_loop``. PySCF may set that
        yielded mask to ``None`` after AO evaluation when sparse contraction is
        unsuitable, even though ``non0tab`` still contains the exact
        shell-screening data. The caller handles a missing ``non0tab`` as the
        genuinely dense case.
        """
        row_start = grid_start // dft.gen_grid.BLKSIZE
        row_end = (grid_end + dft.gen_grid.BLKSIZE - 1) // dft.gen_grid.BLKSIZE
        block_non0tab = non0tab[row_start:row_end]
        if np.all(np.any(block_non0tab, axis=0)):
            return None
        return torch.as_tensor(
            _active_cpu_ao_indices(self.mol, block_non0tab),
            device=self.dm.device,
            dtype=torch.long,
        )

    def __iter__(self) -> Iterator[_AOBlock]:
        non0tab = self.grids.non0tab

        end = 0
        for backend_ao_values, _, block_weights, _ in self.numint.block_loop(
            mol=self.mol,
            grids=self.grids,
            nao=self.mol.nao,
            deriv=self.feature_function.deriv,
            blksize=self.blksize,
            non0tab=non0tab,
        ):
            start, end = end, end + block_weights.size
            ao_values = (
                torch.from_numpy(backend_ao_values)
                .to(device=self.dm.device, dtype=self.dm.dtype)
                .transpose(-1, -2)
            )
            active_ao_indices = (
                None
                if non0tab is None
                else self._active_ao_indices(non0tab, start, end)
            )
            if active_ao_indices is None:
                yield _AOBlock(ao_values, None, slice(start, end))
                continue

            if active_ao_indices.numel() == 0:
                continue
            ao_values = ao_values[..., active_ao_indices, :]
            yield _AOBlock(ao_values, active_ao_indices, slice(start, end))


class _GPUAOBlockLoop:
    """Yield GPU4PySCF AO values and compact indices in sorted AO order."""

    def __init__(
        self,
        dm: Tensor,
        mol: gto.Mole,
        grids: Grid,
        feature_function: feature_math.FeatureFunction,
        blksize: int | None,
    ) -> None:
        check_gpu_imports_were_successful()
        self.dm = dm
        self.mol = mol
        self.grids = grids
        self.feature_function = feature_function
        self.blksize = blksize
        self.numint = dft_gpu.numint.NumInt().build(mol, grids.coords)
        self.numint.grid_blksize = blksize
        self.sort_idx = torch.as_tensor(self.numint.gdftopt._ao_idx, device=dm.device)
        self.unsort_idx = torch.argsort(self.sort_idx)

    def order_aos(self, matrix: Tensor) -> Tensor:
        return matrix[..., self.sort_idx[:, None], self.sort_idx[None, :]]

    def restore_ao_order(self, matrix: Tensor) -> Tensor:
        return matrix[..., self.unsort_idx[:, None], self.unsort_idx[None, :]]

    def __iter__(self) -> Iterator[_AOBlock]:
        end = 0
        for (
            backend_ao_values,
            active_ao_indices,
            block_weights,
            _,
        ) in self.numint.block_loop(
            mol=self.mol,
            grids=self.grids,
            nao=self.mol.nao,
            deriv=self.feature_function.deriv,
            blksize=self.blksize,
            non0tab=None,
            # GPU4PySCF otherwise omits zero-AO blocks, shifting later grid slices.
            strict_grid_order=True,
        ):
            start, end = end, end + block_weights.size
            if active_ao_indices.size == 0:
                continue
            yield _AOBlock(
                torch.from_dlpack(backend_ao_values),
                torch.from_dlpack(active_ao_indices),
                slice(start, end),
            )


def _make_ao_block_loop(
    dm: Tensor,
    mol: gto.Mole,
    grids: Grid,
    feature_function: feature_math.FeatureFunction,
    blksize: int | None,
    gpu: bool,
) -> _CPUAOBlockLoop | _GPUAOBlockLoop:
    loop_type = _GPUAOBlockLoop if gpu else _CPUAOBlockLoop
    return loop_type(dm, mol, grids, feature_function, blksize)


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
        block_loop = _make_ao_block_loop(dm, mol, grids, feature_function, blksize, gpu)

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
            active_dm_submatrix = block.select_active_ao_submatrix(
                evaluation_dm_ordered
            )
            temp_feature = _evaluate_feature_block(
                feature_function,
                block,
                active_dm_submatrix,
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
        block_loop = _make_ao_block_loop(dm, mol, grids, feature_function, blksize, gpu)
        dm_ordered = block_loop.order_aos(dm)

        out = torch.zeros_like(dm)
        for block in block_loop:
            active_dm_submatrix = block.select_active_ao_submatrix(dm_ordered)
            block_result = _evaluate_feature_block(
                feature_function,
                block,
                active_dm_submatrix,
                compile_feature_function,
                feature_cotangent,
            )
            block.add_active_ao_submatrix(out, block_result)
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
