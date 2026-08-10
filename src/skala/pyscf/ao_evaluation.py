# SPDX-License-Identifier: MIT

"""Blockwise atomic-orbital feature evaluation and custom autograd."""

from collections.abc import Iterator
from typing import NamedTuple, Protocol, TypeAlias, cast

import numpy as np
import torch
from pyscf import dft, gto
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from torch.utils.dlpack import from_dlpack

from skala.features import FeatureMap
from skala.pyscf import feature_math
from skala.pyscf.backend import (
    Array,
    Grid,
    check_gpu_imports_were_successful,
    dft_gpu,
    from_numpy_or_cupy,
)

_ScreenIndex: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
_AOIndices: TypeAlias = np.ndarray[tuple[int], np.dtype[np.intp]]


class _ChunkEvalContext(Protocol):
    mol: gto.Mole
    grids: Grid
    feature_function: feature_math.LinearFeature
    blksize: int | None
    compile_feature_function: bool
    spin_shape: torch.Size
    output_device: torch.device


def _active_cpu_ao_indices(mol: gto.Mole, screen_index: _ScreenIndex) -> _AOIndices:
    """Expand active shells in a PySCF screen-index slice to AO indices.

    A shell is active for the grid block if it is nonzero in any of the
    ``BLKSIZE``-point rows covered by that block. ``ao_loc_nr`` maps each shell
    to its contiguous range in PySCF's AO ordering.
    """
    active_shells = np.any(screen_index, axis=0)
    ao_loc = mol.ao_loc_nr()
    return np.flatnonzero(np.repeat(active_shells, np.diff(ao_loc)))


class _AOBlock(NamedTuple):
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
    feature_function: feature_math.LinearFeature,
    block: _AOBlock,
    active_dm_submatrix: Tensor | None,
    compile_feature_function: bool,
    feature_cotangent: Tensor | None = None,
) -> Tensor:
    """Evaluate one active-AO feature block or its feature-space VJP."""
    if feature_cotangent is not None:
        local_cotangent = feature_cotangent[..., block.grid_slice]
        if compile_feature_function:
            return torch.compile(feature_function.vjp)(block.ao_values, local_cotangent)
        return feature_function.vjp(block.ao_values, local_cotangent)

    if active_dm_submatrix is None:
        raise ValueError("Feature evaluation requires a density matrix.")
    if compile_feature_function:
        return torch.compile(feature_function.forward)(
            active_dm_submatrix, block.ao_values
        )
    return feature_function(active_dm_submatrix, block.ao_values)


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
        mol: gto.Mole,
        grids: Grid,
        feature_function: feature_math.LinearFeature,
        blksize: int | None,
    ) -> None:
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
        non0tab: _ScreenIndex,
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
        return torch.from_numpy(_active_cpu_ao_indices(self.mol, block_non0tab))

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
            ao_values = torch.from_numpy(backend_ao_values).transpose(-1, -2)
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
        device: torch.device,
        mol: gto.Mole,
        grids: Grid,
        feature_function: feature_math.LinearFeature,
        blksize: int | None,
    ) -> None:
        check_gpu_imports_were_successful()
        self.mol = mol
        self.grids = grids
        self.feature_function = feature_function
        self.blksize = blksize
        self.numint = dft_gpu.numint.NumInt().build(mol, grids.coords)
        self.numint.grid_blksize = blksize
        self.sort_idx = torch.as_tensor(self.numint.gdftopt._ao_idx, device=device)
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
                from_dlpack(backend_ao_values),
                from_dlpack(active_ao_indices),
                slice(start, end),
            )


def _make_ao_block_loop(
    device: torch.device,
    mol: gto.Mole,
    grids: Grid,
    feature_function: feature_math.LinearFeature,
    blksize: int | None,
) -> _CPUAOBlockLoop | _GPUAOBlockLoop:
    if device.type == "cuda":
        return _GPUAOBlockLoop(device, mol, grids, feature_function, blksize)
    return _CPUAOBlockLoop(mol, grids, feature_function, blksize)


class ChunkEvalForward(Function):
    @staticmethod
    def setup_context(
        ctx: FunctionCtx,
        inputs: tuple[
            Tensor,
            gto.Mole,
            Grid,
            feature_math.LinearFeature,
            int | None,
            bool,
        ],
        output: torch.Tensor,
    ) -> None:
        context = cast(_ChunkEvalContext, ctx)
        (
            dm,
            context.mol,
            context.grids,
            context.feature_function,
            context.blksize,
            context.compile_feature_function,
        ) = inputs
        context.spin_shape = dm.shape[:-2]
        context.output_device = output.device

    @staticmethod
    def forward(
        dm: torch.Tensor,
        mol: gto.Mole,
        grids: Grid,
        feature_function: feature_math.LinearFeature,
        blksize: int | None,
        compile_feature_function: bool,
    ) -> torch.Tensor:
        ngrids = grids.weights.size
        block_loop = _make_ao_block_loop(
            dm.device, mol, grids, feature_function, blksize
        )

        features = torch.zeros(
            *dm.shape[:-2],
            feature_function.nfeats,
            ngrids,
            device=dm.device,
            dtype=dm.dtype,
        )
        evaluation_dm_ordered = block_loop.order_aos(dm)
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
    def jvp(ctx: _ChunkEvalContext, *grad_inputs: torch.Tensor | None) -> torch.Tensor:
        dm_tangent = grad_inputs[0]
        if dm_tangent is None:
            return torch.zeros(
                *ctx.spin_shape,
                ctx.feature_function.nfeats,
                ctx.grids.weights.size,
                device=ctx.output_device,
                dtype=torch.float64,
            )
        return cast(
            Tensor,
            ChunkEvalForward.apply(
                dm_tangent,
                ctx.mol,
                ctx.grids,
                ctx.feature_function,
                ctx.blksize,
                ctx.compile_feature_function,
            ),
        )

    @staticmethod
    def backward(
        ctx: _ChunkEvalContext, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        feature_cotangent = grad_outputs[0]
        dm_cotangent = ChunkEvalBackward.apply(
            feature_cotangent,
            ctx.mol,
            ctx.grids,
            ctx.feature_function,
            ctx.blksize,
            ctx.compile_feature_function,
        )
        # PyTorch expects one gradient slot per forward input; the remaining
        # arguments are AO-evaluation metadata and are not differentiable.
        return dm_cotangent, None, None, None, None, None


class ChunkEvalBackward(Function):
    @staticmethod
    def setup_context(
        ctx: FunctionCtx,
        inputs: tuple[
            torch.Tensor,
            gto.Mole,
            Grid,
            feature_math.LinearFeature,
            int | None,
            bool,
        ],
        output: torch.Tensor,
    ) -> None:
        context = cast(_ChunkEvalContext, ctx)
        (
            feature_cotangent,
            context.mol,
            context.grids,
            context.feature_function,
            context.blksize,
            context.compile_feature_function,
        ) = inputs
        context.spin_shape = feature_cotangent.shape[:-2]
        context.output_device = output.device

    @staticmethod
    def forward(
        feature_cotangent: torch.Tensor,
        mol: gto.Mole,
        grids: Grid,
        feature_function: feature_math.LinearFeature,
        blksize: int | None,
        compile_feature_function: bool,
    ) -> torch.Tensor:
        block_loop = _make_ao_block_loop(
            feature_cotangent.device, mol, grids, feature_function, blksize
        )

        nao = mol.nao_nr()
        out = feature_cotangent.new_zeros(*feature_cotangent.shape[:-2], nao, nao)
        for block in block_loop:
            block_result = _evaluate_feature_block(
                feature_function,
                block,
                None,
                compile_feature_function,
                feature_cotangent,
            )
            block.add_active_ao_submatrix(out, block_result)
        return block_loop.restore_ao_order(out)

    @staticmethod
    def jvp(ctx: _ChunkEvalContext, *grad_inputs: torch.Tensor | None) -> torch.Tensor:
        feature_cotangent_tangent = grad_inputs[0]
        if feature_cotangent_tangent is None:
            nao = ctx.mol.nao_nr()
            return torch.zeros(
                *ctx.spin_shape,
                nao,
                nao,
                device=ctx.output_device,
                dtype=torch.float64,
            )
        return cast(
            Tensor,
            ChunkEvalBackward.apply(
                feature_cotangent_tangent,
                ctx.mol,
                ctx.grids,
                ctx.feature_function,
                ctx.blksize,
                ctx.compile_feature_function,
            ),
        )

    @staticmethod
    def backward(
        ctx: _ChunkEvalContext, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        feature_cotangent_grad = ChunkEvalForward.apply(
            grad_outputs[0],
            ctx.mol,
            ctx.grids,
            ctx.feature_function,
            ctx.blksize,
            ctx.compile_feature_function,
        )
        # PyTorch expects one gradient slot per forward input; the remaining
        # arguments are AO-evaluation metadata and are not differentiable.
        return feature_cotangent_grad, None, None, None, None, None


def evaluate_full_grid(
    dm: torch.Tensor,
    mol: gto.Mole,
    coords: Array,
    feature_function: feature_math.LinearFeature,
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
    feature_function: feature_math.LinearFeature,
    block_size: int | None,
    max_memory: int,
    gpu: bool,
) -> int | None:
    """Resolve an aligned CPU block size or delegate GPU sizing to its backend."""
    if gpu:
        if block_size is None:
            return None
        raise ValueError("Setting custom block size is not supported on GPU.")

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
    feature_function: feature_math.LinearFeature,
    block_size: int | None = None,
    max_memory: int = 2000,
    gpu: bool = False,
) -> FeatureMap:
    """Evaluate raw features with a memory-derived or explicit AO block size."""
    if gpu:
        check_gpu_imports_were_successful()
        if dm.device.type != "cuda":
            raise ValueError("Density matrix must be on the GPU when gpu=True.")

    blksize = _resolve_ao_block_size(mol, feature_function, block_size, max_memory, gpu)

    if blksize is not None and blksize >= grids.weights.shape[0]:
        features = evaluate_full_grid(dm.double(), mol, grids.coords, feature_function)
    else:
        features = ChunkEvalForward.apply(
            dm.double(), mol, grids, feature_function, blksize, False
        )
    return feature_function.to_dict(features)
