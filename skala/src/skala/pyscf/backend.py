# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
    TypeVar,
)

import numpy as np
import torch
from torch import Tensor

from pyscf import dft
from skala.typing import F64

GPU_EXCEPTION: BaseException | None = None

_ShapeT_co = TypeVar("_ShapeT_co", bound=tuple[int, ...], covariant=True)
_DTypeT_co = TypeVar("_DTypeT_co", bound=np.dtype[np.generic], covariant=True)

__all__ = [
    "KS",
    "Array",
    "ArrayF64",
    "Grid",
    "check_gpu_imports_were_successful",
    "dft_gpu",
    "from_numpy_or_cupy",
    "to_cupy",
    "to_numpy",
]


if TYPE_CHECKING:
    # During type checking, we do the same as during normal runtime, but without the try/except.
    import cupy
    from gpu4pyscf import dft as dft_gpu

    GPU_EXCEPTION = None

    Array: TypeAlias = (
        np.ndarray[_ShapeT_co, _DTypeT_co] | cupy.ndarray[_ShapeT_co, _DTypeT_co]
    )
    ArrayF64 = TypeVar(
        "ArrayF64",
        np.ndarray[Any, F64],
        cupy.ndarray[Any, F64],
    )
    Grid: TypeAlias = dft.Grids | dft_gpu.Grids
    KS: TypeAlias = dft.rks.RKS | dft.uks.UKS | dft_gpu.rks.RKS | dft_gpu.uks.UKS
else:
    try:
        import cupy
        from gpu4pyscf import dft as dft_gpu

        from skala.utils.torch_allocator import use_torch_mempool_in_cupy

        # Install this last because GPU4PySCF configures its own CuPy allocator during import.
        # Subsequent CuPy allocations use PyTorch's caching allocator
        if not torch.cuda.is_available():
            raise ImportError(
                "CUDA is not available; cannot configure CuPy to use the PyTorch allocator."
            )
        try:
            use_torch_mempool_in_cupy()
        except RuntimeError as e:
            raise ImportError(
                "Failed to configure CuPy to use the PyTorch allocator."
            ) from e

        Array = np.ndarray | cupy.ndarray
        ArrayF64 = TypeVar(
            "ArrayF64",
            np.ndarray[Any, F64],
            cupy.ndarray[Any, F64],
        )
        Grid = dft.Grids | dft_gpu.Grids
        KS = dft.rks.RKS | dft.uks.UKS | dft_gpu.rks.RKS | dft_gpu.uks.UKS

        GPU_EXCEPTION = None
    except ImportError as e:
        GPU_EXCEPTION = e
        dft_gpu = None

        Array = np.ndarray
        ArrayF64 = TypeVar("ArrayF64", bound=np.ndarray[Any, F64])
        Grid = dft.Grids
        KS = dft.rks.RKS | dft.uks.UKS


def check_gpu_imports_were_successful() -> None:
    if GPU_EXCEPTION is not None:
        raise GPU_EXCEPTION


def from_numpy_or_cupy(
    x: Array[_ShapeT_co, _DTypeT_co],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    transpose: bool = False,
) -> Tensor:
    if isinstance(x, np.ndarray):
        if x.flags.writeable:
            x_torch = torch.from_numpy(x)
        else:
            try:
                x.setflags(write=True)
            except ValueError:
                x_torch = torch.from_numpy(x.copy())
            else:
                try:
                    # Torch does not preserve NumPy's writeability flag. The restored
                    # flag guards NumPy access while this tensor keeps sharing storage.
                    x_torch = torch.from_numpy(x)
                finally:
                    x.setflags(write=False)
    else:
        x_torch = torch.from_dlpack(x)  # type: ignore[attr-defined]
    x_torch = x_torch.to(device=device, dtype=dtype)
    if transpose:
        return x_torch.transpose(-1, -2)
    else:
        return x_torch


def to_numpy(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def to_cupy(x: Tensor) -> cupy.ndarray:
    return cupy.from_dlpack(x.detach())
