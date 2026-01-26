# SPDX-License-Identifier: MIT

from typing import (
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
)

import numpy as np
import torch
from pyscf import dft
from torch import Tensor

GPU_EXCEPTION: BaseException | None = None

__all__ = [
    "Array",
    "Grid",
    "KS",
    "dft_gpu",
    "check_gpu_imports_were_successful",
    "from_numpy_or_cupy",
    "to_numpy",
    "to_cupy",
]


if TYPE_CHECKING:
    # During type checking, we do the same as during normal runtime, but without the try/except.
    import cupy
    from gpu4pyscf import dft as dft_gpu

    GPU_EXCEPTION = None

    Array = TypeVar("Array", np.ndarray, cupy.ndarray)
    Grid: TypeAlias = dft.Grids | dft_gpu.Grids
    KS: TypeAlias = dft.rks.RKS | dft.uks.UKS | dft_gpu.rks.RKS | dft_gpu.uks.UKS

    # Reset the default CuPy memory allocator to avoid memory leak issues
    # that seem to arise when combining the custom allocator of gpu4pyscf with DLPack usage.
    cupy.cuda.set_allocator(cupy.get_default_memory_pool().malloc)
else:
    try:
        import cupy
        from gpu4pyscf import dft as dft_gpu

        Array = TypeVar("Array", np.ndarray, cupy.ndarray)
        Grid: TypeAlias = dft.Grids | dft_gpu.Grids
        KS: TypeAlias = dft.rks.RKS | dft.uks.UKS | dft_gpu.rks.RKS | dft_gpu.uks.UKS

        # Reset the default CuPy memory allocator to avoid memory leak issues
        # that seem to arise when combining the custom allocator of gpu4pyscf with DLPack usage.
        cupy.cuda.set_allocator(cupy.get_default_memory_pool().malloc)
        GPU_EXCEPTION = None
    except ImportError as e:
        GPU_EXCEPTION = e
        dft_gpu = None

        Array = TypeVar("Array", bound=np.ndarray)
        Grid: TypeAlias = dft.Grids
        KS: TypeAlias = dft.rks.RKS | dft.uks.UKS


def check_gpu_imports_were_successful() -> None:
    if GPU_EXCEPTION is not None:
        raise GPU_EXCEPTION


def from_numpy_or_cupy(
    x: Array,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    transpose: bool = False,
) -> Tensor:
    if isinstance(x, np.ndarray):
        x_torch = torch.from_numpy(x)
    else:
        x_torch = torch.from_dlpack(x)
    x_torch = x_torch.to(device=device, dtype=dtype)
    if transpose:
        return x_torch.transpose(-1, -2)
    else:
        return x_torch


def to_numpy(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def to_cupy(x: Tensor) -> "cupy.ndarray":
    return cupy.from_dlpack(x.detach())
