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
    "KS",
    "Array",
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

    Array = TypeVar("Array", np.ndarray, cupy.ndarray)
    Grid: TypeAlias = dft.Grids | dft_gpu.Grids
    KS: TypeAlias = dft.rks.RKS | dft.uks.UKS | dft_gpu.rks.RKS | dft_gpu.uks.UKS
else:
    try:
        import cupy
        from gpu4pyscf import dft as dft_gpu

        from skala.utils.torch_allocator import use_torch_mempool_in_cupy

        # Install this last because GPU4PySCF configures its own CuPy allocator during import.
        # Subsequent CuPy allocations use PyTorch's caching allocator
        use_torch_mempool_in_cupy()

        Array = TypeVar("Array", np.ndarray, cupy.ndarray)
        Grid: TypeAlias = dft.Grids | dft_gpu.Grids
        KS: TypeAlias = dft.rks.RKS | dft.uks.UKS | dft_gpu.rks.RKS | dft_gpu.uks.UKS

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
        x_torch = torch.from_dlpack(x)  # type: ignore[attr-defined]
    x_torch = x_torch.to(device=device, dtype=dtype)
    if transpose:
        return x_torch.transpose(-1, -2)
    else:
        return x_torch


def to_numpy(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def to_cupy(x: Tensor) -> "cupy.ndarray":
    return cupy.from_dlpack(x.detach())
