# SPDX-License-Identifier: MIT
"""Use torch allocator in CuPy.

Adapted from pytorch_pfn_extras.
https://github.com/pfnet/pytorch-pfn-extras/blob/master/pytorch_pfn_extras/cuda/_allocator.py
"""

from typing import Any

import cupy
import torch

_allocator = None


def use_torch_mempool_in_cupy() -> None:
    """Use the PyTorch memory pool in CuPy.

    If non-default streams are used in PyTorch and CuPy,
    the current stream must be set to the same stream in both libraries
    before calling this function.

    Example:
        >>> cupy_stream = cupy.cuda.Stream(non_blocking=True)
        >>> torch_stream = torch.cuda.ExternalStream(
        ...     cupy_stream.ptr,
        ...     device=cupy_stream.device_id,
        ... )
        >>> with torch.cuda.stream(torch_stream), cupy_stream:
        ...     use_torch_mempool_in_cupy()
        ...     torch_array = torch.ones(10, device=torch_stream.device)
        ...     cupy_array = cupy.ones(10)
    """
    global _allocator

    _allocator = cupy.cuda.memory.PythonFunctionAllocator(_torch_alloc, _torch_free)
    cupy.cuda.set_allocator(_allocator.malloc)


def _torch_alloc(size: int, device_id: int) -> Any:
    torch_stream_ptr = torch.cuda.current_stream(device_id).cuda_stream
    cupy_stream_ptr = cupy.cuda.get_current_stream().ptr
    if torch_stream_ptr != cupy_stream_ptr:
        raise RuntimeError("The current stream set in PyTorch and CuPy must be same.")
    return torch.cuda.caching_allocator_alloc(size, device_id, torch_stream_ptr)


def _torch_free(mem_ptr: int, device_id: int) -> None:
    torch.cuda.caching_allocator_delete(mem_ptr)  # type: ignore[no-untyped-call]
