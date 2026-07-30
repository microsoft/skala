# SPDX-License-Identifier: MIT

from typing import Any

import cupy
import torch

_allocator = None


def use_torch_mempool_in_cupy() -> None:
    """Use the PyTorch memory pool in CuPy.

    If you want to use PyTorch's memory pool and non-default CUDA streams,
    streams must be created and managed using PyTorch (using
    `torch.cuda.Stream()` and `pytorch_pfn_extras.cuda.stream(stream)`).
    """
    global _allocator

    _allocator = cupy.cuda.memory.PythonFunctionAllocator(_torch_alloc, _torch_free)
    cupy.cuda.set_allocator(_allocator.malloc)


def _torch_alloc(size: int, device_id: int) -> Any:
    torch_stream_ptr = torch.cuda.current_stream().cuda_stream
    cupy_stream_ptr = cupy.cuda.get_current_stream().ptr
    if torch_stream_ptr != cupy_stream_ptr:
        raise RuntimeError("The current stream set in PyTorch and CuPy must be same.")
    return torch.cuda.caching_allocator_alloc(size, device_id, torch_stream_ptr)


def _torch_free(mem_ptr: int, device_id: int) -> None:
    torch.cuda.caching_allocator_delete(mem_ptr)  # type: ignore[no-untyped-call]
