# SPDX-License-Identifier: MIT
# A copy of useful code from torch scatter
# https://github.com/rusty1s/pytorch_scatter/blob/96aa2e3587123ba4ef31820899d5e62141e9a4c2/torch_scatter/scatter.py

"""
Scatter operations for PyTorch tensors.

This module provides scatter operations similar to pytorch_scatter,
specifically scatter_sum for aggregating values at specified indices.
"""

import torch


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: None | torch.Tensor = None,
    dim_size: None | int = None,
) -> torch.Tensor:
    """
    Sum all values from the src tensor at indices specified in the index tensor.

    Parameters
    ----------
    src : torch.Tensor
        Source tensor containing values to scatter.
    index : torch.Tensor
        Index tensor specifying where to scatter values.
    dim : int, optional
        Dimension along which to scatter. Default: -1.
    out : torch.Tensor or None, optional
        Output tensor. If None, a new tensor is created.
    dim_size : int or None, optional
        Size of the output tensor along the scatter dimension.

    Returns
    -------
    torch.Tensor
        Tensor with scattered and summed values.
    """
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Broadcast src tensor to match the shape of other tensor along specified dimensions.

    Parameters
    ----------
    src : torch.Tensor
        Source tensor to broadcast.
    other : torch.Tensor
        Target tensor whose shape to match.
    dim : int
        Dimension along which to perform broadcasting.

    Returns
    -------
    torch.Tensor
        Broadcasted tensor with shape matching other.
    """
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src
