# SPDX-License-Identifier: MIT

import torch


def pad_ragged(
    data: torch.Tensor, sizes: torch.Tensor, size_bound: int
) -> torch.Tensor:
    """Packs variable-length concatenated sequences into a batched tensor with padding.

    You can think of the static parameter `size_bound` as `sizes.max()`, but it is passed separately
    as an integer to avoid "data-dependent control flow". We want to avoid the shape of the output
    tensor depending on the value of the input tensors.

    Args:
        data: Tensor [total, *rest] where total = sum(sizes)
        sizes: 1D tensor [batch] of sequence lengths
        size_bound: Pad dimension size. If smaller than max(sizes), sequences will be cropped.

    Returns:
        Tensor [batch, size_bound, *rest], zero-padded.

    Example:
        >>> data = torch.tensor([1, 2, 3, 4, 5])  # two sequences: [1,2] and [3,4,5]
        >>> sizes = torch.tensor([2, 3])
        >>> pad_ragged(data, sizes, size_bound=4)
        tensor([[1, 2, 0, 0],
                [3, 4, 5, 0]])
    """
    if (sizes < 0).any():
        raise ValueError("sizes must contain only non-negative values")
    if data.shape[0] != sizes.sum():
        raise ValueError(
            f"data length {data.shape[0]} must equal sum of sizes {sizes.sum().item():.0f}"
        )

    batch_size = sizes.shape[0]
    rest_shape = data.shape[1:]

    # Fast path: single sequence - just pad directly
    if batch_size == 1:
        seq_len = data.shape[0]
        pad_len = size_bound - seq_len
        if pad_len > 0:
            padding = torch.zeros(
                pad_len, *rest_shape, dtype=data.dtype, device=data.device
            )
            return torch.cat([data, padding], dim=0).unsqueeze(0)
        return data[:size_bound].unsqueeze(0)

    col_indices = torch.broadcast_to(
        torch.arange(size_bound, device=data.device),
        (batch_size, size_bound),
    )  # [batch_size, size_bound]

    # Compute source indices
    ends = sizes.cumsum(0)
    starts = ends - sizes
    source_indices = starts.unsqueeze(1) + col_indices  # [batch_size, size_bound]
    clamped_indices = source_indices.clamp(
        0, data.shape[0] - 1
    )  # Don't exceed data size.

    # Gather values from data
    gathered = data[clamped_indices.view(-1)].view(batch_size, size_bound, *rest_shape)

    # Zero out invalid positions - expand mask for broadcasting over rest dimensions
    mask = col_indices < sizes.unsqueeze(1)  # [batch_size, size_bound]
    mask_expanded = mask.view(batch_size, size_bound, *([1] * len(rest_shape)))
    out = gathered * mask_expanded

    return out


def unpad_ragged(
    padded: torch.Tensor, sizes: torch.Tensor, total_size: int
) -> torch.Tensor:
    """Inverse of pad_ragged: extract variable-length sequences from a padded batch tensor.

    You can think of the static parameter ``total_size`` as ``sizes.sum()``, but it is passed
    separately as an integer to avoid data-dependent output shapes, keeping the function compatible
    with ``torch.compile(fullgraph=True)`` and ``torch.export``.

    Args:
        padded: Tensor [batch, size_bound, *rest] with zero-padding.
        sizes: 1D tensor [batch] of true sequence lengths.
        total_size: Total number of valid elements, i.e. sum(sizes). Passed as an integer to
            keep the output shape static.

    Returns:
        Tensor [total_size, *rest], with padding removed.

    Example:
        >>> padded = torch.tensor([[1, 2, 0, 0],
        ...                        [3, 4, 5, 0]])
        >>> sizes = torch.tensor([2, 3])
        >>> unpad_ragged(padded, sizes, total_size=5)
        tensor([1, 2, 3, 4, 5])
    """
    if (sizes < 0).any():
        raise ValueError("sizes must contain only non-negative values")

    batch_size = padded.shape[0]
    size_bound = padded.shape[1]
    rest_shape = padded.shape[2:]

    if total_size == 0:
        return torch.zeros(0, *rest_shape, dtype=padded.dtype, device=padded.device)

    # Fast path: single sequence - just slice
    if batch_size == 1:
        return padded[0, :total_size]

    # For each output position, find which batch it belongs to via binary search,
    # then gather the corresponding element from the padded tensor.
    ends = sizes.cumsum(0)
    starts = ends - sizes
    positions = torch.arange(total_size, device=padded.device)
    batch_id = torch.searchsorted(ends, positions, right=True)
    local_col = positions - starts[batch_id]
    src_idx = batch_id * size_bound + local_col

    flat_padded = padded.reshape(-1, *rest_shape)
    return flat_padded[src_idx]
