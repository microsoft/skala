"""Memory estimators for screened calculations with Skala 1.1."""

import torch

_MODEL_ELEMENTS_PER_GRID_POINT = {
    0: 5830,
    1: 6680,
    2: 24230,
}
_GLOBAL_DENSE_BYTES_PER_AO_SQUARED = {
    0: 36.8,
    1: 37.0,
    2: 9.0,
}


def estimate_max_model_atoms_per_chunk(
    dm: torch.Tensor,
    atomic_grid_sizes: torch.Tensor,
    nfeatures: int,
    max_memory_in_mb: int | None = None,
    safety_fraction: float = 0.8,
    func_deriv: int = 1,
) -> dict[int, int]:
    """Estimate an atom limit for every homogeneous atomic-grid-size group.

    AO evaluation is completed globally before model chunking starts. Its AO-sized
    terms therefore do not scale with each model chunk. Once chunks contain only
    equal-sized atomic grids of size ``g``, the model's padded point count equals
    its real point count, and its chunk-local peak for ``a`` atoms is modelled as::

        chunk_bytes ~= model_bytes_per_point * g * a

    For an explicit memory budget, globally live allocations are estimated from
    ``atomic_grid_sizes`` and subtracted once. A probed CUDA free-memory value
    already excludes allocations currently resident on the device, so the global
    footprint is not subtracted a second time in that case.

    Args:
        dm: Density matrix; its device selects how available memory is probed.
        atomic_grid_sizes: Number of grid points belonging to each atom.
        nfeatures: Number of globally stored raw features per grid point.
        max_memory_in_mb: Memory budget in megabytes (MB). When ``None``, free
            device memory is probed automatically on CUDA.
        safety_fraction: Fraction of the budget the predicted peak is allowed to
            occupy (``0 < safety_fraction <= 1``). Headroom for allocator
            fragmentation and transient buffers.
        func_deriv: Code path driving autograd retention: ``0`` energy only
            (``exc_only``), ``1`` first order (``__call__``/``V_xc``), ``2``
            second order (``gen_response``/Hessian-vector product). Selects the
            calibrated coefficients.
    Returns:
        Mapping from each distinct atomic grid size to the maximum number of atoms
        of that size per model chunk. Values may be non-positive when the global
        footprint exceeds the budget; callers are expected to clamp them to one.

    Raises:
        ValueError: If ``safety_fraction`` is outside ``(0, 1]``, or if
            ``max_memory_in_mb`` is ``None`` and ``dm`` lives on a device type
            other than ``cuda`` or ``cpu`` (supply ``max_memory_in_mb`` instead).
        RuntimeError: If CPU host memory cannot be determined automatically.
    """
    if not 0 < safety_fraction <= 1:
        raise ValueError("safety_fraction must be greater than 0 and at most 1")
    if atomic_grid_sizes.numel() == 0 or torch.any(atomic_grid_sizes <= 0):
        raise ValueError("atomic_grid_sizes must contain positive values")

    if max_memory_in_mb is None:
        match dm.device.type:
            case "cuda":
                free_bytes, _ = torch.cuda.mem_get_info(dm.device)
                free_bytes += torch.cuda.memory_reserved(
                    dm.device
                ) - torch.cuda.memory_allocated(
                    dm.device
                )  # include reserved but unused pytorch memory
            case "cpu":
                raise ValueError(
                    "Automatic CPU memory estimation is not implemented. Supply max_memory_in_mb explicitly."
                )
            case _:
                raise ValueError(
                    f"Unsupported device type: {dm.device.type} for memory estimation. Supply max_memory_in_mb explicitly."
                )
        available_memory = int(free_bytes * safety_fraction)
    else:
        free_bytes = int(max_memory_in_mb * 1000**2)
        available_memory = int(free_bytes * safety_fraction)
        available_memory -= estimate_global_screened_buffer_memory(
            dm, nfeatures, atomic_grid_sizes, func_deriv
        )

    bytes_per_point = estimate_model_memory_per_grid_point(func_deriv)
    return {
        grid_size: available_memory // (grid_size * bytes_per_point)
        for grid_size in map(int, torch.unique(atomic_grid_sizes).tolist())
    }


def estimate_model_memory_per_grid_point(func_deriv: int) -> int:
    """Return calibrated chunk-local Skala memory per homogeneous grid point."""
    try:
        elements_per_point = _MODEL_ELEMENTS_PER_GRID_POINT[func_deriv]
    except KeyError as error:
        raise ValueError("Invalid func_deriv value") from error
    return 8 * elements_per_point


def estimate_global_raw_feature_buffer_memory(
    dm: torch.Tensor,
    nfeatures: int,
    ngrids: int,
    func_deriv: int,
) -> int:
    """Estimate full-grid raw-feature storage for global screened evaluation.

    First order keeps sorted and atom-major feature values plus atom-major and
    sorted cotangents. Second order additionally keeps an atom-major feature JVP,
    an atom-major model Hessian action, and its sorted copy.

    Args:
        dm: Density matrix whose leading dimensions determine the spin batches.
        nfeatures: Number of raw AO-derived features per grid point.
        ngrids: Total number of molecular grid points.
        func_deriv: Functional derivative order, either first or second.

    Returns:
        Estimated bytes occupied by global raw-feature buffers.

    Raises:
        ValueError: If ``func_deriv`` is not first or second order.
    """
    match func_deriv:
        case 1:
            buffer_count = 4
        case 2:
            buffer_count = 5
        case _:
            raise ValueError("Global screened features support func_deriv 1 or 2")

    batch_size = dm.numel() // (dm.shape[-2] * dm.shape[-1])
    return buffer_count * batch_size * nfeatures * ngrids * 8


def estimate_global_screened_buffer_memory(
    dm: torch.Tensor,
    nfeatures: int,
    atomic_grid_sizes: torch.Tensor,
    func_deriv: int,
) -> int:
    """Estimate globally live buffers whose lifetimes overlap model chunks."""
    try:
        dense_bytes_per_ao_squared = _GLOBAL_DENSE_BYTES_PER_AO_SQUARED[func_deriv]
    except KeyError as error:
        raise ValueError("Invalid func_deriv value") from error

    raw_feature_bytes = estimate_global_raw_feature_buffer_memory(
        dm, nfeatures, int(atomic_grid_sizes.sum().item()), func_deriv
    )
    dense_buffer_bytes = int(dense_bytes_per_ao_squared * dm.shape[-1] ** 2)
    return raw_feature_bytes + dense_buffer_bytes
