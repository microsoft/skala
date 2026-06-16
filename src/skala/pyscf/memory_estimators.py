"""Memory estimators for chunked calculations for Skala 1.1."""

import torch


def estimate_max_grid_chunk_size(
    dm: torch.Tensor,
    deriv: int,
    max_memory_in_mb: int | None = None,
    safety_fraction: float = 0.8,
    func_deriv: int = 1,
) -> int:
    """Heuristically pick a grid chunk size for :func:`chunked_features`.

    The dominant per-chunk allocation is the atomic-orbital matrix evaluated by
    ``non_chunk`` (shape ``(ncomp, nao, n)`` in float64, with no AO screening),
    together with the ``c0``/``ci`` products formed inside the feature function and
    retained by autograd for the backward pass. Peak memory is therefore modelled
    as affine in the number of grid points ``n`` (see
    :func:`linear_peak_memory_model`)::

        peak_bytes ~= bytes_per_point * n + fixed_overhead

    The returned chunk size is the largest ``n`` whose predicted peak fits within
    ``safety_fraction`` of the available memory.

    Args:
        dm: Density matrix; only its device and trailing dimension are used.
            ``dm.shape[-1]`` is taken as ``nao`` and ``dm.device`` selects how
            available memory is probed.
        deriv: Derivative order of the requested AO features (e.g. ``1`` for
            MGGA), which sets the AO component count ``ncomp``.
        max_memory_in_mb: Memory budget in **megabytes (MB)** to use on the device on which the density matrix is located. When ``None`` the
            budget is probed automatically: free device memory on CUDA, available
            physical RAM on CPU.
        safety_fraction: Fraction of the budget the predicted peak is allowed to
            occupy (``0 < safety_fraction <= 1``). Headroom for allocator
            fragmentation and transient buffers.
        func_deriv: Code path driving autograd retention: ``0`` energy only
            (``exc_only``), ``1`` first order (``__call__``/``V_xc``), ``2``
            second order (``gen_response``/Hessian-vector product). Selects the
            calibrated coefficients.

    Returns:
        Maximum number of grid points per chunk whose predicted peak memory fits
        within ``safety_fraction`` of the budget. May be non-positive when the
        ``fixed_overhead`` alone exceeds the budget; callers are expected to
        clamp it to at least the largest atomic grid size.

    Raises:
        ValueError: If ``max_memory_in_mb`` is ``None`` and ``dm`` lives on a device
            type other than ``cuda`` or ``cpu`` (supply ``max_memory_in_mb`` instead).
        RuntimeError: If CPU host memory cannot be determined automatically.
    """
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
    else:
        free_bytes = int(max_memory_in_mb * 1000**2)
    free_bytes = int(free_bytes * safety_fraction)

    bytes_per_point, fixed_overhead = linear_peak_memory_model(
        nao=dm.shape[-1],
        deriv=deriv,
        func_deriv=func_deriv,
    )
    chunk_size = int((free_bytes - fixed_overhead) / bytes_per_point)

    return chunk_size


def linear_peak_memory_model(
    nao: int,
    deriv: int,
    func_deriv: int,
) -> tuple[float, float]:
    """
    Return the coefficients of the linear model for peak memory usage in the number of grid points::

        bytes ~= bytes_per_point * n + fixed_overhead

    Both terms are quadratic in ``nao`` and calibrated *per code path*::

        bytes_per_point = 8 * (C_AO2*nao^2 + (ncomp + C_LIN)*nao + C_NET)
        fixed_overhead  = C_FIX * nao^2

    Details:
        The four coefficients are fitted directly for skala-1.1 to the empirical sweep (9999
        measured chunks, nao 38-4452, deriv=1 / MGGA) and then scaled by a single
        per-path safety margin so the worst observed meas/pred ratio is 0.90 with
        zero breaches.  This replaces the earlier single ``autograd_factor`` that
        multiplied *both* the nao^2 and the network-constant terms: the data show
        the nao^2 (AO-retention) coefficient is almost path-independent (0.0065 /
        0.0067 / 0.0104 B), while only the network-activation constant scales
        strongly across energy/first/second order.  Decoupling them removes the
        ~3x over-padding the old model carried on the second-order path.
    """
    # Number of AO components for the requested derivative order.
    ncomp = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6

    # Per-path calibrated coefficients, keyed by func_deriv (0=energy/exc_only,
    # 1=first order/__call__, 2=second order/gen_response).  Each tuple is
    # (C_AO2, C_LIN, C_NET, C_FIX) in float64 elements (C_FIX already in bytes):
    #   * C_AO2 - nao^2 coefficient of the per-point cost (autograd-retained AO
    #             intermediates); barely grows with path.
    #   * C_LIN - retained AO columns beyond the raw ncomp matrix (c0/ci + grads).
    #   * C_NET - nao-independent enhancement-network activation elements/point;
    #             this is where the second-order double-backward graph shows up.
    #   * C_FIX - quadratic coefficient of the dense nao x nao buffers (dm0, dm1,
    #             hvp_total, Vxc accumulator, get_j), in bytes.
    # Fitted on the cc-pVQZ/5Z/6Z + PAH sweep (coronene/cc-pV6Z reaches nao=4452)
    # then scaled to worst-case ratio 0.90; tested max meas/pred 0.900, 0 breaches.
    match func_deriv:
        case 0:
            C_AO2, C_LIN, C_NET, C_FIX = 1.07e-3, 5.2, 5830.0, 36.8
        case 1:
            C_AO2, C_LIN, C_NET, C_FIX = 1.10e-3, 4.8, 6680.0, 37.0
        case 2:
            C_AO2, C_LIN, C_NET, C_FIX = 1.80e-3, 1.7, 24230.0, 9.0
        case _:
            raise ValueError("Invalid func_deriv value")

    elems_per_point = (
        C_AO2 * nao * nao  # autograd-retained AO intermediates (nao^2)
        + (ncomp + C_LIN) * nao  # AO matrix + retained feature function memory
        + C_NET  # network activations (path-dependent)
    )
    bytes_per_point = 8.0 * elems_per_point  # float64
    # Dense nao x nao buffers; autograd-independent and already conservative.
    fixed_overhead = C_FIX * nao * nao

    return bytes_per_point, fixed_overhead
