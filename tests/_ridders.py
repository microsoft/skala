"""Shared Ridders finite-difference helpers for gradient tests.

Used by both ``test_pyscf_gradients.py`` and ``test_gpu4pyscf_gradients.py``.
The helpers are device-aware (CPU or CUDA) via ``x.device``.
"""

from collections.abc import Callable

import torch


def num_diff_ridders(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    initial_step: float = 0.01,
    step_div: float = 1.414,
    max_tab: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Numerical derivative via extrapolation, so this is expensive, but will be accurate, so good for testing.
    The second return value is the estimate error. If this is large, try reducing the initial_step.

    func:         function to differentiate
    x:            position where to evaluate the derivative
    initial_step: initial step size
    max_tab:      amount of different steps tried
    step_div:     amount by which the step is divided
    """
    d_estimate = torch.empty((max_tab, max_tab), dtype=x.dtype, device=x.device)

    step = initial_step
    step_div_2 = step_div**2
    err = torch.tensor(torch.finfo(x.dtype).max, dtype=x.dtype, device=x.device)
    prev_err = err

    d_estimate[0, 0] = (func(x + step) - func(x - step)) / (2 * step)
    prev_deriv = d_estimate[0, 0]
    num_deriv = prev_deriv
    for i in range(1, max_tab):
        step /= step_div
        d_estimate[i, 0] = (func(x + step) - func(x - step)) / (2 * step)
        # use this new central difference estimate to eliminate next leading errors from previous estimates
        factor = step_div_2
        for order in range(i):
            # each step in order eliminates the term of order ~ step**(2order)
            factor *= step_div_2
            d_estimate[i, order + 1] = (
                factor * d_estimate[i, order] - d_estimate[i - 1, order]
            ) / (factor - 1.0)
            # estimate error as the max difference w.r.t. the two lower order options
            err_est = torch.max(
                torch.abs(d_estimate[i, order + 1] - d_estimate[i, order]),
                torch.abs(d_estimate[i, order + 1] - d_estimate[i - 1, order]),
            )
            if err_est <= err:
                err = err_est
                num_deriv = d_estimate[i, order + 1]

        if torch.abs(d_estimate[i, i] - d_estimate[i - 1, i - 1]) >= 2 * err and i > 1:
            # subtracting different step sizes does not work anymore to reduce error
            # suspect last step-size is too small, so don't trust -> stop and return previous best
            return prev_deriv, prev_err

        prev_deriv = num_deriv
        prev_err = err

    return num_deriv, err


def num_grad_ridders(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    initial_step: float = 0.01,
    step_div: float = 1.414,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recursively calculates the partial derivative w.r.t. all elements of x over all dimensions."""

    def func_1d_red(xi: torch.Tensor) -> torch.Tensor:
        x_ = x.clone()
        x_[i] = xi
        return func(x_)

    grad = torch.empty_like(x)
    err = torch.empty_like(x)

    if len(x.size()) == 1:
        for i, xi in enumerate(x):
            grad[i], err[i] = num_diff_ridders(
                func_1d_red, xi, initial_step=initial_step, step_div=step_div
            )
    else:
        for i, xi in enumerate(x):
            grad[i], err[i] = num_grad_ridders(
                func_1d_red, xi, initial_step=initial_step, step_div=step_div
            )

    return grad, err
