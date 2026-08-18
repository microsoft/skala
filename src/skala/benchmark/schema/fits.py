# SPDX-License-Identifier: MIT

"""Serialization helpers for benchmark scaling fits."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skala.benchmark.metrics import is_fit_combination

if TYPE_CHECKING:
    from skala.benchmark.fitting import PiecewiseFit


def fit_to_rows(
    fit: PiecewiseFit,
    *,
    env_id: str,
    basis: str,
    functional: str,
    metric: str,
    x_axis: str,
) -> list[dict[str, object]]:
    """Serialize one fit as one row per segment.

    Args:
        fit: Piecewise scaling fit to serialize.
        env_id: Benchmark environment identifier.
        basis: Basis-set name.
        functional: Functional name.
        metric: Cost metric represented by the fit.
        x_axis: System-size metric represented by x.

    Returns:
        JSON-serializable fit segment dictionaries.

    Raises:
        ValueError: If the metric/x-axis pair is not registered for fitting.
    """
    if not is_fit_combination(metric, x_axis):
        raise ValueError(f"unsupported fit combination: {metric!r}/{x_axis!r}")
    return [
        {
            "env_id": env_id,
            "basis": basis,
            "functional": functional,
            "metric": metric,
            "x_axis": x_axis,
            "segment_index": segment_index,
            "x_start": segment.x_start,
            "x_end": segment.x_end,
            "slope": segment.slope,
            "intercept": segment.intercept,
            "continuous": fit.continuous,
            "cv_score": fit.cv_score,
            "n_points": fit.n_points,
            "breakpoints": list(fit.breakpoints),
        }
        for segment_index, segment in enumerate(fit.segments)
    ]
