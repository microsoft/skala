# SPDX-License-Identifier: MIT

"""Tests for piecewise log-log benchmark scaling fits."""

import math

import numpy as np
import pytest

from skala.benchmark.fitting import fit_piecewise_loglog, smooth_stacked_fractions
from skala.benchmark.schema.fits import fit_to_rows


@pytest.mark.parametrize("exponent", [1.0, 2.0, 3.0])
def test_recovers_single_power_law(exponent: float) -> None:
    x = np.geomspace(10.0, 3000.0, 40)
    y = 2.5 * x**exponent

    fit = fit_piecewise_loglog(x, y, random_state=17)

    assert fit.n_points == len(x)
    assert fit.breakpoints == []
    assert fit.continuous
    assert fit.segments[0].slope == pytest.approx(exponent, abs=1e-10)
    assert fit.cv_score < 1e-20


def test_recovers_continuous_piecewise_power_law() -> None:
    breakpoint = 200.0
    x = np.unique(
        np.concatenate(
            [
                np.geomspace(10.0, breakpoint, 31),
                np.geomspace(breakpoint, 4000.0, 31),
            ]
        )
    )
    y_at_breakpoint = 5.0 * breakpoint
    y = np.where(
        x <= breakpoint,
        5.0 * x,
        y_at_breakpoint * (x / breakpoint) ** 3.0,
    )

    fit = fit_piecewise_loglog(x, y, max_knots=3, random_state=9)

    assert fit.continuous
    assert len(fit.breakpoints) >= 1
    assert fit.breakpoints[0] == pytest.approx(breakpoint, rel=0.08)
    assert fit.segments[0].slope == pytest.approx(1.0, abs=0.08)
    assert fit.segments[-1].slope == pytest.approx(3.0, abs=0.08)


def test_default_fit_has_at_most_one_kink() -> None:
    x = np.geomspace(10.0, 4000.0, 80)
    y = np.where(
        x < 100.0,
        x,
        np.where(x < 800.0, 0.01 * x**2, 1.25e-5 * x**3),
    )

    fit = fit_piecewise_loglog(x, y, random_state=9)

    assert len(fit.breakpoints) <= 1
    assert len(fit.segments) <= 2
    assert fit.max_knots == 1


def test_cross_validation_does_not_wildly_overfit_noisy_power_law() -> None:
    rng = np.random.default_rng(123)
    x = np.geomspace(10.0, 3000.0, 70)
    log_y = math.log10(3.0) + 2.0 * np.log10(x) + rng.normal(0.0, 0.025, x.size)
    y = 10.0**log_y

    fit = fit_piecewise_loglog(x, y, max_knots=4, random_state=41)

    assert len(fit.breakpoints) <= 1
    assert fit.segments[0].slope == pytest.approx(2.0, abs=0.15)


def test_one_standard_error_rule_prefers_constant_model_for_noisy_flat_data() -> None:
    rng = np.random.default_rng(2026)
    x = np.geomspace(10.0, 3000.0, 25)
    y = 8.0 * 10.0 ** rng.normal(0.0, 0.09, x.size)

    fit = fit_piecewise_loglog(x, y, max_knots=3, random_state=12)

    assert fit.breakpoints == []
    assert len(fit.segments) == 1


def test_minimum_segment_point_count_is_respected() -> None:
    breakpoint = 150.0
    x = np.unique(
        np.concatenate(
            [
                np.geomspace(10.0, breakpoint, 25),
                np.geomspace(breakpoint, 3000.0, 25),
            ]
        )
    )
    y = np.where(
        x <= breakpoint,
        2.0 * x**0.5,
        2.0 * breakpoint**0.5 * (x / breakpoint) ** 3.5,
    )
    min_segment_points = 6

    fit = fit_piecewise_loglog(
        x,
        y,
        max_knots=3,
        min_segment_points=min_segment_points,
        random_state=5,
    )
    segment_indices = np.searchsorted(fit.breakpoints, x, side="left")
    counts = np.bincount(segment_indices, minlength=len(fit.segments))

    assert fit.breakpoints
    assert np.all(counts >= min_segment_points)


def test_degenerate_input_and_invalid_values_are_handled() -> None:
    empty = fit_piecewise_loglog([0.0, -1.0, math.nan], [1.0, 2.0, 3.0])
    singleton = fit_piecewise_loglog([1.0, 2.0, 3.0], [-1.0, 4.0, 0.0])

    assert empty.n_points == 0
    assert empty.segments == []
    assert math.isnan(empty.cv_score)
    assert singleton.n_points == 1
    assert len(singleton.segments) == 1
    assert singleton.predict(2.0) == pytest.approx(4.0)


def test_fit_serialization_is_json_ready() -> None:
    x = np.geomspace(10.0, 1000.0, 20)
    fit = fit_piecewise_loglog(x, 7.0 * x**2)

    rows = fit_to_rows(
        fit,
        env_id="cpu",
        basis="def2-svp",
        functional="skala",
        metric="total",
        x_axis="num_aos",
    )
    assert len(rows) == len(fit.segments)
    assert [row["breakpoints"] for row in rows] == [fit.breakpoints] * len(rows)
    assert all(row["metric"] == "total" for row in rows)
    assert all(row["x_axis"] == "num_aos" for row in rows)


def test_predict() -> None:
    x = np.geomspace(10.0, 1000.0, 30)
    fit = fit_piecewise_loglog(x, 4.0 * x**1.5)

    assert fit.predict(100.0) == pytest.approx(4000.0, rel=1e-10)


def test_fit_serialization_rejects_unregistered_combination() -> None:
    fit = fit_piecewise_loglog([10.0, 20.0], [100.0, 400.0])

    with pytest.raises(ValueError, match="unsupported fit combination"):
        fit_to_rows(
            fit,
            env_id="cpu",
            basis="def2-svp",
            functional="skala",
            metric="total",
            x_axis="grid_size",
        )


def test_slopes_are_never_negative() -> None:
    rng = np.random.default_rng(3)
    x = np.geomspace(10.0, 2000.0, 45)
    # A globally decreasing signal must not produce a downward-sloping fit.
    decreasing = 5000.0 / np.sqrt(x) * np.exp(rng.normal(0.0, 0.05, x.size))
    fit = fit_piecewise_loglog(x, decreasing, random_state=3)
    assert all(segment.slope >= 0.0 for segment in fit.segments)

    # A rising signal with a local dip must still yield only non-negative slopes.
    dip = np.where(x < 300.0, x**1.2, (300.0**1.2) * (x / 300.0) ** 0.4)
    dip *= np.exp(rng.normal(0.0, 0.03, x.size))
    for allow in (True, False):
        fit = fit_piecewise_loglog(x, dip, allow_discontinuous=allow, random_state=3)
        assert all(segment.slope >= 0.0 for segment in fit.segments)


def test_slopes_are_never_decreasing() -> None:
    rng = np.random.default_rng(11)
    x = np.geomspace(10.0, 3000.0, 60)
    # A concave (in log-log) signal whose local slope decreases with size: the
    # fit must still report slopes that only increase from left to right.
    concave = np.where(x < 200.0, x**1.6, (200.0**1.6) * (x / 200.0) ** 0.5)
    concave *= np.exp(rng.normal(0.0, 0.03, x.size))
    for allow in (True, False):
        fit = fit_piecewise_loglog(
            x, concave, allow_discontinuous=allow, random_state=11
        )
        slopes = [segment.slope for segment in fit.segments]
        assert all(slope >= 0.0 for slope in slopes)
        assert all(
            later >= earlier - 1e-9
            for earlier, later in zip(slopes[:-1], slopes[1:], strict=True)
        )


def test_fit_is_never_decreasing() -> None:
    rng = np.random.default_rng(7)
    x = np.geomspace(10.0, 3000.0, 60)
    # A rising trend with a sharp downward step that a discontinuous fit could
    # otherwise reproduce as a downward jump between segments.
    step = np.where(x < 200.0, 40.0 * x**1.4, 5.0 * x**1.4)
    step *= np.exp(rng.normal(0.0, 0.04, x.size))
    for allow in (True, False):
        fit = fit_piecewise_loglog(x, step, allow_discontinuous=allow, random_state=7)
        assert all(segment.slope >= 0.0 for segment in fit.segments)
        probe = np.geomspace(float(x.min()), float(x.max()), 400)
        predictions = [fit.predict(value) for value in probe]
        assert all(
            later >= earlier - 1e-9
            for earlier, later in zip(predictions[:-1], predictions[1:], strict=True)
        )


def test_smooth_stacked_fractions_preserves_partition_of_unity() -> None:
    """Smoothed band widths stay non-negative and sum to one at every point."""
    rng = np.random.default_rng(0)
    xs = np.logspace(1, 3, 12)
    raw = rng.uniform(0.05, 1.0, size=(xs.size, 5))
    fractions = raw / raw.sum(axis=1, keepdims=True)

    stack = smooth_stacked_fractions(xs, fractions)

    cumulative = np.asarray(stack.cumulative)
    assert cumulative.shape[0] == 6
    np.testing.assert_allclose(cumulative[0], 0.0)
    np.testing.assert_allclose(cumulative[-1], 1.0)
    widths = np.diff(cumulative, axis=0)
    assert (widths >= -1e-9).all()
    np.testing.assert_allclose(widths.sum(axis=0), 1.0)


@pytest.mark.parametrize("n_points", [1, 2, 3, 4])
def test_smooth_stacked_fractions_returns_raw_points_for_small_series(
    n_points: int,
) -> None:
    """Series shorter than the spline's minimum are returned unsmoothed.

    Four points in particular must not reach ``make_smoothing_spline``, which
    requires five.
    """
    xs = [10.0 * 2**index for index in range(n_points)]
    fractions = [[0.5, 0.5]] * n_points

    stack = smooth_stacked_fractions(xs, fractions)

    assert stack.x == xs
    assert len(stack.cumulative) == 3
