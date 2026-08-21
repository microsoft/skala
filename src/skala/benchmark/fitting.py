# SPDX-License-Identifier: MIT

"""Piecewise-linear scaling fits in log-log space."""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import nnls

FloatArray: TypeAlias = NDArray[np.float64]

#: Smallest input ``scipy.interpolate.make_smoothing_spline`` accepts. Shorter
#: series are returned unsmoothed rather than raising.
MIN_SMOOTHING_POINTS = 5


@dataclass(frozen=True)
class Segment:
    """One linear segment of a log-log scaling fit."""

    x_start: float
    x_end: float
    slope: float
    intercept: float


@dataclass(frozen=True)
class PiecewiseFit:
    """A selected piecewise-linear model expressed in original x units."""

    segments: list[Segment]
    breakpoints: list[float]
    continuous: bool
    cv_score: float
    n_points: int
    max_knots: int

    def predict_loglog(self, x: float) -> float:
        """Predict ``log10(y)`` at an x value in original units.

        Args:
            x: Positive x value in original units.

        Returns:
            The predicted base-10 logarithm of y, or NaN for an empty fit.

        Raises:
            ValueError: If x is non-positive or non-finite.
        """
        if not math.isfinite(x) or x <= 0:
            raise ValueError("x must be positive and finite")
        if not self.segments:
            return math.nan
        segment_index = bisect.bisect_left(self.breakpoints, x)
        segment = self.segments[min(segment_index, len(self.segments) - 1)]
        return segment.intercept + segment.slope * math.log10(x)

    def predict(self, x: float) -> float:
        """Predict y at an x value in original units.

        Args:
            x: Positive x value in original units.

        Returns:
            The prediction in original y units, or NaN for an empty fit.
        """
        prediction = self.predict_loglog(x)
        return 10.0**prediction


def fit_piecewise_loglog(
    x: Sequence[float],
    y: Sequence[float],
    *,
    max_knots: int = 1,
    allow_discontinuous: bool = True,
    cv_folds: int = 5,
    random_state: int = 0,
    min_segment_points: int = 3,
) -> PiecewiseFit:
    """Fit and cross-validate a piecewise-linear scaling model.

    Args:
        x: System sizes in original positive units.
        y: Costs in original positive units.
        max_knots: Maximum number of breakpoints considered.
        allow_discontinuous: Whether independently fitted segments are candidates.
        cv_folds: Requested number of cross-validation folds.
        random_state: Seed used to assign points to folds.
        min_segment_points: Minimum number of training points in every segment.

    Returns:
        The selected fit. Slopes are scaling exponents in log10-log10 space.

    Raises:
        ValueError: If inputs have different shapes, are not one-dimensional, or
            an option is invalid.
    """
    return _fit_piecewise_loglog(
        x,
        y,
        max_knots=max_knots,
        allow_discontinuous=allow_discontinuous,
        cv_folds=cv_folds,
        random_state=random_state,
        min_segment_points=min_segment_points,
    )


@dataclass(frozen=True)
class _LogSegment:
    x_start: float
    x_end: float
    slope: float
    intercept: float


@dataclass(frozen=True)
class _Model:
    segments: list[_LogSegment]
    knots: list[float]
    continuous: bool

    def predict(self, x: FloatArray) -> FloatArray:
        indices = np.searchsorted(np.asarray(self.knots), x, side="left")
        slopes = np.asarray([segment.slope for segment in self.segments])
        intercepts = np.asarray([segment.intercept for segment in self.segments])
        return intercepts[indices] + slopes[indices] * x


def _linear_coefficients(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    if x.size < 2 or float(np.ptp(x)) <= np.finfo(float).eps:
        return 0.0, float(np.mean(y))
    design = np.column_stack((np.ones_like(x), x))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(coefficients[1]), float(coefficients[0])


def _nonneg_linear(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    """Least-squares slope/intercept constrained to a non-negative slope.

    When the unconstrained slope is negative, the constrained optimum for a
    single free intercept collapses to a flat line at the mean.
    """
    slope, intercept = _linear_coefficients(x, y)
    if slope < 0.0:
        return 0.0, float(np.mean(y))
    return slope, intercept


def _candidate_knots(
    x: FloatArray, y: FloatArray, min_segment_points: int
) -> list[float]:
    unique_x = np.unique(x)
    if unique_x.size < 3:
        return []

    candidates: set[float] = set()
    for index in range(1, unique_x.size - 1):
        candidates.add(float(unique_x[index]))

    for left, right in pairwise(unique_x[1:-1]):
        candidates.add(float((left + right) / 2.0))

    sorted_indices = np.argsort(x, kind="stable")
    sorted_x = x[sorted_indices]
    sorted_y = y[sorted_indices]
    for split in range(min_segment_points, x.size - min_segment_points + 1):
        if sorted_x[split - 1] == sorted_x[split]:
            continue
        left_slope, left_intercept = _linear_coefficients(
            sorted_x[:split], sorted_y[:split]
        )
        right_slope, right_intercept = _linear_coefficients(
            sorted_x[split:], sorted_y[split:]
        )
        slope_difference = left_slope - right_slope
        if abs(slope_difference) <= np.finfo(float).eps:
            continue
        intersection = (right_intercept - left_intercept) / slope_difference
        if sorted_x[split - 1] <= intersection <= sorted_x[split]:
            candidates.add(float(intersection))

    x_min = float(unique_x[0])
    x_max = float(unique_x[-1])
    return sorted(
        candidate
        for candidate in candidates
        if x_min < candidate < x_max
        and np.all(_segment_counts(x, [candidate]) >= min_segment_points)
    )


def _segment_counts(x: FloatArray, knots: list[float]) -> NDArray[np.int64]:
    indices = np.searchsorted(np.asarray(knots), x, side="left")
    return np.bincount(indices, minlength=len(knots) + 1)


def _fit_continuous(x: FloatArray, y: FloatArray, knots: list[float]) -> _Model:
    slopes, intercepts = _continuous_convex_lines(x, y, knots)
    boundaries = [float(np.min(x)), *knots, float(np.max(x))]
    segments = [
        _LogSegment(
            boundaries[index], boundaries[index + 1], slopes[index], intercepts[index]
        )
        for index in range(len(slopes))
    ]
    return _Model(segments=segments, knots=knots, continuous=True)


def _continuous_convex_lines(
    x: FloatArray, y: FloatArray, knots: list[float]
) -> tuple[list[float], list[float]]:
    """Fit a continuous piecewise line whose slopes never decrease left to right.

    The model uses a ReLU-hinge basis ``[x, max(0, x - k_0), max(0, x - k_1), …]``
    whose coefficients are the *increments* added to the slope at each knot. A
    non-negative least-squares solve forces every increment ``>= 0``, so each
    segment slope is the running sum of non-negative increments: the slopes are
    non-negative and monotonically non-decreasing (the curve is convex). The free
    intercept is expressed as the difference of two non-negative variables to keep
    the whole system inside NNLS.
    """
    if knots:
        knot_array = np.asarray(knots, dtype=np.float64)
        columns = [x]
        for knot_value in knot_array:
            columns.append(np.maximum(0.0, x - knot_value))
        basis = np.column_stack(columns)
    else:
        basis = x.reshape(-1, 1)

    design = np.column_stack((np.ones_like(x), -np.ones_like(x), basis))
    solution, _ = nnls(design, y)
    intercept = float(solution[0] - solution[1])
    increments = solution[2:]
    segment_slopes = np.cumsum(increments)

    slopes: list[float] = [float(segment_slopes[0])]
    intercepts: list[float] = [intercept]
    running_slope: float = float(segment_slopes[0])
    running_intercept = intercept
    for knot, next_slope in zip(knots, segment_slopes[1:], strict=True):
        hinge = float(next_slope) - running_slope
        running_slope = float(next_slope)
        running_intercept -= hinge * knot
        slopes.append(running_slope)
        intercepts.append(running_intercept)
    return slopes, intercepts


def _forbid_downward_jumps(
    coefficients: list[tuple[float, float] | None],
    boundaries: list[float],
) -> None:
    """Raise segment intercepts so the fit never steps down at a breakpoint.

    Each segment already has a non-negative slope, so lifting any segment that
    would start below its predecessor's end value makes the whole piecewise
    curve non-decreasing from left to right. Slopes are left untouched.
    """
    for index in range(len(coefficients) - 1):
        left = coefficients[index]
        right = coefficients[index + 1]
        assert left is not None and right is not None
        boundary = boundaries[index + 1]
        end_left = left[0] * boundary + left[1]
        start_right = right[0] * boundary + right[1]
        if start_right < end_left:
            coefficients[index + 1] = (right[0], right[1] + (end_left - start_right))


def _enforce_increasing_slopes(
    coefficients: list[tuple[float, float] | None],
    x: FloatArray,
    y: FloatArray,
    indices: NDArray[np.int64],
) -> None:
    """Raise segment slopes so they never decrease from left to right.

    Walking left to right, any segment whose slope falls below its predecessor's
    is clamped up to that predecessor slope; its intercept is then refit through
    the segment's data centroid (fixed slope least squares) when it has points,
    keeping the line close to its data. Because the first slope is already
    non-negative, the whole sequence of slopes stays non-negative and
    non-decreasing.
    """
    running_slope = 0.0
    for index in range(len(coefficients)):
        coefficient = coefficients[index]
        assert coefficient is not None
        slope = coefficient[0]
        if slope < running_slope:
            slope = running_slope
            mask = indices == index
            if np.any(mask):
                mean_x = float(np.mean(x[mask]))
                mean_y = float(np.mean(y[mask]))
                coefficients[index] = (slope, mean_y - slope * mean_x)
            else:
                coefficients[index] = (slope, coefficient[1])
        running_slope = slope


def _fit_discontinuous(x: FloatArray, y: FloatArray, knots: list[float]) -> _Model:
    indices = np.searchsorted(np.asarray(knots), x, side="left")
    coefficients: list[tuple[float, float] | None] = []
    for segment_index in range(len(knots) + 1):
        mask = indices == segment_index
        segment_x = x[mask]
        segment_y = y[mask]
        if segment_x.size >= 2 and float(np.ptp(segment_x)) > np.finfo(float).eps:
            coefficients.append(_nonneg_linear(segment_x, segment_y))
        else:
            coefficients.append(None)

    valid_indices = [
        index for index, coefficient in enumerate(coefficients) if coefficient
    ]
    for index, coefficient in enumerate(coefficients):
        if coefficient is not None:
            continue
        mask = indices == index
        mean_x = float(np.mean(x[mask])) if np.any(mask) else 0.0
        mean_y = float(np.mean(y[mask])) if np.any(mask) else float(np.mean(y))
        if valid_indices:
            neighbor = min(
                valid_indices, key=lambda valid_index: abs(valid_index - index)
            )
            slope = max(coefficients[neighbor][0], 0.0)  # type: ignore[index]
        else:
            slope = 0.0
        coefficients[index] = (slope, mean_y - slope * mean_x)

    boundaries = [float(np.min(x)), *knots, float(np.max(x))]
    _enforce_increasing_slopes(coefficients, x, y, indices)
    _forbid_downward_jumps(coefficients, boundaries)
    segments = [
        _LogSegment(
            boundaries[index],
            boundaries[index + 1],
            coefficients[index][0],  # type: ignore[index]
            coefficients[index][1],  # type: ignore[index]
        )
        for index in range(len(coefficients))
    ]
    return _Model(segments=segments, knots=knots, continuous=False)


def _fit_model(
    x: FloatArray, y: FloatArray, knots: list[float], continuous: bool
) -> _Model:
    if continuous:
        return _fit_continuous(x, y, knots)
    return _fit_discontinuous(x, y, knots)


def _training_error(model: _Model, x: FloatArray, y: FloatArray) -> float:
    residual = model.predict(x) - y
    return float(np.mean(residual * residual))


def _fit_path(
    x: FloatArray,
    y: FloatArray,
    max_knots: int,
    continuous: bool,
    min_segment_points: int,
) -> list[_Model | None]:
    path: list[_Model | None] = [_fit_model(x, y, [], continuous)]
    selected: list[float] = []
    candidates = _candidate_knots(x, y, min_segment_points)

    for _ in range(max_knots):
        best_model: _Model | None = None
        best_error = math.inf
        best_knots: list[float] | None = None
        for candidate in candidates:
            if candidate in selected:
                continue
            knots = sorted([*selected, candidate])
            if np.any(_segment_counts(x, knots) < min_segment_points):
                continue
            model = _fit_model(x, y, knots, continuous)
            error = _training_error(model, x, y)
            if error < best_error and not math.isclose(
                error, best_error, rel_tol=1e-12, abs_tol=1e-15
            ):
                best_model = model
                best_error = error
                best_knots = knots
        if best_model is None or best_knots is None:
            path.extend([None] * (max_knots + 1 - len(path)))
            break
        selected = best_knots
        path.append(best_model)
    return path


def _folds(n_points: int, cv_folds: int, random_state: int) -> list[FloatArray]:
    n_folds = min(cv_folds, n_points)
    permutation = np.random.default_rng(random_state).permutation(n_points)
    return [fold.astype(np.float64) for fold in np.array_split(permutation, n_folds)]


def _cross_validation_scores(
    x: FloatArray,
    y: FloatArray,
    *,
    max_knots: int,
    allow_discontinuous: bool,
    cv_folds: int,
    random_state: int,
    min_segment_points: int,
) -> dict[tuple[int, bool], tuple[float, float]]:
    configurations = [
        (knot_count, continuous)
        for knot_count in range(max_knots + 1)
        for continuous in (
            (True, False) if allow_discontinuous and knot_count > 0 else (True,)
        )
    ]
    fold_errors: dict[tuple[int, bool], list[float]] = {
        configuration: [] for configuration in configurations
    }
    valid = {configuration: True for configuration in configurations}

    for test_indices_float in _folds(x.size, cv_folds, random_state):
        test_indices = test_indices_float.astype(np.int64)
        train_mask = np.ones(x.size, dtype=bool)
        train_mask[test_indices] = False
        train_x = x[train_mask]
        train_y = y[train_mask]
        test_x = x[test_indices]
        test_y = y[test_indices]

        paths = {
            True: _fit_path(
                train_x,
                train_y,
                max_knots,
                True,
                min_segment_points,
            )
        }
        if allow_discontinuous:
            paths[False] = _fit_path(
                train_x,
                train_y,
                max_knots,
                False,
                min_segment_points,
            )

        for configuration in configurations:
            knot_count, continuous = configuration
            model = paths[continuous][knot_count]
            if model is None:
                valid[configuration] = False
                continue
            residual = model.predict(test_x) - test_y
            if not np.all(np.isfinite(residual)):
                valid[configuration] = False
                continue
            fold_errors[configuration].append(float(np.mean(residual * residual)))

    scores: dict[tuple[int, bool], tuple[float, float]] = {}
    for configuration in configurations:
        errors = fold_errors[configuration]
        if not valid[configuration] or not errors:
            scores[configuration] = (math.inf, 0.0)
            continue
        mean = float(np.mean(errors))
        standard_error = (
            float(np.std(errors, ddof=1) / math.sqrt(len(errors)))
            if len(errors) >= 2
            else 0.0
        )
        scores[configuration] = (mean, standard_error)
    return scores


def _public_fit(
    model: _Model,
    *,
    cv_score: float,
    n_points: int,
    max_knots: int,
) -> PiecewiseFit:
    return PiecewiseFit(
        segments=[
            Segment(
                x_start=10.0**segment.x_start,
                x_end=10.0**segment.x_end,
                slope=segment.slope,
                intercept=segment.intercept,
            )
            for segment in model.segments
        ],
        breakpoints=[10.0**knot for knot in model.knots],
        continuous=model.continuous,
        cv_score=cv_score,
        n_points=n_points,
        max_knots=max_knots,
    )


def _fit_piecewise_loglog(
    x: Sequence[float],
    y: Sequence[float],
    *,
    max_knots: int = 1,
    allow_discontinuous: bool = True,
    cv_folds: int = 5,
    random_state: int = 0,
    min_segment_points: int = 3,
) -> PiecewiseFit:
    if max_knots < 0:
        raise ValueError("max_knots must be non-negative")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    if min_segment_points < 3:
        raise ValueError("min_segment_points must be at least 3")

    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.ndim != 1 or y_array.ndim != 1 or x_array.shape != y_array.shape:
        raise ValueError("x and y must be one-dimensional sequences with equal lengths")

    valid_mask = (
        np.isfinite(x_array) & np.isfinite(y_array) & (x_array > 0.0) & (y_array > 0.0)
    )
    log_x = np.log10(x_array[valid_mask])
    log_y = np.log10(y_array[valid_mask])
    n_points = int(log_x.size)

    if n_points == 0:
        return PiecewiseFit([], [], True, math.nan, 0, max_knots)
    if n_points == 1:
        segment = Segment(
            x_start=float(x_array[valid_mask][0]),
            x_end=float(x_array[valid_mask][0]),
            slope=0.0,
            intercept=float(log_y[0]),
        )
        return PiecewiseFit([segment], [], True, math.nan, 1, max_knots)

    effective_max_knots = min(max_knots, max(0, n_points // min_segment_points - 1))
    scores = _cross_validation_scores(
        log_x,
        log_y,
        max_knots=effective_max_knots,
        allow_discontinuous=allow_discontinuous,
        cv_folds=cv_folds,
        random_state=random_state,
        min_segment_points=min_segment_points,
    )

    best_configuration = min(
        scores,
        key=lambda configuration: (
            scores[configuration][0],
            configuration[0],
            not configuration[1],
        ),
    )
    best_mean, best_standard_error = scores[best_configuration]

    selection_threshold = best_mean + best_standard_error
    selected_configuration = best_configuration
    for configuration in sorted(scores, key=lambda item: (item[0], not item[1])):
        mean, _ = scores[configuration]
        if mean <= selection_threshold or math.isclose(
            mean, selection_threshold, rel_tol=1e-9, abs_tol=1e-12
        ):
            selected_configuration = configuration
            break

    knot_count, continuous = selected_configuration
    selected_mean, _ = scores[selected_configuration]
    final_model = _fit_path(
        log_x,
        log_y,
        knot_count,
        continuous,
        min_segment_points,
    )[knot_count]
    if final_model is None:
        final_model = _fit_model(log_x, log_y, [], True)
        selected_mean = scores[(0, True)][0]
    return _public_fit(
        final_model,
        cv_score=selected_mean,
        n_points=n_points,
        max_knots=max_knots,
    )


@dataclass(frozen=True)
class SmoothStack:
    """A smoothed stacked-fraction curve on a dense log-spaced x grid.

    Attributes:
        x: Dense, strictly increasing x grid in original units.
        cumulative: ``(num_bands + 1, len(x))`` cumulative boundaries. Row ``0``
            is all zeros and the last row is all ones, so band ``i`` spans
            ``cumulative[i] .. cumulative[i + 1]`` and every column sums to one.
    """

    x: list[float]
    cumulative: list[list[float]]


def smooth_stacked_fractions(
    x: Sequence[float],
    fractions: Sequence[Sequence[float]],
    grid_points: int = 96,
) -> SmoothStack:
    """Smooth stacked composition fractions with a CV-selected smoothing spline.

    Each cumulative band boundary is smoothed independently against ``log10(x)``
    using :func:`scipy.interpolate.make_smoothing_spline`, whose penalty weight is
    chosen by generalized cross-validation. The smoothed boundaries are then
    clamped to ``[0, 1]`` and made monotone across bands, so the reconstructed
    per-band fractions stay non-negative and sum to one at every point.

    Args:
        x: Positive x positions (need not be sorted or unique).
        fractions: One row of band fractions per x; each row should sum to one.
        grid_points: Number of points on the dense output grid.

    Returns:
        A :class:`SmoothStack`. For fewer than five distinct x positions the
        original points are returned unsmoothed, which is the smallest input
        ``make_smoothing_spline`` accepts.

    Raises:
        ValueError: If ``x`` and ``fractions`` differ in length or are empty.
    """
    xs = np.asarray(x, dtype=np.float64)
    table = np.asarray(fractions, dtype=np.float64)
    if xs.ndim != 1 or table.ndim != 2 or xs.size != table.shape[0]:
        raise ValueError("x and fractions must share a leading dimension")
    if xs.size == 0:
        raise ValueError("at least one point is required")

    order = np.argsort(xs)
    xs, table = xs[order], table[order]
    unique_x, inverse = np.unique(xs, return_inverse=True)
    if unique_x.size != xs.size:
        averaged = np.zeros((unique_x.size, table.shape[1]), dtype=np.float64)
        counts = np.zeros(unique_x.size, dtype=np.float64)
        np.add.at(averaged, inverse, table)
        np.add.at(counts, inverse, 1.0)
        table = averaged / counts[:, None]
        xs = unique_x

    num_bands = table.shape[1]
    cumulative_data = np.cumsum(table, axis=1)[:, : num_bands - 1]
    log_x = np.log10(xs)

    if xs.size < MIN_SMOOTHING_POINTS:
        boundaries = np.clip(cumulative_data, 0.0, 1.0)
        boundaries = np.maximum.accumulate(boundaries, axis=1)
        return _assemble_stack(xs, boundaries)

    dense_log = np.linspace(log_x[0], log_x[-1], grid_points)
    smoothed = np.empty((dense_log.size, num_bands - 1), dtype=np.float64)
    for band in range(num_bands - 1):
        spline = make_smoothing_spline(log_x, cumulative_data[:, band])
        smoothed[:, band] = spline(dense_log)
    smoothed = np.clip(smoothed, 0.0, 1.0)
    smoothed = np.maximum.accumulate(smoothed, axis=1)
    return _assemble_stack(10.0**dense_log, smoothed)


def _assemble_stack(xs: FloatArray, interior: FloatArray) -> SmoothStack:
    """Wrap interior cumulative boundaries with the 0 and 1 rails."""
    zeros = np.zeros((xs.size, 1), dtype=np.float64)
    ones = np.ones((xs.size, 1), dtype=np.float64)
    cumulative = np.concatenate([zeros, interior, ones], axis=1)
    return SmoothStack(
        x=[float(value) for value in xs],
        cumulative=[[float(value) for value in row] for row in cumulative.T],
    )
