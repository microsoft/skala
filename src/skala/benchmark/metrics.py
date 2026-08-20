# SPDX-License-Identifier: MIT

"""Definitions and calculations for benchmark metrics and time composition.

The registries in this module are the single source of truth for report labels,
units, axes, source columns, and fit combinations.

Every quantity is measured directly (see :mod:`skala.benchmark.timing`); nothing
is inferred from sampling. Four layers nest inside one SCF iteration::

    xc_eval  in  numint  in  veff  in  cycle

so the composition bands are differences of measured quantities rather than
attributed fractions:

``xc_eval``
    The exchange-correlation functional itself, without the contraction of its
    derivative into the AO basis. libxc's ``eval_xc_eff(deriv=1)`` returns energy
    and derivative together and cannot be split further; for Skala it is the
    forward pass alone, because its backward performs that contraction (see
    :func:`_xc_eval_value`).
``numint - xc_eval``
    The rest of the exchange-correlation quadrature: atomic orbitals on the
    grid, density assembly, and Vxc assembly.
``veff - numint``
    The J/K build: Coulomb and exact-exchange matrices, contracted from the
    density-fitting integrals.
``cycle - veff``
    Fock diagonalization, DIIS, and density-matrix bookkeeping.

A second composition covers a whole calculation rather than one iteration: the
one-time compilation of a neural functional, the setup before the SCF loop, and
the loop itself (see :func:`run_composition_times`).

Scaling metrics use the **steady-state** cycles rather than an average over all
of them, so that a first iteration which behaved differently cannot distort a
scaling exponent. In practice the measured difference is small, because the
worker pays process-wide first-use costs in an unmeasured SCF beforehand and the
pre-loop Fock build is charged to setup; :func:`warmup_ratio` reports what is
left, and is close to 1 on both CPU and GPU.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

Row = Mapping[str, Any]

#: Cycles from this index on are treated as steady state. Mirrors
#: :data:`skala.benchmark.timing.STEADY_STATE_FROM_CYCLE`.
STEADY_STATE_FROM_CYCLE = 1


@dataclass(frozen=True)
class AxisDefinition:
    """Definition of one supported system-size axis."""

    label: str
    source_column: str


@dataclass(frozen=True)
class MetricDefinition:
    """Display and data requirements for one plotted benchmark metric."""

    label: str
    unit: str
    axes: tuple[str, ...]
    #: Whether the metric needs per-cycle records rather than run-level columns.
    requires_cycles: bool
    source_columns: tuple[str, ...]


@dataclass(frozen=True)
class CompositionDefinition:
    """One band of a stacked time composition."""

    label: str
    short_label: str


@dataclass(frozen=True)
class CompositionSeries:
    """A complete stacked composition: its bands and how to measure them."""

    label: str
    bands: dict[str, CompositionDefinition]
    times: Callable[[Row], dict[str, float] | None]


AXIS_DEFINITIONS: dict[str, AxisDefinition] = {
    "num_aos": AxisDefinition(
        label="Number of atomic orbitals",
        source_column="num_atomic_orbitals",
    ),
    "grid_size": AxisDefinition(label="Grid points", source_column="grid_size"),
}

METRIC_DEFINITIONS: dict[str, MetricDefinition] = {
    "xc_eval": MetricDefinition(
        label="XC functional evaluation, per SCF iteration",
        unit="ms",
        axes=("num_aos", "grid_size"),
        requires_cycles=True,
        source_columns=("cycles",),
    ),
    "numint": MetricDefinition(
        label="Numerical integration, per SCF iteration",
        unit="ms",
        axes=("num_aos", "grid_size"),
        requires_cycles=True,
        source_columns=("cycles",),
    ),
    "jk": MetricDefinition(
        label="J/K build, per SCF iteration",
        unit="ms",
        axes=("num_aos",),
        requires_cycles=True,
        source_columns=("cycles",),
    ),
    "cycle": MetricDefinition(
        label="SCF iteration (steady state)",
        unit="ms",
        axes=("num_aos", "grid_size"),
        requires_cycles=True,
        source_columns=("cycles",),
    ),
    "total": MetricDefinition(
        label="Total SCF wall time",
        unit="ms",
        axes=("num_aos",),
        requires_cycles=False,
        source_columns=("kernel_time_ms",),
    ),
    "iterations": MetricDefinition(
        label="SCF iterations",
        unit="iterations",
        axes=("num_aos",),
        requires_cycles=False,
        source_columns=("num_scf_iterations",),
    ),
    "setup": MetricDefinition(
        label="Setup before the SCF loop",
        unit="ms",
        axes=("num_aos",),
        requires_cycles=False,
        source_columns=("setup_ms",),
    ),
}

CYCLE_COMPOSITION_BANDS: dict[str, CompositionDefinition] = {
    "xc_eval": CompositionDefinition(
        label="XC functional evaluation",
        short_label="XC eval",
    ),
    "numint_rest": CompositionDefinition(
        label="Rest of XC quadrature (AO evaluation, density, Vxc contraction)",
        short_label="XC rest",
    ),
    "jk": CompositionDefinition(
        label="J/K build",
        short_label="J/K",
    ),
    "cycle_rest": CompositionDefinition(
        label="Diagonalization, DIIS, bookkeeping",
        short_label="Diag + DIIS",
    ),
}

RUN_COMPOSITION_BANDS: dict[str, CompositionDefinition] = {
    "setup": CompositionDefinition(
        label="Setup before the SCF loop",
        short_label="Setup",
    ),
    "iterations": CompositionDefinition(
        label="SCF iterations",
        short_label="Iterations",
    ),
}

X_AXES = tuple(AXIS_DEFINITIONS)
X_AXIS_LABELS = {
    axis: definition.label for axis, definition in AXIS_DEFINITIONS.items()
}
METRIC_IDS = tuple(METRIC_DEFINITIONS)
METRIC_LABELS = {
    metric: definition.label for metric, definition in METRIC_DEFINITIONS.items()
}
METRIC_UNITS = {
    metric: definition.unit for metric, definition in METRIC_DEFINITIONS.items()
}
FIT_COMBINATIONS = tuple(
    (metric, definition.axes) for metric, definition in METRIC_DEFINITIONS.items()
)


def _positive_float(value: Any) -> float | None:
    """Return ``value`` as a positive finite float, or ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _non_negative_float(value: Any) -> float:
    """Return ``value`` as a finite non-negative float, or ``0.0``."""
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number) or number < 0.0:
        return 0.0
    return number


def cycles_of(row: Row) -> list[Mapping[str, Any]]:
    """Return the per-cycle records of a row, oldest first.

    Accepts any iterable of mappings, because the nested parquet column
    materializes as a list under pyarrow but as a numpy array under pandas.
    """
    cycles = row.get("cycles")
    if cycles is None or isinstance(cycles, (str, bytes, Mapping)):
        return []
    try:
        items = list(cycles)
    except TypeError:
        return []
    return [cycle for cycle in items if isinstance(cycle, Mapping)]


def steady_state_cycles(row: Row) -> list[Mapping[str, Any]]:
    """Return the cycles that represent recurring cost.

    Cycle 0 absorbs one-time warmup. When a calculation converged too quickly for
    a steady state to exist, all cycles are returned rather than none, so a short
    run still yields a (noisier) data point instead of a gap in the plot.
    """
    cycles = cycles_of(row)
    return cycles[STEADY_STATE_FROM_CYCLE:] or cycles


def _steady_median(row: Row, field: str) -> float | None:
    """Median of one per-cycle field over the steady-state cycles.

    The median rather than the mean, because one descheduled cycle on a shared
    node would otherwise shift the value.
    """
    values = [
        _non_negative_float(cycle.get(field)) for cycle in steady_state_cycles(row)
    ]
    return statistics.median(values) if values else None


def is_row_plottable(row: Row) -> bool:
    """Return whether a measurement row is a converged, successful data point."""
    return bool(row.get("is_converged")) and str(row.get("status")) == "ok"


def x_value(row: Row, x_axis: str) -> float | None:
    """Return the positive x-axis value for a row, or ``None`` if unavailable."""
    try:
        source_column = AXIS_DEFINITIONS[x_axis].source_column
    except KeyError:
        raise ValueError(f"unknown x_axis: {x_axis!r}") from None
    return _positive_float(row.get(source_column))


def warmup_ratio(row: Row) -> float | None:
    """Return cycle 0's cost divided by the steady-state cost.

    A value of 1 means the first iteration cost no more than the rest, which is
    what the benchmark measures in practice: the worker runs an unmeasured
    single-cycle SCF first, so process-wide first-use costs are already paid, and
    the pre-loop Fock build is charged to setup rather than to cycle 0. A value
    meaningfully above 1 means some cost recurs on the first iteration of every
    calculation and is worth investigating. Returns ``None`` when the run has no
    steady state to compare against.
    """
    cycles = cycles_of(row)
    if len(cycles) <= STEADY_STATE_FROM_CYCLE:
        return None
    first = _non_negative_float(cycles[0].get("wall_ms"))
    steady = _steady_median(row, "wall_ms")
    if not steady or first <= 0.0:
        return None
    return first / steady


#: Per-cycle field backing each steady-state duration metric.
_CYCLE_FIELDS = {
    "numint": "numint_ms",
    "cycle": "wall_ms",
}


def _xc_eval_value(row: Row) -> float | None:
    """Return the functional's own evaluation, on the same footing for both kinds.

    ``xc_eval_ms`` is not comparable across implementations: for a neural
    functional it is the forward pass plus a backward that differentiates all
    the way to the density matrix, so it also performs the contraction into the
    AO basis; libxc instead returns the derivative on the grid and pyscf
    contracts it afterwards, outside the timed call.

    Reported here is the functional evaluated to first order *without* that
    contraction: ``forward_ms`` for a neural functional, and the whole
    ``eval_xc_eff`` call for libxc, which returns energy and derivative
    together and cannot be split further.
    """
    forward = _steady_median(row, "forward_ms")
    if forward is not None and forward > 0.0:
        return forward
    # Classical: nothing patches a backward, so xc_eval_ms is the libxc call.
    return _steady_median(row, "xc_eval_ms")


def _jk_value(row: Row) -> float | None:
    """Return the J/K build: the effective potential minus its quadrature part."""
    veff = _steady_median(row, "veff_ms")
    numint = _steady_median(row, "numint_ms")
    if veff is None or numint is None:
        return None
    return max(veff - numint, 0.0)


def metric_value(row: Row, metric: str) -> float | None:
    """Return one registered metric value for a successful, converged row."""
    if metric not in METRIC_DEFINITIONS:
        raise ValueError(f"unknown metric: {metric!r}")
    if not is_row_plottable(row):
        return None

    if metric == "iterations":
        try:
            count = int(row.get("num_scf_iterations"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return float(count) if count > 0 else None
    if metric == "total":
        return _positive_float(row.get("kernel_time_ms"))
    if metric == "setup":
        return _positive_float(row.get("setup_ms"))
    if metric == "xc_eval":
        value = _xc_eval_value(row)
        return value if value is not None and value > 0.0 else None
    if metric == "jk":
        value = _jk_value(row)
        return value if value is not None and value > 0.0 else None

    value = _steady_median(row, _CYCLE_FIELDS[metric])
    return value if value is not None and value > 0.0 else None


def metric_has_grid_size_axis(metric: str) -> bool:
    """Return whether a registered metric is plotted against grid size."""
    try:
        return "grid_size" in METRIC_DEFINITIONS[metric].axes
    except KeyError:
        raise ValueError(f"unknown metric: {metric!r}") from None


def metric_axes(metric: str) -> tuple[str, ...]:
    """Return the supported axes for a registered metric."""
    try:
        return METRIC_DEFINITIONS[metric].axes
    except KeyError:
        raise ValueError(f"unknown metric: {metric!r}") from None


def composition_times(row: Row) -> dict[str, float] | None:
    """Return steady-state milliseconds per band of one SCF iteration.

    The bands are differences of directly measured, nested quantities, so they
    sum to the steady-state iteration time by construction. Measurement noise can
    make a difference marginally negative; those are clamped to zero.
    """
    if not is_row_plottable(row):
        return None
    cycle_ms = _steady_median(row, "wall_ms")
    veff_ms = _steady_median(row, "veff_ms")
    numint_ms = _steady_median(row, "numint_ms")
    # The same definition the xc_eval panel uses, so the two sections agree.
    xc_eval_ms = _xc_eval_value(row)
    if (
        cycle_ms is None
        or veff_ms is None
        or numint_ms is None
        or xc_eval_ms is None
        or cycle_ms <= 0.0
    ):
        return None
    return {
        "xc_eval": xc_eval_ms,
        "numint_rest": max(numint_ms - xc_eval_ms, 0.0),
        "jk": max(veff_ms - numint_ms, 0.0),
        "cycle_rest": max(cycle_ms - veff_ms, 0.0),
    }


def run_composition_times(row: Row) -> dict[str, float] | None:
    """Return milliseconds per band of the measured part of one complete SCF.

    Unlike :func:`composition_times`, whose bands are both recurring
    per-iteration costs, this covers the measured kernel end to end: the setup
    before the SCF loop, and the loop itself. The post-loop convergence check
    (``finalize_ms``) is left out; it is under 1% of a GPU run and never carries
    anything a reader needs to see.

    The costs a process pays before the measured kernel -- start-up and the
    overhead of the first exchange-correlation evaluation -- are excluded, so
    the bands describe the calculation rather than the process around it.
    """
    if not is_row_plottable(row):
        return None
    setup_ms = _non_negative_float(row.get("setup_ms"))
    iterations_ms = sum(
        _non_negative_float(cycle.get("wall_ms")) for cycle in cycles_of(row)
    )
    if iterations_ms <= 0.0:
        return None
    return {
        "setup": setup_ms,
        "iterations": iterations_ms,
    }


COMPOSITION_SERIES: dict[str, CompositionSeries] = {
    "cycle": CompositionSeries(
        label="One steady-state SCF iteration",
        bands=CYCLE_COMPOSITION_BANDS,
        times=composition_times,
    ),
    "run": CompositionSeries(
        label="One complete SCF",
        bands=RUN_COMPOSITION_BANDS,
        times=run_composition_times,
    ),
}
COMPOSITION_SERIES_IDS = tuple(COMPOSITION_SERIES)


def composition_band_ids(series: str = "cycle") -> tuple[str, ...]:
    """Return the band identifiers of a composition series, innermost first."""
    return tuple(COMPOSITION_SERIES[series].bands)


def composition_fractions(row: Row, series: str = "cycle") -> dict[str, float] | None:
    """Return the bands of a composition series, normalized to sum to one."""
    times = COMPOSITION_SERIES[series].times(row)
    if times is None:
        return None
    denominator = sum(times.values())
    if denominator <= 0.0:
        return None
    return {band: times[band] / denominator for band in composition_band_ids(series)}
