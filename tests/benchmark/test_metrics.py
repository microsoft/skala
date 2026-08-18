# SPDX-License-Identifier: MIT

"""Tests for derived benchmark scaling metrics."""

import math

import pytest

from skala.benchmark.metrics import (
    AXIS_DEFINITIONS,
    COMPOSITION_SERIES,
    COMPOSITION_SERIES_IDS,
    FIT_COMBINATIONS,
    METRIC_AXES,
    METRIC_CYCLE_REQUIREMENTS,
    METRIC_DEFINITIONS,
    METRIC_IDS,
    METRIC_LABELS,
    METRIC_SOURCE_COLUMNS,
    METRIC_UNITS,
    X_AXES,
    composition_band_ids,
    composition_fractions,
    composition_times,
    is_row_plottable,
    metric_axes,
    metric_has_grid_size_axis,
    metric_requires_cycles,
    metric_value,
    run_composition_times,
    steady_state_cycles,
    warmup_ratio,
    x_value,
)


def _cycle(
    cycle: int,
    wall: float,
    numint: float,
    xc_eval: float,
    veff: float | None = None,
) -> dict[str, object]:
    return {
        "cycle": cycle,
        "wall_ms": wall,
        # Default the J/K build to half the gap between numint and the iteration.
        "veff_ms": veff if veff is not None else (numint + wall) / 2,
        "numint_ms": numint,
        "xc_eval_ms": xc_eval,
        "forward_ms": 0.4 * xc_eval,
        "backward_ms": 0.6 * xc_eval,
        "veff_calls": 1,
        "numint_calls": 1,
        "xc_eval_calls": 1,
    }


def _ok_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "is_converged": True,
        "status": "ok",
        "kernel_time_ms": 1000.0,
        "setup_ms": 60.0,
        "finalize_ms": 40.0,
        # Cycle 0 is warmup-dominated; the rest are the steady state.
        "cycles": [
            _cycle(0, 400.0, 320.0, 160.0),
            _cycle(1, 100.0, 80.0, 30.0),
            _cycle(2, 100.0, 80.0, 30.0),
            _cycle(3, 104.0, 82.0, 31.0),
        ],
        "num_scf_iterations": 4,
        "num_atomic_orbitals": 300,
        "grid_size": 40000,
    }
    row.update(overrides)
    return row


def test_steady_state_excludes_the_first_cycle() -> None:
    cycles = steady_state_cycles(_ok_row())
    assert [cycle["cycle"] for cycle in cycles] == [1, 2, 3]


def test_steady_state_falls_back_when_there_is_only_one_cycle() -> None:
    row = _ok_row(cycles=[_cycle(0, 400.0, 320.0, 160.0)])
    assert [cycle["cycle"] for cycle in steady_state_cycles(row)] == [0]
    # A single-cycle run still yields a point rather than a gap in the plot.
    assert metric_value(row, "cycle") == pytest.approx(400.0)


def test_duration_metrics_are_steady_state_medians() -> None:
    row = _ok_row()
    # Median of the steady state (100, 100, 104), not the mean over all cycles.
    assert metric_value(row, "cycle") == pytest.approx(100.0)
    assert metric_value(row, "numint") == pytest.approx(80.0)
    # Neural: the forward pass alone, because the backward also contracts.
    assert metric_value(row, "xc_eval") == pytest.approx(12.0)
    assert metric_value(row, "jk") == pytest.approx(10.0)


def test_xc_eval_excludes_the_contraction_for_both_kinds() -> None:
    """The panel must mean the same thing for a neural and a libxc functional.

    A neural functional's backward differentiates to the density matrix and so
    performs the contraction into the AO basis; libxc's call stops at the grid
    and pyscf contracts afterwards. Reporting the neural forward against the
    whole libxc call puts them on the same footing.
    """
    neural = _ok_row()  # forward 12, backward 18
    assert metric_value(neural, "xc_eval") == pytest.approx(12.0)

    classical = _ok_row(
        cycles=[
            dict(cycle, forward_ms=0.0, backward_ms=0.0)
            for cycle in _ok_row()["cycles"]  # type: ignore[union-attr]
        ]
    )
    # Nothing patches a backward for libxc, so the whole call is xc_eval_ms.
    assert metric_value(classical, "xc_eval") == pytest.approx(30.0)


def test_warmup_is_excluded_from_the_scaling_metrics() -> None:
    hot = _ok_row(
        cycles=[
            _cycle(0, 9000.0, 8000.0, 4000.0),  # extreme first-iteration cost
            _cycle(1, 100.0, 80.0, 30.0),
            _cycle(2, 100.0, 80.0, 30.0),
        ]
    )
    assert metric_value(hot, "cycle") == pytest.approx(100.0)
    # The ratio is still available for tooltips, though it is no longer plotted.
    assert warmup_ratio(hot) == pytest.approx(90.0)


def test_warmup_ratio_needs_a_steady_state() -> None:
    assert warmup_ratio(_ok_row(cycles=[_cycle(0, 400.0, 320.0, 160.0)])) is None
    assert warmup_ratio(_ok_row(cycles=[])) is None
    assert warmup_ratio(_ok_row()) == pytest.approx(4.0)


def test_run_level_metrics_read_their_columns() -> None:
    row = _ok_row()
    assert metric_value(row, "total") == pytest.approx(1000.0)
    assert metric_value(row, "iterations") == pytest.approx(4.0)
    assert metric_value(row, "setup") == pytest.approx(60.0)


def test_composition_bands_are_differences_of_nested_measurements() -> None:
    times = composition_times(_ok_row())
    assert times is not None
    assert times["xc_eval"] == pytest.approx(12.0)  # the forward pass alone
    assert times["numint_rest"] == pytest.approx(68.0)  # numint - xc_eval
    assert times["jk"] == pytest.approx(10.0)  # veff - numint
    assert times["cycle_rest"] == pytest.approx(10.0)  # cycle - veff
    assert sum(times.values()) == pytest.approx(100.0)  # the steady-state cycle


def test_composition_bands_clamp_negative_noise() -> None:
    # forward marginally exceeding numint is measurement noise, not a negative band.
    row = _ok_row(
        cycles=[
            _cycle(0, 5.0, 4.0, 2.0),
            # forward_ms is 0.4 * xc_eval, so ask for one marginally above numint.
            _cycle(1, 100.0, 80.0, 201.0, veff=79.0),
        ]
    )
    times = composition_times(row)
    assert times is not None
    assert times["numint_rest"] == 0.0
    assert times["jk"] == 0.0
    assert min(times.values()) >= 0.0


def test_a_zero_xc_eval_yields_no_metric() -> None:
    row = _ok_row(cycles=[_cycle(0, 400.0, 320.0, 0.0), _cycle(1, 100.0, 80.0, 0.0)])
    assert metric_value(row, "xc_eval") is None
    times = composition_times(row)
    assert times is not None
    assert times["xc_eval"] == 0.0
    assert times["numint_rest"] == pytest.approx(80.0)


def test_composition_fractions_sum_to_one() -> None:
    fractions = composition_fractions(_ok_row())
    assert fractions is not None
    assert set(fractions) == set(composition_band_ids("cycle"))
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert fractions["xc_eval"] == pytest.approx(0.12)
    assert fractions["jk"] == pytest.approx(0.1)


def test_rows_without_cycles_yield_no_per_cycle_metrics() -> None:
    row = _ok_row(cycles=[])
    assert metric_value(row, "cycle") is None
    assert metric_value(row, "numint") is None
    assert composition_times(row) is None
    # Run-level metrics remain available.
    assert metric_value(row, "total") == pytest.approx(1000.0)


def test_unconverged_rows_yield_no_values() -> None:
    row = _ok_row(is_converged=False)
    assert not is_row_plottable(row)
    for metric in METRIC_IDS:
        assert metric_value(row, metric) is None
    assert composition_fractions(row) is None


def test_x_value_and_validation() -> None:
    row = _ok_row()
    assert x_value(row, "num_aos") == pytest.approx(300.0)
    assert x_value(row, "grid_size") == pytest.approx(40000.0)
    assert x_value(_ok_row(grid_size=0), "grid_size") is None
    with pytest.raises(ValueError):
        x_value(row, "nope")


def test_metric_requires_cycles_flags() -> None:
    assert metric_requires_cycles("xc_eval")
    assert metric_requires_cycles("numint")
    assert metric_requires_cycles("cycle")
    assert metric_requires_cycles("jk")
    assert not metric_requires_cycles("total")
    assert not metric_requires_cycles("iterations")
    assert not metric_requires_cycles("setup")


def test_metric_has_grid_size_axis_flags() -> None:
    # The three nested per-iteration layers all scale with grid density.
    assert metric_has_grid_size_axis("xc_eval")
    assert metric_has_grid_size_axis("numint")
    assert metric_has_grid_size_axis("cycle")
    assert not metric_has_grid_size_axis("jk")
    assert not metric_has_grid_size_axis("total")
    assert not metric_has_grid_size_axis("iterations")


def test_compatibility_views_derive_from_registries() -> None:
    assert METRIC_IDS == tuple(METRIC_DEFINITIONS)
    assert X_AXES == tuple(AXIS_DEFINITIONS)
    assert METRIC_LABELS == {
        metric: definition.label for metric, definition in METRIC_DEFINITIONS.items()
    }
    assert METRIC_UNITS == {
        metric: definition.unit for metric, definition in METRIC_DEFINITIONS.items()
    }
    assert METRIC_AXES == {
        metric: definition.axes for metric, definition in METRIC_DEFINITIONS.items()
    }
    assert METRIC_CYCLE_REQUIREMENTS == {
        metric: definition.requires_cycles
        for metric, definition in METRIC_DEFINITIONS.items()
    }
    assert METRIC_SOURCE_COLUMNS == {
        metric: definition.source_columns
        for metric, definition in METRIC_DEFINITIONS.items()
    }
    assert FIT_COMBINATIONS == tuple(
        (metric, definition.axes) for metric, definition in METRIC_DEFINITIONS.items()
    )
    assert metric_axes("xc_eval") == ("num_aos", "grid_size")


def test_composition_views_derive_from_registry() -> None:
    assert COMPOSITION_SERIES_IDS == tuple(COMPOSITION_SERIES)
    for series in COMPOSITION_SERIES_IDS:
        assert composition_band_ids(series) == tuple(COMPOSITION_SERIES[series].bands)


def test_run_composition_covers_setup_and_the_loop() -> None:
    times = run_composition_times(_ok_row())
    assert times is not None
    assert set(times) == {"setup", "iterations"}
    assert times["setup"] == pytest.approx(60.0)
    # Every cycle, not just the steady-state ones: 400 + 100 + 100 + 104.
    assert times["iterations"] == pytest.approx(704.0)


def test_run_composition_excludes_what_precedes_the_measured_kernel() -> None:
    """Start-up and first-evaluation overhead are not part of the calculation."""
    fractions = composition_fractions(_ok_row(jit_compile_ms=5000.0), "run")
    assert fractions is not None
    assert sum(fractions.values()) == pytest.approx(1.0)
    # A huge compile does not move the bands, because it is not among them.
    assert fractions == composition_fractions(_ok_row(jit_compile_ms=0.0), "run")


def test_run_composition_needs_cycles() -> None:
    assert run_composition_times(_ok_row(cycles=[])) is None


def test_unknown_metric_raises() -> None:
    with pytest.raises(ValueError):
        metric_value(_ok_row(), "bogus")
    with pytest.raises(ValueError):
        metric_axes("bogus")
    with pytest.raises(ValueError):
        metric_requires_cycles("bogus")


def test_nonfinite_values_are_rejected() -> None:
    assert metric_value(_ok_row(kernel_time_ms=math.inf), "total") is None
    assert metric_value(_ok_row(kernel_time_ms=None), "total") is None


def test_cycles_survive_a_pandas_round_trip() -> None:
    """Parquet list columns arrive as numpy arrays under pandas, not lists."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame([_ok_row()])
    row = frame.iloc[0].to_dict()
    assert metric_value(row, "numint") == pytest.approx(80.0)
    assert composition_times(row) is not None


def test_a_scalar_cycles_value_is_ignored() -> None:
    assert metric_value(_ok_row(cycles=None), "cycle") is None
    assert metric_value(_ok_row(cycles="not-a-list"), "cycle") is None
