# SPDX-License-Identifier: MIT

"""Data preparation for the standalone benchmark report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias, TypedDict

import numpy as np
import pandas as pd

from skala.benchmark import fitting, metrics

from ._normalize import coerce_int, string_list

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class MetricRecord(TypedDict):
    """Metadata for one report metric."""

    label: str
    unit: str


class XModeRecord(TypedDict):
    """Metadata for one report x-axis mode."""

    field: str
    label: str


class PointRecord(TypedDict):
    """One plotted report point."""

    env_id: str
    basis: str
    functional: str
    metric: str
    x_axis: str
    x: float
    y: float
    molecule: JsonValue
    ansatz: JsonValue
    atoms: JsonValue
    electrons: JsonValue
    num_aos: JsonValue
    grid_size: JsonValue
    scf_iterations: JsonValue
    energy: JsonValue
    warmup_ratio: JsonValue


class FitRecord(TypedDict):
    """One fitted scaling segment."""

    env_id: JsonValue
    basis: JsonValue
    functional: JsonValue
    metric: JsonValue
    x_axis: JsonValue
    segment_index: JsonValue
    x_start: JsonValue
    x_end: JsonValue
    slope: JsonValue
    intercept: JsonValue
    continuous: JsonValue
    cv_score: JsonValue
    n_points: JsonValue
    breakpoints: JsonValue


class CompositionBucketRecord(TypedDict):
    """Display metadata for one timing-composition band."""

    id: str
    label: str
    short: str


class CompositionRecord(TypedDict):
    """One measured timing-composition point."""

    env_id: str
    basis: str
    functional: str
    x: float
    y: list[float]


class SmoothedCompositionRecord(TypedDict):
    """One smoothed timing-composition curve."""

    env_id: str
    basis: str
    functional: str
    x: list[float]
    cumulative: list[list[float]]


class AxisDomain(TypedDict):
    """Fixed x and y domains for one chart."""

    x: list[float]
    y: list[float]


METRICS: dict[str, MetricRecord] = {
    metric: {
        "label": metrics.METRIC_LABELS[metric],
        "unit": metrics.METRIC_UNITS[metric],
    }
    for metric in metrics.METRIC_IDS
}
X_MODES: dict[str, XModeRecord] = {
    x_axis: {"field": x_axis, "label": metrics.X_AXIS_LABELS[x_axis]}
    for x_axis in metrics.X_AXES
}

_MEASUREMENT_COLUMNS = {
    "env_id",
    "basis",
    "functional",
    "mol_name",
    "ansatz",
    "num_atoms",
    "num_electrons",
    "num_atomic_orbitals",
    "grid_size",
    "is_converged",
    "num_scf_iterations",
    "total_energy",
    "kernel_time_ms",
    "setup_ms",
    "finalize_ms",
    "cycles",
    "status",
}
_FIT_COLUMNS = {
    "env_id",
    "basis",
    "functional",
    "metric",
    "x_axis",
    "segment_index",
    "x_start",
    "x_end",
    "slope",
    "intercept",
    "continuous",
    "cv_score",
    "n_points",
    "breakpoints",
}


def prepare_points(measurements: pd.DataFrame) -> pd.DataFrame:
    """Compute metric/x-axis point pairs through the shared metrics module.

    Args:
        measurements: Concatenated collected measurement table.

    Returns:
        One row per valid metric/x-axis pair, including tooltip fields.

    Raises:
        ValueError: If required collected-measurement columns are absent.
    """
    _require_columns(measurements, _MEASUREMENT_COLUMNS, "measurements")
    columns = [
        "env_id",
        "basis",
        "functional",
        "metric",
        "x_axis",
        "x",
        "y",
        "molecule",
        "ansatz",
        "atoms",
        "electrons",
        "num_aos",
        "grid_size",
        "scf_iterations",
        "energy",
        "warmup_ratio",
    ]
    points: list[PointRecord] = []
    for raw_row in measurements.to_dict(orient="records"):
        row = {str(key): _json_value(value) for key, value in raw_row.items()}
        if not metrics.is_row_plottable(row):
            continue
        functional = str(row.get("functional"))
        for metric in metrics.METRIC_IDS:
            for x_axis in metrics.metric_axes(metric):
                x_value = metrics.x_value(row, x_axis)
                metric_value = metrics.metric_value(row, metric)
                if x_value is None or metric_value is None:
                    continue
                points.append(
                    {
                        "env_id": str(row.get("env_id")),
                        "basis": str(row.get("basis")),
                        "functional": functional,
                        "metric": metric,
                        "x_axis": x_axis,
                        "x": x_value,
                        "y": metric_value,
                        "molecule": row.get("mol_name"),
                        "ansatz": row.get("ansatz"),
                        "atoms": row.get("num_atoms"),
                        "electrons": row.get("num_electrons"),
                        "num_aos": row.get("num_atomic_orbitals"),
                        "grid_size": row.get("grid_size"),
                        "scf_iterations": row.get("num_scf_iterations"),
                        "energy": row.get("total_energy"),
                        # How much more expensive the first iteration was than
                        # the steady state; surfaced in tooltips so a suspicious
                        # point can be traced to warmup rather than scaling.
                        "warmup_ratio": metrics.warmup_ratio(row),
                    }
                )
    return pd.DataFrame(points, columns=columns)


def compute_domains(
    points: pd.DataFrame,
) -> dict[str, dict[str, AxisDomain]]:
    """Compute fixed global log domains for every supported plot and x mode.

    Args:
        points: Prepared point table from :func:`prepare_points`.

    Returns:
        Nested ``metric -> x mode -> {x, y}`` domains.
    """
    domains: dict[str, dict[str, AxisDomain]] = {}
    for metric in metrics.METRIC_IDS:
        modes = (
            ("num_aos", "grid_size")
            if metrics.metric_has_grid_size_axis(metric)
            else ("num_aos",)
        )
        domains[metric] = {}
        for mode in modes:
            selected = points.loc[
                points["metric"].eq(metric) & points["x_axis"].eq(mode)
            ]
            domains[metric][mode] = {
                "x": _log_domain(selected["x"]),
                "y": _log_domain(selected["y"]),
            }
    return domains


def environment_rows(
    environments: pd.DataFrame, measurement_env_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Normalize environment metadata for every measured environment.

    Args:
        environments: Concatenated environment metadata.
        measurement_env_ids: Environment identifiers in first-seen order.

    Returns:
        JSON-safe environment metadata records.

    Raises:
        ValueError: If environment metadata is missing for a measured environment.
    """
    frame = environments.copy()
    if "env_id" not in frame.columns:
        raise ValueError("environments is missing required columns: env_id")
    frame["env_id"] = frame["env_id"].astype(str)
    duplicate_ids = sorted(
        frame.loc[frame["env_id"].duplicated(keep=False), "env_id"].unique()
    )
    if duplicate_ids:
        raise ValueError(
            "duplicate environment metadata for env_id values: "
            + ", ".join(duplicate_ids)
        )
    frame = frame.set_index("env_id", drop=False)
    missing = [env_id for env_id in measurement_env_ids if env_id not in frame.index]
    if missing:
        raise ValueError(
            "environments is missing metadata for measured env_id values: "
            + ", ".join(missing)
        )

    rows: list[dict[str, Any]] = []
    for env_id in measurement_env_ids:
        selected = frame.loc[env_id]
        raw = (
            selected.iloc[-1].to_dict()
            if isinstance(selected, pd.DataFrame)
            else selected.to_dict()
        )
        row = {str(key): _json_value(value) for key, value in raw.items()}
        row["env_id"] = env_id
        row["label"] = str(row.get("label") or env_id)
        row["control_label"] = _control_label(row)
        rows.append(row)
    return rows


def _control_label(environment: Mapping[str, Any]) -> str:
    """Return a short compute-toggle label, e.g. ``2 threads`` or ``A100 GPU``.

    GPU environments are labelled by accelerator model; CPU environments by their
    thread count. Falls back to the full ``label`` when neither is available.
    """
    gpu_models = string_list(environment.get("gpu_models"))
    gpu_count = coerce_int(environment.get("gpu_count")) or 0
    if gpu_count > 0 and gpu_models:
        short = _gpu_short_name(gpu_models[0])
        prefix = f"{gpu_count}× " if gpu_count > 1 else ""
        return f"{prefix}{short} GPU"
    threads = _thread_count(environment)
    if threads is not None:
        return f"{threads} threads"
    return str(environment.get("label") or environment.get("env_id") or "")


def _gpu_short_name(model: str) -> str:
    """Extract a concise accelerator name (e.g. ``A100``) from a GPU model string."""
    import re

    match = re.search(r"\b([A-Z]{1,4}\d{3,4}[A-Za-z]{0,3})\b", model)
    return match.group(1) if match else model


def _thread_count(environment: Mapping[str, Any]) -> int | None:
    """Return the CPU thread count, preferring ``OMP_NUM_THREADS``.

    Falls back to the logical core count when the threading variable is absent.
    """
    env_vars = environment.get("env_vars")
    pairs: list[tuple[Any, Any]] = []
    if isinstance(env_vars, Mapping):
        pairs = list(env_vars.items())
    elif isinstance(env_vars, (list, tuple, np.ndarray)):
        pairs = [
            (item[0], item[1])
            for item in env_vars
            if isinstance(item, (list, tuple, np.ndarray)) and len(item) == 2
        ]
    for key, value in pairs:
        if str(key) == "OMP_NUM_THREADS":
            threads = coerce_int(value)
            if threads is not None and threads > 0:
                return threads
    return coerce_int(environment.get("cores_logical"))


def normalize_fits(fits: pd.DataFrame) -> list[FitRecord]:
    """Validate and normalize precomputed fit segments.

    Args:
        fits: Concatenated fit-segment table.

    Returns:
        JSON-safe fit segment records.

    Raises:
        ValueError: If a non-empty table lacks required fit columns.
    """
    if fits.empty:
        return []
    _require_columns(fits, _FIT_COLUMNS, "fits")
    frame = fits.copy()
    valid = (
        frame["metric"].isin(metrics.METRIC_IDS)
        & frame["x_axis"].isin(metrics.X_AXES)
        & pd.to_numeric(frame["x_start"], errors="coerce").gt(0)
        & pd.to_numeric(frame["x_end"], errors="coerce").gt(0)
    )
    frame = frame.loc[valid].sort_values(
        ["env_id", "basis", "functional", "metric", "x_axis", "segment_index"]
    )
    records: list[FitRecord] = []
    for record in frame.to_dict(orient="records"):
        normalized = {str(key): _json_value(value) for key, value in record.items()}
        records.append(
            {
                "env_id": normalized["env_id"],
                "basis": normalized["basis"],
                "functional": normalized["functional"],
                "metric": normalized["metric"],
                "x_axis": normalized["x_axis"],
                "segment_index": normalized["segment_index"],
                "x_start": normalized["x_start"],
                "x_end": normalized["x_end"],
                "slope": normalized["slope"],
                "intercept": normalized["intercept"],
                "continuous": normalized["continuous"],
                "cv_score": normalized["cv_score"],
                "n_points": normalized["n_points"],
                "breakpoints": normalized["breakpoints"],
            }
        )
    return records


def point_records(points: pd.DataFrame) -> list[PointRecord]:
    """Convert prepared points to compact JSON-safe records.

    Args:
        points: Prepared point table.

    Returns:
        List of point dictionaries.
    """
    records: list[PointRecord] = []
    for record in points.to_dict(orient="records"):
        records.append(
            {
                "env_id": str(record["env_id"]),
                "basis": str(record["basis"]),
                "functional": str(record["functional"]),
                "metric": str(record["metric"]),
                "x_axis": str(record["x_axis"]),
                "x": float(record["x"]),
                "y": float(record["y"]),
                "molecule": _json_value(record["molecule"]),
                "ansatz": _json_value(record["ansatz"]),
                "atoms": _json_value(record["atoms"]),
                "electrons": _json_value(record["electrons"]),
                "num_aos": _json_value(record["num_aos"]),
                "grid_size": _json_value(record["grid_size"]),
                "scf_iterations": _json_value(record["scf_iterations"]),
                "energy": _json_value(record["energy"]),
                "warmup_ratio": _json_value(record["warmup_ratio"]),
            }
        )
    return records


COMPOSITION_BUCKETS: dict[str, list[CompositionBucketRecord]] = {
    series: [
        {
            "id": band,
            "label": definition.label,
            "short": definition.short_label,
        }
        for band, definition in metrics.COMPOSITION_SERIES[series].bands.items()
    ]
    for series in metrics.COMPOSITION_SERIES_IDS
}


def composition_records(
    measurements: pd.DataFrame, series: str = "cycle"
) -> list[CompositionRecord]:
    """Build per-(env, basis, functional, #AOs) time-composition fractions.

    Rows sharing the same atomic-orbital count are averaged so each environment,
    basis, and functional yields one stacked-area point per system size.

    Args:
        measurements: Concatenated collected measurement table.
        series: Which composition to build, see
            :data:`skala.benchmark.metrics.COMPOSITION_SERIES`.

    Returns:
        Records with ``env_id``, ``basis``, ``functional``, ``x`` (#AOs), and
        ``y`` (fractions ordered by the series' band identifiers).
    """
    buckets = list(metrics.composition_band_ids(series))
    rows: list[dict[str, Any]] = []
    for raw_row in measurements.to_dict(orient="records"):
        row = {str(key): _json_value(value) for key, value in raw_row.items()}
        fractions = metrics.composition_fractions(row, series)
        if fractions is None:
            continue
        functional = str(row.get("functional"))
        x_value = metrics.x_value(row, "num_aos")
        if x_value is None:
            continue
        rows.append(
            {
                "env_id": str(row.get("env_id")),
                "basis": str(row.get("basis")),
                "functional": functional,
                "x": x_value,
                **{bucket: fractions[bucket] for bucket in buckets},
            }
        )
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["env_id", "basis", "functional", "x"], as_index=False)[
        buckets
    ].mean()
    grouped = grouped.sort_values(["env_id", "basis", "functional", "x"])
    return [
        {
            "env_id": str(record["env_id"]),
            "basis": str(record["basis"]),
            "functional": str(record["functional"]),
            "x": float(record["x"]),
            "y": [float(record[bucket]) for bucket in buckets],
        }
        for record in grouped.to_dict(orient="records")
    ]


def composition_smooth(
    records: Sequence[CompositionRecord],
) -> list[SmoothedCompositionRecord]:
    """Smooth each composition series into a dense stacked-fraction curve.

    Groups the per-point fractions from :func:`composition_records` by
    environment, basis, and functional, then fits a cross-validated smoothing
    spline to the cumulative band boundaries (see
    :func:`skala.benchmark.fitting.smooth_stacked_fractions`). The result feeds
    the report's stacked-area charts so the bands read as smooth curves rather
    than piecewise segments.

    Args:
        records: Composition records from :func:`composition_records`.

    Returns:
        One record per (env, basis, functional) with ``x`` (dense grid) and
        ``cumulative`` (band boundaries, each column summing to one).
    """
    grouped: dict[tuple[str, str, str], list[CompositionRecord]] = {}
    for record in records:
        key = (record["env_id"], record["basis"], record["functional"])
        grouped.setdefault(key, []).append(record)

    smoothed: list[SmoothedCompositionRecord] = []
    for (env_id, basis, functional), group in grouped.items():
        ordered = sorted(group, key=lambda item: item["x"])
        xs = [float(item["x"]) for item in ordered]
        fractions = [list(item["y"]) for item in ordered]
        stack = fitting.smooth_stacked_fractions(xs, fractions)
        smoothed.append(
            {
                "env_id": env_id,
                "basis": basis,
                "functional": functional,
                "x": stack.x,
                "cumulative": stack.cumulative,
            }
        )
    return smoothed


def composition_domain(records: Sequence[CompositionRecord]) -> AxisDomain:
    """Return the fixed log x domain and [0, 1] y domain for composition charts.

    Args:
        records: Composition records from :func:`composition_records`.

    Returns:
        ``{"x": [low, high], "y": [0, 1]}``.
    """
    xs = pd.Series([record["x"] for record in records], dtype="float64")
    return {"x": _log_domain(xs), "y": [0.0, 1.0]}


def _log_domain(values: pd.Series) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric) & numeric.gt(0)]
    if numeric.empty:
        return [1.0, 10.0]
    low = float(numeric.min())
    high = float(numeric.max())
    if np.isclose(low, high):
        return [low / 1.2, high * 1.2]
    padding = (np.log10(high) - np.log10(low)) * 0.04
    return [
        float(10 ** (np.log10(low) - padding)),
        float(10 ** (np.log10(high) + padding)),
    ]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
