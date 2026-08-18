# SPDX-License-Identifier: MIT

"""Collect a local benchmark output directory into report-ready JSON files."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from skala.benchmark import metrics
from skala.benchmark.fitting import fit_piecewise_loglog
from skala.benchmark.schema.environment import Environment
from skala.benchmark.schema.measurements import read_dataset

MIN_FIT_POINTS = 4


def collect_results(
    input_dir: str | Path, output_dir: str | Path
) -> tuple[Path, Path, Path]:
    """Collect one local benchmark dataset into report-ready JSON files.

    Args:
        input_dir: Dataset root containing ``environments/`` and
            ``measurements/``.
        output_dir: Directory in which to write the collected JSON files.

    Returns:
        Paths to ``environments.json``, ``measurements.json``, and ``fits.json``.

    Raises:
        ValueError: If a measurement refers to an unknown environment or
            duplicate computation.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    environment_rows = _load_environments(input_dir / "environments")
    measurements = read_dataset(input_dir / "measurements").to_pylist()
    _validate_measurements(measurements, {row["env_id"] for row in environment_rows})

    output_dir.mkdir(parents=True, exist_ok=True)
    environments_path = output_dir / "environments.json"
    measurements_path = output_dir / "measurements.json"
    fits_path = output_dir / "fits.json"
    _write_json(environment_rows, environments_path)
    _write_json(measurements, measurements_path)
    _write_json(_collect_fit_rows(measurements), fits_path)
    return environments_path, measurements_path, fits_path


def _collect_fit_rows(rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["env_id"]), str(row["basis"]), str(row["functional"]))
        groups.setdefault(key, []).append(row)

    fit_rows: list[dict[str, object]] = []
    for (env_id, basis, functional), group_rows in groups.items():
        for metric, x_axes in metrics.FIT_COMBINATIONS:
            for x_axis in x_axes:
                points: list[tuple[float, float]] = []
                for row in group_rows:
                    x = metrics.x_value(row, x_axis)
                    y = metrics.metric_value(row, metric)
                    if x is not None and y is not None:
                        points.append((x, y))
                if len(points) < MIN_FIT_POINTS:
                    continue

                x_values, y_values = zip(*points, strict=True)
                fit = fit_piecewise_loglog(
                    x_values,
                    y_values,
                )
                fit_rows.extend(
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
                )
    return fit_rows


def _load_environments(directory: Path) -> list[dict[str, Any]]:
    rows = [
        Environment.load(path).to_dict() for path in sorted(directory.glob("*.json"))
    ]
    if not rows:
        raise ValueError(f"no environment JSON files found in {directory}")
    env_ids = [str(row["env_id"]) for row in rows]
    if len(env_ids) != len(set(env_ids)):
        raise ValueError("duplicate environment ids")
    return rows


def _validate_measurements(
    rows: list[dict[str, Any]], environment_ids: set[str]
) -> None:
    unknown = sorted({str(row["env_id"]) for row in rows} - environment_ids)
    if unknown:
        raise ValueError(f"measurements refer to unknown environments: {unknown}")
    keys = [
        (row["env_id"], row["basis"], row["functional"], row["mol_hash"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate DFT computations in measurements")


def _write_json(rows: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            _json_value(rows),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_value(value: Any) -> Any:
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
