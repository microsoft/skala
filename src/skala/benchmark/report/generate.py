# SPDX-License-Identifier: MIT

"""Orchestration for the standalone benchmark report generator."""

from __future__ import annotations

import json
import shutil
import warnings
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from .data import (
    COMPOSITION_BUCKETS,
    METRICS,
    X_MODES,
    _string_list,
    composition_domain,
    composition_records,
    composition_smooth,
    compute_domains,
    environment_rows,
    normalize_fits,
    point_records,
    prepare_points,
)
from .prose import load_prose, markdown_to_html

_PACKAGE_DIR = Path(__file__).resolve().parent
_ASSET_DIR = _PACKAGE_DIR / "assets"
_REQUIRED_FILES = (
    "measurements.json",
    "environments.json",
    "fits.json",
)
_BASIS_ORDER = ("def2-svp", "def2-tzvp", "def2-qzvp")
_DEFAULT_BASIS = "def2-tzvp"
_TEMPLATES = Environment(
    loader=FileSystemLoader(_PACKAGE_DIR / "templates"),
    autoescape=select_autoescape(("html",)),
)


def generate(
    output_dir: str | Path,
    collected_dirs: Sequence[str | Path],
    prose_path: str | Path | None = None,
) -> Path:
    """Generate a fully offline static benchmark report.

    Args:
        output_dir: Destination directory for the report bundle.
        collected_dirs: One or more collected directories. Every directory must
            contain ``measurements.json``, ``environments.json``, and
            ``fits.json``.
        prose_path: Optional prose YAML file.

    Returns:
        Path to the generated ``index.html``.

    Raises:
        ValueError: If no collected directories are provided or table columns are absent.
        FileNotFoundError: If an input is not a directory or a required file is missing.
    """
    input_files = _validate_collected_dirs(collected_dirs)

    measurement_rows = _read_json_files(
        [files["measurements.json"] for files in input_files], "measurements"
    )
    environment_records = _read_json_files(
        [files["environments.json"] for files in input_files], "environments"
    )
    fit_records = _read_json_files(
        [files["fits.json"] for files in input_files], "fits"
    )
    measurements = pd.DataFrame(measurement_rows)
    environments_frame = pd.DataFrame(environment_records)
    fits_frame = pd.DataFrame(fit_records)
    points = prepare_points(measurements)
    if points.empty:
        raise ValueError("no converged successful measurements are available to plot")

    prose = load_prose(prose_path)
    env_ids = list(dict.fromkeys(points["env_id"].astype(str).tolist()))
    env_ids = _ordered_environment_ids(env_ids, prose.get("environment_order"))
    environments = environment_rows(environments_frame, env_ids)
    _validate_measurement_rows(measurement_rows)
    fits = normalize_fits(fits_frame)
    _warn_for_missing_fits(points, fits)

    title = str(prose.get("title") or "Skala benchmark scaling report")
    author = str(prose.get("author") or "")
    report_date = str(prose.get("date") or date.today().isoformat())
    abstract = str(prose.get("abstract") or "")
    bases = _ordered_basis_names(points["basis"].astype(str).tolist())
    functionals = list(dict.fromkeys(points["functional"].astype(str).tolist()))

    composition = {
        series: composition_records(measurements, series)
        for series in COMPOSITION_BUCKETS
    }
    data = {
        "meta": {
            "environments": environments,
            "bases": bases,
            "functionals": functionals,
            "metrics": METRICS,
            "x_modes": X_MODES,
            "composition_buckets": COMPOSITION_BUCKETS,
            "fits_available": bool(fits),
            "initial_selection": {
                "environment": env_ids[0],
                "basis": _DEFAULT_BASIS if _DEFAULT_BASIS in bases else bases[0],
            },
        },
        "points": point_records(points),
        "domains": compute_domains(points),
        "composition": composition,
        "composition_smooth": {
            series: composition_smooth(records)
            for series, records in composition.items()
        },
        "composition_domain": {
            series: composition_domain(records)
            for series, records in composition.items()
        },
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_json = _json_dump(data)
    fits_json = _json_dump(fits)
    (output / "data.json").write_text(data_json, encoding="utf-8")

    comparison = _mapping(prose.get("comparison"))
    annotations = _mapping(comparison.get("environments"))
    environment_cards = [
        {
            **environment,
            "facts": _environment_facts(environment),
            "annotation": Markup(
                markdown_to_html(annotations.get(environment["env_id"]))
            ),
        }
        for environment in environments
    ]
    context = {
        "title": title,
        "author": author,
        "date": report_date,
        "abstract": Markup(markdown_to_html(abstract)),
        "intro": Markup(markdown_to_html(prose.get("intro"))),
        "comparison_intro": Markup(
            markdown_to_html(_mapping(prose.get("comparison")).get("intro"))
        ),
        "environments": environment_cards,
        "comparison_notes": [
            Markup(markdown_to_html(note)) for note in comparison.get("notes", [])
        ],
        "numint_label": METRICS["numint"]["label"],
        "xc_eval_label": METRICS["xc_eval"]["label"],
        "cycle_label": METRICS["cycle"]["label"],
        "jk_label": METRICS["jk"]["label"],
        "iterations_label": METRICS["iterations"]["label"],
        "total_label": METRICS["total"]["label"],
        "setup_label": METRICS["setup"]["label"],
        "numint_commentary": Markup(
            markdown_to_html(_mapping(prose.get("plots")).get("numint"))
        ),
        "xc_eval_commentary": Markup(
            markdown_to_html(_mapping(prose.get("plots")).get("xc_eval"))
        ),
        "cycle_commentary": Markup(
            markdown_to_html(_mapping(prose.get("plots")).get("cycle"))
        ),
        "jk_commentary": Markup(
            markdown_to_html(_mapping(prose.get("plots")).get("jk"))
        ),
        "iterations_commentary": Markup(
            markdown_to_html(_mapping(prose.get("plots")).get("iterations"))
        ),
        "total_commentary": Markup(
            markdown_to_html(_mapping(prose.get("plots")).get("total"))
        ),
        "setup_commentary": Markup(
            markdown_to_html(_mapping(prose.get("plots")).get("setup"))
        ),
        "composition_commentary": Markup(
            markdown_to_html(_mapping(prose.get("plots")).get("composition"))
        ),
        "run_composition_commentary": Markup(
            markdown_to_html(_mapping(prose.get("plots")).get("run_composition"))
        ),
        "startup_commentary": Markup(
            markdown_to_html(_mapping(prose.get("plots")).get("startup"))
        ),
        "closing": Markup(markdown_to_html(prose.get("closing"))),
        "data_json": Markup(_script_json(data_json)),
        "fits_json": Markup(_script_json(fits_json)),
    }
    rendered = _TEMPLATES.get_template("index.html").render(**context)
    (output / "index.html").write_text(rendered, encoding="utf-8")

    assets = {
        _ASSET_DIR / "report-base.css": output / "report-base.css",
        _ASSET_DIR / "report.css": output / "report.css",
        _ASSET_DIR / "report.js": output / "report.js",
        _ASSET_DIR / "vendor" / "d3.min.js": output / "d3.min.js",
        _ASSET_DIR / "vendor" / "katex.min.js": output / "katex.min.js",
    }
    for source, destination in assets.items():
        shutil.copyfile(source, destination)
    return output / "index.html"


def _read_json_files(paths: Sequence[Path], kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}: {error}") from error
        if not isinstance(value, list) or not all(
            isinstance(row, Mapping) for row in value
        ):
            raise ValueError(f"{path} must contain a JSON array of {kind} objects")
        rows.extend(dict(row) for row in value)
    return rows


def _validate_measurement_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    required = {"env_id", "basis", "functional", "mol_hash"}
    missing = sorted({field for row in rows for field in required if field not in row})
    if missing:
        raise ValueError(
            "measurements is missing required identity fields: " + ", ".join(missing)
        )
    keys = [
        (
            str(row["env_id"]),
            str(row["basis"]),
            str(row["functional"]),
            str(row["mol_hash"]),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate DFT computations across collected inputs")


def _validate_collected_dirs(
    collected_dirs: Sequence[str | Path],
) -> list[dict[str, Path]]:
    if not collected_dirs:
        raise ValueError("at least one collected directory is required")
    result: list[dict[str, Path]] = []
    for raw_directory in collected_dirs:
        directory = Path(raw_directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"collected input is not a directory: {directory}")
        missing = [
            filename
            for filename in _REQUIRED_FILES
            if not (directory / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"collected directory {directory} is missing required file(s): "
                + ", ".join(missing)
            )
        result.append({filename: directory / filename for filename in _REQUIRED_FILES})
    return result


def _warn_for_missing_fits(
    points: pd.DataFrame, fits: Sequence[Mapping[str, Any]]
) -> None:
    if not fits:
        warnings.warn(
            "No precomputed fits were found; fitted lines will be omitted.",
            stacklevel=2,
        )
        return
    available = {
        (
            str(fit["env_id"]),
            str(fit["basis"]),
            str(fit["functional"]),
            str(fit["metric"]),
            str(fit["x_axis"]),
        )
        for fit in fits
    }
    expected = {
        (
            str(row.env_id),
            str(row.basis),
            str(row.functional),
            str(row.metric),
            str(row.x_axis),
        )
        for row in points[
            ["env_id", "basis", "functional", "metric", "x_axis"]
        ].itertuples(index=False)
    }
    missing = ["/".join(key) for key in sorted(expected - available)]
    if missing:
        sample = ", ".join(missing[:5])
        suffix = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        warnings.warn(
            f"Precomputed fits are missing for {sample}{suffix}; those lines will be omitted.",
            stacklevel=2,
        )


def _environment_facts(environment: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return compact card rows: hardware focus plus one combined software line.

    Processor, cores, memory, and GPU are shown as individual rows; every
    software version is collapsed into a single ``Software`` row so the cards
    stay short enough to place several side by side.
    """
    facts: list[tuple[str, str]] = []
    cpu_model = environment.get("cpu_model")
    if cpu_model:
        facts.append(("Processor", str(cpu_model)))
    cores = _coerce_int(environment.get("cores_physical"))
    logical = _coerce_int(environment.get("cores_logical"))
    if cores is not None:
        cores_text = f"{cores} physical"
        if logical is not None and logical != cores:
            cores_text += f" / {logical} logical"
        facts.append(("CPU cores", cores_text))
    memory = environment.get("mem_total_gb")
    if isinstance(memory, (int, float)):
        facts.append(("Memory", f"{memory:g} GB"))
    gpu_models = _string_list(environment.get("gpu_models"))
    gpu_count = _coerce_int(environment.get("gpu_count")) or 0
    if gpu_models:
        prefix = f"{gpu_count}× " if gpu_count > 1 else ""
        facts.append(("GPU", f"{prefix}{gpu_models[0]}"))
    software = _software_summary(environment)
    if software:
        facts.append(("Software", software))
    return facts


def _software_summary(environment: Mapping[str, Any]) -> str:
    """Collapse all recorded software versions into one compact line."""
    parts: list[str] = []
    skala_version = _skala_version(environment)
    if skala_version:
        parts.append(f"Skala {skala_version}")
    for name, key in (
        ("PySCF", "pyscf_version"),
        ("gpu4pyscf", "gpu4pyscf_version"),
        ("PyTorch", "torch_version"),
        ("CUDA", "cuda_version"),
        ("NumPy", "numpy_version"),
        ("Python", "python_version"),
        ("BLAS", "blas_impl"),
    ):
        value = environment.get(key)
        if value:
            parts.append(f"{name} {value}")
    return " · ".join(parts)


def _skala_version(environment: Mapping[str, Any]) -> str:
    """Return the Skala code version, preferring a first-class field.

    Falls back to the catch-all ``versions`` mapping, which the orchestrator
    populates as a dict or a list of ``[name, version]`` pairs.
    """
    direct = environment.get("skala_version")
    if direct:
        return str(direct)
    versions = environment.get("versions")
    if isinstance(versions, Mapping):
        value = versions.get("skala")
        return str(value) if value else ""
    if isinstance(versions, (list, tuple)):
        for item in versions:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and str(item[0]) == "skala"
                and item[1]
            ):
                return str(item[1])
    return ""


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _ordered_environment_ids(
    measured_ids: list[str], configured_order: Any
) -> list[str]:
    if configured_order is None:
        return measured_ids
    if not isinstance(configured_order, list):
        raise ValueError("environment_order must be a list")

    requested = [str(env_id) for env_id in configured_order]
    unknown = sorted(set(requested) - set(measured_ids))
    if unknown:
        raise ValueError(
            "environment_order contains unmeasured env_id values: " + ", ".join(unknown)
        )
    return list(dict.fromkeys([*requested, *measured_ids]))


def _ordered_basis_names(measured_names: Sequence[str]) -> list[str]:
    available = list(dict.fromkeys(measured_names))
    preferred = [basis for basis in _BASIS_ORDER if basis in available]
    return [*preferred, *(basis for basis in available if basis not in preferred)]


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _script_json(value: str) -> str:
    return value.replace("</", "<\\/")
