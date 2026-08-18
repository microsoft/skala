# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skala.benchmark import metrics
from skala.benchmark.report import generate
from skala.benchmark.report.prose import load_prose


def _cycles(cycle_ms: float, *, forward: bool) -> list[dict[str, object]]:
    """Four iterations: a warmup-heavy first one, then a flat steady state."""
    veff_ms = 0.8 * cycle_ms
    numint_ms = 0.6 * cycle_ms
    # Every functional is timed at the XC layer; only the split into forward and
    # backward is neural-only.
    xc_eval_ms = 0.4 * numint_ms
    return [
        {
            "cycle": index,
            "wall_ms": cycle_ms * scale,
            "veff_ms": veff_ms * scale,
            "numint_ms": numint_ms * scale,
            "xc_eval_ms": xc_eval_ms * scale,
            "forward_ms": 0.4 * xc_eval_ms * scale if forward else 0.0,
            "backward_ms": 0.6 * xc_eval_ms * scale if forward else 0.0,
            "veff_calls": 1,
            "numint_calls": 1,
            "xc_eval_calls": 1,
        }
        for index, scale in enumerate([3.0, 1.0, 1.0, 1.0])
    ]


def _write_inputs(directory: Path) -> Path:
    directory.mkdir(parents=True)
    environments = [
        {
            "env_id": "gpu-env",
            "label": "A100 benchmark",
            "hostname": "cc0d49b2baaf4587ab5b7619264a26fb000000",
            "cpu_model": "AMD EPYC",
            "cores_physical": 32,
            "cores_logical": 64,
            "mem_total_gb": 256.0,
            "gpu_models": ["NVIDIA A100 80GB PCIe"],
            "gpu_count": 1,
            "torch_version": "2.8",
            "pyscf_version": "2.12",
            "gpu4pyscf_version": "1.4",
            "cuda_version": "12.8",
            "blas_impl": "MKL",
            "python_version": "3.11",
            "skala_version": "2026.7",
        },
        {
            "env_id": "cpu-env",
            "hostname": "cpu-node",
            "cpu_model": "Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz",
            "cores_physical": 24,
            "cores_logical": 48,
            "mem_total_gb": 128.0,
            "gpu_models": [],
            "gpu_count": 0,
            "torch_version": "2.8",
            "pyscf_version": "2.12",
            "gpu4pyscf_version": None,
            "cuda_version": None,
            "blas_impl": "OpenBLAS",
            "python_version": "3.11",
            "skala_version": "2026.7",
        },
    ]
    (directory / "environments.json").write_text(
        json.dumps(environments, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    molecules = [
        ("water", 24, 80_000, 3, 10),
        ("benzene", 114, 300_000, 12, 42),
        ("caffeine", 310, 900_000, 24, 102),
    ]
    functionals = ["skala-1.1", "r2scan", "custom-xc"]
    bases = ["def2-svp", "custom-basis"]
    for env_index, env_id in enumerate(["gpu-env", "cpu-env"]):
        for basis_index, basis in enumerate(bases):
            for functional_index, functional in enumerate(functionals):
                for order, (
                    molecule,
                    num_aos,
                    grid_size,
                    num_atoms,
                    electrons,
                ) in enumerate(molecules):
                    iterations = 6 + functional_index
                    kernel = (
                        (num_aos**1.35)
                        * (1.0 + basis_index * 0.25)
                        * (1.0 + functional_index * 0.2)
                        * (1.0 if env_index == 0 else 2.5)
                    )
                    rows.append(
                        {
                            "env_id": env_id,
                            "basis": basis,
                            "functional": functional,
                            "mol_name": molecule,
                            "mol_hash": molecule,
                            # One shard per (basis, functional), running the
                            # molecules in order, as a real sweep does.
                            "shard_index": basis_index * 3 + functional_index,
                            "timestamp": f"2026-07-16T00:0{order}:00+00:00",
                            "ansatz": "RKS",
                            "device": "gpu" if env_index == 0 else "cpu",
                            "num_atoms": num_atoms,
                            "num_electrons": electrons,
                            "num_atomic_orbitals": num_aos,
                            "grid_size": grid_size,
                            "is_converged": True,
                            "num_scf_iterations": iterations,
                            "total_energy": -0.5 * electrons,
                            "kernel_time_ms": kernel,
                            "wall_time_ms": kernel + 5.0,
                            "setup_ms": 25.0,
                            "finalize_ms": 5.0,
                            "load_ms": 200.0 if functional == "skala-1.1" else 0.0,
                            "process_warmup_ms": 12800.0 if order == 0 else 4200.0,
                            "jit_compile_ms": (
                                180.0 if functional == "skala-1.1" else 0.0
                            ),
                            "cycles": _cycles(
                                kernel / iterations,
                                # Only Skala has a network to charge forward to.
                                forward=functional == "skala-1.1",
                            ),
                            "status": "ok",
                        }
                    )
    (directory / "measurements.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    fit_rows: list[dict[str, object]] = []
    metric_modes = {
        "xc_eval": ["num_aos", "grid_size"],
        "numint": ["num_aos", "grid_size"],
        "jk": ["num_aos"],
        "cycle": ["num_aos", "grid_size"],
        "iterations": ["num_aos"],
        "total": ["num_aos"],
        "setup": ["num_aos"],
    }
    for env_id in ["gpu-env", "cpu-env"]:
        for basis in bases:
            for functional in functionals:
                for metric, modes in metric_modes.items():
                    for x_axis in modes:
                        x_start, x_end = (
                            (24.0, 310.0)
                            if x_axis == "num_aos"
                            else (80_000.0, 900_000.0)
                        )
                        fit_rows.append(
                            {
                                "env_id": env_id,
                                "basis": basis,
                                "functional": functional,
                                "metric": metric,
                                "x_axis": x_axis,
                                "segment_index": 0,
                                "x_start": x_start,
                                "x_end": x_end,
                                "slope": 1.2,
                                "intercept": -1.0,
                                "continuous": True,
                                "cv_score": 0.1,
                                "n_points": 3,
                                "breakpoints": [],
                            }
                        )
    (directory / "fits.json").write_text(
        json.dumps(fit_rows, indent=2), encoding="utf-8"
    )
    return directory


def test_generate_offline_report_with_fixed_domains_and_comparison(
    tmp_path: Path,
) -> None:
    collected_dir = _write_inputs(tmp_path / "inputs")
    output = tmp_path / "report"

    index_path = generate(output, [collected_dir])

    assert index_path == output / "index.html"
    for name in [
        "index.html",
        "data.json",
        "fits.json",
        "metadata.yaml",
        "report.js",
        "report.css",
        "d3.min.js",
        "katex.min.js",
        "report-base.css",
    ]:
        assert (output / name).is_file()

    index = (output / "index.html").read_text()
    assert "http://" not in index
    assert 'href="https://microsoft.github.io/skala/benchmarks.html"' in index
    assert 'src="d3.min.js"' in index
    assert 'src="katex.min.js"' in index
    assert 'src="report.js"' in index
    assert 'href="report-base.css"' in index
    assert "Compare your own timings with these results" in index
    assert "LivDFT" not in index
    assert 'href="/reports"' not in index
    assert "lorem ipsum" not in index.lower()

    data = json.loads((output / "data.json").read_text())
    assert set(data["meta"]["metrics"]) == {
        "xc_eval",
        "numint",
        "jk",
        "cycle",
        "iterations",
        "total",
        "setup",
    }
    assert data["meta"]["initial_selection"] == {
        "environment": data["points"][0]["env_id"],
        "basis": data["points"][0]["basis"],
    }
    assert set(data["domains"]["xc_eval"]) == {"num_aos", "grid_size"}
    assert set(data["domains"]["numint"]) == {"num_aos", "grid_size"}
    assert set(data["domains"]["cycle"]) == {"num_aos", "grid_size"}
    assert set(data["domains"]["jk"]) == {"num_aos"}
    assert set(data["domains"]["iterations"]) == {"num_aos"}
    assert set(data["domains"]["total"]) == {"num_aos"}
    assert "comparison" not in data
    environment_labels = {
        environment["env_id"]: environment["label"]
        for environment in data["meta"]["environments"]
    }
    assert environment_labels == {
        "gpu-env": "A100 benchmark",
        "cpu-env": "cpu-env",
    }
    assert '<div class="cards env-cards">' in index
    assert "<span>Software</span>" in index
    assert "Skala 2026.7" in index
    assert "A100 benchmark" in index
    assert "cpu-env" in index
    assert data["meta"]["functionals"] == ["skala-1.1", "r2scan", "custom-xc"]
    assert data["meta"]["bases"] == ["def2-svp", "custom-basis"]
    report_css = (output / "report.css").read_text()
    assert "overflow-x: clip" in report_css
    assert "white-space: nowrap" in report_css
    report_js = (output / "report.js").read_text()
    assert 'attr("class", "functional-series")' in report_js
    assert 'attr("data-functional", functional)' in report_js
    assert '.on("pointerenter focus", focusSeries)' in report_js
    assert "restoreSeriesOrder" in report_js
    assert all(domain["x"][0] > 0 for domain in data["domains"]["numint"].values())
    first = next(
        point
        for point in data["points"]
        if point["metric"] == "numint" and point["x_axis"] == "num_aos"
    )
    raw_first = json.loads((collected_dir / "measurements.json").read_text())[0]
    assert first["x"] == metrics.x_value(raw_first, "num_aos")
    assert first["y"] == metrics.metric_value(raw_first, "numint")
    assert data["meta"]["metrics"]["numint"] == {
        "label": metrics.METRIC_LABELS["numint"],
        "unit": metrics.METRIC_UNITS["numint"],
    }
    # Every functional is timed at the XC-evaluation layer, neural or not.
    xc_point = next(
        point
        for point in data["points"]
        if point["metric"] == "xc_eval" and point["x_axis"] == "grid_size"
    )
    assert {
        point["functional"] for point in data["points"] if point["metric"] == "xc_eval"
    } == {"skala-1.1", "r2scan", "custom-xc"}
    # Cycle 0 costs 3x the steady state in the fixture, and the scaling metrics
    # must report the steady state rather than an average over all iterations.
    assert xc_point["warmup_ratio"] == pytest.approx(3.0)

    # Both compositions are present, each with its own bands, and every stack
    # sums to one.
    assert set(data["composition"]) == set(metrics.COMPOSITION_SERIES_IDS)
    for series in metrics.COMPOSITION_SERIES_IDS:
        bands = metrics.composition_band_ids(series)
        assert [
            bucket["id"] for bucket in data["meta"]["composition_buckets"][series]
        ] == list(bands)
        for record in data["composition"][series]:
            assert sum(record["y"]) == pytest.approx(1.0)
            assert len(record["y"]) == len(bands)

    # Start-up is pooled across bases and molecules, but the first task of each
    # shard is dropped: it also pays what the machine caches after it.


@pytest.mark.parametrize(
    "missing",
    [
        "measurements.json",
        "environments.json",
        "fits.json",
    ],
)
def test_generate_requires_complete_collected_directories(
    tmp_path: Path, missing: str
) -> None:
    collected_dir = _write_inputs(tmp_path / "inputs")
    (collected_dir / missing).unlink()

    with pytest.raises(FileNotFoundError, match=missing):
        generate(tmp_path / "report", [collected_dir])


def test_generate_defaults_are_neutral_for_local_comparisons(tmp_path: Path) -> None:
    collected_dir = _write_inputs(tmp_path / "inputs")
    output = tmp_path / "report"

    generate(output, [collected_dir])

    index = (output / "index.html").read_text()
    assert "Skala benchmark comparison" in index
    assert "No study-specific conclusions were supplied" in index
    assert "--prose" not in index
    assert "On the A100 it is about 6.8 s" not in index
    assert "Toggle the horizontal axis" in index
    assert "Compare your own timings with these results" in index
    assert "https://microsoft.github.io/skala/benchmarks.html" in index
    assert "python -m skala.benchmark.report" not in index


def test_reference_prose_contains_official_analysis() -> None:
    repository_root = Path(__file__).parents[2]
    prose = load_prose(repository_root / "benchmarks" / "reference" / "prose.yaml")

    assert prose["title"] == "Skala performance and scaling benchmark"
    assert "On the A100 it is about 6.8 s" in prose["plots"]["startup"]
    assert (
        "Sixteen-thread CPU reference"
        in prose["comparison"]["environments"]["d32v3-t16"]
    )


def test_generate_rejects_missing_environment_metadata(tmp_path: Path) -> None:
    collected_dir = _write_inputs(tmp_path / "inputs")
    environments_path = collected_dir / "environments.json"
    environments = json.loads(environments_path.read_text())
    environments_path.write_text(
        json.dumps(
            [
                environment
                for environment in environments
                if environment["env_id"] != "cpu-env"
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cpu-env"):
        generate(tmp_path / "report", [collected_dir])


def test_generate_rejects_duplicate_environment_ids(tmp_path: Path) -> None:
    first = _write_inputs(tmp_path / "first")
    second = _write_inputs(tmp_path / "second")

    with pytest.raises(ValueError, match="duplicate environment metadata"):
        generate(tmp_path / "report", [first, second])


def test_generate_rejects_duplicate_computations(tmp_path: Path) -> None:
    collected = _write_inputs(tmp_path / "inputs")
    measurements = json.loads((collected / "measurements.json").read_text())
    measurements.append(dict(measurements[0]))
    (collected / "measurements.json").write_text(json.dumps(measurements))

    with pytest.raises(ValueError, match="duplicate DFT computations"):
        generate(tmp_path / "report", [collected])


def test_generate_rejects_malformed_collected_json(tmp_path: Path) -> None:
    collected = _write_inputs(tmp_path / "inputs")
    (collected / "fits.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        generate(tmp_path / "report", [collected])


def test_prose_yaml_flows_into_report(tmp_path: Path) -> None:
    collected_dir = _write_inputs(tmp_path / "inputs")
    prose = tmp_path / "prose.yaml"
    prose.write_text(
        """
title: "Synthetic scaling study"
author: "Benchmark Team"
date: "2026-07-16"
abstract: "A **compact** abstract."
intro: |
  Intro with `code` and a [relative link](notes.html) and math $E_{xc}[n]$.
comparison:
  intro: "Comparison **context**."
  environments:
    gpu-env: "GPU row *annotation*."
    cpu-env: "CPU row annotation."
  notes:
    - "Matched costs only."
plots:
  numint: "Numint commentary."
  iterations: "Iterations commentary."
  total: "Total commentary."
closing: |
  Final **conclusion**.
""".strip()
        + "\n"
    )
    output = tmp_path / "report"

    generate(output, [collected_dir], prose_path=prose)

    index = (output / "index.html").read_text()
    for expected in [
        "Synthetic scaling study",
        "Benchmark Team",
        "<strong>compact</strong>",
        "Intro with <code>code</code>",
        '<span class="tex" data-tex="E_{xc}[n]">E_{xc}[n]</span>',
        "Comparison <strong>context</strong>",
        "GPU row <em>annotation</em>",
        "CPU row annotation",
        "Matched costs only",
        "Numint commentary",
        "Iterations commentary",
        "Total commentary",
        "Final <strong>conclusion</strong>",
    ]:
        assert expected in index
