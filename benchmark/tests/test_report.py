# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from skala_benchmark.report import generate
from skala_benchmark.report.data import JsonValue
from skala_benchmark.report.prose import load_prose

_REFERENCE_DIR = Path(__file__).parents[1] / "reference"
_EXPECTED_FILES = {
    "data.json",
    "d3.min.js",
    "index.html",
    "katex.min.js",
    "report-base.css",
    "report.css",
    "report.js",
}


def test_reference_report_is_reproducible(tmp_path: Path) -> None:
    outputs = (tmp_path / "first", tmp_path / "second")
    for output in outputs:
        index = generate(
            output,
            [_REFERENCE_DIR],
            prose_path=_REFERENCE_DIR / "prose.yaml",
        )
        assert index == output / "index.html"

    generated_files = {
        path.relative_to(outputs[0]).as_posix()
        for path in outputs[0].rglob("*")
        if path.is_file()
    }
    assert generated_files == _EXPECTED_FILES
    assert generated_files == {
        path.relative_to(outputs[1]).as_posix()
        for path in outputs[1].rglob("*")
        if path.is_file()
    }
    for filename in generated_files:
        assert (outputs[0] / filename).read_bytes() == (
            outputs[1] / filename
        ).read_bytes()

    data = json.loads((outputs[0] / "data.json").read_text())
    index_html = (outputs[0] / "index.html").read_text()
    fits_match = re.search(
        r'<script id="report-fits" type="application/json">(.*?)</script>',
        index_html,
    )
    assert fits_match is not None
    fits = json.loads(fits_match.group(1))
    assert data["points"]
    assert data["meta"]["initial_selection"]["environment"].startswith("a100-")
    assert data["meta"]["initial_selection"]["basis"] == "def2-tzvp"
    assert data["meta"]["environments"][0]["env_id"].startswith("a100-")
    assert data["meta"]["bases"] == ["def2-svp", "def2-tzvp", "def2-qzvp"]
    assert fits
    assert "Skala computational cost benchmark report" in index_html
    assert "machine cold start" in index_html
    assert "Compare your own timings with these results" in index_html
    assert "https://microsoft.github.io/skala/benchmarks.html" in index_html


def test_default_prose_is_neutral() -> None:
    prose = json.dumps(load_prose(None))

    assert "No study-specific interpretation was supplied" in prose
    assert "machine cold start" not in prose
    assert "--prose" not in prose


@pytest.mark.parametrize("closing_tag", ["</script>", "</ScRiPt>"])
def test_report_json_cannot_close_script_elements(
    tmp_path: Path, closing_tag: str
) -> None:
    """Hostile strings stay JSON data in both embedded report blocks."""
    payload = closing_tag + "<script>alert(1)</script>"
    collected = tmp_path / "collected"
    collected.mkdir()
    measurement: dict[str, JsonValue] = {
        "env_id": payload,
        "basis": "def2-tzvp",
        "functional": "skala-1.1",
        "mol_name": payload,
        "mol_hash": "test",
        "ansatz": "rks",
        "num_atoms": 2,
        "num_electrons": 2,
        "num_atomic_orbitals": 10,
        "grid_size": 100,
        "is_converged": True,
        "num_scf_iterations": 2,
        "total_energy": -1.0,
        "kernel_time_ms": 10.0,
        "setup_ms": 1.0,
        "finalize_ms": 1.0,
        "cycles": [],
        "status": "ok",
    }
    fit: dict[str, JsonValue] = {
        "env_id": payload,
        "basis": "def2-tzvp",
        "functional": "skala-1.1",
        "metric": "total",
        "x_axis": "num_aos",
        "segment_index": 0,
        "x_start": 1,
        "x_end": 10,
        "slope": 1.0,
        "intercept": 0.0,
        "continuous": True,
        "cv_score": 0.0,
        "n_points": 2,
        "breakpoints": [],
    }
    for filename, records in (
        ("measurements.json", [measurement]),
        ("environments.json", [{"env_id": payload}]),
        ("fits.json", [fit]),
    ):
        (collected / filename).write_text(json.dumps(records))
    with pytest.warns(UserWarning, match="Precomputed fits are missing"):
        index = generate(tmp_path / "report", [collected])
    html = index.read_text()
    assert payload not in html
    for block in ("report-data", "report-fits"):
        match = re.search(
            rf'<script id="{block}" type="application/json">(.*?)</script>', html
        )
        assert match is not None
        embedded = json.loads(match.group(1))
        if block == "report-data":
            assert embedded["points"][0]["molecule"] == payload
        else:
            assert embedded[0]["env_id"] == payload


def test_generate_requires_complete_collected_directory(tmp_path: Path) -> None:
    collected = tmp_path / "collected"
    collected.mkdir()
    (collected / "environments.json").write_text("[]\n")
    (collected / "fits.json").write_text("[]\n")

    with pytest.raises(FileNotFoundError, match="measurements.json"):
        generate(tmp_path / "report", [collected])
