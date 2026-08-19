# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from skala.benchmark.report import generate
from skala.benchmark.report.prose import load_prose

_REFERENCE_DIR = Path(__file__).parents[2] / "benchmarks" / "reference"
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


def test_generate_requires_complete_collected_directory(tmp_path: Path) -> None:
    collected = tmp_path / "collected"
    collected.mkdir()
    (collected / "environments.json").write_text("[]\n")
    (collected / "fits.json").write_text("[]\n")

    with pytest.raises(FileNotFoundError, match="measurements.json"):
        generate(tmp_path / "report", [collected])
