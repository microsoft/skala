# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skala.benchmark.report import generate
from skala.benchmark.report.prose import load_prose

_REFERENCE_DIR = Path(__file__).parents[2] / "benchmarks" / "reference"
_EXPECTED_FILES = {
    "data.json",
    "d3.min.js",
    "fits.json",
    "index.html",
    "katex.min.js",
    "metadata.yaml",
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
    fits = json.loads((outputs[0] / "fits.json").read_text())
    index = (outputs[0] / "index.html").read_text()
    assert data["points"]
    assert fits
    assert "Skala performance and scaling benchmark" in index
    assert "On the A100 it is about 6.8 s" in index
    assert "Compare your own timings with these results" in index
    assert "https://microsoft.github.io/skala/benchmarks.html" in index


def test_default_prose_is_neutral() -> None:
    prose = json.dumps(load_prose(None))

    assert "No study-specific interpretation was supplied" in prose
    assert "No study-specific conclusions were supplied" in prose
    assert "On the A100 it is about 6.8 s" not in prose
    assert "--prose" not in prose


def test_generate_requires_complete_collected_directory(tmp_path: Path) -> None:
    collected = tmp_path / "collected"
    collected.mkdir()
    (collected / "environments.json").write_text("[]\n")
    (collected / "fits.json").write_text("[]\n")

    with pytest.raises(FileNotFoundError, match="measurements.json"):
        generate(tmp_path / "report", [collected])
