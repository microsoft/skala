# SPDX-License-Identifier: MIT

"""Tests for publishing the benchmark report with Sphinx."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _load_extension() -> Any:
    path = Path(__file__).parents[2] / "website" / "_ext" / "benchmark_report.py"
    spec = importlib.util.spec_from_file_location("benchmark_report_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_html_build_generates_report_and_copies_sources(
    tmp_path: Path, monkeypatch: Any
) -> None:
    module = _load_extension()
    reference = tmp_path / "reference"
    reference.mkdir()
    for filename in module._SOURCE_FILES:
        (reference / filename).write_text(filename, encoding="utf-8")

    calls: list[tuple[Path, list[Path], Path]] = []

    def generate(output: Path, inputs: list[Path], *, prose_path: Path) -> None:
        calls.append((output, inputs, prose_path))
        output.mkdir(parents=True)
        (output / "index.html").write_text("report", encoding="utf-8")

    monkeypatch.setattr(module, "_REFERENCE_DIR", reference)
    monkeypatch.setattr(module, "generate", generate)
    app = SimpleNamespace(
        outdir=str(tmp_path / "html"),
        builder=SimpleNamespace(format="html"),
    )

    module._build_report(app, None)

    output = Path(app.outdir) / "benchmarks"
    assert calls == [(output, [reference], reference / "prose.yaml")]
    assert (output / "index.html").read_text() == "report"
    assert {path.name for path in (output / "source").iterdir()} == set(
        module._SOURCE_FILES
    )


def test_non_html_build_skips_report(tmp_path: Path, monkeypatch: Any) -> None:
    module = _load_extension()
    monkeypatch.setattr(
        module,
        "generate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("called")),
    )
    app = SimpleNamespace(
        outdir=str(tmp_path / "linkcheck"),
        builder=SimpleNamespace(format=""),
    )

    module._build_report(app, None)
