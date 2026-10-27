# SPDX-License-Identifier: MIT

"""Tests for publishing the benchmark report with Sphinx."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _load_extension(name: str = "benchmark_report") -> Any:
    path = Path(__file__).parents[2] / "website" / "_ext" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_test", path)
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


@pytest.mark.parametrize("kind", ["tree", "blob"])
def test_source_links_check_ci_revision(kind: str, monkeypatch: Any) -> None:
    module = _load_extension("source_links")
    monkeypatch.setenv("GITHUB_SHA", "7b589dc5c1f5dfbab133fbfd77fd5bb8eff27e07")
    suffix = "SkalaXC/README.md#building"
    uri = f"https://github.com/microsoft/skala/{kind}/main/{suffix}"

    assert module._check_source_uri(None, uri) == (
        f"https://github.com/microsoft/skala/{kind}/"
        f"7b589dc5c1f5dfbab133fbfd77fd5bb8eff27e07/{suffix}"
    )


@pytest.mark.parametrize(
    "uri",
    [
        "https://github.com/microsoft/skala/tree/other/SkalaXC",
        "https://github.com/other/skala/tree/main/SkalaXC",
        "https://microsoft.github.io/skala/",
    ],
)
def test_source_links_leave_other_urls_unchanged(uri: str, monkeypatch: Any) -> None:
    module = _load_extension("source_links")
    monkeypatch.setenv("GITHUB_SHA", "7b589dc5c1f5dfbab133fbfd77fd5bb8eff27e07")

    assert module._check_source_uri(None, uri) is None


def test_source_links_leave_local_checks_unchanged(monkeypatch: Any) -> None:
    module = _load_extension("source_links")
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    assert (
        module._check_source_uri(
            None, "https://github.com/microsoft/skala/tree/main/SkalaXC"
        )
        is None
    )


def test_non_html_build_skips_report(tmp_path: Path, monkeypatch: Any) -> None:
    module = _load_extension()

    def fail_generate(*args: object, **kwargs: object) -> None:
        raise AssertionError("called")

    monkeypatch.setattr(
        module,
        "generate",
        fail_generate,
    )
    app = SimpleNamespace(
        outdir=str(tmp_path / "linkcheck"),
        builder=SimpleNamespace(format=""),
    )

    module._build_report(app, None)
