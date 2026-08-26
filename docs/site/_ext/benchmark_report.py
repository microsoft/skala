# SPDX-License-Identifier: MIT

"""Generate the standalone benchmark report alongside the Sphinx HTML site."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from skala_benchmark.report import generate

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE_DIR = _REPOSITORY_ROOT / "benchmark" / "reference"
_SOURCE_FILES = (
    "environments.json",
    "measurements.json",
    "fits.json",
)


def _build_report(app: Any, exception: Exception | None) -> None:
    if exception is not None or app.builder.format != "html":
        return

    output = Path(app.outdir) / "benchmarks"
    generate(output, [_REFERENCE_DIR], prose_path=_REFERENCE_DIR / "prose.yaml")

    source_output = output / "source"
    source_output.mkdir(parents=True, exist_ok=True)
    for filename in _SOURCE_FILES:
        shutil.copyfile(_REFERENCE_DIR / filename, source_output / filename)


def setup(app: Any) -> dict[str, object]:
    app.connect("build-finished", _build_report)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
