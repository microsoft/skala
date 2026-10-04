# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path

import pytest
from skala_benchmark.__main__ import main
from skala_benchmark.orchestrator import SweepRequest
from skala_benchmark.protocol import Device


def test_run_routes_a_typed_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[SweepRequest] = []

    def fake_run_sweep(request: SweepRequest) -> Path:
        requests.append(request)
        return tmp_path

    monkeypatch.setattr(
        "skala_benchmark.__main__.run_sweep",
        fake_run_sweep,
    )

    main(
        [
            "run",
            str(tmp_path),
            "--env-id",
            "cpu-local",
            "--env-label",
            "Local CPU",
            "--device",
            "cpu",
            "--time-limit",
            "2m",
        ]
    )

    request = requests[0]
    assert request.output_dir == tmp_path
    assert request.env_label == "Local CPU"
    assert request.device is Device.CPU
    assert request.time_limit_seconds == 120.0


def test_collect_routes_to_the_default_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_collect_results(
        input_dir: str | Path, output_dir: str | Path
    ) -> tuple[Path, Path, Path]:
        calls.append((str(input_dir), str(output_dir)))
        output_path = Path(output_dir)
        return output_path, output_path, output_path

    monkeypatch.setattr(
        "skala_benchmark.collect_results.collect_results",
        fake_collect_results,
    )

    main(["collect", str(tmp_path)])

    assert calls == [(str(tmp_path), str(tmp_path / "collected"))]


def test_report_routes_dry_and_interpreted_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, list[str], str | None]] = []

    def fake_generate(
        output: str | Path,
        inputs: Sequence[str | Path],
        prose_path: str | Path | None = None,
    ) -> Path:
        calls.append(
            (
                str(output),
                [str(input_path) for input_path in inputs],
                str(prose_path) if prose_path is not None else None,
            )
        )
        return Path(output)

    monkeypatch.setattr(
        importlib.import_module("skala_benchmark.report.generate"),
        "generate",
        fake_generate,
    )
    reference = "benchmark/reference"
    local = str(tmp_path / "benchmark-output" / "collected")

    main(["report", str(tmp_path / "local-report"), reference, local])
    main(
        [
            "report",
            str(tmp_path / "official-report"),
            reference,
            "--prose",
            "benchmark/reference/prose.yaml",
        ]
    )

    assert calls == [
        (
            str(tmp_path / "local-report"),
            [reference, local],
            None,
        ),
        (
            str(tmp_path / "official-report"),
            [reference],
            "benchmark/reference/prose.yaml",
        ),
    ]
