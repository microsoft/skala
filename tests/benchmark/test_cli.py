# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from skala.benchmark.__main__ import main
from skala.benchmark.orchestrator import SweepRequest
from skala.benchmark.protocol import Device


def test_run_routes_a_typed_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[SweepRequest] = []
    monkeypatch.setattr(
        "skala.benchmark.__main__.run_sweep", lambda request: requests.append(request)
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
    monkeypatch.setattr(
        "skala.benchmark.collect_results.collect_results",
        lambda input_dir, output_dir: calls.append((input_dir, output_dir)),
    )

    main(["collect", str(tmp_path)])

    assert calls == [(str(tmp_path), str(tmp_path / "collected"))]


def test_report_routes_reference_and_local_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, list[str], str | None]] = []
    monkeypatch.setattr(
        importlib.import_module("skala.benchmark.report.generate"),
        "generate",
        lambda output, inputs, prose_path=None: calls.append(
            (output, inputs, prose_path)
        ),
    )
    reference = "benchmarks/reference"
    local = str(tmp_path / "benchmark-output" / "collected")

    main(["report", str(tmp_path / "local-report"), reference, local])

    assert calls == [
        (
            str(tmp_path / "local-report"),
            [reference, local],
            None,
        )
    ]
