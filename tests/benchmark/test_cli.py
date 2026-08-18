# SPDX-License-Identifier: MIT

import importlib
from pathlib import Path

import pytest

from skala.benchmark.__main__ import main
from skala.benchmark.orchestrator import SweepRequest
from skala.benchmark.protocol import DEFAULT_PROTOCOL, Device


def test_run_requires_explicit_device(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["run", str(tmp_path), "--env-id", "local"])


def test_run_builds_typed_sweep_request(
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


def test_run_restricts_the_protocol_to_selected_bases(
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
            "--device",
            "cpu",
            "--basis",
            "def2-tzvp",
            "--basis",
            "def2-svp",
        ]
    )

    # Order follows the protocol, not the order the flags were given.
    assert requests[0].protocol.bases == ("def2-svp", "def2-tzvp")


def test_run_defaults_to_every_protocol_basis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[SweepRequest] = []
    monkeypatch.setattr(
        "skala.benchmark.__main__.run_sweep", lambda request: requests.append(request)
    )

    main(["run", str(tmp_path), "--env-id", "cpu-local", "--device", "cpu"])

    assert requests[0].protocol is DEFAULT_PROTOCOL
    assert [functional.name for functional in DEFAULT_PROTOCOL.functionals] == [
        "skala-1.1",
        "r2scan",
        "b3lyp5",
        "m06-2x",
    ]


def test_run_rejects_an_unknown_basis(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                str(tmp_path),
                "--env-id",
                "cpu-local",
                "--device",
                "cpu",
                "--basis",
                "sto-3g",
            ]
        )


def test_collect_uses_default_collected_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "skala.benchmark.collect_results.collect_results",
        lambda input_dir, output_dir: calls.append((input_dir, output_dir)),
    )

    main(["collect", str(tmp_path)])

    assert calls == [(str(tmp_path), str(tmp_path / "collected"))]


def test_report_accepts_collected_directories(
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

    main(
        [
            "report",
            str(tmp_path / "report"),
            str(tmp_path / "cpu" / "collected"),
            str(tmp_path / "gpu" / "collected"),
            "--prose",
            "prose.yaml",
        ]
    )

    assert calls == [
        (
            str(tmp_path / "report"),
            [
                str(tmp_path / "cpu" / "collected"),
                str(tmp_path / "gpu" / "collected"),
            ],
            "prose.yaml",
        )
    ]
