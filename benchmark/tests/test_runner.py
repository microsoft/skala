# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch
from skala_benchmark import runner
from skala_benchmark.models import Molecule
from skala_benchmark.protocol import Device, FunctionalKind, FunctionalSpec
from skala_benchmark.runner import RunConfig, RunResult, run_worker


def test_run_config_json_round_trip(tmp_path: Path) -> None:
    config = RunConfig(
        molecule=Molecule(
            atomic_numbers=[1, 8],
            geometry_bohr=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]],
            charge=-1,
            multiplicity=2,
        ),
        basis="def2-svp",
        functional=FunctionalSpec("skala-1.1", FunctionalKind.SKALA),
        device=Device.GPU,
        ansatz="RKS",
        density_fit=False,
        auxbasis="def2-universal-jkfit",
        grid_level=4,
        conv_tol=1e-7,
        conv_tol_grad=1e-5,
    )
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

    assert RunConfig.from_json(path) == config


def test_num_profiled_runs_handles_private_torch_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch._C, "_jit_get_num_profiled_runs", lambda: 3)
    assert runner._num_profiled_runs() == 3

    monkeypatch.setattr(torch._C, "_jit_get_num_profiled_runs", None)
    assert runner._num_profiled_runs(default=2) == 2


def test_real_worker_produces_an_accounted_serializable_measurement() -> None:
    outcome = run_worker(
        RunConfig(
            molecule=Molecule(
                atomic_numbers=[1, 1],
                geometry_bohr=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
            ),
            basis="sto-3g",
            functional=FunctionalSpec("pbe", FunctionalKind.NATIVE),
            device=Device.CPU,
            density_fit=False,
        )
    )

    assert isinstance(outcome, RunResult)
    assert outcome.is_converged
    assert math.isfinite(outcome.total_energy)
    assert outcome.num_scf_iterations == len(outcome.cycles)
    assert outcome.cycles
    assert all(
        cycle["xc_eval_ms"] <= cycle["numint_ms"] <= cycle["wall_ms"]
        for cycle in outcome.cycles
    )
    assert outcome.load_ms + outcome.warmup_ms + outcome.build_ms + (
        outcome.kernel_time_ms
    ) == pytest.approx(outcome.worker_ms, abs=1e-6)
    assert outcome.setup_ms + outcome.finalize_ms + sum(
        cycle["wall_ms"] for cycle in outcome.cycles
    ) == pytest.approx(outcome.kernel_time_ms, rel=0.02)
    assert json.loads(json.dumps(dataclasses.asdict(outcome)))["cycles"]
