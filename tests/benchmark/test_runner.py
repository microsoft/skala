# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from skala.benchmark.models import Molecule
from skala.benchmark.protocol import Device, FunctionalKind, FunctionalSpec
from skala.benchmark.runner import RunConfig, RunResult, run_worker


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
