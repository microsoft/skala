# SPDX-License-Identifier: MIT

"""Tests for the orchestrator's task selection and resume bookkeeping."""

import dataclasses
import subprocess
from pathlib import Path

import pytest

from skala.benchmark.dataset import BenchmarkMolecule
from skala.benchmark.models import Molecule
from skala.benchmark.orchestrator import (
    SweepContext,
    SweepRequest,
    _error_status,
    _execute,
    _filter_tasks_by_max_orbitals,
    _load_checkpoint_state,
    _run_task,
    _sweep_fingerprint,
    _with_process_timings,
    build_tasks,
    parse_duration,
    select_for_shard,
)
from skala.benchmark.protocol import (
    BenchmarkProtocol,
    Device,
    FunctionalKind,
    FunctionalSpec,
)
from skala.benchmark.runner import RunConfig
from skala.benchmark.schema.measurements import make_row, write_shard


def _mol(name: str, num_electrons: int = 1) -> BenchmarkMolecule:
    return BenchmarkMolecule(
        mol_hash=name,
        name=name,
        molecule=Molecule(
            atomic_numbers=[1],
            geometry_bohr=[[0.0, 0.0, 0.0]],
            charge=0,
            multiplicity=2,
        ),
        num_atoms=1,
        num_electrons=num_electrons,
    )


def _protocol() -> BenchmarkProtocol:
    return BenchmarkProtocol(
        bases=("b1", "b2"),
        functionals=(
            FunctionalSpec("f1", FunctionalKind.NATIVE),
            FunctionalSpec("f2", FunctionalKind.NATIVE),
        ),
    )


def _request(tmp_path: Path) -> SweepRequest:
    return SweepRequest(
        output_dir=tmp_path,
        env_id="test",
        env_label="Test",
        device=Device.CPU,
        protocol=_protocol(),
    )


def _checkpoint_row(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "env_id": "test",
        "functional_kind": "native",
        "ansatz": "UKS",
        "density_fit": True,
        "auxbasis": "def2-universal-jkfit",
        "grid_level": 3,
        "conv_tol": 5e-6,
        "device": "cpu",
        "sweep_fingerprint": "fingerprint",
        "status": "oom",
    }
    fields.update(overrides)
    return dict(make_row(**fields))


def test_build_tasks_cardinality_and_order() -> None:
    mols = [_mol("large", 10), _mol("small", 2)]
    functionals = [
        FunctionalSpec(name, FunctionalKind.NATIVE) for name in ("f1", "f2", "f3")
    ]
    tasks = build_tasks(mols, ["b1", "b2"], functionals)
    assert len(tasks) == 2 * 3 * 2
    assert [task[2].name for task in tasks[:6]] == ["small"] * 6
    assert (tasks[0][0], tasks[0][1].name) == ("b1", "f1")
    assert (tasks[1][0], tasks[1][1].name) == ("b1", "f2")
    assert (tasks[3][0], tasks[3][1].name) == ("b2", "f1")


def test_select_for_shard_partitions_disjointly_and_completely() -> None:
    mols = [
        _mol(name, electrons)
        for name, electrons in zip("EDCBA", range(5, 0, -1), strict=True)
    ]
    tasks = build_tasks(
        mols, ["b1", "b2"], [FunctionalSpec("f1", FunctionalKind.NATIVE)]
    )
    num_shards = 3
    shards = [select_for_shard(tasks, s, num_shards) for s in range(num_shards)]

    # Every task appears exactly once across shards.
    recombined = [t for shard in shards for t in shard]
    assert sorted(map(id, recombined)) == sorted(map(id, tasks))
    # Balanced: shard sizes differ by at most one.
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1
    assert all(
        [task[2].num_electrons for task in shard]
        == sorted(task[2].num_electrons for task in shard)
        for shard in shards
    )


def test_single_shard_takes_everything() -> None:
    tasks = build_tasks(
        [_mol("A")],
        ["b1"],
        [
            FunctionalSpec("f1", FunctionalKind.NATIVE),
            FunctionalSpec("f2", FunctionalKind.NATIVE),
        ],
    )
    assert select_for_shard(tasks, 0, 1) == tasks


def test_max_orbitals_filters_basis_specific_tasks_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def count_orbitals(_molecule: object, basis: str) -> int:
        calls.append(basis)
        return {"small": 10, "large": 20}[basis]

    monkeypatch.setattr(
        "skala.benchmark.orchestrator.count_atomic_orbitals", count_orbitals
    )
    tasks = build_tasks(
        [_mol("A")],
        ["small", "large"],
        [
            FunctionalSpec("f1", FunctionalKind.NATIVE),
            FunctionalSpec("f2", FunctionalKind.NATIVE),
        ],
    )

    kept = _filter_tasks_by_max_orbitals(tasks, 10)

    assert [(basis, functional.name) for basis, functional, _ in kept] == [
        ("small", "f1"),
        ("small", "f2"),
    ]
    assert calls == ["small", "large"]


def test_run_task_uses_context_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = BenchmarkProtocol(
        bases=("b1",),
        functionals=(FunctionalSpec("f1", FunctionalKind.NATIVE),),
        ansatz="RKS",
        density_fit=False,
        auxbasis="custom-aux",
        grid_level=7,
        conv_tol=1e-8,
    )
    ctx = SweepContext(
        output_dir=tmp_path,
        workdir=tmp_path,
        python="python",
        shard_index=0,
        num_shards=1,
        time_limit_seconds=None,
        env_id="test",
        commit=None,
        device=Device.CPU,
        protocol=protocol,
        sweep_fingerprint="fingerprint",
    )
    captured: list[RunConfig] = []

    def execute(_ctx: SweepContext, config: RunConfig) -> dict[str, object]:
        captured.append(config)
        return {"error": "expected"}

    monkeypatch.setattr("skala.benchmark.orchestrator._execute", execute)

    _run_task(ctx, ("b1", protocol.functionals[0], _mol("A")))

    config = captured[0]
    assert config.ansatz == "RKS"
    assert not config.density_fit
    assert config.auxbasis == "custom-aux"
    assert config.grid_level == 7
    assert config.conv_tol == 1e-8


def test_existing_task_keys_reads_all_rows_in_shard(tmp_path: Path) -> None:
    measurements = tmp_path / "measurements"
    rows = [
        _checkpoint_row(basis="b1", functional="f1", mol_hash="A", shard_index=2),
        _checkpoint_row(basis="b2", functional="f2", mol_hash="B", shard_index=2),
    ]
    write_shard(rows, measurements, 2)
    write_shard(
        [_checkpoint_row(basis="other", functional="f1", mol_hash="C", shard_index=1)],
        measurements,
        1,
    )

    state = _load_checkpoint_state(measurements, 2, _request(tmp_path), "fingerprint")
    assert state.completed == {
        ("b1", "f1", "A"),
        ("b2", "f2", "B"),
    }


def test_existing_task_keys_reports_interrupted_checkpoint(tmp_path: Path) -> None:
    measurements = tmp_path / "measurements"
    corrupt = measurements / "shard_index=0" / "interrupted.parquet"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("incomplete", encoding="utf-8")

    with pytest.raises(ValueError, match="unreadable benchmark checkpoint"):
        _load_checkpoint_state(measurements, 0, _request(tmp_path), "fingerprint")
    assert corrupt.exists()


def test_existing_task_keys_treats_every_recorded_row_as_complete(
    tmp_path: Path,
) -> None:
    """Timings are written with the row itself, so there is no partial state."""
    measurements = tmp_path / "measurements"
    write_shard(
        [_checkpoint_row(basis="b1", functional="f1", mol_hash="A", shard_index=0)],
        measurements,
        0,
    )
    write_shard(
        [
            _checkpoint_row(
                basis="b2",
                functional="f2",
                mol_hash="B",
                shard_index=0,
                status="oom",
            )
        ],
        measurements,
        0,
    )

    state = _load_checkpoint_state(measurements, 0, _request(tmp_path), "fingerprint")

    assert state.completed == {("b1", "f1", "A"), ("b2", "f2", "B")}


def test_existing_task_keys_rejects_incompatible_sweep(tmp_path: Path) -> None:
    measurements = tmp_path / "measurements"
    write_shard(
        [
            _checkpoint_row(
                basis="b1",
                functional="f1",
                mol_hash="A",
                shard_index=0,
                device="gpu",
            )
        ],
        measurements,
        0,
    )

    with pytest.raises(ValueError, match="incompatible with this sweep"):
        _load_checkpoint_state(measurements, 0, _request(tmp_path), "fingerprint")


def test_sweep_fingerprint_pins_shard_plan(tmp_path: Path) -> None:
    tasks = build_tasks(
        [_mol("A"), _mol("B")],
        ["b1"],
        [FunctionalSpec("f1", FunctionalKind.NATIVE)],
    )
    request = _request(tmp_path)
    different_shards = dataclasses.replace(request, num_shards=2)
    different_orbital_limit = dataclasses.replace(request, max_orbitals=100)

    assert _sweep_fingerprint(request, tasks) != _sweep_fingerprint(
        different_shards, tasks
    )
    assert _sweep_fingerprint(request, tasks) != _sweep_fingerprint(
        different_orbital_limit, tasks
    )


def test_execute_reports_a_runner_that_produced_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = SweepContext(
        output_dir=tmp_path / "output",
        workdir=tmp_path / "work",
        python="python",
        shard_index=3,
        num_shards=4,
        time_limit_seconds=None,
        env_id="test",
        commit=None,
        device=Device.CPU,
        protocol=_protocol(),
        sweep_fingerprint="fingerprint",
    )
    ctx.workdir.mkdir()
    config = RunConfig(
        molecule=Molecule(
            atomic_numbers=[1],
            geometry_bohr=[[0.0, 0.0, 0.0]],
            charge=0,
            multiplicity=2,
        ),
        basis="b1",
        functional=FunctionalSpec("f1", FunctionalKind.NATIVE),
        device=Device.CPU,
    )

    class Process:
        returncode = 1

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            assert timeout is None
            (ctx.workdir / "result.json").write_text('{"wall_time_ms": 1.0}')
            return "", "No child process"

    monkeypatch.setattr(
        "skala.benchmark.orchestrator.subprocess.Popen", lambda *a, **kw: Process()
    )

    outcome = _execute(ctx, config)

    # The runner did write a result, so its payload wins over the non-zero exit.
    assert outcome["wall_time_ms"] == 1.0
    # The orchestrator always records how long the subprocess took.
    assert outcome["process_wall_ms"] > 0.0
    # Without a worker_ms there is nothing to take a startup residual from.
    assert "startup_ms" not in outcome


def test_execute_timeout_keeps_a_persisted_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = SweepContext(
        output_dir=tmp_path / "output",
        workdir=tmp_path / "work",
        python="python",
        shard_index=3,
        num_shards=4,
        time_limit_seconds=1.0,
        env_id="test",
        commit=None,
        device=Device.CPU,
        protocol=_protocol(),
        sweep_fingerprint="fingerprint",
    )
    ctx.workdir.mkdir()
    config = RunConfig(
        molecule=Molecule(
            atomic_numbers=[1],
            geometry_bohr=[[0.0, 0.0, 0.0]],
            charge=0,
            multiplicity=2,
        ),
        basis="b1",
        functional=FunctionalSpec("f1", FunctionalKind.NATIVE),
        device=Device.CPU,
    )

    class Process:
        pid = 123
        returncode = -9
        calls = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            self.calls += 1
            if self.calls == 1:
                (ctx.workdir / "result.json").write_text(
                    '{"wall_time_ms": 1.0}', encoding="utf-8"
                )
                raise subprocess.TimeoutExpired("runner", timeout)
            return "", "probe killed after timeout"

    monkeypatch.setattr(
        "skala.benchmark.orchestrator.subprocess.Popen", lambda *a, **kw: Process()
    )
    monkeypatch.setattr(
        "skala.benchmark.orchestrator._terminate_process_group", lambda _proc: None
    )

    outcome = _execute(ctx, config)

    assert outcome["wall_time_ms"] == 1.0
    assert outcome["process_wall_ms"] > 0.0


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("30", 30.0), ("15s", 15.0), ("2m", 120.0), ("4h", 14400.0)],
)
def test_parse_duration(value: str, seconds: float) -> None:
    assert parse_duration(value) == seconds


def test_timeout_has_distinct_status() -> None:
    assert _error_status("computation timed out after 14400 seconds") == "timeout"


def test_startup_is_the_residual_the_worker_could_not_time() -> None:
    """The phases must close against the subprocess wall time."""
    outcome = _with_process_timings({"worker_ms": 400.0}, process_wall_ms=1000.0)

    assert outcome["startup_ms"] == pytest.approx(600.0)
    assert outcome["startup_ms"] + outcome["worker_ms"] == pytest.approx(
        outcome["process_wall_ms"]
    )


def test_a_failed_run_still_records_the_subprocess_duration() -> None:
    outcome = _with_process_timings({"error": "boom"}, process_wall_ms=1000.0)

    assert outcome["process_wall_ms"] == pytest.approx(1000.0)
    assert "startup_ms" not in outcome


def test_startup_splits_into_boot_and_teardown() -> None:
    """The worker measures its own boot; the parent attributes the rest."""
    outcome = _with_process_timings(
        {"worker_ms": 400.0, "boot_ms": 250.0}, process_wall_ms=1000.0
    )

    assert outcome["startup_ms"] == pytest.approx(600.0)
    assert outcome["teardown_ms"] == pytest.approx(350.0)
    assert outcome["boot_ms"] + outcome["worker_ms"] + outcome["teardown_ms"] == (
        pytest.approx(outcome["process_wall_ms"])
    )


def test_startup_stays_lumped_without_a_boot_measurement() -> None:
    """A worker that could not compare clocks reports no teardown split."""
    outcome = _with_process_timings({"worker_ms": 400.0}, process_wall_ms=1000.0)

    assert outcome["startup_ms"] == pytest.approx(600.0)
    assert "teardown_ms" not in outcome
