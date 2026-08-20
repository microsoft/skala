# SPDX-License-Identifier: MIT

"""Run one deterministic local shard of the benchmark task grid.

This module does not submit cluster jobs. An external scheduler can invoke one
command per shard, while this runner handles the work inside each job.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from skala.benchmark.dataset import BenchmarkMolecule, load_benchmark_molecules
from skala.benchmark.node_info import collect_environment
from skala.benchmark.protocol import (
    DEFAULT_PROTOCOL,
    BenchmarkProtocol,
    Device,
    FunctionalSpec,
)
from skala.benchmark.runner import RunConfig, count_atomic_orbitals, validate_device
from skala.benchmark.schema.measurements import Row, make_row, write_shard

#: One point in the sweep: a (basis, functional, molecule) triple.
Task = tuple[str, FunctionalSpec, BenchmarkMolecule]
TaskKey = tuple[str, str, str]


@dataclasses.dataclass(frozen=True)
class SweepRequest:
    """One user-requested shard of the benchmark protocol."""

    output_dir: Path
    env_id: str
    env_label: str
    device: Device
    shard_index: int = 0
    num_shards: int = 1
    max_atoms: int | None = None
    max_orbitals: int | None = None
    molecule_names: tuple[str, ...] | None = None
    time_limit_seconds: float | None = None
    protocol: BenchmarkProtocol = DEFAULT_PROTOCOL
    #: Where the molecule set was fetched to; ``None`` uses the default.
    dataset_dir: Path | None = None


@dataclasses.dataclass(frozen=True)
class SweepContext:
    """Everything held constant across all tasks of one shard.

    Collecting these invariants here lets the per-task helpers keep two/three
    argument signatures instead of threading a dozen values through each call.
    """

    output_dir: Path  # dataset root: environments/, measurements/
    workdir: Path  # scratch for run.json and the worker's result.json
    python: str  # interpreter that runs the worker
    shard_index: int
    num_shards: int
    time_limit_seconds: float | None
    env_id: str  # -> Environment that this shard's rows link to
    commit: str | None  # skala git commit
    device: Device
    protocol: BenchmarkProtocol
    sweep_fingerprint: str


@dataclasses.dataclass(frozen=True)
class CheckpointState:
    """Tasks already recorded for this shard."""

    completed: frozenset[TaskKey]


def main(argv: list[str]) -> None:
    from skala.benchmark.__main__ import main as benchmark_main

    benchmark_main(["run", *argv])


def run_sweep(request: SweepRequest) -> Path:
    """Run one shard and write its environment and measurements."""
    validate_device(request.device)
    output_dir = request.output_dir

    molecules = _filter_molecules(
        load_benchmark_molecules(request.dataset_dir),
        request.max_atoms,
        request.molecule_names,
    )
    all_tasks = build_tasks(
        molecules,
        request.protocol.bases,
        request.protocol.functionals,
    )
    sweep_fingerprint = _sweep_fingerprint(request, all_tasks)
    selected_tasks = select_for_shard(
        all_tasks, request.shard_index, request.num_shards
    )
    tasks = _filter_tasks_by_max_orbitals(selected_tasks, request.max_orbitals)
    skipped_for_size = len(selected_tasks) - len(tasks)
    if skipped_for_size:
        print(
            f"skipped {skipped_for_size} tasks exceeding --max-orbitals "
            f"{request.max_orbitals}",
            file=sys.stderr,
        )
    if not tasks:
        print(
            "warning: no tasks selected for this shard (check molecule filters, "
            "--max-orbitals, and shard settings).",
            file=sys.stderr,
        )

    checkpoints = _load_checkpoint_state(
        output_dir / "measurements",
        request.shard_index,
        request,
        sweep_fingerprint,
    )
    collect_environment(request.env_id, request.env_label).save(
        output_dir / "environments"
    )
    pending_tasks = [
        task for task in tasks if _task_key(task) not in checkpoints.completed
    ]
    skipped_existing = len(tasks) - len(pending_tasks)
    if skipped_existing:
        print(
            f"resuming shard: skipped {skipped_existing} existing results",
            file=sys.stderr,
        )

    with tempfile.TemporaryDirectory() as tmp:
        ctx = SweepContext(
            output_dir=output_dir,
            workdir=Path(tmp),
            python=sys.executable,
            shard_index=request.shard_index,
            num_shards=request.num_shards,
            time_limit_seconds=request.time_limit_seconds,
            env_id=request.env_id,
            commit=_git_commit(),
            device=request.device,
            protocol=request.protocol,
            sweep_fingerprint=sweep_fingerprint,
        )
        path = output_dir / "measurements" / f"shard_index={request.shard_index}"
        written_rows = 0
        for i, task in enumerate(pending_tasks, start=1):
            row = _run_task(ctx, task)
            path = write_shard(
                [row],
                output_dir / "measurements",
                request.shard_index,
                checkpoint_name=_checkpoint_name(_task_key(task)),
            )
            written_rows += 1
            basis, functional, mol = task
            print(
                f"[{i}/{len(pending_tasks)}] {functional.name} {basis} {mol.name}: "
                f"{row['status']}",
                file=sys.stderr,
                flush=True,
            )

    print(
        f"wrote {written_rows} rows; skipped {skipped_existing} existing -> {path}",
        file=sys.stderr,
    )
    return path


def build_tasks(
    molecules: list[BenchmarkMolecule],
    bases: tuple[str, ...] | list[str],
    functionals: tuple[FunctionalSpec, ...] | list[FunctionalSpec],
) -> list[Task]:
    """Full task grid ordered by increasing molecule electron count."""
    return [
        (basis, functional, molecule)
        for molecule in sorted(molecules, key=lambda item: item.num_electrons)
        for basis in bases
        for functional in functionals
    ]


def select_for_shard(
    tasks: list[Task], shard_index: int, num_shards: int
) -> list[Task]:
    """Round-robin slice of ``tasks`` belonging to ``shard_index``."""
    return tasks[shard_index::num_shards]


def _filter_tasks_by_max_orbitals(
    tasks: list[Task],
    max_orbitals: int | None,
) -> list[Task]:
    """Keep tasks whose molecule/basis pair has at most ``max_orbitals`` AOs."""
    if max_orbitals is None:
        return tasks

    counts: dict[tuple[str, str], int] = {}
    kept: list[Task] = []
    for task in tasks:
        basis, _, mol = task
        molecule_basis = (mol.mol_hash, basis)
        if molecule_basis not in counts:
            counts[molecule_basis] = count_atomic_orbitals(mol.molecule, basis)
        if counts[molecule_basis] <= max_orbitals:
            kept.append(task)
    return kept


def _task_key(task: Task) -> TaskKey:
    """Stable identity for one point in the fixed benchmark grid."""
    basis, functional, mol = task
    return basis, functional.name, mol.mol_hash


def _load_checkpoint_state(
    directory: Path,
    shard_index: int,
    request: SweepRequest,
    sweep_fingerprint: str,
) -> CheckpointState:
    """Load the tasks already completed for a compatible shard."""
    partition = directory / f"shard_index={shard_index}"
    completed: set[TaskKey] = set()
    for path in sorted(partition.glob("*.parquet")):
        try:
            rows = pq.read_table(path).to_pylist()
        except Exception as exc:
            raise ValueError(f"unreadable benchmark checkpoint: {path}") from exc
        for row in rows:
            _validate_checkpoint(row, request, sweep_fingerprint, path)
            key = (str(row["basis"]), str(row["functional"]), str(row["mol_hash"]))
            if key in completed:
                raise ValueError(f"duplicate benchmark checkpoint for {key}: {path}")
            completed.add(key)
    return CheckpointState(frozenset(completed))


def _validate_checkpoint(
    row: dict[str, object],
    request: SweepRequest,
    sweep_fingerprint: str,
    path: Path,
) -> None:
    """Reject attempts to resume a shard with different scientific settings."""
    functional_kinds = {
        functional.name: functional.kind.value
        for functional in request.protocol.functionals
    }
    functional = str(row["functional"])
    expected = {
        "env_id": request.env_id,
        "functional_kind": functional_kinds.get(functional),
        "ansatz": request.protocol.ansatz,
        "density_fit": request.protocol.density_fit,
        "auxbasis": request.protocol.auxbasis,
        "grid_level": request.protocol.grid_level,
        "conv_tol": request.protocol.conv_tol,
        "device": request.device.value,
        "sweep_fingerprint": sweep_fingerprint,
    }
    mismatched = [field for field, value in expected.items() if row.get(field) != value]
    if mismatched:
        raise ValueError(
            f"checkpoint {path} is incompatible with this sweep "
            f"({', '.join(mismatched)}); use a new output directory or environment id"
        )


def _checkpoint_name(key: TaskKey) -> str:
    """Return the deterministic checkpoint file name for one task."""
    digest = hashlib.sha256("\0".join(key).encode()).hexdigest()
    return f"{digest}.parquet"


def _sweep_fingerprint(request: SweepRequest, tasks: list[Task]) -> str:
    """Hash the ordered task grid and settings that determine shard membership."""
    payload = {
        "num_shards": request.num_shards,
        "max_orbitals": request.max_orbitals,
        "device": request.device.value,
        "protocol": {
            "bases": request.protocol.bases,
            "functionals": [
                (functional.name, functional.kind.value)
                for functional in request.protocol.functionals
            ],
            "ansatz": request.protocol.ansatz,
            "density_fit": request.protocol.density_fit,
            "auxbasis": request.protocol.auxbasis,
            "grid_level": request.protocol.grid_level,
            "conv_tol": request.protocol.conv_tol,
        },
        "tasks": [
            (basis, functional.name, functional.kind.value, molecule.mol_hash)
            for basis, functional, molecule in tasks
        ],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _filter_molecules(
    molecules: list[BenchmarkMolecule],
    max_atoms: int | None,
    names: tuple[str, ...] | None,
) -> list[BenchmarkMolecule]:
    """Keep molecules up to ``max_atoms`` and/or in the ``names`` allow-list."""
    if max_atoms is not None:
        molecules = [m for m in molecules if m.num_atoms <= max_atoms]
    if names:
        wanted = set(names)
        molecules = [m for m in molecules if m.name in wanted]
    return molecules


def _run_task(ctx: SweepContext, task: Task) -> Row:
    """Run a single task into a measurement row (never raises)."""
    basis, functional, mol = task
    run_id = str(uuid.uuid4())
    config = _config_for(
        mol,
        basis=basis,
        functional=functional,
        device=ctx.device,
        protocol=ctx.protocol,
    )
    outcome: dict[str, object]
    try:
        outcome = _execute(ctx, config)
    except Exception as exc:  # noqa: BLE001 - one bad task must not sink the shard
        outcome = dict[str, object](error=repr(exc))
    return _build_row(ctx, run_id, mol, config, outcome)


def _config_for(
    mol: BenchmarkMolecule,
    *,
    basis: str,
    functional: FunctionalSpec,
    device: Device,
    protocol: BenchmarkProtocol = DEFAULT_PROTOCOL,
) -> RunConfig:
    """The runner configuration for one molecule at one (basis, functional)."""
    return RunConfig(
        molecule=mol.molecule,
        basis=basis,
        functional=functional,
        device=device,
        ansatz=protocol.ansatz,
        density_fit=protocol.density_fit,
        auxbasis=protocol.auxbasis,
        grid_level=protocol.grid_level,
        conv_tol=protocol.conv_tol,
    )


def _execute(ctx: SweepContext, config: RunConfig) -> dict[str, object]:
    """Run one config's DFT in a fresh worker process.

    Each calculation gets its own interpreter so that a crash, a leak, or an
    out-of-memory kill cannot affect later tasks, and so first-use costs are
    identical for every task.

    Returns the worker's JSON payload, or a synthesized ``{"error": ...}`` if it
    produced no result.
    """
    cfg_path = ctx.workdir / "run.json"
    cfg_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    result_path = ctx.workdir / "result.json"
    result_path.unlink(missing_ok=True)

    started = time.perf_counter()
    command = [
        ctx.python, "-m", "skala.benchmark.runner",
        "--config", str(cfg_path), "--result", str(result_path),
        "--launched-at", repr(started),
    ]  # fmt: skip
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _, stderr = proc.communicate(timeout=ctx.time_limit_seconds)
    except subprocess.TimeoutExpired:
        _terminate_process_group(proc)
        proc.communicate()
        outcome = _read_outcome(result_path) or {
            "error": f"computation timed out after {ctx.time_limit_seconds:g} seconds"
        }
    else:
        outcome = _read_outcome(result_path) or {
            "error": (
                f"runner produced no result (exit {proc.returncode}): {stderr[-500:]}"
            )
        }
    return _with_process_timings(outcome, 1e3 * (time.perf_counter() - started))


def _with_process_timings(
    outcome: dict[str, object], process_wall_ms: float
) -> dict[str, object]:
    """Split the subprocess wall time into boot, worker, and teardown.

    The worker cannot time its own start-up, so it compares the parent's launch
    timestamp against the moment its module finished loading; that is ``boot_ms``
    and it covers process creation, interpreter start-up, and imports. What
    remains after boot and the worker's own phases is ``teardown_ms``: writing
    the result, interpreter shutdown, and the parent reaping the process.

    ``startup_ms`` is kept as their sum, which is what the parent can attribute
    without the worker's cooperation.
    """
    timings: dict[str, object] = {"process_wall_ms": process_wall_ms}
    worker_ms = outcome.get("worker_ms")
    if not isinstance(worker_ms, (int, float)):
        return {**outcome, **timings}

    startup_ms = process_wall_ms - float(worker_ms)
    timings["startup_ms"] = startup_ms
    boot_ms = outcome.get("boot_ms")
    if isinstance(boot_ms, (int, float)):
        timings["teardown_ms"] = startup_ms - float(boot_ms)
    return {**outcome, **timings}


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    """Terminate the worker and anything it spawned, escalating after five seconds."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _build_row(
    ctx: SweepContext,
    run_id: str,
    mol: BenchmarkMolecule,
    config: RunConfig,
    outcome: dict[str, object],
) -> Row:
    """Assemble one measurement row from the context, config, and runner outcome."""
    columns = dict(
        id=run_id,
        env_id=ctx.env_id,
        shard_index=ctx.shard_index,
        num_shards=ctx.num_shards,
        timestamp=datetime.now(UTC),
        skala_commit_hash=ctx.commit,
        sweep_fingerprint=ctx.sweep_fingerprint,
        **_config_columns(config, mol),
    )
    if "error" in outcome:
        return make_row(
            **columns,
            status=_error_status(str(outcome["error"])),
            error=str(outcome["error"]),
        )
    return make_row(**{**columns, **outcome, "status": "ok"})


def _error_status(error: str) -> str:
    """Classify a failure as ``oom`` (out of memory) or plain ``error``.

    Out-of-memory events are recorded distinctly so a report can tell "this
    config is too big for the node" apart from genuine bugs.
    """
    text = error.lower()
    if "timed out" in text:
        return "timeout"
    oom_markers = (
        "out of memory",
        "outofmemory",
        "memoryerror",
        "cuda_error_out_of_memory",
        "cublas_status_alloc_failed",
        "cannot allocate memory",
    )
    return "oom" if any(marker in text for marker in oom_markers) else "error"


def parse_duration(value: str) -> float:
    """Parse seconds or a duration suffixed by ``s``, ``m``, or ``h``."""
    units = {"s": 1.0, "m": 60.0, "h": 3600.0}
    normalized = value.strip().lower()
    suffix = normalized[-1:] if normalized[-1:] in units else ""
    number = normalized[:-1] if suffix else normalized
    try:
        seconds = float(number) * units.get(suffix, 1.0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid duration: {value!r}") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("duration must be a positive finite value")
    return seconds


def _read_outcome(path: Path) -> dict[str, object] | None:
    """Read the worker's explicit result file."""
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _config_columns(config: RunConfig, mol: BenchmarkMolecule) -> dict[str, object]:
    """Measurement columns that echo the run configuration."""
    return dict(
        basis=config.basis,
        functional=config.functional.name,
        functional_kind=config.functional.kind.value,
        mol_name=mol.name,
        mol_hash=mol.mol_hash,
        charge=config.molecule.charge,
        multiplicity=config.molecule.multiplicity,
        ansatz=config.ansatz,
        density_fit=config.density_fit,
        auxbasis=config.auxbasis,
        grid_level=config.grid_level,
        conv_tol=config.conv_tol,
        device=config.device.value,
    )


def _git_commit() -> str | None:
    """The skala repo's current commit hash, or None outside a checkout."""
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    )
    return out.stdout.strip() if out.returncode == 0 else None


if __name__ == "__main__":
    main(sys.argv[1:])
