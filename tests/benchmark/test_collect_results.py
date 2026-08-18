# SPDX-License-Identifier: MIT

"""Tests for benchmark result collection."""

import datetime
import json
from pathlib import Path

import pytest

from skala.benchmark.collect_results import collect_results
from skala.benchmark.schema.environment import Environment
from skala.benchmark.schema.measurements import make_row, write_shard


def _environment(env_id: str = "cpu") -> Environment:
    return Environment(
        env_id=env_id,
        label=f"Environment {env_id}",
        hostname="host",
        timestamp="2026-07-13T00:00:00+00:00",
        cpu_model="CPU",
        num_sockets=1,
        cores_physical=8,
        cores_logical=16,
        numa_nodes=1,
        numa_topology="node 0 cpus: 0-15",
        mem_total_gb=64.0,
        env_vars={"OMP_NUM_THREADS": "8"},
        python_version="3.11",
        pyscf_version="2.12",
        torch_version="2.8",
        numpy_version="2.3",
        blas_impl="OpenBLAS",
        versions={"skala": "2026.6", "pyarrow": "21.0.0"},
    )


def _cycles() -> list[dict[str, object]]:
    """Cycle 0 is warmup-dominated; the rest are the steady state."""
    return [
        {
            "cycle": index,
            "wall_ms": wall,
            "numint_ms": numint,
            "forward_ms": forward,
            "numint_calls": 1,
            "forward_calls": 1,
        }
        for index, (wall, numint, forward) in enumerate(
            [(40.0, 32.0, 16.0), (10.0, 8.0, 3.0), (10.0, 8.0, 3.0)]
        )
    ]


def _measurement(env_id: str, **overrides: object) -> dict[str, object]:
    row = dict(
        id="run-id",
        env_id=env_id,
        shard_index=0,
        num_shards=1,
        timestamp=datetime.datetime.now(datetime.UTC),
        basis="def2-svp",
        functional="skala-1.1",
        mol_name="H2O",
        mol_hash="hash",
        charge=0,
        multiplicity=1,
        ansatz="RKS",
        density_fit=True,
        grid_level=3,
        conv_tol=5e-6,
        num_atomic_orbitals=24,
        grid_size=1000,
        num_scf_iterations=3,
        is_converged=True,
        kernel_time_ms=70.0,
        setup_ms=5.0,
        finalize_ms=5.0,
        cycles=_cycles(),
        status="ok",
    )
    row.update(overrides)
    return row


def test_collect_results_writes_linked_tables(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _environment().save(dataset / "environments")
    write_shard(
        [make_row(**_measurement("cpu"))],
        dataset / "measurements",
        0,
    )

    environments_path, measurements_path, fits_path = collect_results(
        dataset, tmp_path / "collected"
    )

    environments = json.loads(environments_path.read_text())
    measurements = json.loads(measurements_path.read_text())
    assert fits_path.is_file()
    assert environments[0]["env_id"] == "cpu"
    assert environments[0]["env_vars"] == {"OMP_NUM_THREADS": "8"}
    assert measurements[0]["env_id"] == environments[0]["env_id"]
    assert len(measurements[0]["cycles"]) == 3
    assert {path.name for path in measurements_path.parent.iterdir()} == {
        "environments.json",
        "measurements.json",
        "fits.json",
    }


def test_collect_results_preserves_nested_cycles(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _environment().save(dataset / "environments")
    write_shard([make_row(**_measurement("cpu"))], dataset / "measurements", 0)

    _, measurements_path, _ = collect_results(dataset, tmp_path / "collected")

    cycles = json.loads(measurements_path.read_text())[0]["cycles"]
    assert [row["cycle"] for row in cycles] == [0, 1, 2]
    assert cycles[1]["forward_ms"] == pytest.approx(3.0)


def test_collect_results_preserves_unusable_runs_for_diagnostics(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _environment().save(dataset / "environments")
    write_shard(
        [make_row(**_measurement("cpu", is_converged=False))],
        dataset / "measurements",
        0,
    )

    _, measurements_path, _ = collect_results(dataset, tmp_path / "collected")

    rows = json.loads(measurements_path.read_text())
    assert len(rows) == 1
    assert rows[0]["is_converged"] is False


def test_collect_results_rejects_unknown_environment(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _environment().save(dataset / "environments")
    write_shard(
        [make_row(**_measurement("gpu"))],
        dataset / "measurements",
        0,
    )

    with pytest.raises(ValueError, match="unknown environments"):
        collect_results(dataset, tmp_path / "collected")


def test_collect_results_is_deterministic(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _environment().save(dataset / "environments")
    write_shard([make_row(**_measurement("cpu"))], dataset / "measurements", 0)

    first = collect_results(dataset, tmp_path / "first")
    second = collect_results(dataset, tmp_path / "second")

    assert [path.read_bytes() for path in first] == [
        path.read_bytes() for path in second
    ]
