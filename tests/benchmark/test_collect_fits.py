# SPDX-License-Identifier: MIT

"""Tests for scaling-fit generation during benchmark result collection."""

import datetime
import json
import math
from pathlib import Path

from skala.benchmark.collect_results import MIN_FIT_POINTS, collect_results
from skala.benchmark.schema.environment import Environment
from skala.benchmark.schema.measurements import make_row, write_shard

EXPECTED_COMBINATIONS = {
    ("xc_eval", "num_aos"),
    ("xc_eval", "grid_size"),
    ("numint", "num_aos"),
    ("numint", "grid_size"),
    ("cycle", "num_aos"),
    ("cycle", "grid_size"),
    ("iterations", "num_aos"),
    ("total", "num_aos"),
    ("setup", "num_aos"),
}


def _environment() -> Environment:
    return Environment(
        env_id="cpu",
        label="CPU",
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


def _cycles(size: int) -> list[dict[str, object]]:
    """Steady-state cost grows with size; cycle 0 additionally pays warmup."""
    steady = (10.0 * size, 6.0 * size, 2.0 * size)
    return [
        {
            "cycle": index,
            "wall_ms": wall,
            "numint_ms": numint,
            "xc_eval_ms": forward,
            "forward_ms": 0.0,
            "backward_ms": 0.0,
            "numint_calls": 1,
            "xc_eval_calls": 1,
        }
        for index, (wall, numint, forward) in enumerate(
            [tuple(4.0 * value for value in steady), steady, steady, steady]
        )
    ]


def _measurement(
    *,
    index: int,
    basis: str,
) -> dict[str, object]:
    size = index + 1
    return dict(
        id=f"run-{basis}-{index}",
        env_id="cpu",
        shard_index=0,
        num_shards=1,
        timestamp=datetime.datetime.now(datetime.UTC),
        basis=basis,
        functional="skala-1.1",
        mol_name=f"mol-{index}",
        mol_hash=f"{basis}-hash-{index}",
        charge=0,
        multiplicity=1,
        ansatz="RKS",
        density_fit=True,
        grid_level=3,
        conv_tol=5e-6,
        num_atomic_orbitals=10 * size,
        grid_size=1000 * size,
        is_converged=True,
        num_scf_iterations=5 + index,
        kernel_time_ms=100.0 * size**2,
        setup_ms=3.0 * size,
        finalize_ms=1.0 * size,
        cycles=_cycles(size),
        status="ok",
    )


def _write_dataset(
    dataset: Path,
    *,
    populated_points: int,
    sparse_points: int = 0,
) -> None:
    _environment().save(dataset / "environments")
    rows = [
        make_row(**_measurement(index=index, basis="def2-svp"))
        for index in range(populated_points)
    ]
    rows.extend(
        make_row(**_measurement(index=index, basis="def2-tzvp"))
        for index in range(sparse_points)
    )
    write_shard(rows, dataset / "measurements", 0)


def test_collect_results_writes_expected_scaling_fits(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(
        dataset,
        populated_points=MIN_FIT_POINTS + 1,
        sparse_points=MIN_FIT_POINTS - 1,
    )

    *_, fits_path = collect_results(dataset, tmp_path / "collected")

    assert fits_path == tmp_path / "collected" / "fits.json"
    rows = json.loads(fits_path.read_text())
    populated = [row for row in rows if row["basis"] == "def2-svp"]
    assert {
        (row["metric"], row["x_axis"]) for row in populated
    } == EXPECTED_COMBINATIONS
    assert all(row["segment_index"] >= 0 for row in populated)
    assert all(math.isfinite(row["slope"]) for row in populated)
    assert all(math.isfinite(row["intercept"]) for row in populated)
    assert all(row["n_points"] == MIN_FIT_POINTS + 1 for row in populated)
    assert all(len(row["breakpoints"]) <= 1 for row in populated)
    fit_groups: dict[tuple[str, str], int] = {}
    for row in populated:
        key = (row["metric"], row["x_axis"])
        fit_groups[key] = fit_groups.get(key, 0) + 1
    assert all(segment_count <= 2 for segment_count in fit_groups.values())
    assert not any(row["basis"] == "def2-tzvp" for row in rows)


def test_collect_results_writes_empty_fits_table_when_no_group_is_fittable(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset, populated_points=MIN_FIT_POINTS - 1)

    *_, fits_path = collect_results(dataset, tmp_path / "collected")

    assert json.loads(fits_path.read_text()) == []
