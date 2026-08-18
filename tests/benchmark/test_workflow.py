# SPDX-License-Identifier: MIT

import datetime
from pathlib import Path

import pytest

from skala.benchmark.collect_results import collect_results
from skala.benchmark.report import generate
from skala.benchmark.schema.environment import Environment
from skala.benchmark.schema.measurements import make_row, write_shard


def test_raw_dataset_collects_into_offline_report(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    Environment(
        env_id="cpu-local",
        label="Local CPU",
        hostname="host",
        timestamp="2026-07-20T00:00:00+00:00",
        cpu_model="Test CPU",
        num_sockets=1,
        cores_physical=4,
        cores_logical=4,
        numa_nodes=1,
        numa_topology="",
        mem_total_gb=16.0,
        env_vars={"OMP_NUM_THREADS": "4"},
        python_version="3.11",
        pyscf_version="2.12",
        torch_version="2.8",
        numpy_version="2.3",
        blas_impl="OpenBLAS",
        versions={"skala": "test"},
    ).save(raw / "environments")

    write_shard(
        [
            make_row(
                env_id="cpu-local",
                shard_index=0,
                num_shards=1,
                timestamp=datetime.datetime.now(datetime.UTC),
                basis="def2-svp",
                functional="r2scan",
                mol_name="water",
                mol_hash="water",
                charge=0,
                multiplicity=1,
                ansatz="RKS",
                density_fit=True,
                grid_level=3,
                conv_tol=5e-6,
                num_atoms=3,
                num_electrons=10,
                num_atomic_orbitals=24,
                grid_size=80_000,
                is_converged=True,
                num_scf_iterations=6,
                total_energy=-76.0,
                wall_time_ms=110.0,
                kernel_time_ms=100.0,
                setup_ms=8.0,
                finalize_ms=2.0,
                cycles=[
                    {
                        "cycle": index,
                        "wall_ms": 45.0 if index == 0 else 15.0,
                        "numint_ms": 30.0 if index == 0 else 10.0,
                        "forward_ms": 0.0,
                        "numint_calls": 1,
                        "forward_calls": 0,
                    }
                    for index in range(6)
                ],
                status="ok",
            )
        ],
        raw / "measurements",
        0,
    )

    collected = tmp_path / "collected"
    collect_results(raw, collected)

    with pytest.warns(UserWarning, match="No precomputed fits"):
        index = generate(tmp_path / "report", [collected])

    assert index.is_file()
    assert "Local CPU" in index.read_text(encoding="utf-8")
