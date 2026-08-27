# SPDX-License-Identifier: MIT

from __future__ import annotations

import datetime
import json
from pathlib import Path

from skala_benchmark.collect_results import MIN_FIT_POINTS, collect_results
from skala_benchmark.report import generate
from skala_benchmark.schema.environment import Environment
from skala_benchmark.schema.measurements import make_row, write_shard


def test_raw_data_collects_and_reports_reproducibly(tmp_path: Path) -> None:
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

    rows = []
    for index in range(MIN_FIT_POINTS + 1):
        scale = float(index + 1)
        cycles = [
            {
                "cycle": cycle,
                "wall_ms": scale * (45.0 if cycle == 0 else 15.0),
                "veff_ms": scale * (40.0 if cycle == 0 else 13.0),
                "numint_ms": scale * (30.0 if cycle == 0 else 10.0),
                "xc_eval_ms": scale * (12.0 if cycle == 0 else 3.0),
                "forward_ms": 0.0,
                "backward_ms": 0.0,
                "veff_calls": 1,
                "numint_calls": 1,
                "xc_eval_calls": 1,
            }
            for cycle in range(4)
        ]
        rows.append(
            make_row(
                env_id="cpu-local",
                shard_index=0,
                num_shards=1,
                timestamp=datetime.datetime(2026, 7, 20, tzinfo=datetime.UTC),
                basis="def2-svp",
                functional="r2scan",
                functional_kind="native",
                mol_name=f"molecule-{index}",
                mol_hash=f"molecule-{index}",
                charge=0,
                multiplicity=1,
                ansatz="RKS",
                density_fit=True,
                grid_level=3,
                conv_tol=5e-6,
                num_atoms=index + 1,
                num_electrons=10 * (index + 1),
                num_atomic_orbitals=24 * (index + 1),
                grid_size=80_000 * (index + 1),
                is_converged=True,
                num_scf_iterations=4,
                total_energy=-76.0 * scale,
                wall_time_ms=110.0 * scale,
                kernel_time_ms=100.0 * scale,
                setup_ms=8.0 * scale,
                finalize_ms=2.0 * scale,
                cycles=cycles,
                status="ok",
            )
        )
    write_shard(rows, raw / "measurements", 0)

    collected = (tmp_path / "collected-a", tmp_path / "collected-b")
    collected_files = [collect_results(raw, output) for output in collected]
    assert [path.read_bytes() for path in collected_files[0]] == [
        path.read_bytes() for path in collected_files[1]
    ]
    assert json.loads((collected[0] / "fits.json").read_text())

    reports = (tmp_path / "report-a", tmp_path / "report-b")
    for report, inputs in zip(reports, collected, strict=True):
        assert generate(report, [inputs]) == report / "index.html"

    report_files = {
        path.relative_to(reports[0]).as_posix()
        for path in reports[0].rglob("*")
        if path.is_file()
    }
    for filename in report_files:
        assert (reports[0] / filename).read_bytes() == (
            reports[1] / filename
        ).read_bytes()
    assert "Local CPU" in (reports[0] / "index.html").read_text()
