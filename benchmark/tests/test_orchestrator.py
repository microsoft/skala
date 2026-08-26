# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from skala_benchmark import node_info
from skala_benchmark.dataset import BenchmarkMolecule
from skala_benchmark.models import Molecule
from skala_benchmark.orchestrator import (
    SweepRequest,
    _load_checkpoint_state,
    _sweep_fingerprint,
    build_tasks,
    select_for_shard,
)
from skala_benchmark.protocol import (
    BenchmarkProtocol,
    Device,
    FunctionalKind,
    FunctionalSpec,
)
from skala_benchmark.schema.measurements import make_row, write_shard


def _molecule(name: str, electrons: int) -> BenchmarkMolecule:
    return BenchmarkMolecule(
        mol_hash=name,
        name=name,
        molecule=Molecule(
            atomic_numbers=[1],
            geometry_bohr=[[0.0, 0.0, 0.0]],
            multiplicity=2,
        ),
        num_atoms=1,
        num_electrons=electrons,
    )


def _request(tmp_path: Path) -> SweepRequest:
    protocol = BenchmarkProtocol(
        bases=("b1", "b2"),
        functionals=(
            FunctionalSpec("f1", FunctionalKind.NATIVE),
            FunctionalSpec("f2", FunctionalKind.NATIVE),
        ),
    )
    return SweepRequest(
        output_dir=tmp_path,
        env_id="test",
        env_label="Test",
        device=Device.CPU,
        protocol=protocol,
    )


def test_task_order_and_sharding_are_deterministic_and_complete() -> None:
    request = _request(Path("."))
    molecules = [
        _molecule(name, electrons)
        for name, electrons in zip("EDCBA", range(5, 0, -1), strict=True)
    ]
    tasks = build_tasks(
        molecules,
        request.protocol.bases,
        request.protocol.functionals,
    )
    shards = [select_for_shard(tasks, index, 3) for index in range(3)]

    assert [task[2].num_electrons for task in tasks] == sorted(
        task[2].num_electrons for task in tasks
    )
    assert sorted(map(id, (task for shard in shards for task in shard))) == sorted(
        map(id, tasks)
    )
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_resume_rejects_a_changed_sweep(tmp_path: Path) -> None:
    request = _request(tmp_path)
    tasks = build_tasks(
        [_molecule("A", 1), _molecule("B", 2)],
        request.protocol.bases,
        request.protocol.functionals,
    )
    fingerprint = _sweep_fingerprint(request, tasks)
    assert fingerprint != _sweep_fingerprint(
        dataclasses.replace(request, num_shards=2), tasks
    )

    write_shard(
        [
            make_row(
                env_id="test",
                basis="b1",
                functional="f1",
                functional_kind="native",
                mol_hash="A",
                shard_index=0,
                num_shards=1,
                ansatz="UKS",
                density_fit=True,
                auxbasis="def2-universal-jkfit",
                grid_level=3,
                conv_tol=5e-6,
                device="gpu",
                sweep_fingerprint=fingerprint,
                status="ok",
            )
        ],
        tmp_path / "measurements",
        0,
    )

    with pytest.raises(ValueError, match="incompatible with this sweep"):
        _load_checkpoint_state(
            tmp_path / "measurements",
            0,
            request,
            fingerprint,
        )


def test_macos_hardware_probes_degrade_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = {
        "machdep.cpu.brand_string": "Apple M3 Max\n",
        "hw.packages": "1\n",
        "hw.physicalcpu": "12\n",
        "hw.logicalcpu": "16\n",
        "hw.memsize": f"{32 * 1024**3}\n",
    }
    monkeypatch.setattr(
        node_info,
        "_run",
        lambda command: responses.get(command[-1], ""),
    )

    assert node_info._macos_cpu_info() == {
        "Model name": "Apple M3 Max",
        "Socket(s)": "1",
        "Core(s) per socket": "12",
        "CPU(s)": "16",
        "NUMA node(s)": "1",
    }
    assert node_info._macos_mem_total_gb() == 32.0

    responses.clear()
    assert node_info._macos_cpu_info() == {
        "Socket(s)": "1",
        "NUMA node(s)": "1",
    }
    assert node_info._macos_mem_total_gb() == 0.0
