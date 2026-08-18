# SPDX-License-Identifier: MIT

import json
from pathlib import Path

from skala.benchmark.schema.environment import Environment


def _make_env() -> Environment:
    return Environment(
        env_id="cpu-4thread",
        label="CPU · test",
        hostname="node1",
        timestamp="2026-07-10T00:00:00Z",
        cpu_model="Intel Xeon Platinum 8380",
        num_sockets=2,
        cores_physical=32,
        cores_logical=64,
        numa_nodes=2,
        numa_topology="NUMA node0 CPU(s): 0-31\nNUMA node1 CPU(s): 32-63",
        mem_total_gb=256.0,
        gpu_models=["NVIDIA A100"],
        gpu_count=1,
        interconnect="NVLink",
        env_vars={"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
        python_version="3.12.0",
        pyscf_version="2.9.0",
        torch_version="2.4.0",
        numpy_version="2.1.0",
        blas_impl="OpenBLAS 0.3.27",
        cuda_version="12.4",
        gpu4pyscf_version="1.6.0",
        versions={"e3nn": "0.5.1"},
    )


def test_save_load_round_trip(tmp_path: Path) -> None:
    """save() then load() recovers an identical Environment."""
    env = _make_env()
    path = env.save(tmp_path)
    assert path == tmp_path / "cpu-4thread.json"
    assert path.exists()
    loaded = Environment.load(path)
    assert loaded == env


def test_save_creates_missing_directory(tmp_path: Path) -> None:
    """save() creates the destination directory if it does not exist."""
    env = _make_env()
    directory = tmp_path / "nested" / "environments"
    path = env.save(directory)
    assert path == directory / "cpu-4thread.json"
    assert Environment.load(path) == env


def test_save_uses_env_id_as_filename(tmp_path: Path) -> None:
    """The JSON file is named after env_id."""
    env = Environment(**{**_make_env().to_dict(), "env_id": "gpu-a100-single"})
    path = env.save(tmp_path)
    assert path.name == "gpu-a100-single.json"


def test_load_defaults_legacy_label_to_environment_id(tmp_path: Path) -> None:
    data = _make_env().to_dict()
    del data["label"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = Environment.load(path)

    assert loaded.label == loaded.env_id
