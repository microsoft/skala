# SPDX-License-Identifier: MIT

"""A description of the local machine"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Environment:
    """Hardware, threading, and software description of a benchmark node."""

    env_id: str  # also the JSON file stem. Referred to by measurements.env_id
    label: str
    hostname: str
    timestamp: str  # ISO-8601 UTC

    # Hardware
    cpu_model: str
    num_sockets: int
    cores_physical: int
    cores_logical: int  # including hyper-threads
    numa_nodes: int
    numa_topology: str  # raw lscpu/numactl dump
    mem_total_gb: float
    gpu_models: list[str] = field(default_factory=list)
    gpu_count: int = 0
    interconnect: str | None = None  # NVLink/PCIe topology summary

    # Environment
    env_vars: dict[str, str] = field(
        default_factory=dict
    )  # raw threading-related vars (OMP, OpenBLAS, MKL, torch)

    # Software
    python_version: str = ""
    pyscf_version: str = ""
    torch_version: str = ""
    numpy_version: str = ""
    blas_impl: str = ""  # e.g. "OpenBLAS 0.3.x"
    cuda_version: str | None = None
    gpu4pyscf_version: str | None = None
    skala_version: str = ""  # Skala code version being benchmarked
    versions: dict[str, str] = field(
        default_factory=dict
    )  # catch-all for extra packages

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, directory: str | Path) -> Path:
        """Writes this environment as ``<directory>/<env_id>.json``"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.env_id}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Environment":
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("label", data["env_id"])
        return cls(**data)
