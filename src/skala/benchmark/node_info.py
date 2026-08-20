# SPDX-License-Identifier: MIT

"""Collect the local node's hardware + software :class:`Environment`.

Best-effort: every external probe (lscpu, numactl, nvidia-smi, /proc) is guarded
and degrades to an empty/None value so a missing tool never aborts a sweep.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from datetime import UTC, datetime
from importlib import metadata

from skala.benchmark.schema.environment import Environment

#: Threading-related environment variables worth recording verbatim.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "PYSCF_MAX_MEMORY",
)


def _run(cmd: list[str]) -> str:
    """Run a command, returning stdout (empty string on any failure)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "LC_ALL": "C"},  # stable, English lscpu/numactl labels
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def _lscpu() -> dict[str, str]:
    """Parse ``lscpu`` into a ``{field: value}`` dict."""
    fields: dict[str, str] = {}
    for line in _run(["lscpu"]).splitlines():
        key, _, value = line.partition(":")
        if value:
            fields[key.strip()] = value.strip()
    return fields


def _macos_cpu_info() -> dict[str, str]:
    """Read macOS CPU metadata through best-effort ``sysctl`` probes."""
    fields = {
        "Model name": _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip(),
        "Socket(s)": _run(["sysctl", "-n", "hw.packages"]).strip() or "1",
        "Core(s) per socket": _run(["sysctl", "-n", "hw.physicalcpu"]).strip(),
        "CPU(s)": _run(["sysctl", "-n", "hw.logicalcpu"]).strip(),
        "NUMA node(s)": "1",
    }
    return {key: value for key, value in fields.items() if value}


def _int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value)) if value is not None else default
    except ValueError:
        return default


def _mem_total_gb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024 * 1024), 1)
    except OSError:
        pass
    if platform.system() == "Darwin":
        return _macos_mem_total_gb()
    return 0.0


def _macos_mem_total_gb() -> float:
    """Read total macOS memory through a best-effort ``sysctl`` probe."""
    bytes_total = _int(_run(["sysctl", "-n", "hw.memsize"]).strip())
    return round(bytes_total / (1024**3), 1) if bytes_total > 0 else 0.0


def _gpu_models() -> list[str]:
    """GPU names visible to this process, honoring ``CUDA_VISIBLE_DEVICES``."""
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    gpus = [line.strip() for line in out.splitlines() if line.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return gpus
    if not visible.strip():
        return []
    selected: list[str] = []
    for token in visible.split(","):
        token = token.strip()
        if not token.isdigit():
            return gpus  # UUID/MIG spec: can't map by index, report all physical GPUs
        index = int(token)
        if 0 <= index < len(gpus):
            selected.append(gpus[index])
    return selected


def _package_version(*names: str) -> str | None:
    """Return the version of the first installed distribution among ``names``.

    Several names because a package can ship under variants: gpu4pyscf installs
    as ``gpu4pyscf-cuda12x`` or ``gpu4pyscf-cuda11x`` and never under its own
    import name.
    """
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _blas_impl() -> str:
    try:
        import numpy as np

        config = np.show_config(mode="dicts")
        blas = config.get("Build Dependencies", {}).get("blas", {})
        return f"{blas.get('name', '')} {blas.get('version', '')}".strip()
    except Exception:  # noqa: BLE001 - purely informational
        return ""


def _cuda_version() -> str | None:
    try:
        import torch

        return torch.version.cuda
    except Exception:  # noqa: BLE001 - torch may be CPU-only or absent
        return None


def collect_environment(env_id: str, label: str) -> Environment:
    """Probe the local node and return a populated :class:`Environment`."""
    cpu = _lscpu()
    if not cpu and platform.system() == "Darwin":
        cpu = _macos_cpu_info()
    sockets = _int(cpu.get("Socket(s)"), 1)
    cores_per_socket = _int(cpu.get("Core(s) per socket"), 0)
    numa_topology = _run(["numactl", "--hardware"]) or "\n".join(
        line for line in _run(["lscpu"]).splitlines() if line.startswith("NUMA")
    )
    gpu_models = _gpu_models()
    interconnect = (
        _run(["nvidia-smi", "topo", "-m"]).strip() or None
        if len(gpu_models) > 1
        else None
    )

    return Environment(
        env_id=env_id,
        label=label,
        hostname=socket.gethostname(),
        timestamp=datetime.now(UTC).isoformat(),
        cpu_model=cpu.get("Model name", ""),
        num_sockets=sockets,
        cores_physical=sockets * cores_per_socket,
        cores_logical=_int(cpu.get("CPU(s)"), 0),
        numa_nodes=_int(cpu.get("NUMA node(s)"), 1),
        numa_topology=numa_topology,
        mem_total_gb=_mem_total_gb(),
        gpu_models=gpu_models,
        gpu_count=len(gpu_models),
        interconnect=interconnect,
        env_vars={k: os.environ[k] for k in _THREAD_ENV_VARS if k in os.environ},
        python_version=platform.python_version(),
        pyscf_version=_package_version("pyscf") or "",
        torch_version=_package_version("torch") or "",
        numpy_version=_package_version("numpy") or "",
        blas_impl=_blas_impl(),
        cuda_version=_cuda_version(),
        gpu4pyscf_version=_package_version(
            "gpu4pyscf", "gpu4pyscf-cuda12x", "gpu4pyscf-cuda11x"
        ),
        skala_version=_package_version("skala") or "",
        versions={
            name: version
            for name in ("skala", "pyarrow", "dftd3", "ase")
            if (version := _package_version(name)) is not None
        },
    )
