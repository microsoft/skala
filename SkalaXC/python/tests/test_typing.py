from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import skalaxc


def test_installed_stub_matches_native_build() -> None:
    if skalaxc.CUDA_ENABLED and skalaxc.MPI_ENABLED:
        variant = "cuda_mpi"
    elif skalaxc.CUDA_ENABLED:
        variant = "cuda"
    elif skalaxc.MPI_ENABLED:
        variant = "mpi"
    else:
        variant = "cpu"

    package_root = Path(__file__).parent.parent
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "tests/typing/interface.py",
            f"tests/typing/{variant}.py",
        ],
        cwd=package_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
