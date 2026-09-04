#!/usr/bin/env python3
"""Report CMake configuration for the Torch imported by this interpreter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def torch_configuration() -> dict[str, Any]:
    """Return build metadata for the Torch package in this environment."""
    cmake_prefixes = [Path(path) for path in torch.utils.cmake_prefix_path.split(";")]
    torch_dirs = [prefix / "Torch" for prefix in cmake_prefixes]
    torch_dir = next(
        (path for path in torch_dirs if (path / "TorchConfig.cmake").is_file()),
        None,
    )
    if torch_dir is None:
        searched = ", ".join(str(path) for path in torch_dirs)
        raise RuntimeError(f"TorchConfig.cmake was not found under: {searched}")

    return {
        "torch_dir": str(torch_dir.resolve()),
        "version": torch.__version__,
        "cxx11_abi": int(bool(torch._C._GLIBCXX_USE_CXX11_ABI)),
        "cuda": torch.version.cuda,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("json", "cmake"), default="json")
    args = parser.parse_args()
    configuration = torch_configuration()
    if args.format == "cmake":
        print(f"-DTorch_DIR={configuration['torch_dir']}")
        print(f"-DSKALAXC_TORCH_CXX11_ABI={configuration['cxx11_abi']}")
        print(f"-DSKALAXC_TORCH_CUDA_VERSION={configuration['cuda'] or 'none'}")
    else:
        print(json.dumps(configuration, sort_keys=True))


if __name__ == "__main__":
    main()
