"""Direct Python bindings for SkalaXC."""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path

import torch as _torch

from ._build_info import (
    CUDA_ENABLED,
    CUDA_TOOLKIT_VERSION,
    HDF5_ENABLED,
    MPI_ENABLED,
    OPENMP_ENABLED,
    PYTHON_LAYOUT,
    SKALAXC_VERSION,
    TORCH_CUDA_VERSION,
    TORCH_CXX11_ABI,
    TORCH_VERSION,
)


def _major_minor(version: str) -> tuple[int, int]:
    release = version.split("+", maxsplit=1)[0].split(".")
    return int(release[0]), int(release[1])


def _cuda_versions_compatible(build_version: str, runtime_version: str) -> bool:
    build_major = int(build_version.split(".", maxsplit=1)[0])
    runtime_major = int(runtime_version.split(".", maxsplit=1)[0])
    return build_major == runtime_major


if _major_minor(_torch.__version__) != _major_minor(TORCH_VERSION):
    raise ImportError(
        "SkalaXC was built against Torch "
        f"{TORCH_VERSION}, but this environment provides {_torch.__version__}"
    )

if TORCH_CXX11_ABI != "unknown":
    runtime_abi = int(bool(_torch._C._GLIBCXX_USE_CXX11_ABI))
    if runtime_abi != int(TORCH_CXX11_ABI):
        raise ImportError(
            "SkalaXC and Torch use different libstdc++ C++11 ABIs: "
            f"SkalaXC={TORCH_CXX11_ABI}, Torch={runtime_abi}"
        )

if CUDA_ENABLED:
    if _torch.version.cuda is None:
        raise ImportError(
            "CUDA-enabled SkalaXC requires a CUDA-enabled Torch installation"
        )
    if not _cuda_versions_compatible(_torch.version.cuda, TORCH_CUDA_VERSION):
        raise ImportError(
            "SkalaXC was built for the Torch CUDA "
            f"{TORCH_CUDA_VERSION}, but this environment provides "
            f"Torch CUDA {_torch.version.cuda}; their CUDA major versions "
            "must match"
        )

from . import _skalaxc as _native  # noqa: E402

if PYTHON_LAYOUT == "WHEEL":
    MODEL_DIR = _Path(_native.__file__).with_name("models")
elif PYTHON_LAYOUT == "CONDA":
    MODEL_DIR = _Path(_sys.prefix) / "share" / "skalaxc" / "skala_models"
else:
    raise ImportError(f"Unsupported SkalaXC Python layout: {PYTHON_LAYOUT}")
_os.environ.setdefault("SKALAXC_MODEL_PATH", str(MODEL_DIR))

from ._skalaxc import (  # noqa: E402
    Atom,
    AtomicGridSize,
    BasisSet,
    DiagnosticsSnapshot,
    DomainBatchMode,
    ExecutionSpace,
    Functional,
    GradientSettings,
    LoadBalancer,
    LoadBalancerFactory,
    MolecularWeights,
    MolecularWeightsFactory,
    MolecularWeightsSettings,
    Molecule,
    MolGrid,
    MolGridFactory,
    PruningScheme,
    RadialQuad,
    RuntimeEnvironment,
    Shell,
    SkalaXCError,
    TimingMetric,
    TimingSettings,
    TimingStatus,
    TimingValue,
    XCIntegrator,
    XCIntegratorFactory,
    XCWeightAlgorithm,
    native_version,
)

if CUDA_ENABLED:
    DeviceRuntimeSettings = _native.DeviceRuntimeSettings

__version__ = SKALAXC_VERSION

__all__ = [
    "CUDA_ENABLED",
    "CUDA_TOOLKIT_VERSION",
    "HDF5_ENABLED",
    "MODEL_DIR",
    "MPI_ENABLED",
    "OPENMP_ENABLED",
    "PYTHON_LAYOUT",
    "SKALAXC_VERSION",
    "TORCH_CUDA_VERSION",
    "TORCH_VERSION",
    "Atom",
    "AtomicGridSize",
    "BasisSet",
    "DiagnosticsSnapshot",
    "DomainBatchMode",
    "ExecutionSpace",
    "Functional",
    "GradientSettings",
    "LoadBalancer",
    "LoadBalancerFactory",
    "MolGrid",
    "MolGridFactory",
    "MolecularWeights",
    "MolecularWeightsFactory",
    "MolecularWeightsSettings",
    "Molecule",
    "PruningScheme",
    "RadialQuad",
    "RuntimeEnvironment",
    "Shell",
    "SkalaXCError",
    "TimingMetric",
    "TimingSettings",
    "TimingStatus",
    "TimingValue",
    "XCIntegrator",
    "XCIntegratorFactory",
    "XCWeightAlgorithm",
    "native_version",
]

if CUDA_ENABLED:
    __all__.append("DeviceRuntimeSettings")
