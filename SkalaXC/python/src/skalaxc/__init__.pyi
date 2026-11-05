from pathlib import Path

from ._skalaxc import *  # noqa: F403

CUDA_ENABLED: bool
CUDA_TOOLKIT_VERSION: str
HDF5_ENABLED: bool
MPI_ENABLED: bool
MODEL_DIR: Path
OPENMP_ENABLED: bool
PYTHON_LAYOUT: str
SKALAXC_VERSION: str
TORCH_VERSION: str
TORCH_CUDA_VERSION: str
__version__: str

def _cuda_versions_compatible(build_version: str, runtime_version: str) -> bool: ...
