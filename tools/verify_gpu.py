from __future__ import annotations

from importlib.metadata import packages_distributions, version

import cupy
import gpu4pyscf
import torch

EXPECTED_GPU4PYSCF_VERSION = "1.8.1"


def main() -> None:
    """Verify that the selected GPU environment is usable."""
    distributions = packages_distributions().get(gpu4pyscf.__name__, ())
    try:
        distribution = next(
            name for name in distributions if name.startswith("gpu4pyscf-cuda")
        )
    except StopIteration as error:
        raise RuntimeError(
            "Could not identify the installed GPU4PySCF distribution"
        ) from error

    gpu4pyscf_version = version(distribution)
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    print(
        "gpu",
        properties.name,
        "capability",
        f"{properties.major}.{properties.minor}",
        "memory",
        f"{free_memory / 1024**3:.2f}/{total_memory / 1024**3:.2f} GiB free",
    )
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print(
        "cupy",
        cupy.__version__,
        "runtime",
        cupy.cuda.runtime.runtimeGetVersion(),
        "driver",
        cupy.cuda.runtime.driverGetVersion(),
    )
    print(distribution, gpu4pyscf_version)

    assert gpu4pyscf_version == EXPECTED_GPU4PYSCF_VERSION
    assert torch.cuda.is_available()
    cupy.linalg.norm(cupy.arange(3.0)).get()


if __name__ == "__main__":
    main()
