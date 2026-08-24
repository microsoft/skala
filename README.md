# Skala: Accurate and scalable exchange-correlation with deep learning

[![Documentation](https://img.shields.io/badge/docs-microsoft.github.io%2Fskala-blue?logo=read-the-docs&logoColor=white)](https://microsoft.github.io/skala)
[![Tests](https://img.shields.io/github/actions/workflow/status/microsoft/skala/test.yml?branch=main&logo=github&label=build)](https://github.com/microsoft/skala/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/skala?logo=pypi&logoColor=white)](https://pypi.org/project/skala/)
[![Paper](https://img.shields.io/badge/arXiv-2506.14665-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.14665)

Skala is a neural network-based exchange-correlation functional for density functional theory (DFT), developed by Microsoft Research AI for Science. It uses deep learning to predict exchange-correlation energies from electron density features, surpasses state-of-the-art hybrid functionals in accuracy for main group thermochemistry, kinetics and non-covalent interactions, all at a computational cost similar to semi-local DFT.

Trained on a large, diverse dataset — including coupled-cluster atomization energies and public benchmarks — Skala uses scalable message passing and local layers to learn both local and non-local effects. The model has about 385,000 parameters and matches the accuracy of leading hybrid functionals.

The recommended neural functional is `skala-1.1`, which uses per-atom packed grids, multiple non-local layers, and symmetric contraction. The legacy `skala-1.0` traced model is still loadable via `load_functional("skala-1.0")`.

Learn more about Skala in our [ArXiv paper](https://arxiv.org/abs/2506.14665).

## What's in here

This repository contains four components:

1. [`skala/`](skala) is the only published Python package. It contains the runtime needed to load released checkpoints and use Skala through [PySCF](https://pyscf.org/), [GPU4PySCF](https://pyscf.org/user/gpu.html), and [ASE](https://ase-lib.org/).
2. [`model/`](model) contains the trainable model definition, its tests, and compiled-model examples. This development code is not included in the `skala` wheel or source distribution.
3. [`gauxc/`](gauxc) contains the GauXC exporter, native integration examples, tests, and source documentation.
4. [`docs/`](docs) contains the main Sphinx site and benchmark runner/report tooling.

Compiled-code examples include:
    - [Skala in C++ with libtorch](model/examples/cpp/cpp_integration)
   - [Skala in Fortran with FTorch](https://microsoft.github.io/skala/ftorch)
   - [Skala in C++ with GauXC](https://microsoft.github.io/skala/gauxc/cpp-library)
   - [Skala in C with GauXC](https://microsoft.github.io/skala/gauxc/c-library)
   - [Skala in Fortran with GauXC](https://microsoft.github.io/skala/gauxc/fortran-library)

Development-only imports use separate namespaces: `skala_model`, `skala_gauxc`, and
`skala_benchmark`. They are intentionally not compatibility aliases inside the released `skala`
package.

### GauXC development version for PyTorch-based functionals like Skala

[GauXC](https://github.com/wavefunction91/GauXC) is a CPU/GPU C++ library for XC functionals.
A development version with an add-on supporting PyTorch-based functionals like Skala is available in the [`skala` branch of the GauXC repository](https://github.com/wavefunction91/GauXC/tree/skala).
GauXC can be used to integrate Skala into other third-party DFT codes.
For detailed documentation on using GauXC visit the [Skala integration guide](https://microsoft.github.io/skala/gauxc).

## Getting started: PySCF (CPU)

All information below relates to the Python package `skala`.
Skala supports Linux and macOS on Apple Silicon with Python 3.11 through 3.13, the latest
PySCF release (2.14), and the two latest PyTorch release lines (2.12 and 2.13).

`pip install skala` works out of the box and pulls every dependency from PyPI.
If you don't already have PyTorch installed, install the CPU-only wheel first
to avoid pulling a large CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install skala
```

For a reproducible source environment, use the default environment from the
committed Pixi lockfile. It uses Python 3.12, PySCF 2.14, and CPU-only PyTorch 2.13:

```bash
pixi install --locked -e default
pixi run -e default python your_script.py
```

Run an SCF calculation with Skala for a hydrogen molecule:

```python
from pyscf import gto
from skala.pyscf import SkalaKS

mol = gto.M(
    atom="""H 0 0 0; H 0 0 1.4""",
    basis="def2-tzvp",
)
ks = SkalaKS(mol, xc="skala-1.1")
ks.kernel()
```

## Getting started: GPU4PySCF (GPU)

The GPU install is more involved because `gpu4pyscf` ships CUDA-version-specific
wheels that must match your CUDA toolkit. GPU environments and helper packages
use the latest tested GPU4PySCF release, 1.8.1.

To install all dependencies from PyPI, use the GPU specific package with the
matching CUDA version, e.g., for CUDA 12:

```bash
pip install skala-cuda12x
```

The `skala-cuda13x` package is available for CUDA 13.

For a reproducible source environment, choose one of the locked GPU environments:

| Environment | CUDA | PyTorch |
|---|---:|---:|
| `gpu-cuda12-torch212` | 12 | 2.12 |
| `gpu-cuda12-torch213` | 12 | 2.13 |
| `gpu-cuda13-torch213` | 13 | 2.13 |

For example:

```bash
pixi install --locked -e gpu-cuda12-torch213
pixi run -e gpu-cuda12-torch213 python your_script.py
```

The workspace records CUDA 12 and CUDA 13 as explicit platforms, so the lock can
be installed while building a container without an attached GPU. Check your
driver's maximum supported CUDA version with `nvidia-smi`.

Run an SCF calculation with Skala for a hydrogen molecule on GPU:

```python
from pyscf import gto
from skala.gpu4pyscf import SkalaKS

mol = gto.M(
    atom="""H 0 0 0; H 0 0 1.4""",
    basis="def2-tzvp",
)
ks = SkalaKS(mol, xc="skala-1.1")
ks.kernel()
```

### Known issue: multiple visible GPUs

Skala uses a single GPU, but importing `gpu4pyscf` allocates memory on **every**
visible CUDA device. This can conflict with PyTorch and with other processes
sharing those GPUs (e.g. in MPI-parallel workloads).

Restrict CUDA to one device **before** launching Python:

```bash
CUDA_VISIBLE_DEVICES=0 python my_script.py
```

For MPI-parallel runs, assign one GPU per local rank:

```bash
mpirun -np 4 bash -c 'CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK python my_script.py'
```

Tracked upstream at [pyscf/gpu4pyscf#435](https://github.com/pyscf/gpu4pyscf/issues/435).

## Getting started: ASE calculator

Skala also provides an [ASE](https://wiki.fysik.dtu.dk/ase/) calculator for energy, force, and geometry optimization workflows:

```python
from ase.build import molecule
from ase.optimize import LBFGSLineSearch
from skala.ase import Skala

atoms = molecule("H2O")
atoms.calc = Skala(xc="skala-1.1", basis="def2-tzvp")

# Single-point energy (eV)
print(atoms.get_potential_energy())

# Geometry optimization
opt = LBFGSLineSearch(atoms)
opt.run(fmax=0.01)
```

## Documentation and examples

See [microsoft.github.io/skala](https://microsoft.github.io/skala) for a more detailed installation guide and further examples of how to use the Skala functional with PySCF, GPU4PySCF and ASE.

## Security: loading `.fun` files

Skala model files (`.fun`) use TorchScript serialization, which can execute arbitrary code when loaded. **Never load `.fun` files from untrusted sources.**

When loading the official Skala models via `load_functional("skala-1.1")` or `load_functional("skala-1.0")`, file integrity is automatically verified against pinned SHA-256 hashes before deserialization. If you load `.fun` files directly with `TracedFunctional.load()`, pass the `expected_hash` parameter to enable verification:

```python
TracedFunctional.load("model.fun", expected_hash="<sha256-hex-digest>")
```

## Project information

See the following files for more information about contributing, reporting issues, and the code of conduct:

- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`LICENSE.txt`](LICENSE.txt)
- [`SECURITY.md`](SECURITY.md)

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
