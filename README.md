# Skala: Accurate and scalable exchange-correlation with deep learning

[![Documentation](https://img.shields.io/badge/docs-microsoft.github.io%2Fskala-blue?logo=read-the-docs&logoColor=white)](https://microsoft.github.io/skala)
[![Tests](https://img.shields.io/github/actions/workflow/status/microsoft/skala/test.yml?branch=main&logo=github&label=build)](https://github.com/microsoft/skala/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/skala?logo=pypi&logoColor=white)](https://pypi.org/project/skala/)
[![Paper](https://img.shields.io/badge/arXiv-2506.14665-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.14665)

Skala is a neural network-based exchange-correlation functional for density functional theory (DFT), developed by Microsoft Research AI for Science. It uses deep learning to predict exchange-correlation energies from electron density features, achieving chemical accuracy for atomization energies and strong performance on broad thermochemistry and kinetics benchmarks, all at a computational cost similar to semi-local DFT.

Trained on a large, diverse dataset — including coupled-cluster atomization energies and public benchmarks — Skala uses scalable message passing and local layers to learn both local and non-local effects. The model has about 276,000 parameters and matches the accuracy of leading hybrid functionals.

The recommended neural functional is `skala-1.1`, which uses per-atom packed grids, multiple non-local layers, and symmetric contraction. The legacy `skala-1.0` traced model is still loadable via `load_functional("skala-1.0")`.

Learn more about Skala in our [ArXiv paper](https://arxiv.org/abs/2506.14665).

## What's in here

This repository contains two main components:

1. The Python package `skala`, distributed [on PyPI](https://pypi.org/project/skala/) and on conda-forge. It contains a PyTorch implementation of the Skala model and its bindings to the quantum-chemistry packages [PySCF](https://pyscf.org/), [GPU4PySCF](https://pyscf.org/user/gpu.html), and [ASE](https://ase-lib.org/).
2. Examples of using Skala from compiled code through LibTorch and GauXC:
   - [Skala in C++ with libtorch](examples/cpp/cpp_integration)
   - [Skala in Fortran with FTorch](https://microsoft.github.io/skala/ftorch)
   - [Skala in C++ with GauXC](https://microsoft.github.io/skala/gauxc/cpp-library)
   - [Skala in C with GauXC](https://microsoft.github.io/skala/gauxc/c-library)
   - [Skala in Fortran with GauXC](https://microsoft.github.io/skala/gauxc/fortran-library)


### Skala in Azure AI Foundry

The Skala model is also served on [Azure AI Foundry](https://ai.azure.com/catalog/models/Skala).

### GauXC development version for PyTorch-based functionals like Skala

[GauXC](https://github.com/wavefunction91/GauXC) is a CPU/GPU C++ library for XC functionals.
A development version with an add-on supporting PyTorch-based functionals like Skala is available in the [`skala` branch of the GauXC repository](https://github.com/wavefunction91/GauXC/tree/skala).
GauXC is part of the stack that serves Skala in [Azure AI Foundry](https://ai.azure.com/catalog/models/Skala) and can be used to integrate Skala into other third-party DFT codes.
For detailed documentation on using GauXC visit the [Skala integration guide](https://microsoft.github.io/skala/gauxc).

## Getting started: PySCF (CPU)

All information below relates to the Python package `skala`.

`pip install skala` works out of the box and pulls every dependency from PyPI.
If you don't already have PyTorch installed, install the CPU-only wheel first
to avoid pulling a large CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install skala
```

For a reproducible conda environment, use the provided
[`environment-cpu.yml`](environment-cpu.yml), which pins CPU-only PyTorch and
all runtime dependencies:

```bash
mamba env create -n skala -f environment-cpu.yml
mamba activate skala
pip install skala
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
wheels that must match your CUDA toolkit. The recommended path is the provided
[`environment-gpu.yml`](environment-gpu.yml), which pins `pytorch-gpu`,
`cuda-toolkit` 12, `cutensor`, and installs `gpu4pyscf-cuda12x` 1.5 from PyPI:

```bash
mamba env create -n skala -f environment-gpu.yml
mamba activate skala
pip install skala
```

If you are building inside a container without a GPU attached (e.g., CI or a
Docker image built on a CPU-only host), set `CONDA_OVERRIDE_CUDA` so the solver
proceeds without a device:

```bash
CONDA_OVERRIDE_CUDA=12.0 mamba env create -n skala -f environment-gpu.yml
```

For CUDA 11 or 13, adjust `cuda-toolkit`, `cuda-version`, and the
`gpu4pyscf-cuda{11,13}x` pin in `environment-gpu.yml` accordingly. Check your
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

See [microsoft.github.io/skala](https://microsoft.github.io/skala) for a more detailed installation guide and further examples of how to use the Skala functional with PySCF, GPU4PySCF, and ASE, as well as in [Azure AI Foundry](https://ai.azure.com/catalog/models/Skala).

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
