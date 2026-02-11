# Skala: Accurate and scalable exchange-correlation with deep learning

[![Documentation](https://img.shields.io/badge/docs-microsoft.github.io%2Fskala-blue?logo=read-the-docs&logoColor=white)](https://microsoft.github.io/skala)
[![Tests](https://img.shields.io/github/actions/workflow/status/microsoft/skala/test.yml?branch=main&logo=github&label=build)](https://github.com/microsoft/skala/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/skala?logo=pypi&logoColor=white)](https://pypi.org/project/skala/)
[![Paper](https://img.shields.io/badge/arXiv-2506.14665-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.14665)

Skala is a neural network-based exchange-correlation functional for density functional theory (DFT), developed by Microsoft Research AI for Science. It leverages deep learning to predict exchange-correlation energies from electron density features, achieving chemical accuracy for atomization energies and strong performance on broad thermochemistry and kinetics benchmarks, all at a computational cost similar to semi-local DFT.

Trained on a large, diverse dataset—including coupled cluster atomization energies and public benchmarks—Skala uses scalable message passing and local layers to learn both local and non-local effects. The model has about 276,000 parameters and matches the accuracy of leading hybrid functionals.

Learn more about Skala in our [ArXiv paper](https://arxiv.org/abs/2506.14665).

## What's in here

This repository contains two main components:

1. The Python package `skala`, which is also distributed [on PyPI](https://pypi.org/project/skala/) and contains a PyTorch implementation of the Skala model, its hookups to quantum chemistry packages [PySCF](https://pyscf.org/), [GPU4PySCF](https://pyscf.org/user/gpu.html) and [ASE](https://ase-lib.org/).
2. An example of using Skala in C++ CPU applications through LibTorch, see [`examples/cpp/cpp_integration`](examples/cpp/cpp_integration).


### Skala in Azure AI Foundry

The Skala model is also served on [Azure AI Foundry](https://ai.azure.com/catalog/models/Skala).

### GauXC development version for PyTorch-based functionals like Skala

[GauXC](https://github.com/wavefunction91/GauXC) is a CPU/GPU C++ library for XC functionals.
A development version with an add-on supporting PyTorch-based functionals like Skala is available in the [`skala` branch of the GauXC repository](https://github.com/wavefunction91/GauXC/tree/skala).
GauXC is part of the stack that serves Skala in [Azure AI Foundry](https://ai.azure.com/catalog/models/Skala) and can be used to integrate Skala into other third-party DFT codes.
For detailed documentation on using GauXC visit the [Skala integration guide](https://microsoft.github.io/skala/gauxc).

## Getting started: PySCF (CPU)

All information below relates to the Python package `skala`.

Install using Pip:

```bash
# Install CPU-only PyTorch (skip if you already have CPU or GPU-enabled PyTorch installed)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install skala
```

Or using Conda (Mamba):

```bash
mamba install -c conda-forge skala "pytorch=*=cpu*"
```

Run an SCF calculation with Skala for a hydrogen molecule:

```python
from pyscf import gto
from skala.pyscf import SkalaKS

mol = gto.M(
    atom="""H 0 0 0; H 0 0 1.4""",
    basis="def2-tzvp",
)
ks = SkalaKS(mol, xc="skala")
ks.kernel()
```

## Getting started: GPU4PySCF (GPU) 

These instructions use Mamba and pip to install CUDA toolkit, Torch, and CuPy. It supports CUDA version 11, 12 or 13. You can find the most recent CUDA version that is supported on your system using `nvidia-smi`.

```bash
cu_version=12 #or 11 or 13 depending on your CUDA version
mamba env create -n skala -f environment-gpu.yml  "cuda-version==${cu_version}.*" skala
mamba activate skala
pip install --no-deps "gpu4pyscf-cuda${cu_version}x>=1.0,<2" "gpu4pyscf-libxc-cuda${cu_version}x>=0.4,<1"
```

Run an SCF calculation with Skala for a hydrogen molecule on GPU:

```python
from pyscf import gto
from skala.gpu4pyscf import SkalaKS

mol = gto.M(
    atom="""H 0 0 0; H 0 0 1.4""",
    basis="def2-tzvp",
)
ks = SkalaKS(mol, xc="skala")
ks.kernel()
```

## Documentation and examples

Go to [microsoft.github.io/skala](https://microsoft.github.io/skala) for a more detailed installation guide and further examples of how to use the Skala functional with PySCF, GPU4PySCF and ASE and in [Azure AI Foundry](https://ai.azure.com/catalog/models/Skala).

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
