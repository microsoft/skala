# Skala: Accurate and scalable exchange-correlation with deep learning

[![Documentation](https://img.shields.io/badge/docs-microsoft.github.io%2Fskala-blue?logo=read-the-docs&logoColor=white)](https://microsoft.github.io/skala)
[![Tests](https://img.shields.io/github/actions/workflow/status/microsoft/skala/test.yml?branch=main&logo=github&label=build)](https://github.com/microsoft/skala/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/microsoft-skala?logo=pypi&logoColor=white)](https://pypi.org/project/microsoft-skala/)
[![Paper](https://img.shields.io/badge/arXiv-2506.14665-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2506.14665)

Skala is a neural network-based exchange-correlation functional for density functional theory (DFT), developed by Microsoft Research AI for Science. It leverages deep learning to predict exchange-correlation energies from electron density features, achieving chemical accuracy for atomization energies and strong performance on broad thermochemistry and kinetics benchmarks, all at a computational cost similar to semi-local DFT.

Trained on a large, diverse dataset—including coupled cluster atomization energies and public benchmarks—Skala uses scalable message passing and local layers to learn both local and non-local effects. The model has about 276,000 parameters and matches the accuracy of leading hybrid functionals.

Learn more about Skala in our [ArXiv paper](https://arxiv.org/abs/2506.14665).

## What's in here

This repository contains three main components:

1. The Python package `microsoft-skala`, which is also distributed [on PyPI](https://pypi.org/project/microsoft-skala/) and contains a Pytorch implementation of the Skala model, its hookups to quantum chemistry packages [PySCF](https://pyscf.org/) and [ASE](https://ase-lib.org/), and an independent client library for the Skala model served [in Azure AI Foundry](https://ai.azure.com/catalog/models/Skala).
2. A development version of the CPU/GPU C++ library for XC functionals [GauXC](https://github.com/wavefunction91/GauXC) with an add-on supporting Pytorch-based functionals like Skala. GauXC is part of the stack that serves Skala in Azure AI Foundry and can be used to integrate Skala into other third-party DFT codes.
3. An example of using Skala in C++ CPU applications through LibTorch, see [`examples/cpp/cpp_integration`](examples/cpp/cpp_integration).

All information below relates to the Python package, the development version of GauXC including its license and other information can be found in [`third_party/gauxc`](https://github.com/microsoft/skala/tree/main/third_party/gauxc).

## Getting started

Install using Pip:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # unless you already have GPU Pytorch for something else
pip install microsoft-skala
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

Go to [microsoft.github.io/skala](https://microsoft.github.io/skala) for a more detailed installation guide and further examples of how to use Skala functional with PySCF and ASE and in [Azure Foundry](https://ai.azure.com/catalog/models/Skala).

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
