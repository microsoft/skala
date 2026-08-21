# Integrating Skala in C++ code

This example demonstrates how to use the Skala machine learning functional in C++ CPU applications using LibTorch.

## Setup environment

Install the locked Pixi environment from the repository root:

```bash
pixi install --locked -e cpp-integration
```

## Build library

The Pixi task configures and builds the example with CMake and Ninja:

```bash
pixi run -e cpp-integration cpp-build
```

## Run example

Download the Skala model from Hugging Face:

```bash
pixi run -e cpp-integration download-checkpoint
```

Prepare the molecular features for a test molecule (H2) using the provided script:

```bash
pixi run -e default generate-features
```

Finally, run $E_\text{xc}$ and (partial) $V_\text{xc}$ computations with the C++ example:

```bash
pixi run -e cpp-integration cpp-run
```

**Note:** You are expected to add D3 dispersion correction (using b3lyp settings) to the final energy of Skala.

## Performance tuning

[This guide](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html) from Intel provides useful tips on how to tune performance of PyTorch models on CPU.
