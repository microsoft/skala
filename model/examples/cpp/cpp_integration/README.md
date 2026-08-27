# Integrating Skala in C++ code

This example demonstrates how to use the Skala machine learning functional in C++ CPU applications using LibTorch.

## Setup environment

Install the locked Pixi environment from the repository root:

```bash
pixi install --locked -e cpp-integration
```

## Build library

Configure and build the example with CMake and Ninja in the Pixi environment:

```bash
pixi run -e cpp-integration cmake -B build_example -S model/examples/cpp/cpp_integration -G Ninja
pixi run -e cpp-integration cmake --build build_example
```

## Run example

Download the Skala model from Hugging Face:

```bash
pixi run -e cpp-integration hf download microsoft/skala-1.1 \
	skala-1.1-rev1.fun --local-dir .
```

Prepare the molecular features for a test molecule (H2) using the provided script:

```bash
pixi run -e default python examples/cpp/cpp_integration/prepare_inputs.py \
	--output-dir features
```

Finally, run $E_\text{xc}$ and (partial) $V_\text{xc}$ computations with the C++ example:

```bash
pixi run -e cpp-integration ./build_example/skala_cpp_integration ./skala-1.1-rev1.fun ./features
```

**Note:** You are expected to add D3 dispersion correction (using b3lyp settings) to the final energy of Skala.

## Performance tuning

[This guide](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html) from Intel provides useful tips on how to tune performance of PyTorch models on CPU.
