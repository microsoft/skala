# Integrating Skala in C++ code

This example demonstrates how to use the Skala machine learning functional in C++ CPU applications using LibTorch.

## Setup environment

Set up the conda environment using the provided environment file:

```bash
cd examples/cpp/cpp_integration
conda env create -n skala_cpp_integration -f environment.yml
conda activate skala_cpp_integration
```

## Build library

The example can be built using CMake.
The provided environment is configured for CMake to find the required dependencies.

```bash
cmake -S . -B _build -G Ninja
cmake --build _build
```

For any changes to the code, rebuild using the last command.

## Run example

Download the Skala model, as well as a reference LDA functional from HuggingFace
using the provided download script:

```bash
./download_model.py
```

Prepare the molecular features for a test molecule (H2) using the provided script:

```bash
python ./prepare_inputs.py --output-dir H2
```

Finally, run $E_\text{xc}$ and (partial) $V_\text{xc}$ computations with the C++ example:

```bash
./_build/skala_cpp_integration skala-1.0.fun H2
```

**Note:** You are expected to add D3 dispersion correction (using b3lyp settings) to the final energy of Skala.

## Performance tuning

[This guide](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html) from Intel provides useful tips on how to tune performance of PyTorch models on CPU.
