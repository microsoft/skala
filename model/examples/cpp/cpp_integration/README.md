# Integrating Skala in C++ code

This example demonstrates how to use the Skala machine learning functional in C++ CPU applications
using LibTorch.

The full walkthrough, including environment setup, building, input preparation, and expected
output, is available in the [C++ (LibTorch) guide](https://microsoft.github.io/skala/cpp)
(source: [`website/cpp.rst`](../../../../website/cpp.rst)).

Quick start from the repository root:

```bash
pixi shell --locked -e cpp-integration
cmake -B build_example -S model/examples/cpp/cpp_integration -G Ninja
cmake --build build_example
hf download microsoft/skala-1.1 skala-1.1-rev1.fun --local-dir .
python model/examples/prepare_inputs.py --output-dir features
./build_example/skala_cpp_integration ./skala-1.1-rev1.fun ./features
```

The input features are generated with the [`prepare_inputs.py`](../../prepare_inputs.py) script,
which is shared with the [Fortran (FTorch) example](https://microsoft.github.io/skala/ftorch).
