# Integrating Skala in Fortran code

This example demonstrates how to use the Skala machine learning functional in Fortran CPU applications
using FTorch.

When a Conda or Pixi environment is active, CMake looks for OpenSSL in that environment
to avoid mixing system libraries with the environment's compiler sysroot.
Set `-DOPENSSL_ROOT_DIR=/path/to/openssl` to select a different installation explicitly.

The full walkthrough, including environment setup, building, input preparation, and expected
output, is available in the [Fortran (FTorch) guide](https://microsoft.github.io/skala/ftorch)
(source: [`website/ftorch.rst`](../../../../website/ftorch.rst)).
