# SkalaXC

SkalaXC is a standalone C++17 exchange-correlation integration library. It
embeds GauXC's lower-level grid, collocation, and assembly machinery behind
an ABI-isolated public API and owns a TorchScript model per integrator.
GauXC, LibTorch, Eigen, and other implementation types do not cross that API.

This is an additional integration path. The existing [GauXC guide](../gauxc/docs/index.rst),
[LibTorch tutorial](../website/cpp.rst), and [FTorch tutorial](../website/ftorch.rst)
remain available unchanged.

## Build and test

Use the existing locked native toolchain from the repository root:

```bash
git submodule update --init SkalaXC/external/GauXC
pixi install --locked -e gauxc-openmp-fortran
pixi run -e gauxc-openmp-fortran cmake -S SkalaXC -B SkalaXC/build -G Ninja \
  -DSKALAXC_BUILD_TESTS=ON -DSKALAXC_BUILD_EXAMPLES=ON \
  -DCMAKE_INSTALL_PREFIX="$PWD/SkalaXC/install"
pixi run -e gauxc-openmp-fortran cmake --build SkalaXC/build --parallel 4
OMP_NUM_THREADS=4 pixi run -e gauxc-openmp-fortran \
  ctest --test-dir SkalaXC/build --output-on-failure
pixi run -e gauxc-openmp-fortran cmake --install SkalaXC/build
```

Alternatively provide CMake 3.21+, a C++17 compiler, LibTorch, BLAS, HDF5, and
nlohmann_json through your own environment. Eigen, GauXC, and test dependencies
are fetched when unavailable. ExchCXX and IntegratorXX always use GauXC's pinned
source builds to avoid leaking installed shared C++ dependencies.
The GauXC submodule is pinned; an uninitialized
checkout uses the same pinned revision through FetchContent.

Fortran is enabled by default and needs a Fortran compiler. For C/C++ only,
configure `SKALAXC_BUILD_FORTRAN=OFF` and use the `gauxc-openmp` environment.

## API and tutorial

- [Installation and limitations](../website/skalaxc/installation.rst)
- [C++ API](../website/skalaxc/cpp-api.rst)
- [C++ tutorial](../website/skalaxc/cpp.rst)
- [C API](../website/skalaxc/c-api.rst)
- [C tutorial](../website/skalaxc/c.rst)
- [Fortran API](../website/skalaxc/fortran-api.rst)
- [Fortran tutorial](../website/skalaxc/fortran.rst)
- [Python API](../website/skalaxc/python-api.rst)
- [Python tutorial](../website/skalaxc/python.rst)
- [Python binding build and package contract](python/README.md)
- [Test contracts](tests/README.md)

Include `<skalaxc/skalaxc.hpp>` and link only `SkalaXC::skalaxc`.
The installed CMake package supports `find_package(SkalaXC CONFIG REQUIRED
COMPONENTS CXX)`. A consumer does not need GauXC or LibTorch include paths.
The C binding uses `<skalaxc/skalaxc.h>`, opaque handles, status codes, and
caller-owned output buffers. C-only consumers request the `C` CMake component
and link the same public target, without a C++ compiler or private headers.

Python binds the public C++ API directly with nanobind. The optional
`skalaxc-python-host` and `skalaxc-python-mpi` environments add binding build,
typing, and parity-test dependencies without changing the existing toolchains.
The multi-output conda recipe in `recipe/` packages the native library,
Fortran module, and Python binding; it is rendered by the `skalaxc-package`
environment. Source installs of the Python package require an installed
native library selected through `SkalaXC_DIR`.

Models are selected with `LDA`, `PBE`, `TPSS`, or an explicit `.fun` path.
Named models resolve from the installed model directory, with
`SKALAXC_MODEL_PATH` as an explicit runtime override. Models are hash-verified
when downloaded during configuration.

## Supported behavior

Host unrestricted (UKS) energy, scalar/z potential matrices, and nuclear
gradients are validated. Density and potential matrices are double precision,
`nbf x nbf`, column-major; gradients contain `3 * natoms` values in atom-major
xyz order. Both allocating and caller-owned output overloads are supported.
Restricted (RKS) evaluation is not implemented.

MPI is optional (`SKALAXC_ENABLE_MPI=ON`), uses the caller's communicator, and
requires collective evaluation. Use `gauxc-mpi-fortran` for host MPI with Fortran.
CUDA is experimental and off by default (`SKALAXC_ENABLE_CUDA=ON` to enable).
It requires a compatible CUDA compiler, GPU-enabled LibTorch, and GPU runtime.
LDA/PBE gradients and TPSS energy/potentials are covered; the bundled TPSS
backward trace can exceed launch resources for CUDA gradients on sm_86/sm_120.
Task tiles respect both the selected device's grid Y/Z limits and GauXC's
independent memory limit.

Optional HDF5 input can be disabled with `SKALAXC_ENABLE_HDF5=OFF`; construct
molecules and basis sets programmatically in that case. OpenMP defaults on.
`SKALAXC_BUILD_DOCS=ON` adds the strict `skalaxc_doxygen` target when Doxygen
1.9+ is installed. The separate Sphinx guide is built with the main website.
