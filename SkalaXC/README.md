# SkalaXC

SkalaXC is a standalone machine-learning exchange–correlation (XC) functional
library. It reuses GauXC's internal numerical machinery (grid, load balancer, collocation / local work
  driver, molecular weights, HDF5 I/O), **without** using GauXC's public
  `XCIntegrator` XC API, and exposes its **own ABI-isolated public API** (C++, C, and Fortran) for ML XC
  evaluation. No GauXC or LibTorch type ever crosses the SkalaXC boundary.

The complete validated capability is **host (CPU)** evaluation of the
**unrestricted (UKS)** ML XC energy, potential, and nuclear gradient for the
LDA, GGA, and kinetic-energy-dependent meta-GGA Skala models. An optional CUDA
backend is available with the limitations described below.

---

## Layout and naming

| Path | Contents |
| --- | --- |
| `include/skalaxc/` | Installed C++ and C public headers. |
| `src/host/` | Private CPU implementation of the Skala model and host driver. |
| `src/device/` | Private optional CUDA implementation, grouped by backend. |
| `src/c-api/`, `src/fortran/` | Private language-binding implementations. |
| `examples/{c,cpp,fortran}/` | Minimal public-API consumer programs. |
| `tests/` | Test suite, fixture provenance, tolerances, and focused commands. |
| `tests/ref_data/` | SkalaXC-owned HDF5 reference-integration fixtures. |

The `skalaxc_` prefix is reserved for the library's public API and consumer
artifacts. The `skala_` prefix identifies private Skala model/driver code and
its white-box tests. This keeps the boundary visible in both source names and
build targets.

---

## Design decisions

These are the load-bearing decisions; keep them in mind before changing the
build or the public headers.

### 1. GauXC is an internal implementation detail — reused, never leaked
- SkalaXC consumes GauXC as a **source-tree dependency** (`add_subdirectory`),
  not an installed package. This is deliberate: GauXC's `gauxc` target publishes
  `${GauXC}/src` on its `PUBLIC BUILD_INTERFACE` include path, which is what
  grants SkalaXC access to GauXC's **private** headers (the concrete
  `LocalHostWorkDriver` collocation engine, `XCHostData` scratch buffers, and
  `gen_compressed_submat_map`). An *installed* GauXC would not expose these — and
  that is the point: SkalaXC reuses internals, it never ships them.
- **GauXC `master` is never modified.** All SkalaXC code lives under
  `skala/SkalaXC/`. A configure-time guard fails the build if the GauXC tree
  carries any `onedft`/`skala` surface API (`GAUXC_HAS_ONEDFT`).

#### Exactly which internals — and why the public API cannot substitute

The ML path drives GauXC's concrete host work engine directly, so it depends on
three headers that live under `${GauXC}/src` and are therefore **never installed**
by GauXC (its `install()` ships `include/` only):

- `xc_integrator/local_work_driver/host/local_host_work_driver.hpp` —
  `LocalHostWorkDriver`, the concrete collocation / `xmat` / `uvvar` / `zmat` /
  `inc_vxc` / `partition_weights` engine.
- `xc_integrator/replicated/host/xc_host_data.hpp` — `XCHostData`, the per-batch
  host scratch buffers.
- `xc_integrator/integrator_util/integrator_common.hpp` —
  `gen_compressed_submat_map`, the compressed AO-submatrix layout helper.

Everything else SkalaXC takes from GauXC is already public/installed
(`LoadBalancer`, `MolecularWeights`, `BasisSetMap`, the abstract `LocalWorkDriver`
and `LocalWorkDriverFactory`, `MolGridFactory`, `RuntimeEnvironment`,
`Atom` / `Shell` / `Molecule` / `BasisSet`, and, when HDF5 support is enabled,
`read_hdf5_record`).

This is why the source-tree build is **required**, not a convenience:

- GauXC master's public `XCIntegrator::eval_exc_vxc*` only evaluates conventional
  ExchCXX functionals point-wise; it has no hook for a TorchScript model that
  needs the full grid feature set plus autograd back-propagation.
- The one public seam below that — the abstract `LocalWorkDriver` base — is an
  empty interface (a virtual destructor and nothing else). Every operation the
  ML path calls is declared on the non-installed `LocalHostWorkDriver`, reached
  by down-casting the factory's base pointer.
- Because these three headers are neither installed nor a stable API, their types
  can never appear in SkalaXC's public contract. The PIMPL boundary and symbol
  hiding in decision 3 are therefore a **consequence** of reusing them, not an
  independent choice.

### 2. No pass-through to the GauXC XC entry point
SkalaXC reimplements the ML orchestration (collocation → features → model →
`VXC`) directly on GauXC's lowest-level reusable primitives. It never calls
`XCIntegrator::eval_exc_vxc*`. Structures that the ML path needs but that only
existed on the `skala` branch (per-task feature buffers, raw grid weights) are
**owned by SkalaXC** as parallel storage, not patched into GauXC.

On both host and CUDA, each complete atomic grid domain belongs to exactly one
MPI rank. That rank constructs its features, runs its own model and autograd,
maps the derivatives back to local tasks, and assembles local AO or
nuclear-gradient contributions. OpenMP parallelizes atomic-grid generation and
screening; the CUDA backend keeps collocation and assembly work on the selected
device. MPI communication is reserved for the final EXC, VXC, electron-count,
and gradient reductions; model inputs and derivatives are not gathered or
scattered.

During integrator construction, rank zero of the runtime communicator resolves
and reads the selected TorchScript archive. SkalaXC broadcasts the archive
bytes on that communicator, then every rank deserializes and owns its own model
module. Only runtime rank zero therefore needs filesystem access to the `.fun`
file. If rank zero cannot read the model, it broadcasts the error message so
that all ranks fail consistently instead of leaving non-root ranks blocked in
an MPI collective. Every rank still needs enough memory for its local model.

### 3. Strict ABI isolation (C, C++, and Fortran)
The public boundary exposes **zero** GauXC / LibTorch symbols or types:
- **C++** — strict PIMPL: `SkalaXC::XCIntegrator` holds only a forward-declared
  `unique_ptr<Impl>`; all GauXC/Torch usage is confined to the `.cxx` TUs. Its
  header-only matrix facade returns `(EXC, VXCs, VXCz)` using the caller's
  owning column-major dense matrix type. `eval_exc_grad` returns an owning
  `std::vector<double>` with `3 * natoms` atom-major `xyz` values. Overloads
  also write potentials into pre-sized `nbf × nbf` matrices and gradients into
  a pre-sized `std::vector<double>` under the same column-major and atom-major
  contracts. Matrix types must expose contiguous `double` storage through
  `data()` in column-major order; row-major and strided matrix types are not
  supported. Nuclear gradients always include molecular-weight derivatives.
  `SkalaSettings` selects the model when the integrator is
  constructed. The integrator loads and owns that TorchScript module for its
  lifetime, so subsequent evaluations neither reload it nor use a process-wide
  model cache.
- **C** — opaque handle (`skalaxc_calculator_t`) + status codes + POD only.
- **Fortran** — `iso_c_binding` wrapper over the C API; binds only to SkalaXC
  opaque handles. Assumed-shape array wrappers reject noncontiguous storage and
  extents that do not match the native molecule, basis, matrix, or gradient
  contracts before calling C.
- At the binary boundary, densities and potentials cross as raw `double*`
  (`nbf × nbf`, column-major). The C API passes caller-owned output buffers to
  the private non-allocating integration core; it does not allocate or copy
  output matrices. Gradients use caller-owned, atom-major `xyz` storage in C
  and Fortran as well. GauXC exceptions are caught at the boundary and
  re-thrown as `SkalaXC::Exception` (C++) or converted to status codes
  (C/Fortran).
- **Link/runtime:** GauXC and its static dependencies are embedded into
  `libskalaxc.so`; Eigen expressions are compiled only in private translation
  units. Everything is compiled `-fvisibility=hidden`
  and a linker **version script** (`cmake/skalaxc-exports.map`) plus
  `--exclude-libs,ALL` export **only** `skalaxc_*` / `SkalaXC::*` symbols. This
  prevents ODR / symbol-interposition clashes if the host application links its
  own (possibly different) GauXC or LibTorch.

Consumers therefore need **only** this repository's public headers and
`libskalaxc` — not GauXC, LibTorch, or Eigen.

An `XCIntegrator` instance is not safe for concurrent evaluation calls. Use a
separate integrator per calling thread or serialize access to a shared one.
The Python binding releases the GIL during synchronous `eval_exc_vxc` and
`eval_exc_grad` calls because LibTorch autograd must run without it; therefore,
the same restriction applies to Python threads.

### 4. CUDA is optional and private
`SKALAXC_ENABLE_CUDA` (**OFF** by default) builds a private device driver on
GauXC's Scheme1 CUDA work engine. Collocation, density variables, AO assembly,
Pulay terms, and partition-weight derivatives remain on the GPU. SkalaXC owns
the CUDA feature packing, TorchScript/autograd execution, model-potential
unpacking, and Skala-specific VXC transforms. CUDA, GauXC, Torch, and Eigen
types remain behind the same public PIMPL boundary as the host implementation.

The CUDA backend supports UKS `EXC` + scalar/z `VXC` and nuclear gradients for
LDA, GGA, and kinetic-energy-dependent meta-GGA models. Repeated calls reuse
the model owned by the integrator. In an MPI build, every active rank evaluates
its locally owned complete domains with its own CUDA model. Only the final AO
potential, energy, electron-count, and nuclear-gradient values are reduced on
the **runtime communicator**. No CUDA path substitutes `MPI_COMM_WORLD` for the
communicator supplied by the caller.

CUDA support remains experimental: it is disabled by default, and its device
orchestration is not yet a production-supported surface. Enable it only after
compiling and validating the required model, GPU architecture, and runtime
configuration. The complete production-validated backend remains host CPU.

SkalaXC invokes each archive's integrated `get_exc()` method. Autograd provides
the density-feature potentials and `dE/dw`, the derivative of the integrated
energy with respect to each grid weight. GauXC evaluates the partition-weight
response through the chain rule `dE/dR = sum_i (dE/dw_i) (dw_i/dR)`.
This matches the Python Skala execution path and avoids retaining a separate
per-point energy-density model interface.

The bundled TPSS archive remains supported for CUDA `EXC`/`VXC`. TPSS CUDA
gradients are not validated: its current TensorExpr backward trace can exceed
kernel launch resources on `sm_120` at LibTorch's default launch configuration.
TPSS CUDA-gradient validation is deferred until that archive is retraced with
lower register pressure. LDA/PBE host-device gradient parity and neural Skala's
kinetic-density and explicit-geometry gradients remain covered. SkalaXC does
not mutate LibTorch's process-global TensorExpr launch settings.

---

## Prerequisites

- **CMake ≥ 3.21** and a generator (Ninja recommended).
- **C++17 compiler** and a **C** compiler (GCC and Clang 22 are CI-tested);
  **gfortran** only if building the Fortran API (enabled lazily, so
  C/C++-only consumers do not need it).
- **Eigen >= 5.0,<6.0** (found as a private build dependency; Eigen 5.0.1 is
  fetched when no compatible installation is available), **LibTorch** (found
  via `find_package(Torch)`), and **nlohmann_json** (found or fetched).
  **HDF5** and HighFive are required only when `SKALAXC_ENABLE_HDF5=ON`.
- For `SKALAXC_ENABLE_CUDA=ON`: an NVIDIA CUDA toolkit/compiler, a
  CUDA-enabled LibTorch package compatible with that toolkit, and a CUDA
  architecture supported by both. GauXC and the fetched ExchCXX dependency are
  configured with CUDA as part of the same build.
- **GauXC `master` source** — vendored as a **git submodule** at
  `SkalaXC/external/GauXC`, pinned to the exact commit SkalaXC was validated
  against. Initialize it after cloning (see *Getting the source* below). Override
  the location with `-DSKALAXC_GAUXC_SOURCE_DIR=/path/to/GauXC`; if the submodule
  is absent, the same pinned commit is fetched from GitHub as a fallback.

In this repository the dependencies are locked by the root `pixi.toml` and
`pixi.lock`. From the repository root, install the host environment and run the
standard configure, build, and test task:

```bash
pixi install --locked -e skalaxc-host
OMP_NUM_THREADS=4 pixi run -e skalaxc-host skalaxc-test-host
```

`skalaxc-host` is locked for Linux x86-64, Linux ARM64, and macOS ARM64. The
Linux-only `skalaxc-host-clang` environment provides the CI-tested Clang 22
toolchain, and `skalaxc-tools` owns clang-tidy and Doxygen:

```bash
pixi run -e skalaxc-host-clang skalaxc-clang-tidy
pixi run -e skalaxc-tools skalaxc-doxygen
```

CUDA environments use named custom Linux x86-64 platforms so CUDA 12 and 13
remain separate platform solutions in the root lockfile. Pass both the environment
and platform:

```bash
pixi install --locked -e skalaxc-cuda12 -p linux-64-cuda12
pixi install --locked -e skalaxc-cuda13 -p linux-64-cuda13
```

The CUDA host compiler is constrained separately from the unrestricted CPU
toolchain. The root environments currently use these validated combinations:

| Environment | CUDA toolkit | Host compiler |
| --- | --- | --- |
| `skalaxc-cuda12` | `>=12.8,<13` | GCC 14 |
| `skalaxc-cuda13` | `>=13,<14` | GCC 15 |
| `skalaxc-cuda13-clang` | `>=13,<14` | Clang 20 |

The CMake compatibility table follows PyTorch's
[`CUDA_GCC_VERSIONS` and `CUDA_CLANG_VERSIONS`](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/utils/cpp_extension.py).

Use the same `-e` and `-p` pair with `pixi run` when configuring, building, or
testing a CUDA tree.

---

## Build options

| Option | Default | Description |
| --- | --- | --- |
| `SKALAXC_BUILD_FORTRAN` | `ON` | Build the Fortran API (`iso_c_binding`). |
| `SKALAXC_BUILD_TESTS` | `OFF` | Build the test suite. |
| `SKALAXC_BUILD_EXAMPLES` | `OFF` | Build the C/C++/Fortran example programs. |
| `SKALAXC_BUILD_DOCS` | `OFF` | Add the strict project-wide Doxygen target. |
| `SKALAXC_ENABLE_OPENMP` | `ON` | OpenMP threading in the host driver. |
| `SKALAXC_ENABLE_MPI` | `OFF` | MPI support. |
| `SKALAXC_ENABLE_CUDA` | `OFF` | CUDA UKS `EXC`/`VXC`/gradient backend. |
| `SKALAXC_ENABLE_HDF5` | `ON` | HDF5 molecule/basis readers and fixture-driven tests/examples. |
| `SKALAXC_ENABLE_SANITIZERS` | `OFF` | ASan/UBSan instrumentation for GNU/Clang validation builds. |

---

## Compiling

GauXC is a git submodule, so initialize it first (a fresh clone of the `skala`
repo should use `git clone --recursive`, or run this once afterwards):

```bash
git submodule update --init SkalaXC/external/GauXC
```

Configure and build the library, tests, and examples:

```bash
cmake -S . -B build -G Ninja \
  -DSKALAXC_BUILD_TESTS=ON \
  -DSKALAXC_BUILD_EXAMPLES=ON

cmake --build build
```

Install the public headers, libraries, Fortran module, verified baseline
models, and CMake package metadata under a chosen prefix:

```bash
cmake -S . -B build -G Ninja -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --build build
cmake --install build
```

Downstream CMake projects can then use `find_package(SkalaXC CONFIG REQUIRED)`
and link `SkalaXC::skalaxc` or, when enabled, `SkalaXC::skalaxc_fortran`.
Model aliases such as `LDA`, `PBE`, and `TPSS` resolve from
`<prefix>/share/skalaxc/skala_models`; the package also exposes that directory
as `SkalaXC_MODEL_DIR`.

The bundled `skala-1.1.fun` checkpoint contains Skala 1.1 revision 1 for host
evaluation. CUDA builds also bundle the device-compatible revision 1 checkpoint
as `skala-1.1-cuda.fun`. Pass the appropriate installed path or relative filename
to use the learned functional; only `LDA`, `PBE`, and `TPSS` have named aliases.

The installed CMake package also exports the native feature flags,
`SkalaXC_TORCH_VERSION`, `SkalaXC_TORCH_CXX11_ABI`,
`SkalaXC_TORCH_CUDA_VERSION`, and `SkalaXC_CUDA_TOOLKIT_VERSION`. External
language bindings can use these values to reject incompatible runtime
dependencies before loading `libskalaxc`. CUDA-enabled builds require LibTorch
and the compiler toolkit to share a CUDA major compatibility family; minor
toolkit releases within that family may differ.

An existing `.fun` path passed as the model selector always takes precedence.
Otherwise, aliases and relative model names resolve from the directory in the
`SKALAXC_MODEL_PATH` environment variable when it is set, or from the installed
model directory.

Binary packagers can set `SKALAXC_INSTALL_RPATH` to a semicolon-separated list
of package-relative loader paths. The default remains empty for ordinary CMake
consumers; package builds should use relative entries appropriate to their
artifact layout and must not publish staging-prefix RPATHs.

### Building Conda packages

The multi-output recipe in `SkalaXC/recipe` produces `libskalaxc`,
`skalaxc-fortran`, and `skalaxc-python`. Its variant matrix covers `nompi` and
`openmpi`, Python 3.11 through 3.13, CPU packages on Linux x86-64, Linux ARM64,
and macOS ARM64, and CUDA 12/13 packages on Linux x86-64 only. Install the
locked packaging environment and build a representative CPU triplet from the
repository root:

```bash
pixi install --locked -e skalaxc-package
OMP_NUM_THREADS=4 CPU_COUNT=2 pixi run -e skalaxc-package \
  rattler-build build --recipe SkalaXC/recipe/recipe.yaml \
  --target-platform linux-64 \
  --variant mpi=nompi --variant cuda=cpu --variant python=3.12 \
  --output-dir SkalaXC/output-packages --log-style plain
```

Use `linux-aarch64` or `osx-arm64` on the matching native runner. CUDA package
builds select `cuda12` or `cuda13` while retaining `linux-64` as the target
platform. `SkalaXC/pixi.toml` registers the package with the root Pixi workspace,
so package builds and future Python tests share the root `pixi.lock` and environments.

### Querying the library version

The public APIs return the semantic version of the loaded SkalaXC library as a
plain string, leaving logging and output control to the caller:

```cpp
std::cout << "SkalaXC " << SkalaXC::version() << '\n';
```

```c
printf("SkalaXC %s\n", skalaxc_version());
```

```fortran
write (*, '(A)') 'SkalaXC '//skalaxc_version()
```

The C string and the storage referenced by the C++ view have static storage
duration and must not be modified or freed. The Fortran function returns an
allocatable character value.

### Fortran handle ownership

Each Fortran pipeline handle uniquely owns its underlying C object and releases
it automatically when finalized. Handles are non-copyable: intrinsic-looking
assignment terminates with an ownership error. Transfer ownership explicitly
with `call destination%move_from(source)`; the source becomes invalid and may
be checked with `source%is_valid()`. Calling a creation method on an already
valid handle replaces its object only after the new construction succeeds.

For a CUDA build, select the architecture explicitly. For example, Blackwell
(`sm_120`) with a CUDA-enabled LibTorch installation can be configured with:

```bash
cmake -S . -B build-cuda -G Ninja \
  -DSKALAXC_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DBLAS_LIBRARIES=$CONDA_PREFIX/lib/libblas.so

cmake --build build-cuda
```

To validate and generate project-wide API and implementation documentation,
install Doxygen 1.9 or newer and enable the documentation target:

```bash
cmake -S . -B build-doxygen -G Ninja \
  -DSKALAXC_BUILD_DOCS=ON
cmake --build build-doxygen --target skalaxc_doxygen
```

All SkalaXC-owned C, C++, CUDA, and Fortran sources are scanned; build trees and
the external GauXC source tree are excluded. Warnings for undocumented
entities, incomplete parameter documentation, and malformed Doxygen commands
fail the target after the full scan. Generated HTML is written to
`build-doxygen/doxygen/html` and is not installed or published with the public
Skala documentation.

Create a device runtime and use `ExecutionSpace::Device` consistently for the
load balancer, molecular weights, and integrator factories:

```cpp
SkalaXC::DeviceRuntimeSettings settings;
settings.device_id = 0;
settings.memory_fraction = 0.75;
SkalaXC::RuntimeEnvironment runtime{
  SKALAXC_MPI_CODE(comm, ) settings};

SkalaXC::LoadBalancerFactory load_balancer_factory{
  SkalaXC::ExecutionSpace::Device, "Default"};
SkalaXC::MolecularWeightsFactory weights_factory{
  SkalaXC::ExecutionSpace::Device, "Default", {}};
SkalaXC::XCIntegratorFactory<Matrix> integrator_factory{
  SkalaXC::ExecutionSpace::Device};
```

The C API uses `skalaxc_device_runtime_environment_create` and
`SkalaXC_ExecutionSpace_Device`; the Fortran API passes
`skalaxc_device_runtime_settings_default()` to
`skalaxc_runtime_environment_create` and uses
`skalaxc_executionspace%device`.

### Complete-domain batching

Host and CUDA model calls use complete, dense atomic domains. Conservative
batching is the default and evaluates one domain per model call. Aggressive
batching groups all locally owned domains having the same exact grid size into
one call; it never mixes sizes, pads between domains, or presents ragged model
inputs.

```cpp
SkalaXC::XCIntegratorFactory<Matrix> factory{
  SkalaXC::ExecutionSpace::Host,
  SkalaXC::TimingSettings{},
  SkalaXC::DomainBatchMode::Aggressive};
```

The C API uses `skalaxc_integrator_settings_t` with
`skalaxc_xc_integrator_create_with_settings`. The Fortran `xc%create` method
accepts the optional `domain_batch_mode` argument, for example
`skalaxc_domainbatchmode%aggressive`.

Conservative mode minimizes model activation memory but does not guarantee that
one unusually large domain fits. Aggressive mode batches every compatible local
domain and may use substantially more memory. Neither mode predicts available
memory or retries after an allocation failure. Automatic memory-aware batching
is not implemented. `DeviceRuntimeSettings::memory_fraction` controls GauXC's
CUDA arena and does not cap Torch model memory.

### Lightweight diagnostics

Integrator-owned timing and structured setup diagnostics are always enabled.
Host timing uses `std::chrono::steady_clock` around model loading and coarse
evaluation phases; no task-level timers, MPI collectives, or synchronization
calls are added when a host snapshot is read. Retrieve the rank-local snapshot
directly from any integrator:

```cpp
SkalaXC::XCIntegratorFactory<Matrix> factory{
  SkalaXC::ExecutionSpace::Host};
auto integrator = factory.get_instance(functional, load_balancer);

const auto snapshot = integrator.diagnostics();
const auto forward = snapshot.timing(SkalaXC::TimingMetric::ModelForward);
integrator.reset_diagnostics();
```

Snapshots contain fixed phase IDs, call counts, communicator/backend setup,
OpenMP and CUDA settings, local task/point/atom ranges, configured model-batch
geometry, and processed model-batch/domain counts. Empty MPI ranks report zero
for workload ranges. Host snapshots additionally contain last and cumulative
phase timings. Resetting clears evaluation values while preserving all setup
topology and the integrator's model-load timing. Load-balancer construction and
molecular weighting precede integrator construction and are therefore not
included. The C API exposes the same fixed data as POD structs in
`<skalaxc/c/diagnostics.h>` through
`skalaxc_xc_integrator_create_with_timing`,
`skalaxc_xc_integrator_get_diagnostics`, and
`skalaxc_xc_integrator_reset_diagnostics`. The Fortran module provides the
corresponding interoperable types, optional `timing_settings` argument to
`xc%create`, and `xc%diagnostics`/`xc%reset_diagnostics` methods.

Set `TimingSettings::debug_logging=true` to emit a human-readable trace to
`stderr`. Every runtime-communicator rank writes independently and prefixes
each complete line with its backend, local rank/size, and phase. The setup
report identifies the selected model, model features, parallel configuration,
local workload, batching policy, every exact-size model batch, and completed
model-load timing. Evaluation reports contain density-matrix traces/norms/maxima
and final EXC, potential, or gradient summaries. Matrix summaries add
enabled-only $O(n_\mathrm{bf}^2)$ host work. The textual format is intended for
interactive debugging and is not a stable parsing interface; use
`DiagnosticsSnapshot` for programmatic consumers.

CUDA snapshots currently report evaluation calls, local tasks/points, model
batches, domains, setup topology, and model-load timing. With debug logging
enabled, CUDA evaluation timing is reported as unavailable. The planned CUDA
event collector will use query-only harvesting:
completed events contribute device timings and incomplete events remain
pending. `TimingSettings::verbose=true` is reserved for waiting for outstanding
events when a snapshot is read. CUDA event collection is not implemented yet;
host snapshots never require device synchronization.

### Troubleshooting: BLAS selection

GauXC's linalg auto-search may pick up a **system** BLAS (e.g.
`/usr/lib/x86_64-linux-gnu/libopenblas.so`) ahead of the one in your conda
environment. On toolchains where the system multiarch headers
(`/usr/include/x86_64-linux-gnu`) are incompatible with the conda compiler this
shows up as a `libstdc++`/`pthread` error while compiling GauXC (e.g.
`cannot convert … to 'unsigned int' … __GTHREAD_COND_INIT`). Pin the
environment's BLAS/LAPACK to avoid it:

```bash
cmake -S . -B build -G Ninja \
  -DBLAS_LIBRARIES=$CONDA_PREFIX/lib/libblas.so
```

---

## Running the tests

```bash
ctest --test-dir build --output-on-failure
```

CTest discovers the available tests for the enabled language backends and
execution spaces. See [`tests/README.md`](tests/README.md) for the suite's
assurance layers, feature coverage, exact focused commands, fixture provenance,
and numerical tolerance rationale.

CUDA builds also run TPSS device EXC/VXC and PBE device-gradient evaluations
through the pure C and Fortran black-box tests.

The black-box tests double as ABI-isolation proofs: they are given no GauXC
include dirs and do not link GauXC/LibTorch, so a successful compile+link is
itself part of the test.

---

## Running the examples

Enable `-DSKALAXC_BUILD_EXAMPLES=ON` with `SKALAXC_ENABLE_HDF5=ON`, then run any of the three consumer programs
against an HDF5 file containing `/MOLECULE`, `/BASIS`, `/DENSITY_SCALAR`,
`/DENSITY_Z`. The bundled fixtures work out of the box:

```bash
export SKALAXC_MODEL_PATH="$PWD/build/data/skala_models"
./build/examples/skalaxc_eval_cpp     tests/ref_data/skala_he_def2qzvp_pbe_uks.hdf5  PBE
./build/examples/skalaxc_eval_c       tests/ref_data/skala_he_def2qzvp_lda_uks.hdf5  LDA
./build/examples/skalaxc_eval_fortran tests/ref_data/skala_he_def2qzvp_tpss_uks.hdf5 TPSS
```

Each prints `nbf`, the ML `EXC`, and a summary of the potential. The `model`
argument is `"LDA"`, `"PBE"`, `"TPSS"` (resolved using the model lookup rules
above), or a path to a `.fun` model. The examples bind that model when creating
the integrator; evaluation calls provide only densities and output storage.

---

## Verifying ABI isolation

```bash
# Only skalaxc_* / SkalaXC::* symbols are exported (no GauXC/torch):
nm -D --defined-only build/src/libskalaxc.so.0.1.0 | grep -Ev 'skalaxc|SkalaXC'   # → empty

# A pure-C consumer links only the SkalaXC C surface:
readelf -d build/examples/skalaxc_eval_c | grep NEEDED
# → libskalaxc.so.0, libhdf5.so.320, libm.so.6, libc.so.6
```

---

## Status & limitations

- **Host and CUDA UKS** evaluation (`EXC` + scalar/z `VXC` and XC energy
  gradients) is validated across the C, C++, and Fortran APIs for LDA, GGA,
  and kinetic-energy-dependent meta-GGA models.
- **RKS** is not implemented (UKS only).
- Laplacian-dependent meta-GGA models are not supported.
- CUDA MPI inference is rank-local over complete atomic domains, followed by
  final reductions on the supplied runtime communicator. Every rank must
  select an accessible CUDA device; mapping ranks to devices is the caller's
  responsibility.
