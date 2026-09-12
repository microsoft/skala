# AGENTS.md

This file gives coding agents the repository-specific context and commands needed to work safely in
Skala. Run commands from the repository root unless a section says otherwise. Treat `README.md`,
`SkalaXC/README.md`, and the relevant build files as the source of truth when this guide falls behind.

## Repository overview

Skala is a neural network exchange-correlation (XC) functional for density functional theory. This
repository has two distinct implementation surfaces:

1. The Python package `skala`, with PyTorch implementations and PySCF, GPU4PySCF, ASE, and Azure AI
   Foundry integrations.
2. `SkalaXC`, a standalone C++17 shared library that performs host-side ML XC evaluation through its
   own ABI-isolated C++, C, and Fortran APIs. It reuses GauXC internals and LibTorch privately.

The recommended Python functional is `skala-1.1`; `skala-1.0` remains available for compatibility.
SkalaXC's bundled model selectors are `LDA`, `PBE`, and `TPSS`, or an explicit `.fun` path.

| Path | Description |
|------|-------------|
| `skala/` | Published ASE, PySCF, and GPU4PySCF runtime plus tests |
| `model/` | Trainable model definition, tests, and LibTorch/FTorch examples |
| `SkalaXC/` | Standalone C++17 library plus C and Fortran bindings |
| `benchmark/` | Benchmark runner, reference data, report tooling, and tests |
| `website/` | Main Sphinx site |
| `.github/workflows/` | CI workflows (test, docs) |

## Development environment

1. **Python version**: 3.11–3.13 (target 3.11 for compatibility tooling).
2. **Environment setup** (Pixi 0.75):
   ```bash
  pixi install --locked -e default
   ```
3. **Pre-commit hooks** (required before committing):
   ```bash
  pixi run -e default pre-commit install
  pixi run -e default pre-commit run --all-files
   ```

## Code style & linting

- **Formatter/linter**: Ruff (`ruff format`, `ruff check --fix --select I`).
- **Type checking**: mypy in strict mode. Ignore missing type information only for explicitly
  named untyped dependencies in `pyproject.toml`; do not use global `--ignore-missing-imports`.
- Line length: 100 characters (Black-compatible).
- Imports sorted via Ruff's isort rules.
- Generated build, coverage, and documentation output is excluded from static analysis.

When editing code:
- Run `pre-commit run ruff-format --files <file>` and
  `pre-commit run ruff --files <file>` before committing.
- Add type hints to new public functions and classes.
- Use Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections.

## Testing

- Framework: pytest with pytest-cov.
- Run tests:
  ```bash
  OMP_NUM_THREADS=4 pixi run -e default pytest -v --doctest-modules \
    --cov=skala --cov-report=xml --cov-report=term-missing --cov-report=html \
    --durations=50 --durations-min=1.0 skala/src/skala/ skala/tests/
  ```
- Keep tests beside their owning component with a `test_` prefix.
- Use fixtures for expensive setup (molecule construction, model loading).
- Prefer fast unit tests; integration tests that run DFT should be marked or placed separately.

## Documentation

- Engine: Sphinx with myst-nb (executes notebooks during build).
- Build locally:
  ```bash
  pixi run -e docs sphinx-build -b html website website/_build/html
  touch website/_build/html/.nojekyll
  ```
- Notebooks in `website/` should be executable with a 5-minute timeout.
- Use reStructuredText for standalone pages; Jupyter notebooks for tutorials.

## Python architecture

- **Runtime functional API** (`skala/src/skala/functional/`): Loads traced checkpoints and defines
  traditional functionals and the runtime interface.
- **Model definition** (`model/src/skala_model/`): Defines trainable layers and the
  enhancement-factor network; it is not part of release artifacts.
- **PySCF integration** (`skala/src/skala/pyscf/`): Custom `numint` module and `SkalaKS` class hook the
  model into PySCF's DFT machinery.
- **ASE calculator** (`skala/src/skala/ase/`): Provides an ASE-compatible calculator for energy/force
  evaluations and geometry optimizations.

## Common commands

| Task | Command |
|------|---------|
| Format code | `pixi run -e default pre-commit run ruff-format --all-files` |
| Lint code | `pixi run -e default pre-commit run --all-files` |
| Run runtime tests | `OMP_NUM_THREADS=4 pixi run -e default pytest -v --doctest-modules --cov=skala --cov-report=xml --cov-report=term-missing --cov-report=html --durations=50 --durations-min=1.0 skala/src/skala/ skala/tests/` |
| Run component tests | `OMP_NUM_THREADS=4 pixi run -e default pytest -v model/tests/test_model.py model/tests/test_utils.py benchmark/tests/` |
| Build docs | `pixi run -e docs sphinx-build -b html website website/_build/html && touch website/_build/html/.nojekyll` |
| Type check | `pixi run -e default mypy skala/src model/src benchmark/src` |

## SkalaXC architecture

Read `SkalaXC/README.md` before changing the C++ library. The following constraints are intentional:

- The complete, validated backend is host CPU evaluation of unrestricted (UKS) `EXC`, `VXC`, and XC
  energy gradients. RKS is not implemented. CUDA kernels are scaffolded behind
  `SKALAXC_ENABLE_CUDA=OFF`; the device orchestration is incomplete and must not be described as
  production-ready without GPU compilation and validation.
- GauXC is a source-tree implementation dependency, not part of the public API. Do not add Skala or
  OneDFT APIs to the GauXC submodule, and do not route evaluation through GauXC's public
  `XCIntegrator::eval_exc_vxc*` entry points. Reuse its lower-level grid, load-balancing,
  collocation, local-work, molecular-weight, and HDF5 machinery.
- No GauXC, LibTorch, ATen, ExchCXX, IntegratorXX, or Eigen type may cross the public SkalaXC
  boundary. Keep C++ implementation details behind PIMPL, keep the C API opaque/POD-only, and keep
  Fortran on `iso_c_binding` over the C API. A native `MPI_Comm` is the deliberate exception in MPI
  builds.
- Public matrices and raw buffers are `double`, `nbf x nbf`, and column-major. Gradients contain
  exactly `3 * natoms` values in atom-major xyz order. Preserve both allocating and caller-owned
  output overloads.
- `skalaxc::XCIntegrator` owns its selected TorchScript model for its lifetime. Avoid process-wide
  model caches and reloads during evaluation.
- `libskalaxc` statically embeds private dependencies and exports only the SkalaXC surface. Do not
  weaken hidden visibility, the linker version script, or `--exclude-libs,ALL`.
- Use `skalaxc_` for public API and consumer artifacts. Use `skala_` for private model/driver code and
  white-box tests.
- In MPI code, use the `RuntimeEnvironment` communicator for collectives. Do not hardcode
  `MPI_COMM_WORLD` below the API construction boundary.

Public API changes normally require coordinated updates to the C++, C, and Fortran surfaces, their
black-box tests, examples, and `SkalaXC/README.md`. Black-box tests must continue to compile without
private GauXC or LibTorch include directories and link only the public SkalaXC target plus their own
fixture-reading dependency.

## Building SkalaXC

SkalaXC needs CMake 3.21+, C++17, LibTorch, HDF5, and nlohmann_json. Fortran is enabled by default and
needs `gfortran`. The root `skalaxc-host` environment provides the locked host toolchain on Linux
x86-64, Linux ARM64, and macOS ARM64.

Initialize the pinned GauXC source once after cloning:

```bash
git submodule update --init SkalaXC/external/GauXC
```

Configure a host development build. Use a fresh build directory after dependency-source or major
option changes so stale CMake cache values do not select an old GauXC tree.

```bash
pixi install --locked -e skalaxc-host
pixi run -e skalaxc-host skalaxc-configure-host
pixi run -e skalaxc-host skalaxc-build-host
```

Useful options are `SKALAXC_BUILD_FORTRAN`, `SKALAXC_ENABLE_OPENMP`, `SKALAXC_ENABLE_MPI`, and
`SKALAXC_ENABLE_CUDA`. CUDA defaults to off and currently covers kernels only. If GauXC's linalg
search selects an incompatible system OpenBLAS, reconfigure with:

```bash
-DBLAS_LIBRARIES=$CONDA_PREFIX/lib/libblas.so
```

## Testing SkalaXC

Build and run the narrowest relevant Catch2 tag first. The focused command below is the established
workflow for the reorder and array-view code; substitute another tag when appropriate.

```bash
pixi run -e skalaxc-host cmake --build SkalaXC/build-pixi-host \
  --target skalaxc_unit_test --parallel 4
OMP_NUM_THREADS=4 pixi run -e skalaxc-host \
  SkalaXC/build-pixi-host/tests/skalaxc_unit_test "[reorder]"
OMP_NUM_THREADS=4 pixi run -e skalaxc-host \
  SkalaXC/build-pixi-host/tests/skalaxc_unit_test "[array-view]"
```

Run a named CTest slice or the complete suite after the focused test:

```bash
OMP_NUM_THREADS=4 pixi run -e skalaxc-host ctest --test-dir SkalaXC/build-pixi-host \
  -R 'skalaxc_(host_parity|public_api)' --output-on-failure
OMP_NUM_THREADS=4 pixi run -e skalaxc-host skalaxc-test-host
```

The numerical gradient test is intentionally slower than the other host tests. MPI changes require
an MPI-enabled configure and a real multi-rank runtime test; a serial build does not validate the MPI
collective path. CUDA changes require an actual CUDA compiler and GPU validation.

Run the public examples against the bundled fixtures when changing API or integration behavior:

```bash
SkalaXC/build-pixi-host/examples/skalaxc_eval_cpp \
  SkalaXC/tests/ref_data/skala_he_def2qzvp_pbe_uks.hdf5 PBE
SkalaXC/build-pixi-host/examples/skalaxc_eval_c \
  SkalaXC/tests/ref_data/skala_he_def2qzvp_lda_uks.hdf5 LDA
SkalaXC/build-pixi-host/examples/skalaxc_eval_fortran \
  SkalaXC/tests/ref_data/skala_he_def2qzvp_tpss_uks.hdf5 TPSS
```

For C++ formatting, use the checked-in style:

```bash
clang-format -i SkalaXC/src/path/to/file.cxx SkalaXC/tests/path/to/test.cxx
```

## Change discipline

- Start with a focused regression test for the touched behavior, then run the broader relevant suite.
- Keep Python and SkalaXC changes separated unless the contract genuinely spans both implementations.
- Do not edit generated headers under `SkalaXC/build/include`; their templates live under
  `SkalaXC/include` and are configured by CMake.
- Do not commit build products, downloaded models, environment files, or CMake cache output.
- Update user-facing documentation when installation, model selection, public APIs, or limitations
  change.
- Keep commits focused and do not modify the GauXC submodule unless the task explicitly updates the
  pinned upstream dependency.

## Project links

- Issues: https://github.com/microsoft/skala/issues
- Documentation: https://microsoft.github.io/skala
- Security policy: `SECURITY.md`
- Contribution guide and code of conduct: `CONTRIBUTING.md`
