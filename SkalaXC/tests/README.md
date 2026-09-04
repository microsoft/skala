# SkalaXC test suite

This directory contains the first-party tests for the standalone SkalaXC
library. The suite deliberately repeats some numerical behavior across private
drivers and the public C++, C, and Fortran APIs: those checks protect different
linkage, ownership, layout, and exception/status-code boundaries. They should
not be merged merely because they evaluate the same model and fixture.

## Assurance layers

| Layer | Primary tests | Contract protected |
| --- | --- | --- |
| Private units | `*_unit_test.cxx`, tags such as `[reorder]`, `[array-view]`, `[diagnostics]` | Reordering, semantic array views, atomic-domain assignment, model metadata, diagnostics, and MPI wrappers. |
| Build features | CTest `skalaxc_config.hdf5_feature` | Generated public/internal headers and installed-package metadata agree for HDF5-enabled and HDF5-disabled builds. |
| Host numerical integration | `skala_host_test.cxx`, `skala_openmp_test.cxx`, `skala_stress_test.cxx` | LDA/PBE/TPSS EXC, scalar/z VXC, gradients, batching, OpenMP consistency, numerical derivatives, repeated evaluation, and bounded memory growth. |
| Traditional-functional parity | `skala_traditional_integration_test.cxx` | Public SkalaXC results against GauXC/ExchCXX for seven molecules through chlorine, three models, random UKS densities, and serial/MPI execution. |
| CUDA parity | `skala_device_test.cxx` and `test_pyscf_skalaxc_gpu_parity.py` | Native LDA/PBE host-device gradient parity, TPSS EXC/VXC, lightweight batching/stream/MPI infrastructure checks, and dedicated GPU4PySCF–SkalaXC neural-gradient parity. TPSS CUDA-gradient coverage is deferred until its TensorExpr trace is retraced for lower sm_120 register pressure. |
| Public C++ consumer | CTest `skalaxc_public_api` | Installed-style header use, PIMPL isolation, owning outputs, state/error contracts, gradients, batching, and diagnostics. |
| Public C consumer | CTest `skalaxc_c_api` | Opaque handles, status codes, column-major caller-owned buffers, validation, gradients, and diagnostics. |
| Public Fortran consumer | CTest `skalaxc_fortran` and `skalaxc_fortran_assignment.*` | `iso_c_binding`, move-only handle ownership, caller-owned arrays, diagnostics, and gradients. |
| Binary boundary | CTest `skalaxc_abi.exported_symbols` and `skalaxc_abi.c_consumer_dependencies` | Export allowlist and absence of private GauXC/LibTorch dependencies from a C consumer. |

The C and Fortran suites validate the conservative batching mode through their
public bindings. Positive aggressive-batching equivalence is covered at the
private host/device and public C++ layers; it is not repeated in each language
binding.

HDF5 support is enabled by default. With `SKALAXC_ENABLE_HDF5=OFF`, fixture-
driven host/device tests and the C, C++, and Fortran numerical consumer tests
are not built. Constructed-system units, traditional-functional parity, the
Fortran ownership tests, and the exported-symbol ABI check remain available.

## Shared C++ setup

`test_utils.hpp` and `test_utils.cxx` build as the private
`skalaxc_test_utils` static library. They own setup that is data construction
rather than behavior under test: H2/STO-3G systems, public SkalaXC molecule and
basis loading, scalar/z density loading, standard molecular-grid construction,
and normalized matrix errors. The utility target depends only on public
SkalaXC, Eigen, optional HighFive, and MPI interfaces, so the black-box C++ consumer can
reuse it without gaining access to `skalaxc_core`, GauXC, or LibTorch.

Runtime selection, load balancing, molecular-weight modification, and
integrator construction remain in each test. Their settings and stage ordering
are part of the contracts those tests exercise. White-box tests that directly
use GauXC types also keep their fixture setup separate from the public-type
helpers.

## Running focused tests

Build and run from the repository root. Set `SKALAXC_MODEL_PATH` when invoking
the Catch executable directly; CTest registrations set it automatically.
The commands below use the locked host environment and the default Pixi build
tree created by `skalaxc-configure-host`.

```bash
pixi run -e skalaxc-host skalaxc-configure-host
pixi run -e skalaxc-host cmake --build SkalaXC/build-pixi-host \
  --target skalaxc_unit_test --parallel 2

OMP_NUM_THREADS=4 pixi run -e skalaxc-host \
  SkalaXC/build-pixi-host/tests/skalaxc_unit_test "[host-reference-integration]"
OMP_NUM_THREADS=4 pixi run -e skalaxc-host \
  SkalaXC/build-pixi-host/tests/skalaxc_unit_test "[host-gradient]"
OMP_NUM_THREADS=4 pixi run -e skalaxc-host \
  SkalaXC/build-pixi-host/tests/skalaxc_unit_test "[gradient-numerical]"
OMP_NUM_THREADS=4 pixi run -e skalaxc-host \
  SkalaXC/build-pixi-host/tests/skalaxc_unit_test "[openmp]"
OMP_NUM_THREADS=4 pixi run -e skalaxc-host \
  SkalaXC/build-pixi-host/tests/skalaxc_unit_test "[traditional-integration]"

OMP_NUM_THREADS=4 pixi run -e skalaxc-host ctest \
  --test-dir SkalaXC/build-pixi-host -R '^skalaxc_public_api$' --output-on-failure
OMP_NUM_THREADS=4 pixi run -e skalaxc-host ctest \
  --test-dir SkalaXC/build-pixi-host -R '^skalaxc_(c_api|fortran)$' --output-on-failure
OMP_NUM_THREADS=4 pixi run -e skalaxc-host ctest \
  --test-dir SkalaXC/build-pixi-host --output-on-failure
```

Run CUDA-tagged tests with the matching `skalaxc-cuda12` or
`skalaxc-cuda13` environment and custom platform. The focused `[stream]` case
checks that SkalaXC always uses GauXC's master CUDA stream internally and
restores a non-default caller Torch stream after successful and failed
evaluations.

`[gradient-numerical]` is tagged `[.slow]`, so it runs only when selected
explicitly. Cases tagged `[mpi-only]` are excluded from single-process Catch
discovery and are available through these MPI-enabled CTest registrations:

| CTest name | Ranks | Coverage |
| --- | ---: | --- |
| `skalaxc_unit_mpi.subcommunicator` | 3 | MPI wrapper and model-grid subcommunicator units. |
| `skalaxc_unit_mpi.host_subcommunicator` | 4 | Full host TPSS EXC/VXC/gradient isolation between two communicators. |
| `skalaxc_unit_mpi.atomic_domain_ownership` | 3 | Exactly one owner per complete atomic domain. |
| `skalaxc_unit_mpi.model_broadcast` | 3 | Runtime-rank-zero model loading, broadcast, and error propagation. |
| `skalaxc_unit_mpi.traditional_integration` | 3 | Replicated SkalaXC/GauXC parity. |
| `skalaxc_unit_mpi.cuda_idle_rank` | 3 | CUDA evaluation when one rank owns no atomic domain. |
| `skalaxc_unit_mpi.cuda_subcommunicator` | 4 | CUDA isolation between two runtime communicators. |
| `skalaxc_fortran_mpi` | 2 | Fortran binding under MPI. |

For example:

```bash
OMP_NUM_THREADS=4 pixi run -e skalaxc-host ctest \
  --test-dir SkalaXC/build-pixi-host \
  -R '^skalaxc_unit_mpi\.host_subcommunicator$' --output-on-failure
```

Multi-rank registrations set `OMP_NUM_THREADS=1` to avoid oversubscription.
Use the project's normal OpenMP thread setting for serial tests.

## Fixtures and provenance

The three SkalaXC-owned HDF5 files are golden He/def2-QZVP UKS snapshots. Each
contains `/MOLECULE`, `/BASIS`, `/DENSITY_SCALAR`, `/DENSITY_Z`, `/EXC`,
`/VXC_SCALAR`, and `/VXC_Z`; the density and potential matrices are `30 x 30`.
They were moved from the former GauXC/OneDFT fixture paths in commit `a841c9f`.
The files contain no embedded generator version or provenance attributes, so
their checked-in hashes are the artifact identities:

| Fixture | Bytes | SHA-256 |
| --- | ---: | --- |
| `ref_data/skala_he_def2qzvp_lda_uks.hdf5` | 41,304 | `bce8289eac18173575cf867178f8c6751e5095b65dc6fd975497fb9a05b6b76a` |
| `ref_data/skala_he_def2qzvp_pbe_uks.hdf5` | 41,304 | `e1af1d35c5686d6ae448408315b0ebc93f27e96cf74bb297640617bf5eab9ca2` |
| `ref_data/skala_he_def2qzvp_tpss_uks.hdf5` | 41,304 | `1c5742160700ab7f899f7ba60d399b34530c5144ddeacaec19a82c5ef40e349a` |

Gradient, batching, OpenMP, binding, and load-balancer tests also use
`external/GauXC/tests/ref_data/h2o2_def2-tzvp.hdf5`. It is owned by the pinned
GauXC submodule; at GauXC revision
`554bef7495b2a93f16a2fdedabf4e1cdcb3a1faf`, its SHA-256 is
`3d0eb1e98d02b7fe8892f1bde4fbdccd7c0686cbdc17746ebe8fa25cc8ff8302`.
Small H2 gradient and communicator tests construct their geometry, STO-3G
basis, and density directly in the test source.

If a golden fixture is regenerated, record the generator, dependency versions,
and command in this file; update the checksum and the numerical expectation in
the same change. A checksum-only replacement is not sufficient provenance.

## Numerical tolerances

Tolerance families reflect the comparison being made:

| Comparison | Current thresholds | Rationale |
| --- | --- | --- |
| Golden host snapshots | EXC relative `1e-5`; scalar VXC norm/nbf `1e-7`; z VXC norm/nbf `1e-10` | Mirrors the historical GauXC/Skala reference checks while retaining a tighter spin-potential discriminator. |
| Traditional functionals | EXC relative `1e-5`; each VXC relative `2e-5`; gradient relative `1e-4` | The neural baselines reproduce conventional functionals to about `1e-6` in measured worst cases; random densities and gradient accumulation require headroom. |
| CUDA versus host | EXC `1e-10`; scalar VXC norm/nbf `1e-7`; z VXC norm/nbf `1e-10`; maximum gradient component `1e-6` | Covers different reduction order and GPU kernel/autograd implementations without relaxing functional-level regressions. |
| OpenMP thread counts | EXC and VXC normalized errors `1e-12`; gradient RMS/component count `1e-10` | Both runs use the same host implementation; only work partition and reduction order change. |
| Host batching | EXC `1e-12`; VXC `1e-11`; gradient `1e-10` | Conservative and aggressive modes differ only in model batching. |
| Numerical gradient | Ridders estimate error below `1e-7`; analytic directional derivative within absolute `1e-6` | Separately constrains finite-difference convergence and the implemented analytic gradient. |

The constants in the test source are authoritative. Update this table whenever
a threshold changes, and document measured evidence rather than widening a
tolerance solely to make one platform pass.

## Randomized parity controls

`skala_traditional_integration_test.cxx` chooses a random density seed on
runtime rank zero and broadcasts it. CI fixes the seed. These variables support
local diagnosis and exact replay:

| Variable | Effect |
| --- | --- |
| `SKALAXC_TEST_SEED=<integer>` | Generate the same random UKS densities. |
| `SKALAXC_TEST_DENSITY_DIR=<dir>` | Replay `density_<molecule>_{scalar,z}.mtx` files emitted after a mismatch. |
| `SKALAXC_TEST_VERBOSE=1` | Print per-molecule/model numerical errors on rank zero. |

On failure, preserve the printed seed in the issue or pull request. Store a
replay density only when it adds a stable regression case that a seed cannot
reproduce.

## Adding or merging tests

- Prefer a focused unit test for private transformations and an integration
  test only where ownership, communication, numerical assembly, or ABI behavior
  crosses a boundary.
- Keep public C++, C, and Fortran checks separate when compilation, linking,
  storage ownership, or error translation differs.
- Tag a Catch case `[mpi-only]` when it requires multiple ranks, then add one
  explicit CTest registration with the required rank count.
- Do not add aggregate CTests that rerun already discovered Catch cases. Use a
  direct Catch tag expression in focused CI jobs instead.
