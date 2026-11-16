# SkalaXC tests

The C layer adds a pure-C numerical consumer, dependency isolation checks, and
malformed HDF5 density-shape regression tests. Installed consumers cover both
C and C++, while the export allowlist admits only those public surfaces.

Fortran coverage adds numerical and installed-module consumers, scalar and
rank-1/rank-2 assignment rejection for all eight handle types, and exact-once
array finalization tests. MPI builds additionally run the Fortran numerical
test on two ranks. Disable the optional module with `SKALAXC_BUILD_FORTRAN=OFF`
when testing a C/C++-only installation.

The Catch2 units protect array views, atomic-domain ordering, task limits,
model interfaces, diagnostics, and communicator behavior. Host numerical
tests cover LDA/PBE/TPSS energy, both potential channels, nuclear gradients,
finite-difference gradients, batching, and OpenMP invariance.

`skalaxc_public_api` is a black-box C++ consumer linking only the public target
and its own Eigen/HDF5 fixture readers. `tests/installed` independently checks
the exported CMake package. ABI tests reject leaked private dependency symbols.
CUDA tests, when enabled, cover device parity, batching, streams, idle ranks,
and subcommunicators. MPI tests require actual multi-rank execution.

Run the narrowest Catch2 tag first, then CTest:

```bash
OMP_NUM_THREADS=4 SkalaXC/build/tests/skalaxc_unit_test "[task-batching]"
OMP_NUM_THREADS=4 ctest --test-dir SkalaXC/build --output-on-failure
```

Native CI covers serial and MPI host builds. The serial job exercises the
GauXC FetchContent fallback; MPI uses the pinned submodule. Parser sources
and fixtures use the resolved source directory for both routes. Existing
GauXC, LibTorch, and FTorch workflows remain separate and unchanged.
