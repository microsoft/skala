# SkalaXC Python bindings

This package binds the public SkalaXC C++ API directly with nanobind. It does
not route through the C or Fortran wrappers. Density and potential matrices are
CPU NumPy arrays with shape `(nbf, nbf)` and are converted to column-major
`float64` storage. Gradients are returned as atom-major `(natoms, 3)` arrays.
The selected native SkalaXC build may execute internally on the host or CUDA.

## Binary and source packages

The build is package-manager neutral. The selected Python interpreter supplies
Torch, while `SkalaXC_DIR` identifies an exact installed native SkalaXC CMake
package. The binding CMake project never searches for or links Torch itself.
Binary artifacts carry `_skalaxc`, their matching `libskalaxc`, model files,
and native build metadata. They require NumPy and the supported Torch 2.13
minor line from the target environment.

A source build first installs SkalaXC into an isolated prefix using the
`Torch_DIR` and C++ ABI reported by `tools/torch_config.py`, then builds this
directory with `SkalaXC_DIR` set to that prefix's CMake package directory.
This contract is the same for pip, uv, conda, pixi, or another frontend.
Binary recipes also set `SKALAXC_INSTALL_RPATH` while building the native stage:
wheel layouts use `$ORIGIN/../torch/lib`, while conda/pixi layouts use
`$ORIGIN/../../..` to reach the environment's `lib` directory. Multiple
relative entries may be supplied as a CMake list. Packaged binaries must not
retain staging-prefix paths.

## Evaluation

`Molecule.from_hdf5()` and `BasisSet.from_hdf5()` are present when the native
library was built with `SKALAXC_ENABLE_HDF5=ON` (the default), reported as
`skalaxc.HDF5_ENABLED`. HDF5-disabled builds construct molecules and basis sets
through `Atom`, `Shell`, and `append()` instead.

```python
import h5py
import numpy as np
import skalaxc

molecule = skalaxc.Molecule.from_hdf5("system.hdf5")
basis = skalaxc.BasisSet.from_hdf5("system.hdf5")
grid = skalaxc.MolGridFactory.create_default(molecule)
runtime = skalaxc.RuntimeEnvironment()
load_balancer = skalaxc.LoadBalancerFactory(
	skalaxc.ExecutionSpace.HOST
).get_instance(runtime, molecule, grid, basis)
weights = skalaxc.MolecularWeightsFactory(
	skalaxc.ExecutionSpace.HOST
).get_instance()
weights.modify_weights(load_balancer)
integrator = skalaxc.XCIntegratorFactory(
	skalaxc.ExecutionSpace.HOST
).get_instance(skalaxc.Functional("PBE"), load_balancer)

with h5py.File("system.hdf5") as handle:
	scalar = np.asfortranarray(handle["/DENSITY_SCALAR"])
	spin = np.asfortranarray(handle["/DENSITY_Z"])

energy, scalar_potential, spin_potential = integrator.eval_exc_vxc(
	scalar, spin
)
gradient = integrator.eval_exc_grad(scalar, spin)
```

The binding releases the Python GIL while `eval_exc_vxc` and `eval_exc_grad`
run. Do not call one `XCIntegrator` instance concurrently from multiple
threads; use one integrator per thread or serialize access to a shared one.

Named models resolve from the packaged `skalaxc.MODEL_DIR` unless
`SKALAXC_MODEL_PATH` is already set. Diagnostics are rank-local and available
through `integrator.diagnostics()` and `integrator.reset_diagnostics()`.

## CUDA and MPI variants

CUDA variants still accept and return CPU NumPy arrays. They expose
`ExecutionSpace.DEVICE` and `DeviceRuntimeSettings` only when the native
SkalaXC package was built with CUDA, and require the runtime Torch CUDA major
compatibility family recorded by that native build. Minor toolkit releases
within one CUDA major family are accepted.

MPI variants require `mpi4py` and an explicit intracommunicator:

```python
from mpi4py import MPI
import skalaxc

runtime = skalaxc.RuntimeEnvironment(MPI.COMM_SELF)
```

`None`, `MPI.COMM_NULL`, freed communicators, and intercommunicators are
rejected. The communicator is not duplicated or freed by SkalaXC. Python
objects retain it through load balancing and integration, but callers must not
explicitly free it while dependent objects remain alive. All evaluation calls
are collective over that exact communicator.

Run the CUDA+MPI communicator checks from this directory with:

```bash
OMP_NUM_THREADS=4 mpiexec -n 3 python -m pytest \
	tests/test_cuda.py::test_cuda_uses_mpi_world_with_an_idle_domain_rank -v
OMP_NUM_THREADS=4 mpiexec -n 4 python -m pytest \
	tests/test_cuda.py::test_cuda_uses_runtime_mpi_subcommunicator -v
```

The first command covers an idle domain rank. The second splits four ranks into
two independent communicators with distinct inputs, detecting accidental
collectives over `MPI.COMM_WORLD`.

## Typing

Every package includes `py.typed` and a generated native stub. The stub matches
the selected native build: CPU, CUDA, MPI, and CUDA+MPI packages expose only
their supported `RuntimeEnvironment` constructors. MPI constructors accept
`mpi4py.MPI.Intracomm`; evaluation matrices are typed as two-dimensional NumPy
`float64` arrays. At runtime, nanobind accepts NumPy arrays with other
convertible dtypes or layouts and materializes Fortran-contiguous `float64`
temporaries before evaluation.

Run the common interface fixture and the fixture matching the installed build:

```bash
mypy tests/typing/interface.py tests/typing/cuda_mpi.py
```

The other build fixtures are `cpu.py`, `cuda.py`, and `mpi.py` in the same
directory. Unused-ignore checking ensures each fixture also verifies that
constructors unavailable in that build remain rejected.