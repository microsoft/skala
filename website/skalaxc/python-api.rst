Python API
============================================================

The ``skalaxc`` package binds the public C++ API directly through nanobind,
not through the C or Fortran wrappers. NumPy inputs have shape ``(nbf, nbf)``
and are converted to Fortran-contiguous ``float64`` arrays as needed.
``eval_exc_vxc`` returns an energy and the scalar and z-spin potential
matrices. ``eval_exc_grad`` returns a ``(natoms, 3)`` nuclear gradient,
not forces. Inputs are scalar/z-spin channels, not alpha/beta matrices.

Installation
------------------------------------------------------------

Use the locked binding environment, build and install the native library,
then select that exact installation for the Python build:

.. code-block:: bash

   git submodule update --init SkalaXC/external/GauXC
   pixi install --locked -e skalaxc-python-host
   pixi run -e skalaxc-python-host cmake -S SkalaXC -B SkalaXC/build-python -G Ninja \
     -DSKALAXC_BUILD_FORTRAN=OFF -DSKALAXC_BUILD_TESTS=ON \
     -DCMAKE_INSTALL_PREFIX="$PWD/SkalaXC/install-python"
   pixi run -e skalaxc-python-host cmake --build SkalaXC/build-python --parallel 4
   pixi run -e skalaxc-python-host cmake --install SkalaXC/build-python
   SkalaXC_DIR="$PWD/SkalaXC/install-python/lib/cmake/SkalaXC" \
     pixi run -e skalaxc-python-host python -m pip install SkalaXC/python \
       --no-build-isolation --no-deps
   OMP_NUM_THREADS=4 pixi run -e skalaxc-python-host \
     python -m pytest -v SkalaXC/python/tests

The package requires Python 3.11--3.13 and the supported Torch 2.13 minor line.
The test suite uses additional numerical fixtures from the GauXC submodule.
The native library and runtime Torch must have compatible versions and C++
ABIs. The binding itself does not link Torch directly. The default wheel
layout bundles the native library and models; conda packages use the
environment prefix. See ``SkalaXC/python/README.md`` for packaging details.

Runtime behavior
------------------------------------------------------------

``HDF5_ENABLED``, ``MPI_ENABLED``, ``CUDA_ENABLED``, and ``OPENMP_ENABLED``
describe the installed native build. HDF5-free builds use ``Atom``, ``Shell``
and container ``append()`` methods instead of file readers.

Container indexing returns independent values. To change an atom, retrieve
it, modify it, then assign it back to the molecule. Keep one integrator per
thread or serialize access: evaluation releases the GIL but a single
integrator must not be evaluated concurrently.

MPI builds require ``mpi4py`` and an explicit live intracommunicator, for
example ``RuntimeEnvironment(MPI.COMM_SELF)``. All evaluations are collective
over that communicator; do not explicitly free it while dependent objects
exist. Multi-rank pytest launches must use ``-p no:randomly``.

CUDA builds still accept and return CPU NumPy arrays. Device runtime
constructors are present only in CUDA-enabled builds. See :doc:`installation`
for the experimental CUDA and TPSS gradient limitations.

The package ships ``py.typed`` and build-specific generated stubs. The
public namespace is:

.. literalinclude:: ../../SkalaXC/python/src/skalaxc/__init__.pyi
   :language: python

See :doc:`python` for the complete evaluation pipeline.
