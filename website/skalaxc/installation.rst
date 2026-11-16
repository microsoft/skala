Installation and execution contract
===================================

Build from the repository root with the locked native toolchain:

.. code-block:: bash

   git submodule update --init SkalaXC/external/GauXC
   pixi install --locked -e gauxc-openmp
   pixi run -e gauxc-openmp cmake -S SkalaXC -B SkalaXC/build -G Ninja \
     -DSKALAXC_BUILD_TESTS=ON -DSKALAXC_BUILD_EXAMPLES=ON \
     -DCMAKE_INSTALL_PREFIX="$PWD/SkalaXC/install"
   pixi run -e gauxc-openmp cmake --build SkalaXC/build --parallel 4
   OMP_NUM_THREADS=4 pixi run -e gauxc-openmp \
     ctest --test-dir SkalaXC/build --output-on-failure
   pixi run -e gauxc-openmp cmake --install SkalaXC/build

The public library requires C++17 and CMake 3.21 or newer. LibTorch, BLAS, and
HDF5 must be available to the build. GauXC is a private source dependency, not
part of the public interface; consumers link only the installed SkalaXC target.
The pinned GauXC revision is fetched automatically when its submodule is absent.

``SKALAXC_ENABLE_HDF5=OFF`` removes HDF5 input support.
``SKALAXC_ENABLE_OPENMP`` defaults to ``ON``.
For MPI, use ``gauxc-mpi`` and configure ``SKALAXC_ENABLE_MPI=ON``; all
evaluations are collective on the communicator supplied to the runtime.

Models and array layout
-----------------------

Select ``LDA``, ``PBE``, ``TPSS``, or an explicit TorchScript ``.fun`` path.
Configuration downloads and verifies bundled model hashes. Named models use
the installed model directory; ``SKALAXC_MODEL_PATH`` overrides that directory.
Each integrator owns its model for its lifetime.

Unrestricted calculations use scalar and z-spin density matrices. Matrices
are double precision, square ``nbf x nbf``, and column-major. Nuclear
gradients are atom-major xyz values, not forces; negate them for forces.

Limitations
-----------

The validated production backend is host UKS evaluation of energy, potentials,
and gradients. RKS is not implemented. CUDA is experimental and disabled by
default; enabling it requires a compatible CUDA compiler, GPU-enabled
LibTorch, and validation on the target GPU. TPSS CUDA energy and potentials are
tested, but its archived backward trace can exceed kernel resources for
gradients. LDA/PBE CUDA gradients are tested separately.
