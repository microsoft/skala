.. _gauxc_install:

Installing GauXC
================

In this section, we will install GauXC with Skala support using the locked Pixi
environments in the Skala repository.
As part of this tutorial we will be

* installing dependencies for building GauXC
* configuring GauXC with different options
* testing the Skala implementation in GauXC
* installing the GauXC library
* reusing GauXC from the CMake build system


Prerequisites
-------------

Install `Pixi 0.75 <https://pixi.sh>`__ and clone Skala. The repository provides
five native environments:

.. list-table::
   :header-rows: 1

   * - Environment
     - Backend
     - Bindings
   * - ``gauxc-openmp``
     - OpenMP
     - C/C++
   * - ``gauxc-openmp-fortran``
     - OpenMP
     - C/C++/Fortran
   * - ``gauxc-mpi``
     - OpenMPI with MPI HDF5
     - C/C++
   * - ``gauxc-mpi-fortran``
     - OpenMPI with MPI HDF5
     - C/C++/Fortran
   * - ``gauxc-cuda12``
     - CUDA 12 with GPU LibTorch
     - C/C++

.. note::

   A full list of dependencies can be found at :ref:`gauxc-cmake-deps` in the CMake configuration documentation.

Install the environment for the backend you need, for example:

.. code-block:: bash

   git clone https://github.com/microsoft/skala
   cd skala
   pixi install --locked -e gauxc-openmp

Verify that the toolchain is visible:

.. code-block:: bash

   pixi run -e gauxc-openmp cmake --version
   pixi run -e gauxc-openmp python -c "import torch; print(torch.__version__)"


Obtain GauXC with Skala
-----------------------

Download the pre-packaged source bundle from the Skala release page:

.. code-block:: none

   curl -L https://github.com/microsoft/skala/releases/download/v1.1.1/gauxc-skala-r2.tar.gz | tar xzv

.. tip::

   To verify the downloaded tarball you can obtain a checksum

   .. code-block:: none

      curl -L https://github.com/microsoft/skala/releases/download/v1.1.1/gauxc-skala-r2.tar.gz > gauxc-skala-r2.tar.gz
      curl -L https://github.com/microsoft/skala/releases/download/v1.1.1/gauxc-skala-r2.tar.gz.sha256 | sha256sum -c
      tar xzvf gauxc-skala-r2.tar.gz

The archive expands into a ``gauxc`` directory that already contains the Skala
patches. Place it beside the Skala checkout because the CMake commands below
assume this layout:

.. code-block:: text

   work/
   ├── skala/
   ├── gauxc/
   ├── build_gauxc/
   └── build_example/

.. note::

   You can also obtain the latest version of GauXC with Skala support by downloading the `skala branch of GauXC <https://github.com/wavefunction91/gauxc/tree/skala>`__.

   .. code-block:: none

      curl -L https://github.com/wavefunction91/GauXC/archive/refs/heads/skala.tar.gz | tar xzv


Configure and build
-------------------

From the Skala repository root, pick the environment and CMake options that
match your backend. The following commands build the C++ API; enable
:cmake:variable:`GAUXC_ENABLE_C` or :cmake:variable:`GAUXC_ENABLE_FORTRAN` when
building the C or Fortran examples.

.. tab-set::
   :sync-group: config

   .. tab-item:: OpenMP

         .. code-block:: bash

             pixi run -e gauxc-openmp bash -c \
                'cmake -B ../build_gauxc -S ../gauxc -G Ninja \
                   -DGAUXC_ENABLE_OPENMP=ON \
                   -DGAUXC_ENABLE_MPI=OFF \
                   -DGAUXC_ENABLE_CUDA=OFF \
                   -DGAUXC_ENABLE_ONEDFT=ON \
                   -DGAUXC_ENABLE_C=OFF \
                   -DGAUXC_ENABLE_FORTRAN=OFF \
                   -DGAUXC_ENABLE_TESTS=OFF \
                   -DBUILD_SHARED_LIBS=ON \
                   -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX"'
             pixi run -e gauxc-openmp cmake --build ../build_gauxc
             pixi run -e gauxc-openmp cmake --install ../build_gauxc

   .. tab-item:: MPI

         .. code-block:: bash

             pixi run -e gauxc-mpi bash -c \
                'cmake -B ../build_gauxc -S ../gauxc -G Ninja \
                   -DGAUXC_ENABLE_OPENMP=OFF \
                   -DGAUXC_ENABLE_MPI=ON \
                   -DGAUXC_ENABLE_CUDA=OFF \
                   -DGAUXC_ENABLE_ONEDFT=ON \
                   -DGAUXC_ENABLE_C=OFF \
                   -DGAUXC_ENABLE_FORTRAN=OFF \
                   -DGAUXC_ENABLE_TESTS=OFF \
                   -DBUILD_SHARED_LIBS=ON \
                   -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX"'
             pixi run -e gauxc-mpi cmake --build ../build_gauxc
             pixi run -e gauxc-mpi cmake --install ../build_gauxc

   .. tab-item:: CUDA

      .. code-block:: bash

         pixi run -e gauxc-cuda12 bash -c \
           'cmake -B ../build_gauxc -S ../gauxc -G Ninja \
             -DGAUXC_ENABLE_OPENMP=ON \
             -DGAUXC_ENABLE_MPI=OFF \
             -DGAUXC_ENABLE_CUDA=ON \
             -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-60}" \
             -DGAUXC_ENABLE_ONEDFT=ON \
             -DGAUXC_ENABLE_C=OFF \
             -DGAUXC_ENABLE_FORTRAN=OFF \
             -DGAUXC_ENABLE_TESTS=OFF \
             -DBUILD_SHARED_LIBS=ON \
             -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX"'
         pixi run -e gauxc-cuda12 cmake --build ../build_gauxc
         pixi run -e gauxc-cuda12 cmake --install ../build_gauxc

      This defaults to compute capability 6.0, GauXC's minimum for FP64
      atomics. Set ``CMAKE_CUDA_ARCHITECTURES`` before the configure command for
      the deployment GPU, for example ``CMAKE_CUDA_ARCHITECTURES=80``.

.. note::

   To enable the C or Fortran bindings, set :cmake:variable:`GAUXC_ENABLE_C` or :cmake:variable:`GAUXC_ENABLE_FORTRAN` in your CMake configuration step.
   For a full list of available CMake options, see :ref:`gauxc-cmake-options` in the CMake configuration documentation.

.. tip::

   Pixi exposes the selected environment as ``${CONDA_PREFIX}`` for compatibility
   with conda build tools. If CMake cannot find LibTorch, pass
   ``-DTorch_DIR=${CONDA_PREFIX}/share/cmake/Torch``.


Quick verification
------------------

After the build finishes, run the bundled regression test to confirm that Skala-enabled functionals
are working correctly. The Skala implementation can run different traditional functionals, like PBE and TPSS,
which can be compared against other libraries.

.. code-block:: bash

   pixi run -e gauxc-openmp Skala \
     ../gauxc/tests/ref_data/onedft_he_def2qzvp_tpss_uks.hdf5 --model TPSS

Expected output includes the total TPSS energy computed using a checkpoint compatible for the Skala implementation
for the reference density matrix.

.. tip::
   
   If the executable cannot locate libtorch or other shared libraries, double-check
   that ``LD_LIBRARY_PATH`` includes ``${CONDA_PREFIX}/lib``
   (activating the environment usually handles this).


Install the library
-------------------

The CMake install command installs into the selected Pixi environment so
downstream projects can discover its CMake config files.

.. code-block:: bash

   pixi run -e gauxc-openmp cmake --install ../build_gauxc

This installs headers, libraries, and CMake config.

.. note::

   For using GauXC in your own CMake project, check out :ref:`gauxc-cmake-integration` in the CMake configuration documentation.
   Alternatively, you can follow the instructions in the :ref:`gauxc-cpp-library` tutorial for a full standalone example.


Troubleshooting
---------------

Torch not found
  ensure ``Torch_DIR`` points to the libtorch CMake package inside the active environment,
  or export ``Torch_DIR`` before running CMake.

CUDA mismatch
   use ``gauxc-cuda12`` as a unit. Its lock selects matching CUDA 12 compiler,
   ExchCXX, and PyTorch builds.

Linker errors for BLAS/MPI
   run CMake through ``pixi run -e <environment>`` and verify that it picked the
   toolchain from ``${CONDA_PREFIX}`` via ``CMAKE_PREFIX_PATH``.

Standalone driver cannot find densities
  run it from ``gauxc/tests/ref_data`` since paths in density files are specified relative to the
  current directory.

.. note::

   Need help? Open an issue on the `Skala repository <https://github.com/microsoft/skala/issues>`__.