.. _gauxc_install:

Installing GauXC
================

In this section, we will provide instructions on how to install GauXC with Skala support based on the conda-forge ecosystem.
As part of this tutorial we will be

* installing dependencies for building GauXC
* configuring GauXC with different options
* testing our the Skala implementation in GauXC
* installing the GauXC library
* reusing GauXC from the CMake build system


Prerequisites
-------------

For this tutorial, we will use the `mamba <https://mamba.readthedocs.io/en/latest/>`__ package manager for setting up the environment and installing dependencies.
If you do not have mamba installed, you can download the `miniforge <https://conda-forge.org/download/>`__ installer.

First, we will create a new environment with all the required dependencies for building GauXC with Skala support.
We provide three different configurations depending on whether you want to build GauXC with OpenMP, MPI, or CUDA support.

.. dropdown:: GauXC dependencies

   The following dependencies are required for building GauXC with Skala support:

   - C/C++ compiler (with C++17 support)
   - CMake (version 3.15 or higher)
   - `exchcxx <https://github.com/wavefunction91/exchcxx>`__\ * (version 1 or higher)
   - `libxc <https://libxc.gitlab.io/>`__\ * (version 7 or higher)
   - `integratorxx <https://github.com/wavefunction91/integratorxx>`__\ * (version 1 or higher)
   - `gau2grid <https://github.com/psi4/gau2grid>`__\ * (version 2.0.6 or higher)
   - `libtorch <https://docs.pytorch.org/cppdocs/installing.html>`__ (CPU or CUDA version depending on your configuration)
   - `nlohmann_json <https://github.com/nlohmann/json>`__\ * (version 3.9.1 or higher)
   - BLAS library (like OpenBLAS, MKL, etc.)

   When building with MPI support via ``-DGAUXC_ENABLE_MPI=on`` (default ``off``),
   the following dependencies are also required:

   - MPI implementation (like OpenMPI, MPICH, etc.)

   When building with Cuda support via ``-DGAUXC_ENABLE_CUDA=on`` (default ``off``),
   the following dependencies are also required:

   - CUDA toolkit
   - `cuBLAS library <https://developer.nvidia.com/cublas>`__
   - `Cutlass library <https://github.com/NVIDIA/cutlass>`__\ *
   - `CUB library <https://github.com/NVIDIA/cccl/tree/main/cub>`__\ *

   When building with HDF5 support via ``-DGAUXC_ENABLE_HDF5=on`` (default ``on``),
   the following dependencies are also required:

   - `HDF5 <https://support.hdfgroup.org/documentation>`__
   - `HighFive <https://github.com/highfive-dev/highfive>`__\ * (version 2.4.0 or higher)

   All libraries marked with a * can be automatically fetched by the GauXC build system
   and do not need to be installed manually.

For this, create a file named `environment.yml` with the following content:

.. tab-set::
   :sync-group: config

   .. tab-item:: OpenMP

      .. literalinclude:: ../../examples/cpp/gauxc_integration/environment-openmp.yml
         :caption: environment.yml
         :language: yaml

   .. tab-item:: MPI

      .. literalinclude:: ../../examples/cpp/gauxc_integration/environment-mpi.yml
         :caption: environment.yml
         :language: yaml

   .. tab-item:: CUDA

      .. literalinclude:: ../../examples/cpp/gauxc_integration/environment-cuda.yml
         :caption: environment.yml
         :language: yaml

Create and activate the environment:

.. code-block:: none

   mamba env create -n gauxc-dev -f environment.yml
   mamba activate gauxc-dev

Verify that the toolchain is visible:

.. code-block:: bash

   cmake --version
   python -c "import torch; print(torch.__version__)"


Obtain GauXC with Skala
-----------------------

Download the pre-packaged source bundle from the Skala release page:

.. code-block:: none

   curl -L https://github.com/microsoft/skala/releases/download/v1.1.0/gauxc-skala.tar.gz | tar xzv

.. tip::

   To verify the downloaded tarball you can obtain a checksum

   .. code-block:: none

      curl -L https://github.com/microsoft/skala/releases/download/v1.1.0/gauxc-skala.tar.gz > gauxc-skala.tar.gz
      curl -L https://github.com/microsoft/skala/releases/download/v1.1.0/gauxc-skala.tar.gz.sha256 | sha256sum -c
      tar xzvf gauxc-skala.tar.gz

The archive expands into a ``gauxc`` directory that already contains the Skala patches.


Configure and build
-------------------

Create an out-of-tree build directory and pick the configuration that matches your backend.

.. tab-set::
   :sync-group: config

   .. tab-item:: OpenMP

      .. code-block:: none

         cmake -B build -S gauxc -G Ninja \
           -DGAUXC_ENABLE_OPENMP=on \
           -DGAUXC_ENABLE_MPI=off \
           -DGAUXC_ENABLE_CUDA=off \
           -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
         cmake --build build

   .. tab-item:: MPI

      .. code-block:: none

         cmake -B build -S gauxc -G Ninja \
           -DGAUXC_ENABLE_OPENMP=on \
           -DGAUXC_ENABLE_MPI=on \
           -DGAUXC_ENABLE_CUDA=off \
           -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
         cmake --build build

   .. tab-item:: CUDA

      .. code-block:: none

         cmake -B build -S gauxc -G Ninja \
           -DGAUXC_ENABLE_OPENMP=on \
           -DGAUXC_ENABLE_MPI=off \
           -DGAUXC_ENABLE_CUDA=on \
           -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
         cmake --build build

.. tip::

   If CMake cannot find libtorch, the ``Torch_DIR`` variable can be set to help discover the package.
   For conda-forge installed pytorch this should be set as ``-DTorch_DIR=${CONDA_PREFIX}/share/cmake/Torch``
   and for pip installed pytorch the CMake config file will be in ``${CONDA_PREFIX}/lib/python3.11/site-packages/torch/share/cmake/Torch``
   where the Python version should be adjusted accordingly to the environment.


Quick verification
------------------

After the build finishes, run the bundled regression test to confirm that Skala-enabled functionals
are working correctly. The Skala implementation can run different traditional functionals, like PBE and TPSS,
which can be compared against other libraries.

.. code-block:: bash

   cd gauxc/tests/ref_data
   ../../../build/tests/standalone_driver onedft_input.inp

Expected output includes the total TPSS energy computed using a checkpoint compatible for the Skala implementation
for the reference density matrix.

.. tip::
   
   If the executable cannot locate libtorch or other shared libraries, double-check
   that ``LD_LIBRARY_PATH`` includes ``${CONDA_PREFIX}/lib``
   (activating the environment usually handles this).


Install the library
-------------------

Install into the active conda environment so downstream projects can pick up the CMake config files.

.. code-block:: bash

   cmake --install build

This installs headers, libraries, and CMake config.


Integrate with your codebase
----------------------------

Using an installed GauXC
~~~~~~~~~~~~~~~~~~~~~~~~

Add the following to your CMake project, ensuring that ``CMAKE_PREFIX_PATH`` contains
``${CONDA_PREFIX}`` (activation scripts typically set this).

.. code-block:: cmake

   find_package(gauxc CONFIG REQUIRED)

   if(NOT gauxc_HAS_ONEDFT)
     message(FATAL_ERROR "GauXC found but Skala/OneDFT was not enabled during the build")
   endif()

   target_link_libraries(my_dft_driver PRIVATE gauxc::gauxc)

The imported target propagates include directories, compile definitions, and linkage against BLAS,
Torch, and optional MPI/CUDA components.

Embedding GauXC via FetchContent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to vend GauXC directly from your build, use ``FetchContent`` while mirroring the
options chosen above.

.. code-block:: cmake

   set(Skala_GauXC_URL "https://github.com/microsoft/skala/releases/download/v1.1.0/gauxc-skala.tar.gz")
   set(Skala_GauXC_SHA256 "e0346c62453eef58ba3ee52c257370ee8abcbf00fbb1b4ea2e0bb879225e06be")

   option(Skala_GauXC_ENABLE_OPENMP "Enable OpenMP support in GauXC" ON)
   option(Skala_GauXC_ENABLE_MPI "Enable MPI support in GauXC" OFF)
   option(Skala_GauXC_ENABLE_CUDA "Enable CUDA support in GauXC" OFF)

   find_package(gauxc QUIET CONFIG)
   if(NOT gauxc_FOUND)
     include(FetchContent)

     message(STATUS "Could not find GauXC... Building GauXC from source")
     message(STATUS "GAUXC URL: ${Skala_GauXC_URL}")

     set(GAUXC_ENABLE_ONEDFT ON CACHE BOOL "" FORCE)
     set(GAUXC_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
     set(GAUXC_ENABLE_OPENMP ${Skala_GauXC_ENABLE_OPENMP} CACHE BOOL "" FORCE)
     set(GAUXC_ENABLE_MPI ${Skala_GauXC_ENABLE_MPI} CACHE BOOL "" FORCE)
     set(GAUXC_ENABLE_CUDA ${Skala_GauXC_ENABLE_CUDA} CACHE BOOL "" FORCE)

     FetchContent_Declare(
       gauxc
       URL ${Skala_GauXC_URL}
       URL_HASH SHA256=${Skala_GauXC_SHA256}
       DOWNLOAD_EXTRACT_TIMESTAMP ON
     )
     FetchContent_MakeAvailable(gauxc)

   else()
     if(NOT ${GAUXC_HAS_ONEDFT})
       message(FATAL_ERROR "GauXC found but without Skala support enabled")
     endif()
     if(${Skala_GauXC_ENABLE_OPENMP} AND NOT ${GAUXC_HAS_OPENMP})
       message(WARNING "GauXC Found without OpenMP support but Skala_GauXC_ENABLE_OPENMP is ON")
     endif()
     if(${Skala_GauXC_ENABLE_MPI} AND NOT ${GAUXC_HAS_MPI})
       message(WARNING "GauXC Found without MPI support but Skala_GauXC_ENABLE_MPI is ON")
     endif()
     if(${Skala_GauXC_ENABLE_CUDA} AND NOT ${GAUXC_HAS_CUDA})
       message(WARNING "GauXC Found without CUDA support but Skala_GauXC_ENABLE_CUDA is ON")
     endif()
   endif()

Troubleshooting
---------------

Torch not found
  ensure ``Torch_DIR`` points to the libtorch CMake package inside the active environment,
  or export ``Torch_DIR`` before running CMake.

CUDA mismatch
  the CUDA toolkit selected by conda must match the version baked into the
  ``pytorch`` build; reinstall ``pytorch`` if necessary (e.g., ``pytorch ==2.3.* cuda118*``).

Linker errors for BLAS/MPI
  verify that the conda environment stayed active during the build and that ``cmake`` picked
  the toolchain from ``${CONDA_PREFIX}`` via ``CMAKE_PREFIX_PATH``.

Standalone driver cannot find densities
  run it from ``gauxc/tests/ref_data`` since paths in density files are specified relative to the
  current directory.

.. note::

   Need help? Open an issue on the `Skala repository <https://github.com/microsoft/skala/issues>`__.