Installing GauXC
================

In this section, we will provide instructions on how to install GauXC with Skala support based on the conda-forge ecosystem.
For this tutorial, we will use the `mamba <https://mamba.readthedocs.io/en/latest/>`__ package manager for setting up the environment and installing dependencies.
If you do not have mamba installed, you can download the `miniforge <https://conda-forge.org/download/>`__ installer.

First, we will create a new environment with all the required dependencies for building GauXC with Skala support.
We provide three different configurations depending on whether you want to build GauXC with OpenMP, MPI, or CUDA support.

.. dropdown:: GauXC dependencies

   The following dependencies are required for building GauXC with Skala support:

   - C/C++ compiler (with C++17 support)
   - CMake (version 3.15 or higher)
   - `exchcxx <https://github.com/wavefunction91/exchcxx>`__* (version 1 or higher)
   - `libxc <https://libxc.gitlab.io/>`__* (version 7 or higher)
   - `integratorxx <https://github.com/wavefunction91/integratorxx>`__* (version 1 or higher)
   - `gau2grid <https://github.com/psi4/gau2grid>`__* (version 2.0.6 or higher)
   - `libtorch <https://docs.pytorch.org/cppdocs/installing.html>`__ (CPU or CUDA version depending on your configuration)
   - `nlohmann_json <https://github.com/nlohmann/json>`__* (version 3.9.1 or higher)
   - BLAS library (like OpenBLAS, MKL, etc.)

   When building with MPI support via ``-DGAUXC_ENABLE_MPI=on`` (default ``off``),
   the following dependencies are also required:

   - MPI implementation (like OpenMPI, MPICH, etc.)

   When building with Cuda support via ``-DGAUXC_ENABLE_CUDA=on`` (default ``off``),
   the following dependencies are also required:

   - CUDA toolkit
   - `cuBLAS library <https://developer.nvidia.com/cublas>`__
   - `Cutlass library <https://github.com/NVIDIA/cutlass>`__*
   - `CUB library <https://github.com/NVIDIA/cccl/tree/main/cub>`__*

   When building with HDF5 support via ``-DGAUXC_ENABLE_HDF5=on`` (default ``on``),
   the following dependencies are also required:

   - `HDF5 <https://support.hdfgroup.org/documentation>`__
   - `HighFive <https://github.com/highfive-dev/highfive>`__* (version 2.4.0 or higher)

   All libraries marked with a * can be automatically fetched by the GauXC build system
   and do not need to be installed manually.

For this, create a file named `environment.yml` with the following content:

.. tab-set::
   :sync-group: config

   .. tab-item:: OpenMP

      .. code-block:: yaml
         :caption: environment.yml

         name: gauxc-dev
         channels:
         - conda-forge
         dependencies:
         # build requirements
         - c-compiler
         - cxx-compiler
         - cmake
         - ninja
         - nlohmann_json
         # host requirements
         - exchcxx >=1.0
         - gau2grid >=2.0.6
         - hdf5
         - libblas
         - pytorch * cpu*

   .. tab-item:: MPI

      .. code-block:: yaml
         :caption: environment.yml

         name: gauxc-dev
         channels:
         - conda-forge
         dependencies:
         # build requirements
         - c-compiler
         - cxx-compiler
         - cmake
         - ninja
         - nlohmann_json
         # host requirements
         - openmpi  # or mpich
         - exchcxx >=1.0
         - gau2grid >=2.0.6
         - hdf5 * mpi_*
         - libblas
         - pytorch * cpu*

   .. tab-item:: CUDA

      .. code-block:: yaml
         :caption: environment.yml

         name: gauxc-dev
         channels:
         - conda-forge
         dependencies:
         # build requirements
         - c-compiler
         - cxx-compiler
         - cuda-compiler
         - cmake
         - ninja
         - nlohmann_json
         # host requirements
         - gau2grid >=2.0.6
         - hdf5
         - libblas
         - pytorch * cuda*

Then, run the following commands to create and activate the new environment:

.. code-block:: none

   mamba env create -n gauxc-dev -f environment.yml
   mamba activate gauxc-dev

Next, we will download the GauXC source code with Skala support.
We provide a pre-packaged version of GauXC with Skala integration that can be downloaded from the Skala releases page.
Run the following command to download and extract the source code:

.. code-block:: none

   curl -JL https://github.com/microsoft/skala/releases/download/v1.0.0/gauxc-skala.tar.gz > gauxc-skala.tar.gz | tar xzv

Now we can build GauXC with Skala support using CMake and Ninja.
Create a build directory and run the following commands to configure and build the library:

.. tab-set::
   :sync-group: config

   .. tab-item:: OpenMP

      .. code-block:: none

         cmake -B _build -S gauxc -G Ninja -DGAUXC_ENABLE_MPI=no -DGAUXC_ENABLE_OPENMP=yes -DGAUXC_ENABLE_CUDA=no -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
         cmake --build _build

   .. tab-item:: MPI

      .. code-block:: none

         cmake -B _build -S gauxc -G Ninja -DGAUXC_ENABLE_MPI=yes -DGAUXC_ENABLE_OPENMP=yes -DGAUXC_ENABLE_CUDA=no -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
         cmake --build _build

   .. tab-item:: CUDA

      .. code-block:: none

         cmake -B _build -S gauxc -G Ninja -DGAUXC_ENABLE_MPI=no -DGAUXC_ENABLE_OPENMP=yes -DGAUXC_ENABLE_CUDA=yes -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
         cmake --build _build

We can check the Skala implementation by evaluating a traditional density functional via the Skala driver.
In the GauXC test directory we have checkpoints for the PBE and TPSS functionals that can be used for testing.
To run the test via the standalone driver of GauXC, we can use the following command:

.. code-block:: none

   cd gauxc/tests/ref_data
   ../../../_build/tests/standalone_driver onedft_input.inp

This will evaluate the TPSS exchange-correlation functional via the Skala driver on a provided density matrix.

To use GauXC in your project install the built library into your conda environment:

.. code-block:: none

   cmake --install _build

Finally, you can link against the GauXC library in your own electronic structure package to use Skala as an exchange-correlation functional.
This can be done by linking against the `gauxc` target in your CMake build system.

.. code-block:: cmake

   find_package(gauxc REQUIRED)
   if(NOT ${GAUXC_HAS_ONEDFT})
     message(FATAL_ERROR "GauXC found but without Skala support enabled")
   endif()
   target_link_libraries(${PROJECT_NAME} PRIVATE gauxc::gauxc)

To build GauXC as part of your own CMake project, you can use the following snippet to include it via FetchContent.

.. code-block:: cmake

   set(Skala_GauXC_URL "https://github.com/microsoft/skala/releases/download/v1.0.0/gauxc-skala.tar.gz")
   set(Skala_GauXC_SHA "3368a4f8c968a295ad03363c5ccbee787f24f703df42c7b929a40e76f48bd324")
   option(Skala_GauXC_ENABLE_OPENMP "Enable OpenMP support in GauXC" ON)
   option(Skala_GauXC_ENABLE_MPI "Enable MPI support in GauXC" OFF)
   option(Skala_GauXC_ENABLE_CUDA "Enable CUDA support in GauXC" OFF)

   find_package(gauxc QUIET)
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
       URL_HASH SHA256=${Skala_GauXC_SHA}
       DOWNLOAD_EXTRACT_TIMESTAMP ON
     )
     FetchContent_MakeAvailable(gauxc)

   else()
     if(NOT ${GAUXC_HAS_ONEDFT})
       message(FATAL_ERROR "GauXC found but without Skala support enabled")
     endif()
     if(${Skala_GauXC_ENABLE_OPENMP} AND NOT ${GAUXC_HAS_OPENMP})
       message(WARNING "GauXC Found with OpenMP support but Skala_GauXC_ENABLE_OPENMP is OFF")
     endif()
     if(${Skala_GauXC_ENABLE_MPI} AND NOT ${GAUXC_HAS_MPI})
       message(WARNING "GauXC Found with MPI support but Skala_GauXC_ENABLE_MPI is OFF")
     endif()
     if(${Skala_GauXC_ENABLE_CUDA} AND NOT ${GAUXC_HAS_CUDA})
       message(WARNING "GauXC Found with CUDA support but Skala_GauXC_ENABLE_CUDA is OFF")
     endif()
   endif()