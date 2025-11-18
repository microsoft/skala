Installing GauXC
================

In this section, we will provide instructions on how to install GauXC with Skala support based on the conda-forge ecosystem.
For this tutorial, we will use the `mamba <https://mamba.readthedocs.io/en/latest/>`__ package manager for setting up the environment and installing dependencies.
If you do not have mamba installed, you can download the `miniforge <https://conda-forge.org/download/>`__ installer.

First, we will create a new environment with all the required dependencies for building GauXC with Skala support.
For this, create a file named `environment.yml` with the following content:

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
   - pytorch * cpu_*
   - libblas

Then, run the following commands to create and activate the new environment:

.. code-block:: none

   mamba env create -n gauxc-dev -f environment.yml
   mamba activate gauxc-dev

Next, we will download the GauXC source code with Skala support and setup the build using CMake:

.. code-block:: none

   curl -JL https://github.com/microsoft/skala/releases/download/v1.0.0/gauxc-skala.tar.gz > gauxc-skala.tar.gz | tar xzv
   cmake -B _build -S gauxc -G Ninja -DGAUXC_ENABLE_MPI=no -DGAUXC_ENABLE_OPENMP=yes -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
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
   target_link_libraries(${PROJECT_NAME} PRIVATE gauxc::gauxc)

To build GauXC as part of your own CMake project, you can use the following snippet to include it via FetchContent.

.. code-block:: cmake

   set(Skala_GauXC_URL "https://github.com/microsoft/skala/releases/download/v1.0.0/gauxc-skala.tar.gz")
   set(Skala_GauXC_SHA "3368a4f8c968a295ad03363c5ccbee787f24f703df42c7b929a40e76f48bd324")

   find_package(gauxc QUIET)
   if(NOT gauxc_FOUND)
     include(FetchContent)

     message(STATUS "Could not find GauXC... Building GauXC from source")
     message(STATUS "GAUXC URL: ${Skala_GauXC_URL}")

     set(GAUXC_ENABLE_ONEDFT ON CACHE BOOL "" FORCE)
     set(GAUXC_ENABLE_TESTS OFF CACHE BOOL "" FORCE)

     FetchContent_Declare(
       gauxc
       URL ${Skala_GauXC_URL}
       URL_HASH SHA256=${Skala_GauXC_SHA}
       DOWNLOAD_EXTRACT_TIMESTAMP ON
     )
     FetchContent_MakeAvailable(gauxc)

   else()
     if(NOT ${GAUXC_HAS_ONEDFT})
       message(FATAL_ERROR "GauXC Found but without OneDFT Support Enabled")
     endif()
   endif()