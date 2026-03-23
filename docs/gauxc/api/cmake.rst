CMake Configuration Options
===========================

This section provides an overview of the available CMake configuration options for building GauXC with Skala support, as well as the required dependencies for each configuration.


.. _gauxc-cmake-deps:

GauXC dependencies
------------------

The following dependencies are required for building GauXC with Skala support:

- C/C++ compiler (with C++17 support)
- CMake (version 3.20 or higher)
- `exchcxx <https://github.com/wavefunction91/exchcxx>`__\ * (version 1 or higher)
- `libxc <https://libxc.gitlab.io/>`__\ * (version 7 or higher)
- `integratorxx <https://github.com/wavefunction91/integratorxx>`__\ * (version 1 or higher)
- `gau2grid <https://github.com/psi4/gau2grid>`__\ * (version 2.0.6 or higher)
- `libtorch <https://docs.pytorch.org/cppdocs/installing.html>`__ (CPU or CUDA version depending on your configuration)
- `nlohmann_json <https://github.com/nlohmann/json>`__\ * (version 3.9.1 or higher)
- BLAS library (like OpenBLAS, MKL, etc.)

When building with Fortran support via :cmake:variable:`GAUXC_ENABLE_FORTRAN` (default ``off``), a Fortran compiler is also required.

When building with MPI support via :cmake:variable:`GAUXC_ENABLE_MPI` (default ``off``), the following dependencies are also required:

- MPI implementation (like OpenMPI, MPICH, etc.)

When building with CUDA support via :cmake:variable:`GAUXC_ENABLE_CUDA` (default ``off``), the following dependencies are also required:

- CUDA toolkit
- `cuBLAS library <https://developer.nvidia.com/cublas>`__
- `Cutlass library <https://github.com/NVIDIA/cutlass>`__\ *
- `CUB library <https://github.com/NVIDIA/cccl/tree/main/cub>`__\ *

When building with HDF5 support via :cmake:variable:`GAUXC_ENABLE_HDF5` (default ``on``), the following dependencies are also required:

- `HDF5 <https://support.hdfgroup.org/documentation>`__
- `HighFive <https://github.com/highfive-dev/highfive>`__\ * (version 2.4.0 or higher)

All libraries marked with a * can be automatically fetched by the GauXC build system and do not need to be installed manually.


.. _gauxc-cmake-options:

Available configurations for CMake build
----------------------------------------

.. cmake:variable:: GAUXC_ENABLE_OPENMP

   Enable OpenMP support in GauXC (default: ON)

.. cmake:variable:: GAUXC_ENABLE_MPI

   Enable MPI support in GauXC (default: OFF)

.. cmake:variable:: GAUXC_ENABLE_ONEDFT

   Enable Skala support in GauXC (default: OFF)

.. cmake:variable:: GAUXC_ENABLE_CUDA

   Enable CUDA support in GauXC (default: OFF)
   Requires ExchCXX to be built with CUDA support as well (:cmake:variable:`EXCHCXX_ENABLE_CUDA` CMake option).
   Cannot be enabled with HIP support at the same time.

.. cmake:variable:: GAUXC_ENABLE_HIP

   Enable HIP support in GauXC (default: OFF)
   Requires ExchCXX to be built with HIP support as well (:cmake:variable:`EXCHCXX_ENABLE_HIP` CMake option).
   Cannot be enabled with CUDA support at the same time.

.. cmake:variable:: GAUXC_ENABLE_C

   Enable C bindings for GauXC (default: OFF)

.. cmake:variable:: GAUXC_ENABLE_FORTRAN

   Enable Fortran bindings for GauXC (default: OFF)
   Requires Fortran compiler and :cmake:variable:`GAUXC_ENABLE_C` to be enabled as well.

.. cmake:variable:: GAUXC_ENABLE_TESTS

   Enable building of GauXC tests (default: ON)
   Requires catch2 library to be installed and available.

.. cmake:variable:: GAUXC_ENABLE_HDF5

   Enable HDF5 support in GauXC (default: ON)
   Requires HDF5 library and HighFive library to be installed and available.

.. cmake:variable:: GAUXC_ENABLE_MAGMA

   Enable MAGMA support in GauXC (default: OFF)
   Requires MAGMA library to be installed and available.
   Requires CUDA or HIP support to be enabled as well.

.. cmake:variable:: GAUXC_ENABLE_NCCL

   Enable NCCL support in GauXC (default: OFF)
   Requires NCCL library to be installed and available.
   Requires CUDA support to be enabled as well.

.. cmake:variable:: GAUXC_ENABLE_CUTLASS

   Enable CUTLASS support in GauXC (default: OFF)
   Requires CUTLASS library to be installed and available.
   Requires CUDA support to be enabled as well.

.. cmake:variable:: GAUXC_ENABLE_GAU2GRID

   Enable Gau2Grid support in GauXC (default: ON)
   Always enabled since Gau2Grid is a required dependency for GauXC.

.. cmake:variable:: EXCHCXX_ENABLE_CUDA

   Enable CUDA support in ExchCXX (default: OFF)
   Required for GauXC CUDA support.
   Cannot be enabled with HIP support at the same time.

.. cmake:variable:: EXCHCXX_ENABLE_HIP

   Enable HIP support in ExchCXX (default: OFF)
   Required for GauXC HIP support.
   Cannot be enabled with CUDA support at the same time.


.. _gauxc-cmake-integration:

Integrating GauXC into your build system
----------------------------------------

Using an installed GauXC
~~~~~~~~~~~~~~~~~~~~~~~~

To integrate GauXC into your build system, you can use CMake's ``find_package`` command to locate the GauXC package and link against it in your ``CMakeLists.txt`` file.
Make sure that the ``CMAKE_PREFIX_PATH`` variable includes the path to your GauXC installation (e.g., ``${CONDA_PREFIX}`` if installed via Conda).

.. code-block:: cmake

   find_package(gauxc CONFIG REQUIRED)

   if(NOT GAUXC_HAS_ONEDFT)
     message(FATAL_ERROR "GauXC found but Skala/OneDFT was not enabled during the build")
   endif()

   target_link_libraries(my_dft_driver PRIVATE gauxc::gauxc)

The imported target propagates include directories, compile definitions, and linkage against BLAS,
Torch, and optional MPI/CUDA components.

.. note::

   Use the provided CMake variables like ``GAUXC_HAS_ONEDFT`` to check for specific features or configurations in GauXC before linking against it.


Embedding GauXC via FetchContent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to vendor GauXC directly from your build, use ``FetchContent`` while mirroring the options chosen above.
A possible approach for fetching GauXC with Skala support during the CMake configuration step is shown below.

.. literalinclude:: ../../../examples/cpp/gauxc_integration/cmake/skala-gauxc.cmake
   :language: cmake
   :caption: cmake/skala-gauxc.cmake

It is recommended to define the GauXC source URL with its SHA256 hash in a separate CMake file (e.g., `skala-gauxc-versions.cmake`).

.. literalinclude:: ../../../examples/cpp/gauxc_integration/cmake/skala-dep-versions.cmake
   :language: cmake
   :caption: cmake/skala-dep-versions.cmake
   :lines: 1-2

In the main ``CMakeLists.txt``, include the version definitions and the GauXC fetching logic.

.. code-block:: cmake

   include(cmake/skala-gauxc.cmake)

   target_link_libraries(my_dft_driver PRIVATE gauxc::gauxc)