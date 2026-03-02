Macro definitions
=================

GauXC provides a number of compile-time macros to indicate available features and configuration options.

.. c:macro:: GAUXC_HAS_C

   Defines whether C bindings are available in this build of GauXC.
   Enabled with :cmake:variable:`GAUXC_ENABLE_C` CMake option.

.. c:macro:: GAUXC_HAS_FORTRAN

   Defines whether Fortran bindings are available in this build of GauXC.
   Enabled with :cmake:variable:`GAUXC_ENABLE_FORTRAN` CMake option.

.. c:macro:: GAUXC_HAS_HOST

   Defines whether host support is available in this build of GauXC.

.. c:macro:: GAUXC_HAS_DEVICE

   Defines whether any device support (CUDA or HIP) is available in this build of GauXC.
   Enabled with :cmake:variable:`GAUXC_ENABLE_CUDA` or :cmake:variable:`GAUXC_ENABLE_HIP` CMake options.

.. c:macro:: GAUXC_HAS_CUDA

   Defines whether CUDA support is available in this build of GauXC.
   Enabled with :cmake:variable:`GAUXC_ENABLE_CUDA` CMake option.

.. c:macro:: GAUXC_HAS_HIP

   Defines whether HIP support is available in this build of GauXC.
   Enabled with :cmake:variable:`GAUXC_ENABLE_HIP` CMake option.

.. c:macro:: GAUXC_HAS_MPI

   Defines whether MPI support is available in this build of GauXC.
   Enabled with :cmake:variable:`GAUXC_ENABLE_MPI` CMake option.

.. c:macro:: GAUXC_HAS_MAGMA

   Defines whether MAGMA support is available in this build of GauXC.
   Enabled with :cmake:variable:`GAUXC_ENABLE_MAGMA` CMake option.

.. c:macro:: GAUXC_HAS_NCCL

   Defines whether NCCL support is available in this build of GauXC.
   Enabled with :cmake:variable:`GAUXC_ENABLE_NCCL` CMake option.

.. c:macro:: GAUXC_HAS_CUTLASS

   Defines whether CUTLASS support is available in this build of GauXC.
   Enabled with :cmake:variable:`GAUXC_ENABLE_CUTLASS` CMake option.

.. c:macro:: GAUXC_HAS_GAU2GRID

   Defines whether Gau2Grid support is available in this build of GauXC.
   Enabled with :cmake:variable:`GAUXC_ENABLE_GAU2GRID` CMake option.

.. c:macro:: GAUXC_HAS_HDF5

   Defines whether HDF5 support is available in this build of GauXC.
   Enabled with :cmake:variable:`GAUXC_ENABLE_HDF5` CMake option.

.. c:macro:: GAUXC_CPU_XC_MAX_AM

   Maximum angular momentum supported for CPU exchange-correlation calculations.
   Default is 6 (i.e., up to i-type functions).

.. c:macro:: GAUXC_CPU_SNLINK_MAX_AM

   Maximum angular momentum supported for CPU seminumerical exchange calculations.
   Default is 6 (i.e., up to i-type functions).

.. c:macro:: GAUXC_GPU_XC_MAX_AM

   Maximum angular momentum supported for GPU exchange-correlation calculations.
   Default is 4 (i.e., up to g-type functions).

.. c:macro:: GAUXC_GPU_SNLINK_MAX_AM

   Maximum angular momentum supported for GPU seminumerical exchange calculations.
   Default is 2 (i.e., up to d-type functions).