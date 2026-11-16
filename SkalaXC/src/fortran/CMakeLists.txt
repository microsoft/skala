# SkalaXC Fortran API.
#
# A thin iso_c_binding wrapper over the SkalaXC C API. It links ONLY the public
# `skalaxc` shared library -- never GauXC or LibTorch -- so a Fortran consumer's
# ABI surface is limited to the SkalaXC symbols exported by libskalaxc.

set(SKALAXC_Fortran_MODULE_DIR ${CMAKE_CURRENT_BINARY_DIR}/mod)

add_library(skalaxc_fortran STATIC skalaxc.F90)
add_library(SkalaXC::skalaxc_fortran ALIAS skalaxc_fortran)

# The Fortran source is preprocessed so it can guard the MPI-communicator
# plumbing behind SKALAXC_HAS_MPI, exactly as the C/C++ headers do. When MPI is
# OFF the guarded blocks disappear and the Fortran API is byte-for-byte the
# serial one; when ON, the create procedures take an integer MPI communicator
# handle that the C layer converts via MPI_Comm_f2c.
set_source_files_properties(skalaxc.F90 PROPERTIES Fortran_PREPROCESS ON)
if(SKALAXC_ENABLE_MPI)
  target_compile_definitions(skalaxc_fortran PRIVATE SKALAXC_HAS_MPI)
endif()
if(SKALAXC_HAS_HDF5)
  target_compile_definitions(skalaxc_fortran PRIVATE SKALAXC_HAS_HDF5)
endif()

# Public C API only. `skalaxc` propagates just its public include dir (no GauXC).
target_link_libraries(skalaxc_fortran PUBLIC skalaxc)

set_target_properties(skalaxc_fortran PROPERTIES
  Fortran_MODULE_DIRECTORY ${SKALAXC_Fortran_MODULE_DIR}
  POSITION_INDEPENDENT_CODE ON
)

# Expose the generated `.mod` to consumers.
target_include_directories(skalaxc_fortran PUBLIC
  $<BUILD_INTERFACE:${SKALAXC_Fortran_MODULE_DIR}>
  $<INSTALL_INTERFACE:include>
)
