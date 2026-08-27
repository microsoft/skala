cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# Enable C/C++ first so conda's compiler wrappers lead CMake to the matching
# gfortran instead of the system f95, whose module format may be incompatible.
project("Skala" LANGUAGES C CXX Fortran)

include(GNUInstallDirs)
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Disable FC Updates")

list(PREPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(Skala_GauXC_ENABLE_OPENMP "Enable OpenMP support in GauXC" ON)
option(Skala_GauXC_ENABLE_MPI "Enable MPI support in GauXC" OFF)
option(Skala_GauXC_ENABLE_CUDA "Enable CUDA support in GauXC" OFF)

include(skala-dep-versions)
include(skala-hdf5)
include(skala-gauxc)
include(skala-flap)

add_executable(
  "${PROJECT_NAME}"
  "app/main.F90"
)
# GauXC links to C++ libraries such as LibTorch, so use the C++ driver while
# CMake adds the Fortran runtime required by this target's Fortran source.
set_property(TARGET "${PROJECT_NAME}" PROPERTY LINKER_LANGUAGE CXX)
target_link_libraries(
  "${PROJECT_NAME}"
  PRIVATE
  gauxc::gauxc
  FLAP::FLAP
  HDF5::HDF5
)
if(GAUXC_HAS_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(
    "${PROJECT_NAME}"
    PRIVATE
    MPI::MPI_Fortran
  )
endif()

install(
  TARGETS
  "${PROJECT_NAME}"
  DESTINATION
  "${CMAKE_INSTALL_BINDIR}"
)

list(REMOVE_AT CMAKE_MODULE_PATH 0)
