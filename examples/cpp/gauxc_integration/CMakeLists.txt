cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project("Skala" LANGUAGES CXX)

include(GNUInstallDirs)
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Disable FC Updates")

list(PREPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(Skala_GauXC_ENABLE_OPENMP "Enable OpenMP support in GauXC" ON)
option(Skala_GauXC_ENABLE_MPI "Enable MPI support in GauXC" OFF)
option(Skala_GauXC_ENABLE_CUDA "Enable CUDA support in GauXC" OFF)

include(skala-dep-versions)
include(skala-gauxc)
include(skala-cli11)
include(skala-eigen3)

add_executable(
  "${PROJECT_NAME}"
  "app/main.cxx"
)
target_link_libraries(
  "${PROJECT_NAME}"
  PRIVATE
  gauxc::gauxc
  CLI11::CLI11
  Eigen3::Eigen
)
install(
  TARGETS
  "${PROJECT_NAME}"
  DESTINATION
  "${CMAKE_INSTALL_BINDIR}"
)

list(REMOVE_AT CMAKE_MODULE_PATH 0)