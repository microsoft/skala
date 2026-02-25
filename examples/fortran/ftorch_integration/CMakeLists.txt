cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project("Skala" LANGUAGES Fortran C CXX)

include(GNUInstallDirs)
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Disable FC Updates")

list(PREPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(skala-dep-versions)
include(skala-ftorch)
find_package(Torch REQUIRED)

add_executable(
  ${PROJECT_NAME}
  app/main.f90
  src/skala_ftorch.f90
  src/skala_ftorch.cxx
)
target_link_libraries(
  ${PROJECT_NAME}
  FTorch::ftorch
  ${TORCH_LIBRARIES}
)

install(
  TARGETS
  ${PROJECT_NAME}
  DESTINATION
  ${CMAKE_INSTALL_BINDIR}
)

list(REMOVE_AT CMAKE_MODULE_PATH 0)

