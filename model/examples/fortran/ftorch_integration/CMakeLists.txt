cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project("Skala" LANGUAGES Fortran C CXX)

include(GNUInstallDirs)
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Disable FC Updates")

list(PREPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(skala-dep-versions)
include(skala-ftorch)
find_package(Torch REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

add_library(skala-ftorch-bindings STATIC src/skala_ftorch.cxx)
target_link_libraries(skala-ftorch-bindings PUBLIC ${TORCH_LIBRARIES} OpenSSL::Crypto)
target_compile_features(skala-ftorch-bindings PUBLIC cxx_std_17)

add_executable(
  ${PROJECT_NAME}
  app/main.f90
  src/skala_ftorch.f90
)
target_link_libraries(
  ${PROJECT_NAME}
  FTorch::ftorch
  skala-ftorch-bindings
)

include(CTest)
if(BUILD_TESTING)
  add_executable(test_model_load tests/test_model_load.cxx)
  target_link_libraries(test_model_load PRIVATE skala-ftorch-bindings)
  add_test(NAME model_load_integrity COMMAND test_model_load)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test-modules)
  add_executable(test_model_load_fortran tests/test_model_load.f90 src/skala_ftorch.f90)
  set_target_properties(test_model_load_fortran PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test-modules)
  target_include_directories(test_model_load_fortran PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/test-modules)
  target_link_libraries(test_model_load_fortran PRIVATE FTorch::ftorch skala-ftorch-bindings)
  add_test(NAME model_load_fortran_error_status COMMAND test_model_load_fortran)
endif()
# Resolve FTorch and LibTorch from the sibling install library directory while
# keeping the installed executable relocatable.
set_property(
  TARGET ${PROJECT_NAME}
  PROPERTY INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
)

install(
  TARGETS
  ${PROJECT_NAME}
  DESTINATION
  ${CMAKE_INSTALL_BINDIR}
)

list(REMOVE_AT CMAKE_MODULE_PATH 0)
