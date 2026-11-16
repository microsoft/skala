cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(SkalaXCInstalledConsumers LANGUAGES C CXX)

option(SKALAXC_TEST_FORTRAN "Build the installed Fortran consumer" ON)
option(SKALAXC_EXPECT_FORTRAN_UNAVAILABLE
  "Verify that the installed package does not provide Fortran" OFF)
if(SKALAXC_TEST_FORTRAN AND SKALAXC_EXPECT_FORTRAN_UNAVAILABLE)
  message(FATAL_ERROR
    "SKALAXC_TEST_FORTRAN and SKALAXC_EXPECT_FORTRAN_UNAVAILABLE are mutually exclusive")
endif()

if(SKALAXC_EXPECT_FORTRAN_UNAVAILABLE)
  find_package(SkalaXC REQUIRED CONFIG COMPONENTS C CXX)
  find_package(SkalaXC QUIET CONFIG COMPONENTS Fortran)
  if(SkalaXC_Fortran_FOUND OR TARGET SkalaXC::skalaxc_fortran)
    message(FATAL_ERROR
      "The core-only SkalaXC install unexpectedly provides Fortran")
  endif()
elseif(SKALAXC_TEST_FORTRAN)
  enable_language(Fortran)
  find_package(SkalaXC REQUIRED CONFIG COMPONENTS C CXX Fortran)
else()
  find_package(SkalaXC REQUIRED CONFIG COMPONENTS C CXX)
endif()

enable_testing()

add_executable(installed_consumer_cpp consumer.cpp)
target_link_libraries(installed_consumer_cpp PRIVATE SkalaXC::skalaxc)
target_compile_features(installed_consumer_cpp PRIVATE cxx_std_17)
add_test(NAME installed_consumer_cpp COMMAND installed_consumer_cpp)
set_tests_properties(installed_consumer_cpp PROPERTIES
  ENVIRONMENT "SKALAXC_MODEL_PATH=${SkalaXC_MODEL_DIR}")

add_executable(installed_consumer_c consumer.c)
target_link_libraries(installed_consumer_c PRIVATE SkalaXC::skalaxc)
add_test(NAME installed_consumer_c COMMAND installed_consumer_c)
set_tests_properties(installed_consumer_c PROPERTIES
  ENVIRONMENT "SKALAXC_MODEL_PATH=${SkalaXC_MODEL_DIR}")

if(SKALAXC_TEST_FORTRAN)
  add_executable(installed_consumer_fortran consumer.F90)
  set_source_files_properties(consumer.F90 PROPERTIES Fortran_PREPROCESS ON)
  if(SkalaXC_MPI_ENABLED)
    target_compile_definitions(installed_consumer_fortran PRIVATE SKALAXC_HAS_MPI)
  endif()
  target_link_libraries(installed_consumer_fortran PRIVATE
    SkalaXC::skalaxc_fortran)
  add_test(NAME installed_consumer_fortran COMMAND installed_consumer_fortran)
  set_tests_properties(installed_consumer_fortran PROPERTIES
    ENVIRONMENT "SKALAXC_MODEL_PATH=${SkalaXC_MODEL_DIR}")
endif()
