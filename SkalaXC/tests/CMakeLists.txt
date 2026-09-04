# SkalaXC tests. White-box reference-integration harness: links skalaxc_core
# (which brings GauXC in-build) to validate the host ML port against the HDF5
# reference fixtures.

include(FetchContent)

if(NOT EXISTS "${SKALAXC_MODEL_PATH}")
  message(FATAL_ERROR
    "SKALAXC_BUILD_TESTS=ON requires model files, but SKALAXC_MODEL_PATH does "
    "not exist: ${SKALAXC_MODEL_PATH}. Set -DSKALAXC_MODEL_PATH=/path/to/models "
    "containing ldax.fun, pbe.fun, tpss.fun, and skala-1.1.fun.")
endif()

set(_skalaxc_required_model_files ldax.fun pbe.fun tpss.fun skala-1.1.fun)
if(SKALAXC_ENABLE_CUDA)
  list(APPEND _skalaxc_required_model_files skala-1.1-cuda.fun)
endif()
set(_skalaxc_missing_model_files)
foreach(_skalaxc_model_file IN LISTS _skalaxc_required_model_files)
  if(NOT EXISTS "${SKALAXC_MODEL_PATH}/${_skalaxc_model_file}")
    list(APPEND _skalaxc_missing_model_files
      "${SKALAXC_MODEL_PATH}/${_skalaxc_model_file}")
  endif()
endforeach()

if(_skalaxc_missing_model_files)
  string(JOIN "\n  " _skalaxc_missing_lines ${_skalaxc_missing_model_files})
  message(FATAL_ERROR
    "SKALAXC_BUILD_TESTS=ON requires bundled model files. Missing:\n  "
    "${_skalaxc_missing_lines}\n"
    "Set -DSKALAXC_MODEL_PATH=/path/to/models containing ldax.fun, pbe.fun, "
    "tpss.fun, and skala-1.1.fun.")
endif()

find_package(Catch2 CONFIG QUIET)
if(Catch2_FOUND AND Catch2_VERSION VERSION_GREATER_EQUAL 3)
  message(FATAL_ERROR
    "Catch2 v${Catch2_VERSION} found, but SkalaXC unit tests require Catch2 v2 "
    "(catch2/catch.hpp). Please install v2 or let FetchContent provide it.")
endif()

if(NOT Catch2_FOUND)
  FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.10
  )

  set(CATCH_BUILD_TESTING OFF CACHE BOOL "Build Catch2 self tests" FORCE)
  set(CATCH_INSTALL_DOCS OFF CACHE BOOL "Install Catch2 docs" FORCE)
  set(CATCH_INSTALL_HELPERS OFF CACHE BOOL "Install Catch2 helpers" FORCE)

  if(DEFINED CACHE{CMAKE_WARN_DEPRECATED})
    set(_skalaxc_catch2_warn_deprecated_defined TRUE)
    get_property(_skalaxc_catch2_warn_deprecated
      CACHE CMAKE_WARN_DEPRECATED PROPERTY VALUE)
  else()
    set(_skalaxc_catch2_warn_deprecated_defined FALSE)
  endif()
  set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL
      "Suppress deprecation warnings while configuring Catch2" FORCE)

  FetchContent_MakeAvailable(catch2)

  if(_skalaxc_catch2_warn_deprecated_defined)
    set(CMAKE_WARN_DEPRECATED "${_skalaxc_catch2_warn_deprecated}"
        CACHE BOOL "Whether to issue warnings for deprecated functionality" FORCE)
  else()
    set(CMAKE_WARN_DEPRECATED ON CACHE BOOL
        "Whether to issue warnings for deprecated functionality" FORCE)
    unset(CMAKE_WARN_DEPRECATED CACHE)
  endif()
  unset(_skalaxc_catch2_warn_deprecated)
  unset(_skalaxc_catch2_warn_deprecated_defined)
endif()

# Make Catch2's `catch_discover_tests` helper available. An installed Catch2
# exposes the module through its package; a FetchContent build ships it under
# the source tree's contrib/ directory.
if(NOT Catch2_FOUND)
  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/contrib")
endif()
include(Catch)

# --- Shared test setup -------------------------------------------------------
add_library(skalaxc_test_utils STATIC test_utils.cxx)
target_include_directories(skalaxc_test_utils PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(skalaxc_test_utils PUBLIC
  skalaxc
  Eigen3::Eigen)
if(SKALAXC_HAS_HDF5)
  target_link_libraries(skalaxc_test_utils PUBLIC HighFive)
endif()
if(SKALAXC_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_link_libraries(skalaxc_test_utils PUBLIC MPI::MPI_CXX)
endif()
target_compile_features(skalaxc_test_utils PUBLIC cxx_std_17)

add_test(NAME skalaxc_config.hdf5_feature
  COMMAND ${CMAKE_COMMAND}
    -DEXPECT_HDF5=${SKALAXC_HAS_HDF5}
    -DPUBLIC_CONFIG_HEADER=${PROJECT_BINARY_DIR}/include/skalaxc/c/config.h
    -DINTERNAL_CONFIG_HEADER=${PROJECT_BINARY_DIR}/include/skalaxc/skalaxc_config.hpp
    -DPACKAGE_CONFIG=${PROJECT_BINARY_DIR}/SkalaXCConfig.cmake
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_hdf5_feature.cmake)

# --- C++ unit tests (Catch2) -------------------------------------------------
# This target is for focused white-box regression/unit coverage (e.g. MPI
# subgroup and reorder-helper tests). It coexists with existing executable
# black-box tests below.
add_executable(skalaxc_unit_test
  atomic_domain_assignment_unit_test.cxx
  diagnostics_unit_test.cxx
  skala_unit_main.cxx
  skala_util_unit_test.cxx
  skala_model_unit_test.cxx
  skala_stress_test.cxx
  spin_gradient_unit_test.cxx
  skala_host_mpi_test.cxx
  skala_reorder_unit_test.cxx
  skala_mpi_subcomm_unit_test.cxx
  skala_traditional_integration_test.cxx
  # GauXC's Gaussian-94 basis parser lives in its test tree and is not part of
  # the `gauxc` library target. SKALAXC_GAUXC_SOURCE_DIR expands to the vendored
  # GauXC source tree, so listing this file here compiles the parser directly
  # into the integration-test executable without modifying the submodule.
  ${SKALAXC_GAUXC_SOURCE_DIR}/tests/basis/parse_basis.cxx
)
if(SKALAXC_HAS_HDF5)
  target_sources(skalaxc_unit_test PRIVATE
    atomic_domain_load_balancer_unit_test.cxx
    skala_host_test.cxx
    skala_openmp_test.cxx)
endif()
if(MSVC)
  set_source_files_properties(
    ${SKALAXC_GAUXC_SOURCE_DIR}/tests/basis/parse_basis.cxx
    PROPERTIES COMPILE_OPTIONS /w)
else()
  set_source_files_properties(
    ${SKALAXC_GAUXC_SOURCE_DIR}/tests/basis/parse_basis.cxx
    PROPERTIES COMPILE_OPTIONS -w)
endif()
if(SKALAXC_ENABLE_CUDA)
  if(SKALAXC_HAS_HDF5)
    target_sources(skalaxc_unit_test PRIVATE skala_device_test.cxx)
  endif()
  target_link_libraries(skalaxc_unit_test PRIVATE CUDA::cudart)
endif()
target_include_directories(skalaxc_unit_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/src/host
  ${SKALAXC_GAUXC_SOURCE_DIR}/tests/basis
)
target_link_libraries(skalaxc_unit_test PRIVATE
  skalaxc_test_utils
  skalaxc_core
  skalaxc
  Catch2::Catch2
  ${TORCH_LIBRARIES}
)
target_compile_features(skalaxc_unit_test PRIVATE cxx_std_17)
target_compile_definitions(skalaxc_unit_test PRIVATE
  SKALAXC_TEST_REF_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ref_data"
  SKALAXC_GAUXC_REF_DATA_PATH="${SKALAXC_GAUXC_REF_DATA_DIR}"
  SKALAXC_TEST_BASIS_PATH="${SKALAXC_GAUXC_SOURCE_DIR}/tests/basis/old"
  SKALAXC_MODEL_PATH="${SKALAXC_MODEL_PATH}"
)
# Register every Catch2 TEST_CASE as its own CTest test (prefixed with
# "skalaxc_unit.") so results and failures are reported per case instead of as a
# single aggregate run. Individual tags can still be run directly, e.g.
# `skalaxc_unit_test "[skala][gradient-numerical]"`.
catch_discover_tests(skalaxc_unit_test
  TEST_PREFIX "skalaxc_unit."
  TEST_SPEC "~[mpi-only]"
  PROPERTIES ENVIRONMENT "SKALAXC_MODEL_PATH=${SKALAXC_MODEL_PATH}")

# The traditional-integration test is written to validate the replicated MPI path
# as well as the serial one. Register an explicit multi-rank CTest for it when MPI
# is enabled. Three ranks each pinned to a single OpenMP thread fit comfortably
# on a four-slot CI runner without oversubscription.
if(SKALAXC_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  add_test(NAME skalaxc_unit_mpi.subcommunicator
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:skalaxc_unit_test>
      "[skala][mpi][subcomm]" ${MPIEXEC_POSTFLAGS})
  set_tests_properties(skalaxc_unit_mpi.subcommunicator PROPERTIES
    ENVIRONMENT
      "OMP_NUM_THREADS=1;SKALAXC_MODEL_PATH=${SKALAXC_MODEL_PATH}")
  add_test(NAME skalaxc_unit_mpi.host_subcommunicator
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:skalaxc_unit_test>
      "[host-subcomm]" ${MPIEXEC_POSTFLAGS})
  set_tests_properties(skalaxc_unit_mpi.host_subcommunicator PROPERTIES
    ENVIRONMENT
      "OMP_NUM_THREADS=1;SKALAXC_MODEL_PATH=${SKALAXC_MODEL_PATH}")
  add_test(NAME skalaxc_unit_mpi.atomic_domain_ownership
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:skalaxc_unit_test>
      "[atomic-domain-ownership]" ${MPIEXEC_POSTFLAGS})
  set_tests_properties(skalaxc_unit_mpi.atomic_domain_ownership PROPERTIES
    ENVIRONMENT
      "OMP_NUM_THREADS=1;SKALAXC_MODEL_PATH=${SKALAXC_MODEL_PATH}")
  add_test(NAME skalaxc_unit_mpi.model_broadcast
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:skalaxc_unit_test>
      "[model-broadcast]" ${MPIEXEC_POSTFLAGS})
  set_tests_properties(skalaxc_unit_mpi.model_broadcast PROPERTIES
    ENVIRONMENT
      "OMP_NUM_THREADS=1;SKALAXC_MODEL_PATH=${SKALAXC_MODEL_PATH}")
  add_test(NAME skalaxc_unit_mpi.traditional_integration
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:skalaxc_unit_test>
      "[skala][traditional-integration]" ${MPIEXEC_POSTFLAGS})
  set_tests_properties(skalaxc_unit_mpi.traditional_integration PROPERTIES
    ENVIRONMENT
      "OMP_NUM_THREADS=1;SKALAXC_MODEL_PATH=${SKALAXC_MODEL_PATH}")
  if(SKALAXC_ENABLE_CUDA)
    add_test(NAME skalaxc_unit_mpi.cuda_idle_rank
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:skalaxc_unit_test>
        "[device-idle-rank]" ${MPIEXEC_POSTFLAGS})
    set_tests_properties(skalaxc_unit_mpi.cuda_idle_rank PROPERTIES
      ENVIRONMENT
        "OMP_NUM_THREADS=1;SKALAXC_MODEL_PATH=${SKALAXC_MODEL_PATH}")
    add_test(NAME skalaxc_unit_mpi.cuda_subcommunicator
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:skalaxc_unit_test>
        "[device-subcomm]" ${MPIEXEC_POSTFLAGS})
    set_tests_properties(skalaxc_unit_mpi.cuda_subcommunicator PROPERTIES
      ENVIRONMENT
        "OMP_NUM_THREADS=1;SKALAXC_MODEL_PATH=${SKALAXC_MODEL_PATH}")
  endif()
endif()

# --- black-box public-API test ----------------------------------------------
# A pure consumer: it and skalaxc_test_utils include only
# <skalaxc/skalaxc.hpp> from SkalaXC and link the public `skalaxc` shared
# library. Eigen owns matrices, and HighFive loads fixture input. Neither target
# receives GauXC include dirs or links skalaxc_core/gauxc, so a successful
# compile+link proves the public API remains ABI-isolated.
if(SKALAXC_HAS_HDF5)
  add_executable(skalaxc_public_api_test skalaxc_public_api_test.cxx)
  target_link_libraries(skalaxc_public_api_test PRIVATE
    skalaxc_test_utils
    skalaxc
    HighFive
    Eigen3::Eigen
  )
  target_compile_features(skalaxc_public_api_test PRIVATE cxx_std_17)
  if(SKALAXC_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_link_libraries(skalaxc_public_api_test PRIVATE MPI::MPI_CXX)
  endif()
  target_compile_definitions(skalaxc_public_api_test PRIVATE
    SKALAXC_EXPECTED_VERSION="${PROJECT_VERSION}"
    SKALAXC_TEST_REF_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ref_data"
    SKALAXC_GAUXC_REF_DATA_PATH="${SKALAXC_GAUXC_REF_DATA_DIR}"
  )
  add_test(NAME skalaxc_public_api COMMAND skalaxc_public_api_test)
  set_tests_properties(skalaxc_public_api PROPERTIES
    ENVIRONMENT "SKALAXC_MODEL_PATH=${SKALAXC_MODEL_PATH}")
else()
  message(STATUS "SkalaXC: HDF5 support disabled; skipping fixture-driven C++ API test")
endif()

# --- black-box C public-API test --------------------------------------------
# A pure C consumer: includes ONLY <skalaxc/skalaxc.h>, compiled as C, linking
# ONLY libskalaxc (+ the C HDF5 lib to load its own density input). Proves the C
# API is usable and ABI-isolated without any C++/GauXC/torch symbols.
if(SKALAXC_HAS_HDF5)
  find_package(HDF5 QUIET COMPONENTS C)
endif()
if(SKALAXC_HAS_HDF5 AND HDF5_FOUND AND HDF5_C_LIBRARIES)
  add_executable(skalaxc_c_api_test skalaxc_c_api_test.c)
  target_link_libraries(skalaxc_c_api_test PRIVATE skalaxc ${HDF5_C_LIBRARIES})
  target_include_directories(skalaxc_c_api_test PRIVATE ${HDF5_INCLUDE_DIRS})
  if(SKALAXC_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS C)
    target_link_libraries(skalaxc_c_api_test PRIVATE MPI::MPI_C)
  endif()
  target_compile_definitions(skalaxc_c_api_test PRIVATE
    SKALAXC_EXPECTED_VERSION="${PROJECT_VERSION}"
    SKALAXC_TEST_REF_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ref_data"
    SKALAXC_GAUXC_REF_DATA_PATH="${SKALAXC_GAUXC_REF_DATA_DIR}")
  add_test(NAME skalaxc_c_api COMMAND skalaxc_c_api_test)
  set_tests_properties(skalaxc_c_api PROPERTIES
    ENVIRONMENT "SKALAXC_MODEL_PATH=${SKALAXC_MODEL_PATH}")

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_program(SKALAXC_READELF_EXECUTABLE NAMES readelf llvm-readelf REQUIRED)
    add_test(NAME skalaxc_abi.c_consumer_dependencies
      COMMAND ${CMAKE_COMMAND}
        -DREADELF_EXECUTABLE=${SKALAXC_READELF_EXECUTABLE}
        -DCONSUMER_PATH=$<TARGET_FILE:skalaxc_c_api_test>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_c_consumer_dependencies.cmake)
  elseif(APPLE)
    find_program(SKALAXC_OTOOL_EXECUTABLE NAMES otool REQUIRED)
    add_test(NAME skalaxc_abi.c_consumer_dependencies
      COMMAND ${CMAKE_COMMAND}
        -DOTOOL_EXECUTABLE=${SKALAXC_OTOOL_EXECUTABLE}
        -DCONSUMER_PATH=$<TARGET_FILE:skalaxc_c_api_test>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_c_consumer_dependencies.cmake)
  endif()
else()
  message(STATUS "SkalaXC: HDF5 support or its C component unavailable; skipping C API test")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR APPLE)
  add_test(NAME skalaxc_abi.exported_symbols
    COMMAND ${CMAKE_COMMAND}
      -DNM_EXECUTABLE=${CMAKE_NM}
      -DNM_MODE=$<IF:$<BOOL:${APPLE}>,MACHO,ELF>
      -DLIBRARY_PATH=$<TARGET_FILE:skalaxc>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_exported_symbols.cmake)

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_program(SKALAXC_READELF_EXECUTABLE NAMES readelf llvm-readelf REQUIRED)
    add_test(NAME skalaxc_abi.library_dependencies
      COMMAND ${CMAKE_COMMAND}
        -DREADELF_EXECUTABLE=${SKALAXC_READELF_EXECUTABLE}
        -DLIBRARY_PATH=$<TARGET_FILE:skalaxc>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_library_dependencies.cmake)
  else()
    find_program(SKALAXC_OTOOL_EXECUTABLE NAMES otool REQUIRED)
    add_test(NAME skalaxc_abi.library_dependencies
      COMMAND ${CMAKE_COMMAND}
        -DOTOOL_EXECUTABLE=${SKALAXC_OTOOL_EXECUTABLE}
        -DLIBRARY_PATH=$<TARGET_FILE:skalaxc>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_library_dependencies.cmake)
  endif()
endif()

# --- black-box Fortran public-API test --------------------------------------
# A pure Fortran consumer: `use`s ONLY the `skalaxc` module and links ONLY
# libskalaxc (via skalaxc_fortran). HDF5-Fortran is linked solely so the test
# can load its own density input (unrelated to the SkalaXC ABI).
if(SKALAXC_BUILD_FORTRAN AND SKALAXC_HAS_HDF5)
  find_package(HDF5 QUIET COMPONENTS Fortran)
  if(HDF5_FOUND AND HDF5_Fortran_FOUND)
    add_executable(skalaxc_fortran_test skalaxc_fortran_test.F90)
    set_source_files_properties(skalaxc_fortran_test.F90 PROPERTIES
      Fortran_PREPROCESS ON)
    if(SKALAXC_ENABLE_MPI)
      find_package(MPI REQUIRED COMPONENTS Fortran)
      target_compile_definitions(skalaxc_fortran_test PRIVATE SKALAXC_HAS_MPI)
      target_link_libraries(skalaxc_fortran_test PRIVATE MPI::MPI_Fortran)
    endif()
    if(SKALAXC_ENABLE_CUDA)
      target_compile_definitions(skalaxc_fortran_test PRIVATE SKALAXC_HAS_CUDA)
    endif()
    target_compile_definitions(skalaxc_fortran_test PRIVATE
      SKALAXC_EXPECTED_VERSION="${PROJECT_VERSION}")
    target_link_libraries(skalaxc_fortran_test PRIVATE
      skalaxc_fortran ${HDF5_Fortran_LIBRARIES})
    target_include_directories(skalaxc_fortran_test PRIVATE ${HDF5_Fortran_INCLUDE_DIRS})
    add_test(NAME skalaxc_fortran
      COMMAND skalaxc_fortran_test ${CMAKE_CURRENT_SOURCE_DIR}/ref_data
        ${SKALAXC_GAUXC_REF_DATA_DIR})
    set_tests_properties(skalaxc_fortran PROPERTIES
      ENVIRONMENT "SKALAXC_MODEL_PATH=${SKALAXC_MODEL_PATH}")

    if(SKALAXC_ENABLE_MPI)
      add_test(NAME skalaxc_fortran_mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:skalaxc_fortran_test>
          ${CMAKE_CURRENT_SOURCE_DIR}/ref_data ${SKALAXC_GAUXC_REF_DATA_DIR}
          ${MPIEXEC_POSTFLAGS})
      set_tests_properties(skalaxc_fortran_mpi PROPERTIES
        ENVIRONMENT
          "OMP_NUM_THREADS=1;SKALAXC_MODEL_PATH=${SKALAXC_MODEL_PATH}")
    endif()
  else()
    message(STATUS "SkalaXC: HDF5 Fortran component not found; skipping Fortran test")
  endif()
endif()

if(SKALAXC_BUILD_FORTRAN)
  add_executable(skalaxc_fortran_assignment_test
    skalaxc_fortran_assignment_test.F90)
  target_link_libraries(skalaxc_fortran_assignment_test PRIVATE
    skalaxc_fortran)
  foreach(_skalaxc_handle_type IN ITEMS
      runtime molecule basis grid load-balancer molecular-weights functional
      integrator)
    add_test(NAME skalaxc_fortran_assignment.${_skalaxc_handle_type}
      COMMAND ${CMAKE_COMMAND}
        -DEXECUTABLE=$<TARGET_FILE:skalaxc_fortran_assignment_test>
        -DHANDLE_TYPE=${_skalaxc_handle_type}
        "-DEXPECTED_MESSAGE=SkalaXC handles are non-copyable; use move_from"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_expected_failure.cmake)
  endforeach()
  unset(_skalaxc_handle_type)
endif()
