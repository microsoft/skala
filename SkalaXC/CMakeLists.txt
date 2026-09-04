cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# SkalaXC: standalone ML exchange-correlation functional library.

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release RelWithDebInfo MinSizeRel)
endif()

project(SkalaXC VERSION 0.1.0 LANGUAGES C CXX)

list(PREPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(GNUInstallDirs)

function(_skalaxc_enable_warnings target_name)

  target_compile_options(${target_name} PRIVATE
    "$<$<COMPILE_LANGUAGE:C,CXX>:-Wall>"
    "$<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>"
    "$<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>"
    "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=-Wall>"
    "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=-Wextra>"
    "$<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-Wall>"
    "$<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-Wextra>"
  )


  target_compile_options(${target_name} PRIVATE
    "$<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-Wall>"
    "$<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-Wextra>"
    "$<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-Wpedantic>"
  )
endfunction()

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
option(SKALAXC_ENABLE_CUDA   "Enable the CUDA (GPU) backend"                 OFF)
option(SKALAXC_ENABLE_MPI    "Enable MPI support"                            OFF)
option(SKALAXC_ENABLE_OPENMP "Enable OpenMP support"                         ON)
option(SKALAXC_ENABLE_HDF5   "Enable HDF5 molecule and basis-set I/O"         ON)
option(SKALAXC_BUILD_FORTRAN "Build the SkalaXC Fortran API (iso_c_binding)" ON)
option(SKALAXC_BUILD_TESTS   "Build SkalaXC tests"                           OFF)
option(SKALAXC_BUILD_EXAMPLES "Build the SkalaXC C/C++/Fortran example programs" OFF)
option(SKALAXC_BUILD_DOCS "Build project-wide SkalaXC Doxygen documentation" OFF)
option(SKALAXC_DOWNLOAD_MODELS "Download baseline .fun models when required and missing" ON)
option(SKALAXC_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
set(SKALAXC_INSTALL_RPATH "" CACHE STRING
  "Optional relocatable install RPATH for libskalaxc")

if(SKALAXC_ENABLE_SANITIZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR
      "SKALAXC_ENABLE_SANITIZERS requires a GNU or Clang toolchain")
  endif()
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C,CXX,Fortran>:-fsanitize=address,undefined>"
    "$<$<COMPILE_LANGUAGE:C,CXX,Fortran>:-fno-omit-frame-pointer>")
  add_link_options(-fsanitize=address,undefined)
endif()

set(SKALAXC_MODEL_PATH "${PROJECT_SOURCE_DIR}/data/skala_models"
  CACHE PATH "Path to directory containing bundled SkalaXC .fun models")
set(SKALAXC_MODEL_DOWNLOAD_DIR "${PROJECT_BINARY_DIR}/data/skala_models"
  CACHE PATH "Directory used when SKALAXC_DOWNLOAD_MODELS=ON")
set(SKALAXC_MODEL_PATH_INSTALL "${CMAKE_INSTALL_FULL_DATADIR}/skalaxc/skala_models"
  CACHE PATH "Installed directory containing SkalaXC .fun models")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT BLAS_LIBRARIES AND DEFINED ENV{CONDA_PREFIX})
  find_library(_skalaxc_conda_blas
    NAMES blas
    HINTS "$ENV{CONDA_PREFIX}/lib"
    NO_DEFAULT_PATH)
  if(_skalaxc_conda_blas)
    set(BLAS_LIBRARIES "${_skalaxc_conda_blas}" CACHE STRING
      "BLAS library selected from the active Conda or Pixi environment")
  endif()
  unset(_skalaxc_conda_blas CACHE)
endif()

if(SKALAXC_BUILD_DOCS)
  find_package(Doxygen 1.9 REQUIRED)
  set(SKALAXC_DOXYGEN_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
  set(SKALAXC_DOXYGEN_BINARY_DIR "${PROJECT_BINARY_DIR}")
  set(SKALAXC_DOXYGEN_PROJECT_VERSION "${PROJECT_VERSION}")
  include(skalaxc-doxygen)
  add_custom_target(skalaxc_doxygen
    COMMAND "${DOXYGEN_EXECUTABLE}" "${PROJECT_BINARY_DIR}/Doxyfile"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Generating strict project-wide SkalaXC documentation"
    VERBATIM
  )
endif()

# The Fortran API binds only to the C API via iso_c_binding; enable the language
# lazily so C/C++-only consumers never need a Fortran compiler.
if(SKALAXC_BUILD_FORTRAN)
  include(CheckLanguage)
  check_language(Fortran)
  if(NOT CMAKE_Fortran_COMPILER)
    message(FATAL_ERROR
      "SKALAXC_BUILD_FORTRAN=ON but no Fortran compiler was found. "
      "Install a Fortran compiler (for example gfortran) or rerun with "
      "-DSKALAXC_BUILD_FORTRAN=OFF.")
  endif()
  enable_language(Fortran)
endif()

# The CUDA (GPU) backend is optional and OFF by default. Enable the CUDA language
# lazily and only if a CUDA compiler is actually present, so the host build is
# never affected on machines without a CUDA toolchain. If the option is ON but no
# compiler is found we downgrade to a warning rather than failing the configure.
if(SKALAXC_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    include(skalaxc-check-cuda-compiler)
    _skalaxc_check_cuda_compiler_compatibility("${CUDAToolkit_VERSION}")
  endif()

  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(WARNING
      "SKALAXC_ENABLE_CUDA=ON but no CUDA compiler was found; "
      "the SkalaXC CUDA backend will be skipped.")
    set(SKALAXC_ENABLE_CUDA OFF)
  endif()
endif()

# Ensure baseline models are available when requested. Runtime users can
# override named-model lookup with the SKALAXC_MODEL_PATH environment variable.
set(_skalaxc_required_models ldax.fun pbe.fun tpss.fun skala-1.1.fun)
if(SKALAXC_ENABLE_CUDA)
  list(APPEND _skalaxc_required_models skala-1.1-cuda.fun)
endif()
set(_skalaxc_models_missing FALSE)
foreach(_skalaxc_model IN LISTS _skalaxc_required_models)
  if(NOT EXISTS "${SKALAXC_MODEL_PATH}/${_skalaxc_model}")
    set(_skalaxc_models_missing TRUE)
  endif()
endforeach()
if(SKALAXC_DOWNLOAD_MODELS AND
   (SKALAXC_BUILD_TESTS OR SKALAXC_BUILD_EXAMPLES OR
    _skalaxc_models_missing))
  include(skalaxc-models)
  skalaxc_download_baseline_models(
    "${SKALAXC_MODEL_DOWNLOAD_DIR}"
    ${_skalaxc_required_models}
  )
  set(SKALAXC_MODEL_PATH "${SKALAXC_MODEL_DOWNLOAD_DIR}" CACHE PATH
      "Path to directory containing bundled SkalaXC .fun models" FORCE)
  message(STATUS "SkalaXC: using downloaded verified models at ${SKALAXC_MODEL_PATH}")
endif()
unset(_skalaxc_model)
unset(_skalaxc_models_missing)
unset(_skalaxc_required_models)

# ----------------------------------------------------------------------------
# ABI isolation: hide all symbols by default. Only the SkalaXC public API
# (skalaxc_* / SkalaXC_*) is exported, enforced by the version script wired onto
# the final shared library in src/. This prevents GauXC / LibTorch symbol
# clashes with a host application that links its own GauXC.
# ----------------------------------------------------------------------------
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET   hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# ----------------------------------------------------------------------------
# GauXC master (source-tree dependency) -> provides target `gauxc`
# ----------------------------------------------------------------------------
include(skalaxc-gauxc)
if(SKALAXC_ENABLE_HDF5 AND NOT GAUXC_HAS_HDF5)
  message(FATAL_ERROR
    "SKALAXC_ENABLE_HDF5=ON, but GauXC could not enable HDF5 support. "
    "Install HDF5 or configure with -DSKALAXC_ENABLE_HDF5=OFF.")
endif()
set(SKALAXC_HAS_HDF5 ${GAUXC_HAS_HDF5})
message(STATUS "SkalaXC: HDF5 support ${SKALAXC_HAS_HDF5}")
# ML VXC/gradient CUDA kernels are defined before the core so the latter can
# link their private launchers when CUDA is enabled.
if(SKALAXC_ENABLE_CUDA)
add_subdirectory(src/device)
endif()
# ----------------------------------------------------------------------------
# SkalaXC library: the ML exchange-correlation functional built on GauXC's
# reusable internals. Requires LibTorch + nlohmann_json.
# ----------------------------------------------------------------------------
include(skalaxc-skala)   # LibTorch + nlohmann_json
add_subdirectory(src)
if(SKALAXC_BUILD_FORTRAN)
add_subdirectory(src/fortran)
endif()
# ML VXC/gradient CUDA kernels (option-gated; see src/device/CMakeLists.txt).
# OFF by default -- never compiled on a host-only toolchain.
# Minimal C/C++/Fortran consumer examples (option-gated; see examples/).
if(SKALAXC_BUILD_EXAMPLES)
add_subdirectory(examples)
endif()



# ----------------------------------------------------------------------------
# Tests (white-box reference-integration harness; may use GauXC internals in-build).
# ----------------------------------------------------------------------------
if(SKALAXC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

set(_skalaxc_owned_targets
  skalaxc_cuda
  skalaxc_core
  skalaxc
  skalaxc_fortran
  skalaxc_eval_cpp
  skalaxc_eval_c
  skalaxc_eval_fortran
  skalaxc_unit_test
  skalaxc_public_api_test
  skalaxc_c_api_test
  skalaxc_fortran_test
)
foreach(_skalaxc_target IN LISTS _skalaxc_owned_targets)
  if(TARGET ${_skalaxc_target})
    _skalaxc_enable_warnings(${_skalaxc_target})
  endif()
endforeach()

# Catch2 v2 uses __COUNTER__ for test registration, which Clang 22 diagnoses
# as a C2y extension under -Wpedantic.
if(TARGET skalaxc_unit_test)
  target_compile_options(skalaxc_unit_test PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wno-c2y-extensions>"
  )
endif()

unset(_skalaxc_target)
unset(_skalaxc_owned_targets)

# ----------------------------------------------------------------------------
# Installable public package. Private GauXC, Torch, Eigen, and core targets are
# deliberately absent from this export.
# ----------------------------------------------------------------------------
set(SKALAXC_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/SkalaXC")
set(SKALAXC_INSTALL_MODEL_DIR
  "${CMAKE_INSTALL_DATADIR}/skalaxc/skala_models")

install(TARGETS skalaxc
  EXPORT SkalaXCCoreTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Core
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Core
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Core
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(SKALAXC_BUILD_FORTRAN)
  install(TARGETS skalaxc_fortran
    EXPORT SkalaXCFortranTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Fortran
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  get_target_property(_skalaxc_fortran_module_dir skalaxc_fortran
    Fortran_MODULE_DIRECTORY)
  install(DIRECTORY "${_skalaxc_fortran_module_dir}/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT Fortran
    OPTIONAL
    FILES_MATCHING PATTERN "*.mod")
  unset(_skalaxc_fortran_module_dir)
endif()

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/skalaxc"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT Core
  PATTERN "*.in" EXCLUDE)
install(FILES "${PROJECT_BINARY_DIR}/include/skalaxc/c/config.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/skalaxc/c
  COMPONENT Core)
install(DIRECTORY "${SKALAXC_MODEL_PATH}/"
  DESTINATION ${SKALAXC_INSTALL_MODEL_DIR}
  COMPONENT Core
  OPTIONAL
  FILES_MATCHING PATTERN "*.fun")

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/SkalaXCConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/SkalaXCConfig.cmake"
  INSTALL_DESTINATION "${SKALAXC_INSTALL_CMAKEDIR}"
  PATH_VARS SKALAXC_INSTALL_MODEL_DIR)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/SkalaXCConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(EXPORT SkalaXCCoreTargets
  FILE SkalaXCCoreTargets.cmake
  NAMESPACE SkalaXC::
  DESTINATION "${SKALAXC_INSTALL_CMAKEDIR}"
  COMPONENT Core)
if(SKALAXC_BUILD_FORTRAN)
  install(EXPORT SkalaXCFortranTargets
    FILE SkalaXCFortranTargets.cmake
    NAMESPACE SkalaXC::
    DESTINATION "${SKALAXC_INSTALL_CMAKEDIR}"
    COMPONENT Fortran)
endif()
install(FILES
  "${PROJECT_BINARY_DIR}/SkalaXCConfig.cmake"
  "${PROJECT_BINARY_DIR}/SkalaXCConfigVersion.cmake"
  DESTINATION "${SKALAXC_INSTALL_CMAKEDIR}"
  COMPONENT Core)
