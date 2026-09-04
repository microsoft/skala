cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(SkalaXCPython VERSION 0.1.0 LANGUAGES CXX)

find_package(Python 3.11...<3.14 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(SkalaXC 0.1.0 EXACT CONFIG REQUIRED)

set(SKALAXC_PYTHON_LAYOUT "WHEEL" CACHE STRING
  "Python package layout: WHEEL bundles native assets; CONDA uses the environment prefix")
set_property(CACHE SKALAXC_PYTHON_LAYOUT PROPERTY STRINGS WHEEL CONDA)
set(_skalaxc_python_layouts WHEEL CONDA)
if(NOT SKALAXC_PYTHON_LAYOUT IN_LIST _skalaxc_python_layouts)
  message(FATAL_ERROR
    "SKALAXC_PYTHON_LAYOUT must be WHEEL or CONDA, got '${SKALAXC_PYTHON_LAYOUT}'")
endif()
unset(_skalaxc_python_layouts)

if(NOT DEFINED SkalaXC_TORCH_VERSION OR SkalaXC_TORCH_VERSION STREQUAL "")
  message(FATAL_ERROR
    "The selected SkalaXC package does not publish its LibTorch version")
endif()
if(NOT DEFINED SkalaXC_TORCH_CUDA_VERSION OR
   SkalaXC_TORCH_CUDA_VERSION STREQUAL "")
  set(SkalaXC_TORCH_CUDA_VERSION "none")
endif()
if(NOT DEFINED SkalaXC_CUDA_TOOLKIT_VERSION OR
   SkalaXC_CUDA_TOOLKIT_VERSION STREQUAL "")
  set(SkalaXC_CUDA_TOOLKIT_VERSION "none")
endif()

nanobind_add_module(_skalaxc src/bindings/module.cxx)
target_link_libraries(_skalaxc PRIVATE SkalaXC::skalaxc)
target_compile_features(_skalaxc PRIVATE cxx_std_17)
set(_skalaxc_stub_common_patterns
  "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/stubgen-common.patterns")
set(_skalaxc_stub_variant_patterns)
if(SkalaXC_MPI_ENABLED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
      "import mpi4py; print(mpi4py.get_include(), end='')"
    RESULT_VARIABLE _skalaxc_mpi4py_result
    OUTPUT_VARIABLE _skalaxc_mpi4py_include
    ERROR_VARIABLE _skalaxc_mpi4py_error)
  if(NOT _skalaxc_mpi4py_result EQUAL 0 OR
     NOT EXISTS "${_skalaxc_mpi4py_include}/mpi4py/mpi4py.h")
    message(FATAL_ERROR
      "MPI-enabled SkalaXC Python bindings require mpi4py for the selected "
      "Python interpreter: ${_skalaxc_mpi4py_error}")
  endif()
  target_include_directories(_skalaxc PRIVATE "${_skalaxc_mpi4py_include}")
  if(SkalaXC_CUDA_ENABLED)
    set(_skalaxc_stub_variant_patterns
      "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/stubgen-cuda-mpi.patterns")
  else()
    set(_skalaxc_stub_variant_patterns
      "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/stubgen-mpi.patterns")
  endif()
endif()
file(READ "${_skalaxc_stub_common_patterns}" _skalaxc_stub_patterns_content)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${_skalaxc_stub_common_patterns}")
if(_skalaxc_stub_variant_patterns)
  file(READ "${_skalaxc_stub_variant_patterns}"
    _skalaxc_stub_variant_patterns_content)
  string(APPEND _skalaxc_stub_patterns_content
    "\n${_skalaxc_stub_variant_patterns_content}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_skalaxc_stub_variant_patterns}")
endif()
set(_skalaxc_stub_patterns
  "${CMAKE_CURRENT_BINARY_DIR}/_skalaxc-stubgen.patterns")
file(WRITE "${_skalaxc_stub_patterns}" "${_skalaxc_stub_patterns_content}")
nanobind_add_stub(_skalaxc_stub
  MODULE _skalaxc
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_skalaxc.pyi"
  PYTHON_PATH "$<TARGET_FILE_DIR:_skalaxc>"
  PATTERN_FILE "${_skalaxc_stub_patterns}"
  DEPENDS _skalaxc)
if(APPLE)
  set(_skalaxc_install_origin "@loader_path")
else()
  set(_skalaxc_install_origin "$ORIGIN")
endif()
if(SKALAXC_PYTHON_LAYOUT STREQUAL "CONDA")
  string(APPEND _skalaxc_install_origin "/../../..")
endif()
set_target_properties(_skalaxc PROPERTIES
  BUILD_RPATH "$<TARGET_FILE_DIR:SkalaXC::skalaxc>"
  INSTALL_RPATH "${_skalaxc_install_origin}"
  INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE)
unset(_skalaxc_install_origin)

set(_skalaxc_python_cuda False)
if(SkalaXC_CUDA_ENABLED)
  set(_skalaxc_python_cuda True)
endif()
set(_skalaxc_python_mpi False)
if(SkalaXC_MPI_ENABLED)
  set(_skalaxc_python_mpi True)
endif()
set(_skalaxc_python_openmp False)
if(SkalaXC_OpenMP_ENABLED)
  set(_skalaxc_python_openmp True)
endif()
set(_skalaxc_python_hdf5 False)
if(SkalaXC_HDF5_ENABLED)
  set(_skalaxc_python_hdf5 True)
endif()

configure_file(
  src/skalaxc/_build_info.py.in
  generated/skalaxc/_build_info.py
  @ONLY)

install(TARGETS _skalaxc LIBRARY DESTINATION skalaxc)
if(SKALAXC_PYTHON_LAYOUT STREQUAL "WHEEL")
  install(IMPORTED_RUNTIME_ARTIFACTS SkalaXC::skalaxc
    LIBRARY DESTINATION skalaxc
    RUNTIME DESTINATION skalaxc)
endif()
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/generated/skalaxc/_build_info.py"
  "${CMAKE_CURRENT_BINARY_DIR}/_skalaxc.pyi"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/skalaxc/__init__.pyi"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/skalaxc/py.typed"
  DESTINATION skalaxc)
if(SKALAXC_PYTHON_LAYOUT STREQUAL "WHEEL" AND
  EXISTS "${SkalaXC_MODEL_DIR}")
  install(DIRECTORY "${SkalaXC_MODEL_DIR}/"
    DESTINATION skalaxc/models
    FILES_MATCHING PATTERN "*.fun")
endif()

unset(_skalaxc_mpi4py_error)
unset(_skalaxc_mpi4py_include)
unset(_skalaxc_mpi4py_result)