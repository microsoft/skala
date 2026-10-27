# skalaxc-skala.cmake
#
# SkalaXC ML-functional dependencies: Eigen + LibTorch + nlohmann_json

if(TARGET Eigen3::Eigen)
  if(DEFINED Eigen3_VERSION)
    set(_skalaxc_eigen_version "${Eigen3_VERSION}")
  elseif(DEFINED EIGEN3_VERSION_STRING)
    set(_skalaxc_eigen_version "${EIGEN3_VERSION_STRING}")
  else()
    get_target_property(_skalaxc_eigen_major Eigen3::Eigen
      INTERFACE_EIGEN3_MAJOR_VERSION)
    if(_skalaxc_eigen_major)
      set(_skalaxc_eigen_version "${_skalaxc_eigen_major}.0.0")
    endif()
  endif()

  if(NOT _skalaxc_eigen_version OR
     _skalaxc_eigen_version VERSION_LESS 5.0.0 OR
     NOT _skalaxc_eigen_version VERSION_LESS 6.0.0)
    message(FATAL_ERROR
      "SkalaXC requires Eigen >=5.0.0,<6.0.0, but Eigen3::Eigen already "
      "exists with version '${_skalaxc_eigen_version}'.")
  endif()
else()
  find_package(Eigen3 5.0...<6 CONFIG QUIET NO_MODULE)
  if(NOT Eigen3_FOUND)
    message(STATUS "SkalaXC: Eigen 5 not found... fetching Eigen 5.0.1")
    include(FetchContent)
    FetchContent_Declare(
      eigen3
      URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(eigen3)
    set_property(DIRECTORY "${eigen3_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
  endif()
endif()

unset(_skalaxc_eigen_major)
unset(_skalaxc_eigen_version)

if(NOT TARGET nlohmann_json::nlohmann_json)
  find_package(nlohmann_json QUIET)
  if(NOT nlohmann_json_FOUND)
    message(STATUS "SkalaXC: could not find nlohmann_json... fetching")
    include(FetchContent)
    FetchContent_Declare(
      nlohmann_json
      GIT_REPOSITORY https://github.com/nlohmann/json.git
      GIT_TAG        v3.11.3
    )
    FetchContent_MakeAvailable(nlohmann_json)
  endif()
endif()

# Store and restore CMAKE_CUDA_ARCHITECTURES in case Torch clobbers it.
set(_SKALAXC_PREV_CUDA_ARCHS "${CMAKE_CUDA_ARCHITECTURES}")

# CUDA-enabled TorchConfig expects CUDAToolkit imported targets to exist.
if(SKALAXC_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# Some LibTorch packages enable optional Kineto support but do not ship the
# corresponding library. TorchConfig warns and continues successfully; keep
# that third-party warning local while preserving package lookup errors.
if(DEFINED CMAKE_MESSAGE_LOG_LEVEL)
  set(_SKALAXC_PREV_MESSAGE_LOG_LEVEL_DEFINED TRUE)
  set(_SKALAXC_PREV_MESSAGE_LOG_LEVEL "${CMAKE_MESSAGE_LOG_LEVEL}")
else()
  set(_SKALAXC_PREV_MESSAGE_LOG_LEVEL_DEFINED FALSE)
endif()
set(CMAKE_MESSAGE_LOG_LEVEL ERROR)
find_package(Torch REQUIRED)
if(_SKALAXC_PREV_MESSAGE_LOG_LEVEL_DEFINED)
  set(CMAKE_MESSAGE_LOG_LEVEL "${_SKALAXC_PREV_MESSAGE_LOG_LEVEL}")
else()
  unset(CMAKE_MESSAGE_LOG_LEVEL)
endif()
unset(_SKALAXC_PREV_MESSAGE_LOG_LEVEL)
unset(_SKALAXC_PREV_MESSAGE_LOG_LEVEL_DEFINED)

set(SKALAXC_TORCH_VERSION "${Torch_VERSION}" CACHE INTERNAL
  "LibTorch version used to build SkalaXC")
if(DEFINED SKALAXC_TORCH_CUDA_VERSION)
  set(_skalaxc_torch_cuda_version "${SKALAXC_TORCH_CUDA_VERSION}")
elseif(DEFINED CUDAToolkit_VERSION)
  set(_skalaxc_torch_cuda_version "${CUDAToolkit_VERSION}")
else()
  set(_skalaxc_torch_cuda_version "none")
endif()
set(SKALAXC_TORCH_CUDA_VERSION "${_skalaxc_torch_cuda_version}" CACHE STRING
  "CUDA runtime line used by LibTorch, or none" FORCE)
unset(_skalaxc_torch_cuda_version)
if(SKALAXC_ENABLE_CUDA)
  set(SKALAXC_CUDA_TOOLKIT_VERSION "${CMAKE_CUDA_COMPILER_VERSION}"
    CACHE INTERNAL "CUDA toolkit version used to build SkalaXC")
  if(SKALAXC_TORCH_CUDA_VERSION STREQUAL "none")
    message(FATAL_ERROR
      "CUDA-enabled SkalaXC requires CUDA-enabled LibTorch metadata")
  endif()
  string(REGEX MATCH "^[0-9]+" _skalaxc_torch_cuda_major
    "${SKALAXC_TORCH_CUDA_VERSION}")
  string(REGEX MATCH "^[0-9]+" _skalaxc_toolkit_cuda_major
    "${SKALAXC_CUDA_TOOLKIT_VERSION}")
  if(NOT _skalaxc_torch_cuda_major OR NOT _skalaxc_toolkit_cuda_major OR
     NOT _skalaxc_torch_cuda_major STREQUAL _skalaxc_toolkit_cuda_major)
    message(FATAL_ERROR
      "SkalaXC CUDA toolkit ${SKALAXC_CUDA_TOOLKIT_VERSION} does not match "
      "the LibTorch CUDA ${SKALAXC_TORCH_CUDA_VERSION} major compatibility "
      "family")
  endif()
  unset(_skalaxc_toolkit_cuda_major)
  unset(_skalaxc_torch_cuda_major)
else()
  set(SKALAXC_CUDA_TOOLKIT_VERSION "none" CACHE INTERNAL
    "CUDA toolkit version used to build SkalaXC")
endif()
if(DEFINED SKALAXC_TORCH_CXX11_ABI)
  set(_skalaxc_torch_cxx11_abi "${SKALAXC_TORCH_CXX11_ABI}")
else()
  set(_skalaxc_torch_cxx11_abi "unknown")
endif()
set(SKALAXC_TORCH_CXX11_ABI "${_skalaxc_torch_cxx11_abi}" CACHE STRING
  "LibTorch libstdc++ C++11 ABI (0, 1, or unknown)" FORCE)
unset(_skalaxc_torch_cxx11_abi)
set_property(CACHE SKALAXC_TORCH_CXX11_ABI PROPERTY STRINGS 0 1 unknown)
if(SKALAXC_TORCH_CXX11_ABI STREQUAL "unknown" AND
   TORCH_CXX_FLAGS MATCHES "_GLIBCXX_USE_CXX11_ABI=([01])")
  set(SKALAXC_TORCH_CXX11_ABI "${CMAKE_MATCH_1}" CACHE STRING
    "LibTorch libstdc++ C++11 ABI (0, 1, or unknown)" FORCE)
endif()
if(NOT SKALAXC_TORCH_CXX11_ABI MATCHES "^(0|1|unknown)$")
  message(FATAL_ERROR
    "SKALAXC_TORCH_CXX11_ABI must be 0, 1, or unknown, got "
    "'${SKALAXC_TORCH_CXX11_ABI}'")
endif()

# Restore CMAKE_CUDA_ARCHITECTURES (Torch may set it to OFF).
if(NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "${_SKALAXC_PREV_CUDA_ARCHS}")
  set(CMAKE_CUDA_ARCHITECTURES "${_SKALAXC_PREV_CUDA_ARCHS}" CACHE STRING "" FORCE)
  message(WARNING "SkalaXC: Torch changed CMAKE_CUDA_ARCHITECTURES. Restored: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
unset(_SKALAXC_PREV_CUDA_ARCHS)

# Strip Torch-injected -gencode flags from CMAKE_CUDA_FLAGS (PyTorch issue #71379).
string(REGEX REPLACE " -gencode [^ ]+" "" _cleaned_cuda_flags "${CMAKE_CUDA_FLAGS}")
if(NOT "${_cleaned_cuda_flags}" STREQUAL "${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CUDA_FLAGS "${_cleaned_cuda_flags}" CACHE STRING "" FORCE)
  message(WARNING "SkalaXC: stripped Torch-injected -gencode flags from CMAKE_CUDA_FLAGS")
endif()

if(TARGET torch::nvtoolsext)
  list(REMOVE_ITEM TORCH_LIBRARIES torch::nvtoolsext)
endif()
message(STATUS "SkalaXC: Torch libraries: ${TORCH_LIBRARIES}")
