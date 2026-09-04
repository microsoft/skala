# SkalaXC core library.
#
# Phase 1: the ML functional core (model resolution/loading, feature dict,
# get_exc, and the MPI gather/scatter/reorder helpers), ported from
# GauXC/skala onedft_util.* with onedft -> skala. Later phases add the host and
# device orchestration and the ABI-isolated public API.

# --- generated config header (installed model path + feature flags) --------
if(NOT EXISTS "${SKALAXC_MODEL_PATH}")
  message(WARNING
    "SkalaXC model directory not found: ${SKALAXC_MODEL_PATH}. "
    "Model aliases (LDA/PBE/TPSS) will fail at runtime unless models are "
    "installed under SKALAXC_MODEL_PATH_INSTALL, SKALAXC_MODEL_PATH is set "
    "in the runtime environment, or an explicit .fun path is provided when "
    "creating the integrator.")
endif()
set(SKALAXC_HAS_SKALA ON)
if(SKALAXC_ENABLE_CUDA)
  set(SKALAXC_HAS_CUDA ON)
endif()
if(SKALAXC_ENABLE_MPI)
  set(SKALAXC_HAS_MPI ON)
endif()
if(SKALAXC_ENABLE_OPENMP)
  set(SKALAXC_HAS_OPENMP ON)
endif()

configure_file(
  ${PROJECT_SOURCE_DIR}/include/skalaxc/skalaxc_config.hpp.in
  ${PROJECT_BINARY_DIR}/include/skalaxc/skalaxc_config.hpp
)

# Public C-compatible feature-flag header (SKALAXC_HAS_MPI etc.). Unlike the
# internal skalaxc_config.hpp above, this one is included by the public C / C++
# headers to gate the MPI_Comm-taking constructors, so it lives under skalaxc/c/.
configure_file(
  ${PROJECT_SOURCE_DIR}/include/skalaxc/c/config.h.in
  ${PROJECT_BINARY_DIR}/include/skalaxc/c/config.h
)

# --- core object library ----------------------------------------------------
add_library(skalaxc_core STATIC
  atomic_domain_assignment.cxx
  debug_log.cxx
  diagnostics.cxx
  model_grid_layout.cxx
  skala_driver.cxx
  skala_model.cxx
  host/mpi_wrapper.cxx
  host/atomic_domain_load_balancer.cxx
  host/model_grid_exchange.cxx
  host/spin_gradient.cxx
  host/skala_util.cxx
  host/skala_host_driver.cxx
)
if(SKALAXC_ENABLE_CUDA)
  target_sources(skalaxc_core PRIVATE
    device/model_grid_exchange.cxx
    device/skala_device_driver.cxx
  )
endif()
add_library(SkalaXC::core ALIAS skalaxc_core)

target_include_directories(skalaxc_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/host
)

# `gauxc` is PUBLIC so the core inherits GauXC's include tree (public + private
# src/ headers). Torch/nlohmann are PRIVATE implementation details -- they must
# never leak to SkalaXC's public consumers.
target_link_libraries(skalaxc_core
  PUBLIC
    gauxc
    Eigen3::Eigen
  PRIVATE
    ${TORCH_LIBRARIES}
    nlohmann_json::nlohmann_json
)
if(SKALAXC_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(skalaxc_core PRIVATE skalaxc_cuda CUDA::cudart)
endif()

if(SKALAXC_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(skalaxc_core PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_features(skalaxc_core PUBLIC cxx_std_17)

set_target_properties(skalaxc_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# --- public shared library (ABI-isolated) -----------------------------------
# The public API TUs are compiled directly into the shared object (NOT via an
# archive) so their skalaxc_* / SkalaXC::* symbols survive --exclude-libs,ALL.
# skalaxc_core (and, transitively, GauXC) is linked PRIVATE so neither GauXC nor
# its include tree ever reaches a SkalaXC consumer.
add_library(skalaxc SHARED
  skalaxc_api.cxx
)
add_library(SkalaXC::skalaxc ALIAS skalaxc)

target_include_directories(skalaxc
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(skalaxc PRIVATE skalaxc_core)

# Recent GNU linkers do not use an input shared library's RPATH to locate its
# transitive dependencies while linking an executable. Keep the private
# dependencies off the public link line, but make their directory available to
# build-tree consumers for link-time resolution.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TARGET libtorch)
  target_link_options(skalaxc INTERFACE
    "$<BUILD_INTERFACE:LINKER:-rpath-link,$<TARGET_FILE_DIR:libtorch>>")
endif()

foreach(_skalaxc_interface_property
    INTERFACE_INCLUDE_DIRECTORIES
    INTERFACE_LINK_LIBRARIES)
  get_target_property(_skalaxc_interface_value skalaxc
                      ${_skalaxc_interface_property})
  if(_skalaxc_interface_value MATCHES
      "[Gg]au[Xx][Cc]|[Tt]orch|c10|[Ee]igen|[Ee]xch[Cc][Xx]|[Ii]ntegrator[Xx][Xx]|[Nn]lohmann")
    message(FATAL_ERROR
      "Private dependencies must not appear in skalaxc's ${_skalaxc_interface_property}: "
      "${_skalaxc_interface_value}")
  endif()
endforeach()
unset(_skalaxc_interface_property)
unset(_skalaxc_interface_value)

target_compile_features(skalaxc PUBLIC cxx_std_17)
target_compile_definitions(skalaxc PRIVATE SKALAXC_BUILDING_LIBRARY)

set_target_properties(skalaxc PROPERTIES
  CXX_VISIBILITY_PRESET       hidden
  VISIBILITY_INLINES_HIDDEN   ON
  VERSION                     ${PROJECT_VERSION}
  SOVERSION                   ${PROJECT_VERSION_MAJOR}
)
if(SKALAXC_INSTALL_RPATH)
  set_target_properties(skalaxc PROPERTIES
    INSTALL_RPATH "${SKALAXC_INSTALL_RPATH}"
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE)
endif()

# ABI isolation (link-time): export ONLY the SkalaXC public API; hide every
# statically embedded GauXC / LibTorch symbol so a host that links its own GauXC
# cannot clash with ours.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(_skalaxc_export_control
    ${PROJECT_SOURCE_DIR}/cmake/skalaxc-exports.map)
  target_link_options(skalaxc PRIVATE
    "-Wl,--version-script=${_skalaxc_export_control}"
    "-Wl,--exclude-libs,ALL"
  )
elseif(APPLE)
  set(_skalaxc_export_control
    ${PROJECT_SOURCE_DIR}/cmake/skalaxc-exports-macos.txt)
  target_link_options(skalaxc PRIVATE
    "-Wl,-exported_symbols_list,${_skalaxc_export_control}"
  )
endif()
if(_skalaxc_export_control)
  set_target_properties(skalaxc PROPERTIES
    LINK_DEPENDS ${_skalaxc_export_control})
endif()
unset(_skalaxc_export_control)

# Public C API sources (added to the `skalaxc` target above).
add_subdirectory(c-api)
