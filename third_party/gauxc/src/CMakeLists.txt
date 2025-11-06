#
# GauXC Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy).
#
# (c) 2024-2025, Microsoft Corporation
#
# All rights reserved.
#
# See LICENSE.txt for details
#

# Parallelism
if( GAUXC_ENABLE_OPENMP ) 
  message( STATUS "GauXC Enabling OpenMP" )
  find_package( OpenMP  REQUIRED )
  set(GAUXC_HAS_OPENMP TRUE CACHE BOOL "GauXC Has OpenMP" FORCE)
else()
  message( STATUS "GauXC Disabling OpenMP" )
endif()

if( GAUXC_ENABLE_MPI )
  message( STATUS "GauXC Enabling MPI" )
  find_package( MPI REQUIRED )
  set(GAUXC_HAS_MPI TRUE CACHE BOOL "GauXC Has MPI" FORCE)
else()
  message( STATUS "GauXC Disabling MPI" )
endif()

if ( GAUXC_ENABLE_ONEDFT )
  message( STATUS "GauXC Enabling OneDFT" )
  include( gauxc-onedft      )
  set(GAUXC_HAS_ONEDFT TRUE CACHE BOOL "GauXC Has OneDFT" FORCE)
  set(GAUXC_ONEDFT_MODEL_PATH "${PROJECT_SOURCE_DIR}/data/onedft_models" )
else()
  message( STATUS "GauXC Disabling OneDFT" )
endif()

# Required Dependencies
include( gauxc-integratorxx )
include( gauxc-exchcxx      )

add_library( gauxc 
  grid.cxx 
  grid_impl.cxx 
  grid_factory.cxx
  molmeta.cxx 
  molgrid.cxx 
  molgrid_impl.cxx 
  molgrid_defaults.cxx 
  atomic_radii.cxx 
)

target_include_directories( gauxc
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

include( CheckCXXCompilerFlag )
check_cxx_compiler_flag( -Wall              GAUXC_CXX_HAS_WALL              )
check_cxx_compiler_flag( -Wextra            GAUXC_CXX_HAS_WEXTRA            )
check_cxx_compiler_flag( -Wpedantic         GAUXC_CXX_HAS_WPEDANTIC         )
check_cxx_compiler_flag( -Wnon-virtual-dtor GAUXC_CXX_HAS_WNON_VIRTUAL_DTOR )
check_cxx_compiler_flag( -Wshadow           GAUXC_CXX_HAS_WSHADOW           )

if( GAUXC_CXX_HAS_WALL )
  target_compile_options( gauxc PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wall> )
endif()

if( GAUXC_CXX_HAS_WEXTRA )
  target_compile_options( gauxc PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wextra> )
endif()

if( GAUXC_CXX_HAS_WPEDANTIC )
  target_compile_options( gauxc PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wpedantic> )
endif()

if( GAUXC_CXX_HAS_WNON_VIRTUAL_DTOR )
  target_compile_options( gauxc PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wnon-virtual-dtor -Werror=non-virtual-dtor> )
endif()

if( GAUXC_CXX_HAS_WSHADOW )
  target_compile_options( gauxc PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wshadow> )
endif()

target_link_libraries( gauxc PUBLIC 
  ExchCXX::ExchCXX 
  IntegratorXX::IntegratorXX 
)
if( TARGET OpenMP::OpenMP_CXX )
  target_link_libraries( gauxc PUBLIC OpenMP::OpenMP_CXX )
else()
  find_package(Threads REQUIRED)
  target_link_libraries( gauxc PUBLIC Threads::Threads )
endif()


if( GAUXC_HAS_MPI )
  target_link_libraries( gauxc PUBLIC MPI::MPI_C MPI::MPI_CXX )
endif()

if ( GAUXC_HAS_ONEDFT )
  target_link_libraries( gauxc PUBLIC "${TORCH_LIBRARIES}")
endif()

add_subdirectory( runtime_environment )
add_subdirectory( molecular_weights )
add_subdirectory( xc_integrator )
add_subdirectory( load_balancer )
add_subdirectory( reduction_driver )
add_subdirectory( external )


add_library( gauxc::gauxc ALIAS gauxc )


if(GAUXC_HAS_HOST)
  set(GAUXC_CPU_XC_MAX_AM     6) 
  set(GAUXC_CPU_SNLINK_MAX_AM 6)
endif()

if(GAUXC_HAS_DEVICE)
  set(GAUXC_GPU_XC_MAX_AM     4) 
  set(GAUXC_GPU_SNLINK_MAX_AM 2)
endif()

if (GAUXC_HAS_ONEDFT)
  set(GAUXC_ONEDFT_MODEL_PATH_INSTALL "${CMAKE_INSTALL_PREFIX}/share/gauxc/onedft_models")
  install( 
    DIRECTORY  ${GAUXC_ONEDFT_MODEL_PATH}
    DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gauxc"
  )
endif()

# Generate config file
configure_file( 
  ${PROJECT_SOURCE_DIR}/include/gauxc/gauxc_config.hpp.in
  ${PROJECT_BINARY_DIR}/include/gauxc/gauxc_config.hpp
)

include( GNUInstallDirs )

# TARGETS

install( TARGETS gauxc ${GAUXC_EXPORT_DEPENDENCIES}
  EXPORT gauxc-targets 
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

set_target_properties( gauxc PROPERTIES EXPORT_NAME gauxc )

set(export_properties
  # currently configurable properties 
  "GAUXC_HAS_HOST"
  "GAUXC_HAS_DEVICE"
  "GAUXC_HAS_CUDA"
  "GAUXC_HAS_HIP"
  "GAUXC_HAS_MAGMA"
  "GAUXC_HAS_CUTLASS"
  "GAUXC_HAS_NCCL"
  "GAUXC_HAS_MPI"
  "GAUXC_HAS_OPENMP"
  "GAUXC_HAS_HDF5"
  "GAUXC_HAS_ONEDFT"
  
  # may become configurable in the future
  "GAUXC_CPU_XC_MAX_AM"
  "GAUXC_CPU_SNLINK_MAX_AM"
  "GAUXC_GPU_XC_MAX_AM"
  "GAUXC_GPU_SNLINK_MAX_AM"
)

set_target_properties(gauxc
  PROPERTIES
    "GAUXC_HAS_HOST"       ${GAUXC_HAS_HOST}
    "GAUXC_HAS_DEVICE"     ${GAUXC_HAS_DEVICE}
    "GAUXC_HAS_CUDA"       ${GAUXC_HAS_CUDA}
    "GAUXC_HAS_HIP"        ${GAUXC_HAS_HIP}
    "GAUXC_HAS_MAGMA"      ${GAUXC_HAS_MAGMA}
    "GAUXC_HAS_CUTLASS"    ${GAUXC_HAS_CUTLASS}
    "GAUXC_HAS_NCCL"       ${GAUXC_HAS_NCCL}
    "GAUXC_HAS_MPI"        ${GAUXC_HAS_MPI}
    "GAUXC_HAS_OPENMP"     ${GAUXC_HAS_OPENMP}
    "GAUXC_HAS_HDF5"       ${GAUXC_HAS_HDF5}
    "GAUXC_HAS_ONEDFT"     ${GAUXC_HAS_ONEDFT}
    "GAUXC_CPU_XC_MAX_AM"     ${GAUXC_CPU_XC_MAX_AM}     
    "GAUXC_CPU_SNLINK_MAX_AM" ${GAUXC_CPU_SNLINK_MAX_AM}
    "GAUXC_GPU_XC_MAX_AM"     ${GAUXC_GPU_XC_MAX_AM}     
    "GAUXC_GPU_SNLINK_MAX_AM" ${GAUXC_GPU_SNLINK_MAX_AM}
)
set_property(TARGET gauxc APPEND PROPERTY EXPORT_PROPERTIES "${export_properties}")

# Export build tree
export(EXPORT gauxc-targets
      NAMESPACE gauxc::
      FILE "${PROJECT_BINARY_DIR}/gauxc-targets.cmake")

# Install static headers
install( 
  DIRECTORY   ${PROJECT_SOURCE_DIR}/include
  DESTINATION .
  FILES_MATCHING PATTERN "*.hpp"
)

# Install generated headers
install(
  FILES       ${PROJECT_BINARY_DIR}/include/gauxc/gauxc_config.hpp
  DESTINATION include/gauxc
)
  
   
# Export target to script
set( INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/gauxc )
install( EXPORT gauxc-targets
  FILE         gauxc-targets.cmake
  NAMESPACE    gauxc::
  DESTINATION  ${INSTALL_CONFIGDIR}
)

# Create config-version.cmake file
include( CMakePackageConfigHelpers )
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/gauxc-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)


# Setup gauxc-config.cmake
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/gauxc-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/gauxc-config.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

# Install CMake files
install( FILES
  ${PROJECT_SOURCE_DIR}/cmake/modules/FindMAGMA.cmake
  ${PROJECT_SOURCE_DIR}/cmake/modules/FindNCCL.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/gauxc-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/gauxc-config-version.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

# Install Custom Find Modules
include( ${linalg-cmake-modules_SOURCE_DIR}/LinAlgModulesMacros.cmake )
install_linalg_modules( INSTALL_CONFIGDIR )
