cmake_minimum_required( VERSION 3.20 FATAL_ERROR )

include(FetchContent)
set( FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Disable FC Updates" )

project( GauXC VERSION 1.0.0 LANGUAGES C CXX )

# Place local modules in the path
list( PREPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )
list( PREPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules )
include( gauxc-linalg-modules )

# Guard some options settings to only default when not a subproject
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # Populate BUILD_TESTING prior to dependencies to avoid clash
  include(CTest)

  # Default the built type
  if( NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES )
    set( CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build" FORCE )
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
  endif()
endif()



# GauXC Options
option( GAUXC_ENABLE_HOST       "Enable Host Integrator"      ON  )
option( GAUXC_ENABLE_CUDA       "Enable CUDA Bindings"        OFF )
option( GAUXC_ENABLE_HIP        "Enable HIP Bindings"         OFF )
option( GAUXC_ENABLE_MPI        "Enable MPI Bindings"         ON  )
option( GAUXC_ENABLE_OPENMP     "Enable OpenMP Compilation"   ON  )
option( GAUXC_ENABLE_TESTS      "Enable Unit Tests"           ON  )
option( GAUXC_ENABLE_GAU2GRID   "Enable Gau2Grid Collocation" ON  )
option( GAUXC_ENABLE_HDF5       "Enable HDF5 Bindings"        ON  )
option( GAUXC_ENABLE_ONEDFT     "Enable ONEDFT Functional"    ON )
option( GAUXC_USE_FAST_RSQRT    "Enable Fast RSQRT"           OFF )
option( GAUXC_BLAS_PREFER_ILP64 "Prefer ILP64 for host BLAS"  OFF )
option( GAUXC_LINK_CUDA_STATIC  "Link GauXC with static CUDA libs"  OFF )

include(CMakeDependentOption)
cmake_dependent_option( GAUXC_ENABLE_MAGMA    
  "Enable MAGMA Linear Algebra"           ON
  "GAUXC_ENABLE_CUDA OR GAUXC_ENABLE_HIP" OFF  
)
cmake_dependent_option( GAUXC_ENABLE_NCCL 
  "Enable NCCL Collectives" OFF  
  "GAUXC_ENABLE_CUDA"       OFF 
)
cmake_dependent_option( GAUXC_ENABLE_CUTLASS  
  "Enable CUTLASS Linear Algebra" OFF  
  "GAUXC_ENABLE_CUDA"             OFF 
)

# Default the feature variables
set( GAUXC_HAS_HOST       FALSE CACHE BOOL "" FORCE )
set( GAUXC_HAS_CUDA       FALSE CACHE BOOL "" FORCE )
set( GAUXC_HAS_HIP        FALSE CACHE BOOL "" FORCE )
set( GAUXC_HAS_MPI        FALSE CACHE BOOL "" FORCE )
set( GAUXC_HAS_OPENMP     FALSE CACHE BOOL "" FORCE )
set( GAUXC_HAS_GAU2GRID   FALSE CACHE BOOL "" FORCE )
set( GAUXC_HAS_HDF5       FALSE CACHE BOOL "" FORCE )
set( GAUXC_HAS_MAGMA      FALSE CACHE BOOL "" FORCE )
set( GAUXC_HAS_NCCL       FALSE CACHE BOOL "" FORCE )
set( GAUXC_HAS_CUTLASS    FALSE CACHE BOOL "" FORCE )
set( GAUXC_HAS_ONEDFT    FALSE CACHE BOOL "" FORCE )
set( GAUXC_BLAS_IS_LP64   FALSE CACHE BOOL "" FORCE )

mark_as_advanced( FORCE 
  GAUXC_HAS_HOST     
  GAUXC_HAS_CUDA     
  GAUXC_HAS_HIP      
  GAUXC_HAS_MPI      
  GAUXC_HAS_OPENMP   
  GAUXC_HAS_GAU2GRID 
  GAUXC_HAS_HDF5     
  GAUXC_HAS_MAGMA    
  GAUXC_HAS_NCCL     
  GAUXC_HAS_CUTLASS  
  GAUXC_HAS_ONEDFT
  GAUXC_BLAS_IS_LP64 
)


if( NOT GAUXC_ENABLE_GAU2GRID )
  message( FATAL_ERROR "Gau2Grid is currently a required dependency which 
                        will be made optional in a future release of GauXC [WIP]" ) 
endif()


if( GAUXC_ENABLE_HOST ) 
  set(GAUXC_HAS_HOST TRUE CACHE BOOL "GauXC has Host Bindings" FORCE)
endif()

if( GAUXC_ENABLE_CUDA )
  enable_language( CUDA )
  set( GAUXC_HAS_CUDA TRUE CACHE BOOL "GauXC has CUDA and will build CUDA bindings" FORCE )
endif()

if( GAUXC_ENABLE_HIP )
  enable_language( HIP )
  set( GAUXC_HAS_HIP TRUE CACHE BOOL "GauXC has HIP and will build HIP bindings" FORCE )
endif()

# Decided if we're compiling device bindings
if( GAUXC_HAS_CUDA OR GAUXC_HAS_HIP )
  set( GAUXC_HAS_DEVICE TRUE CACHE BOOL "Enable Device Code" )
else()
  set( GAUXC_HAS_DEVICE FALSE CACHE BOOL "Enable Device Code" )
endif()



if( NOT (${GAUXC_HAS_HOST} OR ${GAUXC_HAS_DEVICE}) )
  message( FATAL_ERROR "Neither Host nor Device Integrators have been enabled!" )
endif()


add_subdirectory( src )

if( CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND GAUXC_ENABLE_TESTS AND BUILD_TESTING )
  add_subdirectory( tests )
endif()

list(REMOVE_AT CMAKE_MODULE_PATH 0)
list(REMOVE_AT CMAKE_MODULE_PATH 0)

if( linalg-cmake-modules_POPULATED )
  list(REMOVE_AT CMAKE_MODULE_PATH 0)
endif()
