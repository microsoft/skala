if(NOT DEFINED Skala_GauXC_URL)
  include(skala-dep-versions)
endif()
find_package(gauxc QUIET CONFIG)
if(NOT gauxc_FOUND)
  include(FetchContent)

  message(STATUS "Could not find GauXC... Building GauXC from source")
  message(STATUS "GAUXC URL: ${Skala_GauXC_URL}")

  set(GAUXC_ENABLE_ONEDFT ON CACHE BOOL "" FORCE)
  set(GAUXC_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(GAUXC_ENABLE_OPENMP ${Skala_GauXC_ENABLE_OPENMP} CACHE BOOL "" FORCE)
  set(GAUXC_ENABLE_MPI ${Skala_GauXC_ENABLE_MPI} CACHE BOOL "" FORCE)
  set(GAUXC_ENABLE_CUDA ${Skala_GauXC_ENABLE_CUDA} CACHE BOOL "" FORCE)

  FetchContent_Declare(
    gauxc
    URL ${Skala_GauXC_URL}
    URL_HASH SHA256=${Skala_GauXC_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  FetchContent_MakeAvailable(gauxc)

else()
  if(NOT ${GAUXC_HAS_ONEDFT})
    message(FATAL_ERROR "GauXC found but without Skala support enabled")
  endif()
  if(${Skala_GauXC_ENABLE_OPENMP} AND NOT ${GAUXC_HAS_OPENMP})
    message(FATAL_ERROR "GauXC found without OpenMP support but Skala_GauXC_ENABLE_OPENMP is ON")
  endif()
  if(${Skala_GauXC_ENABLE_MPI} AND NOT ${GAUXC_HAS_MPI})
    message(FATAL_ERROR "GauXC found without MPI support but Skala_GauXC_ENABLE_MPI is ON")
  endif()
  if(${Skala_GauXC_ENABLE_CUDA} AND NOT ${GAUXC_HAS_CUDA})
    message(FATAL_ERROR "GauXC found without CUDA support but Skala_GauXC_ENABLE_CUDA is ON")
  endif()
endif()

if(GAUXC_HAS_GAU2GRID AND NOT TARGET gau2grid::gg)
  find_package(gau2grid CONFIG REQUIRED)
endif()