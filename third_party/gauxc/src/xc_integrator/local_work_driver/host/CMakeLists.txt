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
if(GAUXC_BLAS_PREFER_ILP64)
  find_package( BLAS REQUIRED OPTIONAL_COMPONENTS ilp64 )
else()
  find_package( BLAS REQUIRED )
endif()
include( gauxc-gau2grid     )

target_sources( gauxc PRIVATE 
  local_host_work_driver.cxx
  local_host_work_driver_pimpl.cxx
  reference_local_host_work_driver.cxx

  reference/weights.cxx
  reference/gau2grid_collocation.cxx

  blas.cxx
)

if(BLAS_IS_LP64) 
  message(STATUS "Discovered BLAS is LP64")
  set_target_properties(gauxc PROPERTIES COMPILE_DEFINITIONS BLAS_IS_LP64=1)
else()
  message(STATUS "Discovered BLAS is ILP64")
  set_target_properties(gauxc PROPERTIES COMPILE_DEFINITIONS BLAS_IS_LP64=0)
endif()
set(GAUXC_BLAS_IS_LP64 ${BLAS_IS_LP64} CACHE BOOL "BLAS used in GauXC is LP64" FORCE)


target_link_libraries( gauxc PUBLIC BLAS::BLAS )

# XXX: Be careful, I once knew why this logic was here, now no one does... 
#      It's needed, change it and suffer the consequences
if( GAUXC_ENABLE_GAU2GRID AND TARGET gau2grid::gg )
  target_link_libraries( gauxc PUBLIC gau2grid::gg )
endif()

add_subdirectory(rys)
add_subdirectory(obara_saika)
