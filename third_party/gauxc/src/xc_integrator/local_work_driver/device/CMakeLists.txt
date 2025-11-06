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
target_sources( gauxc PRIVATE 
  local_device_work_driver.cxx
  local_device_work_driver_pimpl.cxx

  scheme1_base.cxx
  scheme1_data_base.cxx
)

if( GAUXC_ENABLE_MAGMA )
  find_package( MAGMA REQUIRED )
  target_link_libraries( gauxc PUBLIC MAGMA::magma )
  set(GAUXC_HAS_MAGMA TRUE CACHE BOOL "GauXC has MAGMA" FORCE)

  target_sources( gauxc PRIVATE 
    scheme1_magma_base.cxx
    scheme1_magma_data_base.cxx
  )
endif()

if(GAUXC_HAS_CUDA)
  add_subdirectory( cuda )
endif()

if(GAUXC_HAS_HIP)
  add_subdirectory( hip )
endif()
