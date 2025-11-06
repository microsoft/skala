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
  device_reduction_driver.cxx
)

if( GAUXC_ENABLE_NCCL )
  target_sources( gauxc PRIVATE nccl_reduction_driver.cxx )
  message( STATUS "NCCL Has Been Enabled" )
  find_package( NCCL REQUIRED )
  target_link_libraries( gauxc PUBLIC NCCL::nccl )
  set(GAUXC_HAS_NCCL TRUE CACHE BOOL "GauXC has NCCL" FORCE)
endif()

