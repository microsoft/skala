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
  load_balancer_device_factory.cxx
)

if( GAUXC_HAS_CUDA )
  add_subdirectory( cuda )
endif()

if( GAUXC_HAS_HIP )
  add_subdirectory( hip )
endif()
