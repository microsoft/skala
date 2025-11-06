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
  load_balancer.cxx 
  load_balancer_impl.cxx 
  load_balancer_factory.cxx
  rebalance.cxx

  host/load_balancer_host_factory.cxx
  host/replicated_host_load_balancer.cxx 
  host/petite_replicated_load_balancer.cxx 
  host/fillin_replicated_load_balancer.cxx 
)

target_include_directories( gauxc
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
)

if( GAUXC_HAS_DEVICE )
  add_subdirectory( device )
endif()

