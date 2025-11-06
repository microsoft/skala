/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 *
 * All rights reserved.
 *
 * See LICENSE.txt for details
 */
#pragma once
#include <gauxc/xc_integrator/replicated/replicated_xc_device_integrator.hpp>
#include "incore_replicated_xc_device_integrator.hpp"
#include "shell_batched_replicated_xc_integrator.hpp"

namespace GauXC {
namespace detail {

template <typename ValueType>
class ShellBatchedReplicatedXCDeviceIntegrator : 
  public ShellBatchedReplicatedXCIntegrator<
    ReplicatedXCDeviceIntegrator<ValueType>,
    IncoreReplicatedXCDeviceIntegrator<ValueType>
  > {

  using base_type  = ShellBatchedReplicatedXCIntegrator<
    ReplicatedXCDeviceIntegrator<ValueType>,
    IncoreReplicatedXCDeviceIntegrator<ValueType>
  >;

public:

  template <typename... Args>
  ShellBatchedReplicatedXCDeviceIntegrator( Args&&... args ) :
    base_type( std::forward<Args>(args)... ) { }

  virtual ~ShellBatchedReplicatedXCDeviceIntegrator() noexcept;

};

extern template class ShellBatchedReplicatedXCDeviceIntegrator<double>;

}
}

