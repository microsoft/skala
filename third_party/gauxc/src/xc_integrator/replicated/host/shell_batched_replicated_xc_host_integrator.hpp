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
#include <gauxc/xc_integrator/replicated/replicated_xc_host_integrator.hpp>
#include "reference_replicated_xc_host_integrator.hpp"
#include "shell_batched_replicated_xc_integrator.hpp"

namespace GauXC {
namespace detail {

template <typename ValueType>
class ShellBatchedReplicatedXCHostIntegrator : 
  public ShellBatchedReplicatedXCIntegrator<
    ReplicatedXCHostIntegrator<ValueType>,
    ReferenceReplicatedXCHostIntegrator<ValueType>
  > {

  using base_type  = ShellBatchedReplicatedXCIntegrator<
    ReplicatedXCHostIntegrator<ValueType>,
    ReferenceReplicatedXCHostIntegrator<ValueType>
  >;

public:

  template <typename... Args>
  ShellBatchedReplicatedXCHostIntegrator( Args&&... args ) :
    base_type( std::forward<Args>(args)... ) { }

  virtual ~ShellBatchedReplicatedXCHostIntegrator() noexcept;

};

extern template class ShellBatchedReplicatedXCHostIntegrator<double>;

}
}

