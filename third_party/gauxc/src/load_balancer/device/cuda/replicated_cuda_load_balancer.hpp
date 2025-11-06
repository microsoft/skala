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

#include "load_balancer_impl.hpp"

namespace GauXC  {
namespace detail {

class DeviceReplicatedLoadBalancer : public LoadBalancerImpl {

protected:

  using basis_type = BasisSet<double>;
  std::vector< XCTask > create_local_tasks_() const override;

public:

  DeviceReplicatedLoadBalancer() = delete;
  template <typename... Args>
  DeviceReplicatedLoadBalancer( Args&&... args ):
    LoadBalancerImpl( std::forward<Args>(args)... ) { }

  DeviceReplicatedLoadBalancer( const DeviceReplicatedLoadBalancer& );
  DeviceReplicatedLoadBalancer( DeviceReplicatedLoadBalancer&& ) noexcept;

  virtual ~DeviceReplicatedLoadBalancer() noexcept;

  std::unique_ptr<LoadBalancerImpl> clone() const override;

};

}
}
