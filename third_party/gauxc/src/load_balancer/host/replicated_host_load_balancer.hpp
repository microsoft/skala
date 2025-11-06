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

class HostReplicatedLoadBalancer : public LoadBalancerImpl {

protected:

  using basis_type = BasisSet<double>;
  std::vector< XCTask > create_local_tasks_() const override;

public:

  HostReplicatedLoadBalancer() = delete;
  template <typename... Args>
  HostReplicatedLoadBalancer( Args&&... args ):
    LoadBalancerImpl( std::forward<Args>(args)... ) { }

  HostReplicatedLoadBalancer( const HostReplicatedLoadBalancer& );
  HostReplicatedLoadBalancer( HostReplicatedLoadBalancer&& ) noexcept;

  virtual ~HostReplicatedLoadBalancer() noexcept;

  virtual std::pair< std::vector<int32_t>, size_t > micro_batch_screen(
    const BasisSet<double>&, const std::array<double,3>&,
    const std::array<double,3>& ) const = 0;

};

}
}
