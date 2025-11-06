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
#include "replicated_host_load_balancer.hpp"

namespace GauXC  {
namespace detail {

struct FillInHostReplicatedLoadBalancer : public HostReplicatedLoadBalancer {

  template <typename... Args>
  FillInHostReplicatedLoadBalancer( Args&&... args ):
    HostReplicatedLoadBalancer( std::forward<Args>(args)... ) { }

  FillInHostReplicatedLoadBalancer( const FillInHostReplicatedLoadBalancer& );
  FillInHostReplicatedLoadBalancer( FillInHostReplicatedLoadBalancer&& ) noexcept;

  ~FillInHostReplicatedLoadBalancer() noexcept;

  std::unique_ptr<LoadBalancerImpl> clone() const override final;

  std::pair< std::vector<int32_t>, size_t > micro_batch_screen(
    const BasisSet<double>&, const std::array<double,3>&,
    const std::array<double,3>& ) const override final;

};

}
}
