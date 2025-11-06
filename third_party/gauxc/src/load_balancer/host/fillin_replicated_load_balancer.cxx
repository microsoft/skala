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
#include "fillin_replicated_load_balancer.hpp"
#include <gauxc/util/geometry.hpp>

namespace GauXC  {
namespace detail {

FillInHostReplicatedLoadBalancer::FillInHostReplicatedLoadBalancer( const FillInHostReplicatedLoadBalancer& ) = default;
FillInHostReplicatedLoadBalancer::FillInHostReplicatedLoadBalancer( FillInHostReplicatedLoadBalancer&& ) noexcept = default;

FillInHostReplicatedLoadBalancer::~FillInHostReplicatedLoadBalancer() noexcept = default;

std::unique_ptr<LoadBalancerImpl> FillInHostReplicatedLoadBalancer::clone() const {
  return std::make_unique<FillInHostReplicatedLoadBalancer>(*this);
}





std::pair<std::vector<int32_t>,size_t> FillInHostReplicatedLoadBalancer::micro_batch_screen(
  const BasisSet<double>&      bs,
  const std::array<double,3>&  box_lo,
  const std::array<double,3>&  box_up
) const {


  int32_t first_shell = -1;
  int32_t last_shell  = -1;
  for(auto iSh = 0ul; iSh < bs.size(); ++iSh) {

    const auto& center = bs[iSh].O();
    const auto  crad   = bs[iSh].cutoff_radius();
    const bool intersect = 
      geometry::cube_sphere_intersect( box_lo, box_up, center, crad );
    
    if( intersect ) {
      if( first_shell < 0 ) first_shell = iSh;
      last_shell = iSh;
    }

  }

  if( first_shell < 0 ) {
    return std::pair( std::vector<int32_t>{}, 0ul );
  }

  int32_t nshells = last_shell - first_shell + 1;
  std::vector<int32_t> shell_list(nshells);
  std::iota( shell_list.begin(), shell_list.end(), first_shell );

  size_t nbe = std::accumulate( shell_list.begin(), shell_list.end(), 0ul,
    [&](const auto& a, const auto& b) { return a + bs[b].size(); } );

  return std::pair( std::move( shell_list ), nbe );

}

}
}
