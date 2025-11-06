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
#include "petite_replicated_load_balancer.hpp"
#include <gauxc/util/geometry.hpp>

namespace GauXC  {
namespace detail {

PetiteHostReplicatedLoadBalancer::PetiteHostReplicatedLoadBalancer( const PetiteHostReplicatedLoadBalancer& ) = default;
PetiteHostReplicatedLoadBalancer::PetiteHostReplicatedLoadBalancer( PetiteHostReplicatedLoadBalancer&& ) noexcept = default;

PetiteHostReplicatedLoadBalancer::~PetiteHostReplicatedLoadBalancer() noexcept = default;

std::unique_ptr<LoadBalancerImpl> PetiteHostReplicatedLoadBalancer::clone() const {
  return std::make_unique<PetiteHostReplicatedLoadBalancer>(*this);
}





std::pair<std::vector<int32_t>,size_t> PetiteHostReplicatedLoadBalancer::micro_batch_screen(
  const BasisSet<double>&      bs,
  const std::array<double,3>&  box_lo,
  const std::array<double,3>&  box_up
) const {


  std::vector<int32_t> shell_list; shell_list.reserve(bs.nshells());
  for(auto iSh = 0ul; iSh < bs.size(); ++iSh) {

    const auto& center = bs[iSh].O();
    const auto  crad   = bs[iSh].cutoff_radius();
    const bool intersect = 
      geometry::cube_sphere_intersect( box_lo, box_up, center, crad );
    

    //std::cout << "  MBS: " << iSh << ", " << 
    //          center[0] << ", " << center[1] << ", " << center[2] << ", " <<
    //          box_up[0] << ", " << box_up[1] << ", " << box_up[2] << ", " <<
    //          box_lo[0] << ", " << box_lo[1] << ", " << box_lo[2] << ", " <<
    //          crad << std::boolalpha << ", " << intersect << std::endl;
              

    // Add shell to list if need be
    if( intersect )
      shell_list.emplace_back( iSh );

  }

  size_t nbe = std::accumulate( shell_list.begin(), shell_list.end(), 0ul,
    [&](const auto& a, const auto& b) { return a + bs[b].size(); } );

  return std::pair( std::move( shell_list ), nbe );

}

}
}
