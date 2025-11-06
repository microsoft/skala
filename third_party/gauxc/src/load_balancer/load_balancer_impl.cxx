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
#include "load_balancer_impl.hpp"

namespace GauXC::detail {

LoadBalancerImpl::LoadBalancerImpl( const RuntimeEnvironment& rt, const Molecule& mol, 
  const MolGrid& mg, const basis_type& basis, std::shared_ptr<MolMeta> molmeta ) :
  runtime_(rt), 
  mol_( std::make_shared<Molecule>(mol) ),
  mg_( std::make_shared<MolGrid>(mg)  ),
  basis_( std::make_shared<basis_type>(basis) ),
  molmeta_( molmeta ) { 

  basis_map_   = std::make_shared<basis_map_type>(*basis_, mol);

}

LoadBalancerImpl::LoadBalancerImpl( const RuntimeEnvironment& rt, const Molecule& mol, 
  const MolGrid& mg, const basis_type& basis, const MolMeta& molmeta ) :
  LoadBalancerImpl( rt, mol, mg, basis, std::make_shared<MolMeta>(molmeta) ) { }

LoadBalancerImpl::LoadBalancerImpl( const RuntimeEnvironment& rt, const Molecule& mol, 
  const MolGrid& mg, const basis_type& basis ) :
  LoadBalancerImpl( rt, mol, mg, basis, std::make_shared<MolMeta>(mol) ) { }


LoadBalancerImpl::LoadBalancerImpl( const LoadBalancerImpl& ) = default;
LoadBalancerImpl::LoadBalancerImpl( LoadBalancerImpl&& ) noexcept = default;

LoadBalancerImpl::~LoadBalancerImpl() noexcept = default;

const std::vector<XCTask>& LoadBalancerImpl::get_tasks() const {
  if( not local_tasks_.size() ) GAUXC_GENERIC_EXCEPTION("No Tasks Created");
  return local_tasks_;
}

std::vector<XCTask>& LoadBalancerImpl::get_tasks() {

  if( not local_tasks_.size() ) {
    auto create_tasks_st = std::chrono::high_resolution_clock::now();
    local_tasks_ = create_local_tasks_();
    auto create_tasks_en = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> create_tasks_dr = create_tasks_en - create_tasks_st; 
    timer_.add_timing("LoadBalancer.CreateTasks", create_tasks_dr);
  }


  return local_tasks_;
}

const util::Timer& LoadBalancerImpl::get_timings() const {
  return timer_;
}


size_t LoadBalancerImpl::total_npts() const {

  return std::accumulate( local_tasks_.cbegin(), local_tasks_.cend(), 0ul,
    []( const auto& a, const auto& b ) {
      return a + b.points.size();
    });

}
size_t LoadBalancerImpl::max_npts() const {

  if( not local_tasks_.size() ) return 0ul;

  return std::max_element( local_tasks_.cbegin(), local_tasks_.cend(),
    []( const auto& a, const auto& b ) {
      return a.points.size() < b.points.size();
    })->points.size();

}
size_t LoadBalancerImpl::max_nbe() const {

  if( not local_tasks_.size() ) return 0ul;

  return std::max_element( local_tasks_.cbegin(), local_tasks_.cend(),
    []( const auto& a, const auto& b ) {
      return a.bfn_screening.nbe < b.bfn_screening.nbe;
    })->bfn_screening.nbe;

}
size_t LoadBalancerImpl::max_npts_x_nbe() const {

  if( not local_tasks_.size() ) return 0ul;

  auto it = std::max_element( local_tasks_.cbegin(), local_tasks_.cend(),
    []( const auto& a, const auto& b ) {
      return a.bfn_screening.nbe * a.points.size() < b.bfn_screening.nbe * b.points.size();
    });

  return it->bfn_screening.nbe * it->points.size();

}




const Molecule& LoadBalancerImpl::molecule() const {
  return *mol_;
}

const MolMeta& LoadBalancerImpl::molmeta() const {
  return *molmeta_;
}

const LoadBalancerImpl::basis_type& LoadBalancerImpl::basis() const {
  return *basis_;
}
const LoadBalancerImpl::basis_map_type& LoadBalancerImpl::basis_map() const {
  return *basis_map_;
}
const LoadBalancerImpl::shell_pair_type& LoadBalancerImpl::shell_pairs() const {
  if(!shell_pairs_) GAUXC_GENERIC_EXCEPTION("ShellPairs must be pregenerated for const-context");
  return *shell_pairs_;
}
const LoadBalancerImpl::shell_pair_type& LoadBalancerImpl::shell_pairs() {
  if(!shell_pairs_) {
    shell_pairs_ = std::make_shared<shell_pair_type>(*basis_);
  }
  return *shell_pairs_;
}

const RuntimeEnvironment& LoadBalancerImpl::runtime() const {
  return runtime_;
}

LoadBalancerState& LoadBalancerImpl::state() {
  return state_;
}

}
