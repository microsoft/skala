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

namespace GauXC {

LoadBalancer::LoadBalancer( std::unique_ptr<pimpl_type>&& pimpl ): 
  pimpl_( std::move(pimpl) ) { }

LoadBalancer::LoadBalancer() : LoadBalancer( nullptr ) { }

LoadBalancer::LoadBalancer( const LoadBalancer& other ) :
  LoadBalancer(other.pimpl_->clone()){ }

LoadBalancer::LoadBalancer( LoadBalancer&& ) noexcept = default;
              
LoadBalancer::~LoadBalancer() noexcept = default;


const std::vector<XCTask>& LoadBalancer::get_tasks() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->get_tasks();
}
std::vector<XCTask>& LoadBalancer::get_tasks() {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->get_tasks();
}

void LoadBalancer::rebalance_weights() {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  pimpl_->rebalance_weights();
}

void LoadBalancer::rebalance_exc_vxc() {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  pimpl_->rebalance_exc_vxc();
}

void LoadBalancer::rebalance_exx() {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  pimpl_->rebalance_exx();
}

const util::Timer& LoadBalancer::get_timings() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->get_timings();
}

size_t LoadBalancer::total_npts() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->total_npts();
}
size_t LoadBalancer::max_npts() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->max_npts();
}
size_t LoadBalancer::max_nbe() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->max_nbe();
}
size_t LoadBalancer::max_npts_x_nbe() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->max_npts_x_nbe();
}



const Molecule& LoadBalancer::molecule() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->molecule();
}
const MolMeta& LoadBalancer::molmeta() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->molmeta();
}

const LoadBalancer::basis_type& LoadBalancer::basis() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->basis();
}
const LoadBalancer::basis_map_type& LoadBalancer::basis_map() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->basis_map();
}
const LoadBalancer::shell_pair_type& LoadBalancer::shell_pairs() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->shell_pairs();
}

const LoadBalancer::shell_pair_type& LoadBalancer::shell_pairs() {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->shell_pairs();
}

LoadBalancerState& LoadBalancer::state() {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->state();
}

const RuntimeEnvironment& LoadBalancer::runtime() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->runtime();
}


bool LoadBalancer::operator==( const LoadBalancer& other ) const {
  return (&other) == this;
}

}
