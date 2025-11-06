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
#include "host_molecular_weights.hpp"
#include "host/local_host_work_driver.hpp"

namespace GauXC::detail {

void HostMolecularWeights::modify_weights( LoadBalancer& lb ) const {

  if(lb.state().modified_weights_are_stored)
    GAUXC_GENERIC_EXCEPTION("Attempting to Overwrite Modified Weights");

  // Cast LWD to LocalHostWorkDriver
  auto* lwd = dynamic_cast<LocalHostWorkDriver*>(this->local_work_driver_.get());

  // (Possibly) Generate tasks
  auto& tasks = lb.get_tasks();

  // Sort tasks on size (XXX: maybe doesnt matter?)
  auto task_comparator = []( const XCTask& a, const XCTask& b ) {
    return (a.points.size() * a.bfn_screening.nbe) > (b.points.size() * b.bfn_screening.nbe);
  };
  std::stable_sort( tasks.begin(), tasks.end(), task_comparator );

  // Modify the weights
  const auto& mol  = lb.molecule();
  const auto& meta = lb.molmeta();
  lwd->partition_weights( this->settings_.weight_alg, mol, meta, 
    tasks.begin(), tasks.end() );

  lb.state().modified_weights_are_stored = true;
  lb.state().weight_alg = this->settings_.weight_alg;
}

}
