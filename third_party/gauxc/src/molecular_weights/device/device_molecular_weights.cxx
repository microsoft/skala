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
#include "device_molecular_weights.hpp"
#include "device/local_device_work_driver.hpp"
#include "device/device_backend.hpp"

namespace GauXC::detail {

void DeviceMolecularWeights::modify_weights( LoadBalancer& lb ) const {

  if(lb.state().modified_weights_are_stored)
    GAUXC_GENERIC_EXCEPTION("Attempting to Overwrite Modified Weights");
  if(this->settings_.weight_alg != XCWeightAlg::SSF)
    GAUXC_GENERIC_EXCEPTION("Non-SSF Weights NYI for Device Integration");

  // Cast LWD to LocalDeviceWorkDriver
  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get() );
  auto rt   = detail::as_device_runtime(lb.runtime());

  // Create device data
  auto device_data_ptr = lwd->create_device_data(rt);
  auto& device_data = *device_data_ptr;
  device_data.reset_allocations();

  // (Possibly) Generate tasks
  auto& tasks = lb.get_tasks();

  auto task_begin = tasks.begin();
  auto task_end   = tasks.end();

  // Sort tasks on size (XXX: maybe doesnt matter?)
  auto task_comparator = []( const XCTask& a, const XCTask& b ) {
    return (a.points.size() * a.bfn_screening.nbe) > (b.points.size() * b.bfn_screening.nbe);
  };
  std::stable_sort(task_begin, task_end, task_comparator );

  const auto& mol  = lb.molecule();
  const auto natoms = mol.natoms();
  const auto& meta = lb.molmeta();

  // Setup device data
  device_data.allocate_static_data_weights( natoms );
  device_data.send_static_data_weights( mol, meta );

  // TODO: this shouldn't be needed for Weights
  const auto& basis = lb.basis();
  BasisSetMap basis_map(basis,mol);

  // Modify the weights
  integrator_term_tracker enabled_terms;
  enabled_terms.weights = true;

  // Processes batches in groups that saturadate available device memory
  auto task_it = task_begin;
  while( task_it != task_end ) {
    
    // Determine next task batch, send relevant data to device 
    auto task_batch_end = 
      device_data.generate_buffers( enabled_terms, basis_map, 
        task_it, task_end );

    // Apply partition weights 
    lwd->partition_weights( &device_data );
    
    // Copy back to host data
    device_data.copy_weights_to_tasks( task_it, task_batch_end );

    // Update iterator
    task_it = task_batch_end;

  } // End loop over batches

  // Synchronize
  rt.device_backend()->master_queue_synchronize();
 
  lb.state().modified_weights_are_stored = true;
  lb.state().weight_alg = this->settings_.weight_alg;

}

}
