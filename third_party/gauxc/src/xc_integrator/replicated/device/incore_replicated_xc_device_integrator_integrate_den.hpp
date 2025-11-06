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
#include "incore_replicated_xc_device_integrator.hpp"
#include "device/local_device_work_driver.hpp"
#include "device/xc_device_aos_data.hpp"
#include <fstream>
#include <gauxc/exceptions.hpp>

namespace GauXC  {
namespace detail {

template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  integrate_den_( int64_t m, int64_t n, const value_type* P,
                 int64_t ldp, value_type* N_EL ) {


  const auto& basis = this->load_balancer_->basis();

  // Check that P / VXC are sane
  const int64_t nbf = basis.nbf();
  if( m != n ) 
    GAUXC_GENERIC_EXCEPTION("P Must Be Square");
  if( m != nbf ) 
    GAUXC_GENERIC_EXCEPTION("P Must Have Same Dimension as Basis");
  if( ldp < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDP");


  // Get Tasks
  auto& tasks = this->load_balancer_->get_tasks();

  // Allocate Device memory
  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get() );
  auto rt  = detail::as_device_runtime(this->load_balancer_->runtime());
  auto device_data_ptr = lwd->create_device_data(rt);


  if( this->reduction_driver_->takes_device_memory() ) {
    GAUXC_GENERIC_EXCEPTION("Device Reduction + Integrate Den NYI");
  } else {

    // Compute local contributions to N_EL and retrieve
    // data from device 
    this->timer_.time_op("XCIntegrator.LocalWork_Den", [&](){
      integrate_den_local_work_( basis, P, ldp, N_EL,
        tasks.begin(), tasks.end(), *device_data_ptr);
    });

    GAUXC_MPI_CODE(
    this->timer_.time_op("XCIntegrator.ImbalanceWait",[&](){
      MPI_Barrier(this->load_balancer_->runtime().comm());
    });  
    )

    // Reduce Results in host mem
    this->timer_.time_op("XCIntegrator.Allreduce", [&](){
      this->reduction_driver_->allreduce_inplace( N_EL, 1, ReductionOp::Sum );
    });

  }
}

template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  integrate_den_local_work_( const basis_type& basis, const value_type* P, 
                       int64_t ldp, value_type* N_EL,
                       host_task_iterator task_begin, host_task_iterator task_end,
                       XCDeviceData& device_data ) {

  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get() );

  // Setup Aliases
  const auto& mol   = this->load_balancer_->molecule();

  // Get basis map
  BasisSetMap basis_map(basis,mol);

  // Populate submat maps
  device_data.populate_submat_maps( basis.nbf(), task_begin, task_end, basis_map );


  // Sort tasks 
  auto task_comparator = []( const XCTask& a, const XCTask& b ) {
    return (a.points.size() * a.bfn_screening.nbe) > (b.points.size() * b.bfn_screening.nbe);
  };
  std::sort( task_begin, task_end, task_comparator );


  // Check that Partition Weights have been calculated
  auto& lb_state = this->load_balancer_->state();
  if( not lb_state.modified_weights_are_stored ) {
    GAUXC_GENERIC_EXCEPTION("Weights Have Not Been Modified"); 
  }

  // Do XC integration in task batches
  const auto nbf     = basis.nbf();
  const auto nshells = basis.nshells();
  device_data.reset_allocations();
  device_data.allocate_static_data_den( nbf, nshells );
  device_data.send_static_data_density_basis( P, ldp, nullptr, 0, nullptr, 0, nullptr, 0,  basis );

  // Zero integrands
  device_data.zero_den_integrands();

  // Processes batches in groups that saturadate available device memory
  integrator_term_tracker enabled_terms;
  enabled_terms.den = true;

  auto task_it = task_begin;
  while( task_it != task_end ) {

    // Determine next task batch, send relevant data to device (Density only)
    task_it = 
      device_data.generate_buffers( enabled_terms, basis_map, task_it, task_end );

    /*** Process the batches ***/

    // Evaluate collocation
    lwd->eval_collocation( &device_data );

    // Evaluate X matrix
    const bool do_xmat_grad = false;
    lwd->eval_xmat( 1.0, &device_data, do_xmat_grad, DEN_S );

    // Evaluate the density
    const bool do_vvar_grad = false;
    lwd->eval_vvars_lda( &device_data, DEN_S );

    // Do scalar N_EL integration
    lwd->inc_nel( &device_data );

  } // Loop over batches of batches 

  // Receive N_EL from device
  this->timer_.time_op("XCIntegrator.DeviceToHostCopy",[&](){
    device_data.retrieve_den_integrands( N_EL );
  });
}

}
}

