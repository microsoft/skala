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
#include <stdexcept>
#include "device/xc_device_aos_data.hpp"
#include <fstream>

namespace GauXC  {
namespace detail {

template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  eval_exc_grad_( int64_t m, int64_t n, const value_type* P,
                 int64_t ldp, value_type* EXC_GRAD, const IntegratorSettingsXC& settings) { 
                 
  const auto& basis = this->load_balancer_->basis();

  // Check that P is sane
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
  auto device_data_ptr = 
    this->timer_.time_op("XCIntegrator.DeviceAlloc",
      [&](){ return lwd->create_device_data(rt); });

  const auto& mol = this->load_balancer_->molecule();
  const auto natoms = mol.size();
  if( this->reduction_driver_->takes_device_memory() ) {
    GAUXC_GENERIC_EXCEPTION("Device Reduction + EXC Grad NYI");
  } else {

    // Compute local contributions to EXC Gradient and retrieve
    // data from device 
    this->timer_.time_op("XCIntegrator.LocalWork", [&](){
      eval_exc_grad_local_work_( basis, P, ldp, nullptr, 0, EXC_GRAD, tasks.begin(),
        tasks.end(), *device_data_ptr, settings );
    });

    GAUXC_MPI_CODE(
    this->timer_.time_op("XCIntegrator.ImbalanceWait",[&](){
      MPI_Barrier(this->load_balancer_->runtime().comm());
    });  
    )

    this->timer_.time_op("XCIntegrator.Allreduce", [&](){
      this->reduction_driver_->allreduce_inplace( EXC_GRAD, 3*natoms, 
        ReductionOp::Sum );
    });

  }

}


template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  eval_exc_grad_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps, 
                  const value_type* Pz, int64_t ldpz, value_type* EXC_GRAD, const IntegratorSettingsXC& settings ) { 
                 
  const auto& basis = this->load_balancer_->basis();

  // Check that P is sane
  const int64_t nbf = basis.nbf();
  if( m != n ) 
    GAUXC_GENERIC_EXCEPTION("P Must Be Square");
  if( m != nbf ) 
    GAUXC_GENERIC_EXCEPTION("P Must Have Same Dimension as Basis");
  if( ldps < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPS");
  if( ldpz < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPZ");

  // Get Tasks
  auto& tasks = this->load_balancer_->get_tasks();

  // Allocate Device memory
  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get() );
  auto rt  = detail::as_device_runtime(this->load_balancer_->runtime());
  auto device_data_ptr = 
    this->timer_.time_op("XCIntegrator.DeviceAlloc",
      [&](){ return lwd->create_device_data(rt); });

  const auto& mol = this->load_balancer_->molecule();
  const auto natoms = mol.size();
  if( this->reduction_driver_->takes_device_memory() ) {
    GAUXC_GENERIC_EXCEPTION("Device Reduction + EXC Grad NYI");
  } else {

    // Compute local contributions to EXC Gradient and retrieve
    // data from device 
    this->timer_.time_op("XCIntegrator.LocalWork", [&](){
      eval_exc_grad_local_work_( basis, Ps, ldps, Pz, ldpz, EXC_GRAD, tasks.begin(),
        tasks.end(), *device_data_ptr, settings );
    });

    GAUXC_MPI_CODE(
    this->timer_.time_op("XCIntegrator.ImbalanceWait",[&](){
      MPI_Barrier(this->load_balancer_->runtime().comm());
    });  
    )

    this->timer_.time_op("XCIntegrator.Allreduce", [&](){
      this->reduction_driver_->allreduce_inplace( EXC_GRAD, 3*natoms, 
        ReductionOp::Sum );
    });

  }

}

template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  eval_exc_grad_local_work_( const basis_type& basis, 
    const value_type* Ps, int64_t ldps,
    const value_type* Pz, int64_t ldpz,
    host_task_iterator task_begin, host_task_iterator task_end,
    XCDeviceData& device_data, const IntegratorSettingsXC& settings ) {

  const bool is_uks = Pz != nullptr;
  const bool is_rks = not is_uks;

  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get() );

  // Setup Aliases
  const auto& func  = *this->func_;
  const auto& mol   = this->load_balancer_->molecule();
  const auto& meta  = this->load_balancer_->molmeta();

  // Sanity gates
  if(func.needs_laplacian()) {
    GAUXC_GENERIC_EXCEPTION("Device EXC Gradients + Laplacian Dependent MGGAs Not Yet Implemented");
  }

  // Get basis map
  BasisSetMap basis_map(basis,mol);

  // Populate submat maps
  device_data.populate_submat_maps( basis.nbf(), task_begin, task_end, basis_map );

  // Sort tasks 
  auto task_comparator = []( const XCTask& a, const XCTask& b ) {
    return (a.points.size() * a.bfn_screening.nbe) > (b.points.size() * b.bfn_screening.nbe);
  };
  std::sort( task_begin, task_end, task_comparator );

  // Misc KS settings
  IntegratorSettingsEXC_GRAD exc_grad_settings;
  if( auto* tmp = dynamic_cast<const IntegratorSettingsEXC_GRAD*>(&settings) ) {
    exc_grad_settings = *tmp;
  }

  // Check that Partition Weights have been calculated
  auto& lb_state = this->load_balancer_->state();
  if( not lb_state.modified_weights_are_stored ) {
    GAUXC_GENERIC_EXCEPTION("Weights Have Not Been Modified"); 
  }
  XCWeightAlg& weight_alg = lb_state.weight_alg;


  // Processes batches in groups that saturadate available device memory
  integrator_term_tracker enabled_terms;
  enabled_terms.exc_grad = true;
  enabled_terms.weights  = true;

  if (is_rks) enabled_terms.ks_scheme = RKS;
  else if (is_uks) enabled_terms.ks_scheme = UKS;

  if( func.is_lda() )      enabled_terms.xc_approx = integrator_xc_approx::LDA; 
  else if( func.is_gga() ) enabled_terms.xc_approx = integrator_xc_approx::GGA; 
  else if( func.needs_laplacian() ) enabled_terms.xc_approx = integrator_xc_approx::MGGA_LAPL;
  else enabled_terms.xc_approx = integrator_xc_approx::MGGA_TAU;

  // Do XC integration in task batches
  const auto nbf     = basis.nbf();
  const auto nshells = basis.nshells();
  const auto natoms  = mol.size();
  device_data.reset_allocations();
  device_data.allocate_static_data_exc_grad( nbf, nshells, natoms, enabled_terms );
  device_data.send_static_data_density_basis( Ps, ldps, Pz, ldpz, nullptr, 0, nullptr, 0, basis );
  // for weight contribution
  device_data.allocate_static_data_weights( natoms );
  device_data.send_static_data_weights( mol, meta );

  // Zero integrands
  device_data.zero_exc_grad_integrands();


  auto task_it = task_begin;
  while( task_it != task_end ) {

    // Determine next task batch, send relevant data to device (EXC Gradient only)
    task_it = 
      device_data.generate_buffers( enabled_terms, basis_map, task_it, task_end );

    /*** Process the batches ***/

    // Evaluate collocation
    if( func.needs_laplacian() ) lwd->eval_collocation_lapgrad ( &device_data );
    else if( !func.is_lda() )    lwd->eval_collocation_hessian ( &device_data );
    else                         lwd->eval_collocation_gradient( &device_data );

    // Evaluate X matrix and V vars
    const auto xmat_fac = is_rks ? 2.0 : 1.0;
    const auto need_lapl = func.needs_laplacian();
    const auto need_xmat_grad = not func.is_lda();
    auto do_xmat_vvar = [&](density_id den_id) {
      lwd->eval_xmat( xmat_fac, &device_data, need_xmat_grad, den_id );
      if(func.is_lda())      lwd->eval_vvars_lda( &device_data, den_id );
      else if(func.is_gga()) lwd->eval_vvars_gga( &device_data, den_id ); 
      else                   lwd->eval_vvars_mgga( &device_data, den_id, need_lapl );

      // Save XMat for EXC gradient assembly
      if(is_uks) lwd->save_xmat( &device_data, need_xmat_grad, den_id );
    };

    do_xmat_vvar(DEN_S);
    if (not is_rks) {
      do_xmat_vvar(DEN_Z);
    }

    // Evaluate U variables
    if( func.is_mgga() )     lwd->eval_uvars_mgga( &device_data, enabled_terms.ks_scheme, need_lapl );
    else if( func.is_gga() ) lwd->eval_uvars_gga ( &device_data, enabled_terms.ks_scheme );
    else                     lwd->eval_uvars_lda ( &device_data, enabled_terms.ks_scheme );

    // Evaluate XC functional (we need VXC for EXC Gradient)
    if( func.is_mgga() )     lwd->eval_kern_exc_vxc_mgga( func, &device_data );
    else if( func.is_gga() ) lwd->eval_kern_exc_vxc_gga ( func, &device_data );
    else                     lwd->eval_kern_exc_vxc_lda ( func, &device_data );


    // Do scalar N_EL integration    
    lwd->inc_nel( &device_data );

    // Increment EXC Gradient
    if( func.is_mgga() )     lwd->inc_exc_grad_mgga( &device_data, enabled_terms.ks_scheme, need_lapl, exc_grad_settings.include_weight_derivatives );
    else if( func.is_gga() ) lwd->inc_exc_grad_gga ( &device_data, enabled_terms.ks_scheme, exc_grad_settings.include_weight_derivatives );
    else                     lwd->inc_exc_grad_lda ( &device_data, enabled_terms.ks_scheme, exc_grad_settings.include_weight_derivatives );

    // weight contribution
    if(exc_grad_settings.include_weight_derivatives)
      lwd->eval_weight_1st_deriv_contracted( &device_data, weight_alg );

  } // Loop over batches of batches 

}

template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  eval_exc_grad_local_work_( const basis_type& basis, 
    const value_type* Ps, int64_t ldps, 
    const value_type* Pz, int64_t ldpz, 
    value_type* EXC_GRAD, 
    host_task_iterator task_begin, host_task_iterator task_end,
    XCDeviceData& device_data, const IntegratorSettingsXC& settings ) {

  // Compute XC gradient and keep data on the device
  eval_exc_grad_local_work_( basis, Ps, ldps, Pz, ldpz, task_begin, task_end, device_data, settings );

  // Receive XC gradient from host
  double N_EL;
  device_data.retrieve_exc_grad_integrands( EXC_GRAD, &N_EL );

  //std::cout << N_EL << std::endl;
}


}
}
