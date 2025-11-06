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
#include "incore_replicated_xc_device_integrator.hpp"
#include <gauxc/util/misc.hpp>
#include <gauxc/util/unused.hpp>

namespace GauXC::detail {

  template <typename ValueType>
  void IncoreReplicatedXCDeviceIntegrator<ValueType>::
    eval_fxc_contraction_( int64_t m, int64_t n, 
                          const value_type* P, int64_t ldp,
                          const value_type* tP, int64_t ldtp,
                          value_type* FXC, int64_t ldfxc,
                          const IntegratorSettingsXC& ks_settings ) {
    
    eval_fxc_contraction_( m, n, P, ldp, nullptr, 0, tP, ldtp, nullptr, 0,
                          FXC, ldfxc, nullptr, 0, ks_settings );
  }

    
  template <typename ValueType>
  void IncoreReplicatedXCDeviceIntegrator<ValueType>::
    eval_fxc_contraction_( int64_t m, int64_t n, 
                          const value_type* Ps, int64_t ldps,
                          const value_type* Pz, int64_t ldpz,
                          const value_type* tPs, int64_t ldtps,
                          const value_type* tPz, int64_t ldtpz,
                          value_type* FXCs, int64_t ldfxcs,
                          value_type* FXCz, int64_t ldfxcz,
                          const IntegratorSettingsXC& ks_settings ) {
    const bool is_uks = (Pz != nullptr);
    const bool is_rks = !is_uks;

    const auto& basis = this->load_balancer_->basis();

    // Check that P / FXC are sane
    const int64_t nbf = basis.nbf();
    if( m != n ) 
      GAUXC_GENERIC_EXCEPTION("P/FXC Must Be Square");
    if( m != nbf ) 
      GAUXC_GENERIC_EXCEPTION("P/FXC Must Have Same Dimension as Basis");
    if( ldps < nbf )
      GAUXC_GENERIC_EXCEPTION("Invalid LDPs");
    if( ldtps < nbf )
      GAUXC_GENERIC_EXCEPTION("Invalid LDTps");
    if( ldfxcs < nbf )
      GAUXC_GENERIC_EXCEPTION("Invalid LDFXCs");

    if( not is_rks ) {
      if( ldpz < nbf )
        GAUXC_GENERIC_EXCEPTION("Invalid LDPz");
      if( ldtpz < nbf )
        GAUXC_GENERIC_EXCEPTION("Invalid LDTpz");
      if( ldfxcz < nbf )
        GAUXC_GENERIC_EXCEPTION("Invalid LDFXCz");
    }

    // Get Tasks
    auto& tasks = this->load_balancer_->get_tasks();

    // Allocate Device memory
    auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get() );
    auto rt  = detail::as_device_runtime(this->load_balancer_->runtime());
    auto device_data_ptr = lwd->create_device_data(rt);

    GAUXC_MPI_CODE( MPI_Barrier(rt.comm());) 

    // Temporary electron count to judge integrator accuracy
    value_type N_EL;
  
    if( this->reduction_driver_->takes_device_memory() ) {

      // If we can do reductions on the device (e.g. NCCL)
      // Don't communicate data back to the host before reduction
      this->timer_.time_op("XCIntegrator.LocalWork_FXC", [&](){
        fxc_contraction_local_work_( basis, Ps, ldps, Pz, ldpz, tPs, ldtps, tPz, ldtpz,
          tasks.begin(), tasks.end(), *device_data_ptr);
      });

      GAUXC_MPI_CODE(
      this->timer_.time_op("XCIntegrator.ImbalanceWait_FXC",[&](){
        MPI_Barrier(this->load_balancer_->runtime().comm());
      });  
      )

      // Reduce results in device memory
      double* fxc_s_device = device_data_ptr->fxc_s_device_data();
      double* fxc_z_device;
      auto nel_device = device_data_ptr->nel_device_data();
      auto queue = device_data_ptr->queue();
      
      if( not is_rks ) {
        fxc_z_device = device_data_ptr->fxc_z_device_data();
        // UKS
        this->timer_.time_op("XCIntegrator.Allreduce_FXC", [&](){
          this->reduction_driver_->allreduce_inplace( fxc_s_device, nbf*nbf, ReductionOp::Sum, queue );
          this->reduction_driver_->allreduce_inplace( fxc_z_device, nbf*nbf, ReductionOp::Sum, queue );
          this->reduction_driver_->allreduce_inplace( nel_device, 1,       ReductionOp::Sum, queue );
        });
      } else {
        // RKS
        this->timer_.time_op("XCIntegrator.Allreduce_FXC", [&](){
          this->reduction_driver_->allreduce_inplace( fxc_s_device, nbf*nbf, ReductionOp::Sum, queue );
          this->reduction_driver_->allreduce_inplace( nel_device, 1,       ReductionOp::Sum, queue );
        });
      }

      // Retrieve data to host
      this->timer_.time_op("XCIntegrator.DeviceToHostCopy_FXC",[&](){
        device_data_ptr->retrieve_fxc_contraction_integrands(&N_EL, FXCs, ldfxcs, FXCz, ldfxcz, nullptr, 0, nullptr, 0);
      });

    } else {

      // Compute local contributions to FXC and retrieve
      // data from device 
      this->timer_.time_op("XCIntegrator.LocalWork_FXC", [&](){
        fxc_contraction_local_work_( basis, Ps, ldps, Pz, ldpz, tPs, ldtps, tPz, ldtpz, &N_EL, 
                              FXCs, ldfxcs, FXCz, ldfxcz, tasks.begin(), tasks.end(), *device_data_ptr);
      });

      GAUXC_MPI_CODE(
      this->timer_.time_op("XCIntegrator.ImbalanceWait_FXC",[&](){
        MPI_Barrier(this->load_balancer_->runtime().comm());
      });  
      )

      // Reduce Results in host mem
      if( is_rks ) {
        this->timer_.time_op("XCIntegrator.Allreduce_FXC", [&](){
          this->reduction_driver_->allreduce_inplace( FXCs, nbf*nbf, ReductionOp::Sum );
        this->reduction_driver_->allreduce_inplace( &N_EL, 1,       ReductionOp::Sum );
        });
      } else {
        // UKS
        this->timer_.time_op("XCIntegrator.Allreduce_FXC", [&](){
          this->reduction_driver_->allreduce_inplace( FXCs, nbf*nbf, ReductionOp::Sum );
          this->reduction_driver_->allreduce_inplace( FXCz, nbf*nbf, ReductionOp::Sum );
          this->reduction_driver_->allreduce_inplace( &N_EL, 1,       ReductionOp::Sum );
        });
      }
    }
  }

  template <typename ValueType>
  void IncoreReplicatedXCDeviceIntegrator<ValueType>::
    fxc_contraction_local_work_( const basis_type& basis, const value_type* Ps, int64_t ldps,
                            const value_type* Pz, int64_t ldpz,
                            const value_type* tPs, int64_t ldtps,
                            const value_type* tPz, int64_t ldtpz,
                            host_task_iterator task_begin, host_task_iterator task_end,
                            XCDeviceData& device_data) {
    const bool is_uks = (Pz != nullptr);
    const bool is_rks = !is_uks;
    if (not is_rks and not is_uks) {
      GAUXC_GENERIC_EXCEPTION("MUST BE EITHER RKS OR UKS!");
    }
    

    // Cast LWD to LocalDeviceWorkDriver
    auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get() );

    // Setup Aliases
    const auto& func  = *this->func_;
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
    

    integrator_term_tracker enabled_terms;
    enabled_terms.fxc_contraction = true;

    if (is_rks) enabled_terms.ks_scheme = RKS;
    else if (is_uks) enabled_terms.ks_scheme = UKS;

    if( func.is_lda() )      
      enabled_terms.xc_approx = integrator_xc_approx::LDA; 
    else if( func.is_gga() ) 
      enabled_terms.xc_approx = integrator_xc_approx::GGA; 
    else if( func.needs_laplacian() )                    
      GAUXC_GENERIC_EXCEPTION("FXC contraction does not support MGGA with Laplacian");
    else
      enabled_terms.xc_approx = integrator_xc_approx::MGGA_TAU;
    
    // Do XC integration in task batches
    const auto nbf     = basis.nbf();
    const auto nshells = basis.nshells();
    device_data.reset_allocations();
    device_data.allocate_static_data_fxc_contraction( nbf, nshells, enabled_terms);
    
    device_data.send_static_data_density_basis( Ps, ldps, Pz, ldpz, nullptr, 0, nullptr, 0, basis );
    device_data.send_static_data_trial_density( tPs, ldtps, tPz, ldtpz, nullptr, 0, nullptr, 0 );


    // Zero integrands
    device_data.zero_fxc_contraction_integrands();


    auto task_it = task_begin;
    while( task_it != task_end ) {

      // Determine next task batch, send relevant data to device (FXC only)
      task_it = 
        device_data.generate_buffers( enabled_terms, basis_map, task_it, task_end );

      /*** Process the batches ***/
      
      const bool need_lapl = func.needs_laplacian();
      // Evaluate collocation
      if( func.is_mgga() ) {
        if(need_lapl) lwd->eval_collocation_laplacian( &device_data );
        else          lwd->eval_collocation_gradient( &device_data );
      }
      else if( func.is_gga() ) lwd->eval_collocation_gradient( &device_data );
      else                     lwd->eval_collocation( &device_data );
        
      const double xmat_fac = is_rks ? 2.0 : 1.0;
      const bool need_xmat_grad = func.is_mgga();

      // Evaluate X matrix and V vars
      auto do_xmat_vvar = [&](density_id den_id) {
        lwd->eval_xmat( xmat_fac, &device_data, need_xmat_grad, den_id );
        if(func.is_lda())      lwd->eval_vvars_lda( &device_data, den_id );
        else if(func.is_gga()) lwd->eval_vvars_gga( &device_data, den_id ); 
        else                   lwd->eval_vvars_mgga( &device_data, den_id, need_lapl );
      };

      do_xmat_vvar(DEN_S);
      if (not is_rks) {
        do_xmat_vvar(DEN_Z);
      }

      // Evaluate U variables
      if( func.is_mgga() )      lwd->eval_uvars_mgga( &device_data, enabled_terms.ks_scheme, need_lapl );
      else if( func.is_gga() )  lwd->eval_uvars_gga ( &device_data, enabled_terms.ks_scheme );
      else                      lwd->eval_uvars_lda ( &device_data, enabled_terms.ks_scheme );

      // Evaluate XC functional
      if( func.is_mgga() )     lwd->eval_kern_vxc_fxc_mgga( func, &device_data );
      else if( func.is_gga() ) lwd->eval_kern_vxc_fxc_gga ( func, &device_data );
      else                     lwd->eval_kern_vxc_fxc_lda ( func, &device_data );      

      // Do scalar N_EL integrations
      lwd->inc_nel( &device_data );

      
      // Evaluate X matrix and V vars from trial density
      auto do_xmat_vvar_trial = [&](density_id den_id) {
        lwd->eval_xmat_trial( xmat_fac, &device_data, need_xmat_grad, den_id );
        if(func.is_lda())      lwd->eval_vvars_lda_trial( &device_data, den_id );
        else if(func.is_gga()) lwd->eval_vvars_gga_trial( &device_data, den_id ); 
        else                   lwd->eval_vvars_mgga_trial( &device_data, den_id, need_lapl );
      };

      do_xmat_vvar_trial(DEN_S);
      if (not is_rks) {
        do_xmat_vvar_trial(DEN_Z);
      }

      // Evaluate tmat (it contains the trial u variable evaluation inside)
      if( func.is_mgga() )      lwd->eval_tmat_mgga( &device_data, enabled_terms.ks_scheme, need_lapl );
      else if( func.is_gga() )  lwd->eval_tmat_gga ( &device_data, enabled_terms.ks_scheme );
      else                      lwd->eval_tmat_lda ( &device_data, enabled_terms.ks_scheme );

      auto do_zmat_fxc = [&](density_id den_id) {
        if( func.is_mgga() ) {
          lwd->eval_zmat_mgga_fxc( &device_data, need_lapl, den_id);
          lwd->eval_mmat_mgga_fxc( &device_data, need_lapl, den_id);
        }
        else if( func.is_gga() ) 
          lwd->eval_zmat_gga_fxc( &device_data, den_id );
        else 
          lwd->eval_zmat_lda_fxc( &device_data, den_id );
        lwd->inc_fxc( &device_data, den_id, func.is_mgga() );
      };

      do_zmat_fxc(DEN_S);
      if(not is_rks) {
        do_zmat_fxc(DEN_Z);
      } 

    } // Loop over batches of batches 

    // Symmetrize FXC in device memory
    lwd->symmetrize_fxc( &device_data, DEN_S );
    if (not is_rks) {
      lwd->symmetrize_fxc( &device_data, DEN_Z );
      }
  }

  template <typename ValueType>
  void IncoreReplicatedXCDeviceIntegrator<ValueType>::
    fxc_contraction_local_work_( const basis_type& basis, const value_type* Ps, int64_t ldps,
                            const value_type* Pz, int64_t ldpz,
                            const value_type* tPs, int64_t ldtps,
                            const value_type* tPz, int64_t ldtpz,
                            value_type *N_EL,
                            value_type* FXCs, int64_t ldfxcs,
                            value_type* FXCz, int64_t ldfxcz,
                            host_task_iterator task_begin, host_task_iterator task_end,
                            XCDeviceData& device_data ) {
    
    // Get integrate and keep data on device
    fxc_contraction_local_work_( basis, Ps, ldps, Pz, ldpz, tPs, ldtps, tPz, ldtpz, 
                              task_begin, task_end, device_data);
    auto rt  = detail::as_device_runtime(this->load_balancer_->runtime());
    rt.device_backend()->master_queue_synchronize();

    // Receive FXC terms from host
    this->timer_.time_op("XCIntegrator.DeviceToHostCopy_FXC",[&](){
      device_data.retrieve_fxc_contraction_integrands( N_EL, FXCs, ldfxcs, FXCz, ldfxcz, nullptr, 0, nullptr, 0 ); 
    });
  }
}
