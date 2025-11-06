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

#include "reference_replicated_xc_host_integrator.hpp"
#include "integrator_util/integrator_common.hpp"
#include "host/local_host_work_driver.hpp"
#include "host/blas.hpp"
#include <stdexcept>

namespace GauXC::detail {

/**
 *  Generic implementation of FXC contraction for RKS/UKS/GKS
 *  
 */
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_fxc_contraction_( int64_t m, int64_t n, 
                        const value_type* Ps, int64_t ldps,
                        const value_type* Pz, int64_t ldpz,
                        const value_type* tPs, int64_t ldtps,
                        const value_type* tPz, int64_t ldtpz,
                        value_type* FXCs, int64_t ldfxcs,
                        value_type* FXCz, int64_t ldfxcz,
                        const IntegratorSettingsXC& ks_settings ){

  const auto& basis = this->load_balancer_->basis();

  // Check that P / FXC are sane
  const int64_t nbf = basis.nbf();
  if( m != n )
    GAUXC_GENERIC_EXCEPTION("P/FXC Must Be Square");
  if( m != nbf )
    GAUXC_GENERIC_EXCEPTION("P/FXC Must Have Same Dimension as Basis");
    
  if( ldps < nbf )
  GAUXC_GENERIC_EXCEPTION("Invalid LDPS");
  if( ldpz and ldpz < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPZ");
  if( ldtps and ldtps < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDTPS");
  if( ldtpz and ldtpz < nbf ) 
    GAUXC_GENERIC_EXCEPTION("Invalid LDTZP");
  if( ldfxcs < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDFXCS");
  if( ldfxcz and ldfxcz < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDFXCZ");


  // Get Tasks
  auto& tasks = this->load_balancer_->get_tasks();

  // Temporary electron count to judge integrator accuracy
  value_type N_EL;
   
  // Compute Local contributions to FXC contraction
  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
    fxc_contraction_local_work_( basis, Ps, ldps, Pz, ldpz, 
                                             tPs, ldtps, tPz, ldtpz,
                                             FXCs, ldfxcs, FXCz, ldfxcz,
                                             &N_EL, ks_settings,
                                             tasks.begin(), tasks.end() );
  });


  // Reduce Results
  this->timer_.time_op("XCIntegrator.Allreduce", [&](){

    if( not this->reduction_driver_->takes_host_memory() )
      GAUXC_GENERIC_EXCEPTION("This Module Only Works With Host Reductions");

    this->reduction_driver_->allreduce_inplace( FXCs, nbf*nbf, ReductionOp::Sum );
    if( FXCz ) this->reduction_driver_->allreduce_inplace( FXCz, nbf*nbf, ReductionOp::Sum );

    this->reduction_driver_->allreduce_inplace( &N_EL, 1    , ReductionOp::Sum );

  });


}

template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  fxc_contraction_local_work_( const basis_type& basis, const value_type* Ps, int64_t ldps,
                            const value_type* Pz, int64_t ldpz,
                            const value_type* tPs, int64_t ldtps,
                            const value_type* tPz, int64_t ldtpz,
                            value_type* FXCs, int64_t ldfxcs,
                            value_type* FXCz, int64_t ldfxcz,
                            value_type *N_EL, const IntegratorSettingsXC& settings,
                            task_iterator task_begin, task_iterator task_end ) {
                                    
  const bool is_uks = Pz != nullptr;
  const bool is_rks = not is_uks;

  // Misc KS settings
  IntegratorSettingsKS ks_settings;
  if( auto* tmp = dynamic_cast<const IntegratorSettingsKS*>(&settings) ) {
    ks_settings = *tmp;
  }

  // Cast LWD to LocalHostWorkDriver
  auto* lwd = dynamic_cast<LocalHostWorkDriver*>(this->local_work_driver_.get());

  // Setup Aliases
  const auto& func  = *this->func_;
  const auto& mol   = this->load_balancer_->molecule();

  const bool needs_laplacian = func.needs_laplacian(); 
  // not suppport laplacian yet
  if( needs_laplacian ) {
    GAUXC_GENERIC_EXCEPTION("Laplacian Not Supported Yet for FXC Contraction");
  }

  // Get basis map
  BasisSetMap basis_map(basis,mol);

  const int32_t nbf = basis.nbf();

  // Sort tasks on size (XXX: maybe doesnt matter?)
  auto task_comparator = []( const XCTask& a, const XCTask& b ) {
    return (a.points.size() * a.bfn_screening.nbe) > (b.points.size() * b.bfn_screening.nbe);
  };

  auto& tasks = this->load_balancer_->get_tasks();
  std::sort( task_begin, task_end, task_comparator );

  // Check that Partition Weights have been calculated
  auto& lb_state = this->load_balancer_->state();
  if( not lb_state.modified_weights_are_stored ) {
    GAUXC_GENERIC_EXCEPTION("Weights Have Not Been Modified");
  }


  // Zero out integrands
  for( auto j = 0; j < nbf; ++j ) 
    for( auto i = 0; i < nbf; ++i ) 
      FXCs[i + j*ldfxcs] = 0.;
    
  if(FXCz)
    for( auto j = 0; j < nbf; ++j ) 
      for( auto i = 0; i < nbf; ++i ) 
        FXCz[i + j*ldfxcz] = 0.;


  // Use FXCs and FXCz  to store FXCa and FXCb temporarily
  value_type* FXCa = FXCs;
  value_type* FXCb = FXCz;
  int64_t ldfxca = ldfxcs;
  int64_t ldfxcb = ldfxcz;
 
  double NEL_WORK = 0.0;
    
  // Loop over tasks
  const size_t ntasks = std::distance(task_begin, task_end);

  #pragma omp parallel
  {

  XCHostData<value_type> host_data; // Thread local host data

  #pragma omp for schedule(dynamic)
  for( size_t iT = 0; iT < ntasks; ++iT ) {
     
    //std::cout << iT << "/" << ntasks << std::endl;
    //if(is_exc_only) printf("%lu / %lu\n", iT, ntasks);
    // Alias current task
    const auto& task = *(task_begin + iT);

    // Get tasks constants
    const int32_t  npts    = task.points.size();
    const int32_t  nbe     = task.bfn_screening.nbe;
    const int32_t  nshells = task.bfn_screening.shell_list.size();

    const auto* points      = task.points.data()->data();
    const auto* weights     = task.weights.data();
    const int32_t* shell_list = task.bfn_screening.shell_list.data();

    // Allocate enough memory for batch
   
    const size_t spin_dim_scal = is_rks ? 1 : 2; 
    const size_t sds          = is_rks ? 1 : 2;
    const size_t mgga_dim_scal = func.is_mgga() ? 4 : 1; // basis + d1basis
    // for second derivatives
    const size_t spin_dim_rhorho = is_rks ? 1 : 3;
    const size_t spin_dim_gammagamma = is_rks ? 1 : 6; 
    const size_t spin_dim_rhogamma = is_rks ? 1 : 6;
    const size_t spin_dim_rhotau = is_rks ? 1 : 4;

    // Things that every calc needs
    host_data.nbe_scr .resize(nbe  * nbe);
    host_data.zmat    .resize(npts * nbe * spin_dim_scal * mgga_dim_scal); 
    host_data.vrho    .resize(npts * spin_dim_scal);
    host_data.v2rho2  .resize(npts * spin_dim_rhorho);
    host_data.FXC_A       .resize(npts * spin_dim_scal);

    // LDA data requirements
    if( func.is_lda() ){
      host_data.basis_eval .resize( npts * nbe );
      host_data.den_scr    .resize( npts * spin_dim_scal);
      host_data.tden_scr   .resize( npts * spin_dim_scal);
    }
     
    // GGA data requirements
    const size_t gga_dim_scal = is_rks ? 1 : 3;
    if( func.is_gga() ){
      host_data.basis_eval .resize( 4 * npts * nbe );
      host_data.den_scr    .resize( spin_dim_scal * 4 * npts );
      host_data.tden_scr   .resize( spin_dim_scal * 4 * npts );
      host_data.gamma      .resize( gga_dim_scal * npts );
      host_data.vgamma     .resize( gga_dim_scal * npts );

      // second derivatives
      host_data.v2rhogamma .resize(npts * spin_dim_rhogamma);
      host_data.v2gamma2   .resize(npts * spin_dim_gammagamma);
      host_data.FXC_B          .resize(npts * 3 * spin_dim_scal);
    }

    if( func.is_mgga() ){

      host_data.den_scr    .resize( spin_dim_scal * 4 * npts );
      host_data.tden_scr   .resize( spin_dim_scal * 4 * npts );
      host_data.gamma      .resize( gga_dim_scal * npts );
      host_data.vgamma     .resize( gga_dim_scal * npts );
      host_data.tau        .resize( npts * spin_dim_scal );
      host_data.vtau       .resize( npts * spin_dim_scal );
      
      // second derivatives
      host_data.v2rhogamma .resize(npts * spin_dim_rhogamma);
      host_data.v2rhotau   .resize(npts * spin_dim_rhotau);
      host_data.v2gamma2   .resize(npts * spin_dim_gammagamma);
      host_data.v2gammatau .resize(npts * spin_dim_rhogamma);
      host_data.v2tau2     .resize(npts * spin_dim_rhorho);
      host_data.ttau       .resize(npts * spin_dim_scal);
      host_data.FXC_B          .resize(npts * 3 * spin_dim_scal);
      host_data.FXC_C          .resize(npts * spin_dim_scal);

      if ( needs_laplacian ) {
        host_data.basis_eval .resize( 11 * npts * nbe ); // basis + grad (3) + hess (6) + lapl 
        host_data.lapl       .resize( spin_dim_scal * npts );
        host_data.vlapl      .resize( spin_dim_scal * npts );
        host_data.v2lapl2    .resize(npts * spin_dim_rhorho);
        host_data.v2rholapl  .resize(npts * spin_dim_rhotau);
        host_data.v2gammalapl.resize(npts * spin_dim_rhogamma);
        host_data.v2lapltau  .resize(npts * spin_dim_rhotau);
        host_data.tlapl      .resize(npts * spin_dim_scal);

      } else {
        host_data.basis_eval .resize( 4 * npts * nbe ); // basis + grad (3)
      }
    }


    // Alias/Partition out scratch memory
    auto* basis_eval = host_data.basis_eval.data();
    auto* den_eval   = host_data.den_scr.data();
    auto* tden_eval   = host_data.tden_scr.data(); // trial density and gradient
    auto* nbe_scr    = host_data.nbe_scr.data();
    auto* zmat       = host_data.zmat.data();

    decltype(zmat) zmat_z = nullptr;
    if(!is_rks) {
      zmat_z = zmat + mgga_dim_scal * nbe * npts;
    }
     
    auto* eps        = host_data.eps.data();
    auto* gamma      = host_data.gamma.data();
    auto* tau        = host_data.tau.data();
    auto* lapl       = host_data.lapl.data();
    auto* vrho       = host_data.vrho.data();
    auto* vgamma     = host_data.vgamma.data();
    auto* vtau       = host_data.vtau.data();
    auto* vlapl      = host_data.vlapl.data();

    // second derivatives
    auto* v2rho2     = host_data.v2rho2.data();
    auto* v2rhogamma = host_data.v2rhogamma.data();
    auto* v2gamma2   = host_data.v2gamma2.data();
    auto* v2gammatau = host_data.v2gammatau.data();
    auto* v2rhotau   = host_data.v2rhotau.data();
    auto* v2lapl2    = host_data.v2lapl2.data();
    auto* v2rholapl  = host_data.v2rholapl.data();
    auto* v2gammalapl= host_data.v2gammalapl.data();
    auto* v2lapltau  = host_data.v2lapltau.data();
    auto* v2tau2     = host_data.v2tau2.data();
    auto* ttau       = host_data.ttau.data();
    auto* tlapl      = host_data.tlapl.data();
    auto* FXC_A          = host_data.FXC_A.data();
    auto* FXC_B          = host_data.FXC_B.data();
    auto* FXC_C          = host_data.FXC_C.data();


    value_type* dbasis_x_eval = nullptr;
    value_type* dbasis_y_eval = nullptr;
    value_type* dbasis_z_eval = nullptr;
    value_type* d2basis_xx_eval = nullptr;
    value_type* d2basis_xy_eval = nullptr;
    value_type* d2basis_xz_eval = nullptr;
    value_type* d2basis_yy_eval = nullptr;
    value_type* d2basis_yz_eval = nullptr;
    value_type* d2basis_zz_eval = nullptr;
    value_type* lbasis_eval = nullptr;
    value_type* dden_x_eval = nullptr;
    value_type* dden_y_eval = nullptr;
    value_type* dden_z_eval = nullptr;
    value_type* tdden_x_eval = nullptr;
    value_type* tdden_y_eval = nullptr;
    value_type* tdden_z_eval = nullptr;
    value_type* mmat_x      = nullptr;
    value_type* mmat_y      = nullptr;
    value_type* mmat_z      = nullptr;
    value_type* mmat_x_z    = nullptr;
    value_type* mmat_y_z    = nullptr;
    value_type* mmat_z_z    = nullptr;

    if( func.is_gga() || func.is_mgga() ) {
      dbasis_x_eval = basis_eval    + npts * nbe;
      dbasis_y_eval = dbasis_x_eval + npts * nbe;
      dbasis_z_eval = dbasis_y_eval + npts * nbe;
      dden_x_eval   = den_eval    + spin_dim_scal * npts;
      dden_y_eval   = dden_x_eval + spin_dim_scal * npts;
      dden_z_eval   = dden_y_eval + spin_dim_scal * npts;
      tdden_x_eval  = tden_eval   + spin_dim_scal * npts;
      tdden_y_eval  = tdden_x_eval+ spin_dim_scal * npts;
      tdden_z_eval  = tdden_y_eval+ spin_dim_scal * npts;
    }

    if ( func.is_mgga() ) {
      mmat_x        = zmat + npts * nbe;
      mmat_y        = mmat_x + npts * nbe;
      mmat_z        = mmat_y + npts * nbe;
      if ( needs_laplacian ) {
        d2basis_xx_eval = dbasis_z_eval + npts * nbe;
        d2basis_xy_eval = d2basis_xx_eval + npts * nbe;
        d2basis_xz_eval = d2basis_xy_eval + npts * nbe;
        d2basis_yy_eval = d2basis_xz_eval + npts * nbe;
        d2basis_yz_eval = d2basis_yy_eval + npts * nbe;
        d2basis_zz_eval = d2basis_yz_eval + npts * nbe;
        lbasis_eval     = d2basis_zz_eval + npts * nbe;
      }
      if(is_uks) {
        mmat_x_z = zmat_z + npts * nbe;
        mmat_y_z = mmat_x_z + npts * nbe;
        mmat_z_z = mmat_y_z + npts * nbe;
      }
    }


    // Get the submatrix map for batch
    std::vector< std::array<int32_t, 3> > submat_map;
    std::tie(submat_map, std::ignore) =
          gen_compressed_submat_map(basis_map, task.bfn_screening.shell_list, nbf, nbf);

    // Evaluate Collocation (+ Grad and Hessian)
    if( func.is_mgga() ) {
      if ( needs_laplacian ) {
        // TODO: Modify gau2grid to compute Laplacian instead of full hessian
        lwd->eval_collocation_hessian( npts, nshells, nbe, points, basis, shell_list,
          basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval, d2basis_xx_eval,
          d2basis_xy_eval, d2basis_xz_eval, d2basis_yy_eval, d2basis_yz_eval,
          d2basis_zz_eval);
        blas::lacpy( 'A', nbe, npts, d2basis_xx_eval, nbe, lbasis_eval, nbe );
        blas::axpy( nbe * npts, 1., d2basis_yy_eval, 1, lbasis_eval, 1);
        blas::axpy( nbe * npts, 1., d2basis_zz_eval, 1, lbasis_eval, 1);
      } else {
        lwd->eval_collocation_gradient( npts, nshells, nbe, points, basis, shell_list,
          basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval );
      }
    }
    // Evaluate Collocation (+ Grad)
    else if( func.is_gga() )
      lwd->eval_collocation_gradient( npts, nshells, nbe, points, basis, shell_list,
        basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval );
    else
      lwd->eval_collocation( npts, nshells, nbe, points, basis, shell_list,
        basis_eval );

     
    // Evaluate X matrix (fac * P * B) -> store in Z
    const auto xmat_fac = is_rks ? 2.0 : 1.0; // TODO Fix for spinor RKS input
    lwd->eval_xmat( mgga_dim_scal * npts, nbf, nbe, submat_map, xmat_fac, Ps, ldps, basis_eval, nbe,
      zmat, nbe, nbe_scr );
    // X matrix for Pz
    if(not is_rks) {
      lwd->eval_xmat( mgga_dim_scal * npts, nbf, nbe, submat_map, 1.0, Pz, ldpz, basis_eval, nbe,
        zmat_z, nbe, nbe_scr);
    }     
     
    // Evaluate U and V variables
    if( func.is_mgga() ) {
      if (is_rks) {
        lwd->eval_uvvar_mgga_rks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, lbasis_eval, zmat, nbe, mmat_x, mmat_y, mmat_z, 
          nbe, den_eval, dden_x_eval, dden_y_eval, dden_z_eval, gamma, tau, lapl);
      } else if (is_uks) {
        lwd->eval_uvvar_mgga_uks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, lbasis_eval, zmat, nbe, zmat_z, nbe, 
          mmat_x, mmat_y, mmat_z, nbe, mmat_x_z, mmat_y_z, mmat_z_z, nbe, 
          den_eval, dden_x_eval, dden_y_eval, dden_z_eval, gamma, tau, lapl);
      }
    } else if ( func.is_gga() ) {
      if(is_rks) {
        lwd->eval_uvvar_gga_rks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, zmat, nbe, den_eval, dden_x_eval, dden_y_eval, dden_z_eval,
          gamma );
      } else if(is_uks) {
        lwd->eval_uvvar_gga_uks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, zmat, nbe, zmat_z, nbe, den_eval, dden_x_eval, 
          dden_y_eval, dden_z_eval, gamma );
      }
  
     } else {
      if(is_rks) {
        lwd->eval_uvvar_lda_rks( npts, nbe, basis_eval, zmat, nbe, den_eval );
      } else if(is_uks) {
        lwd->eval_uvvar_lda_uks( npts, nbe, basis_eval, zmat, nbe, zmat_z, nbe,
          den_eval );
      }
     }

    // Evaluate XC functional
    if( func.is_mgga() )
      func.eval_vxc_fxc( npts, den_eval, gamma, lapl, tau, vrho, vgamma, vlapl, vtau,
        v2rho2, v2rhogamma, v2rholapl, v2rhotau, v2gamma2, 
        v2gammalapl, v2gammatau, v2lapl2, v2lapltau, v2tau2);
    else if( func.is_gga() )
      func.eval_vxc_fxc( npts, den_eval, gamma, vrho, vgamma, v2rho2, v2rhogamma, v2gamma2 );
    else
      func.eval_vxc_fxc( npts, den_eval, vrho, v2rho2 );

    //calculate the trial density variables
    // Evaluate X matrix (fac * tP * B) -> store in Z
    lwd->eval_xmat( mgga_dim_scal * npts, nbf, nbe, submat_map, xmat_fac, tPs, ldps, basis_eval, nbe,
      zmat, nbe, nbe_scr );
    // X matrix for tPz
    if(not is_rks) {
      lwd->eval_xmat( mgga_dim_scal * npts, nbf, nbe, submat_map, 1.0, tPz, ldpz, basis_eval, nbe,
        zmat_z, nbe, nbe_scr);
    }
    // Evaluate U and V trial variables
    if( func.is_mgga() ) {
      if (is_rks) {
        lwd->eval_uvvar_mgga_rks(  npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, lbasis_eval, zmat, nbe, mmat_x, mmat_y, mmat_z, 
          nbe, tden_eval, tdden_x_eval, tdden_y_eval, tdden_z_eval, gamma, ttau, tlapl);
      lwd->eval_tmat_mgga_vxc_rks( npts, vgamma, v2rho2, v2rhogamma, v2rholapl, v2rhotau, v2gamma2, 
        v2gammalapl, v2gammatau, v2lapl2, v2lapltau, v2tau2, tden_eval, tdden_x_eval, 
        tdden_y_eval, tdden_z_eval, ttau, dden_x_eval, dden_y_eval, dden_z_eval, FXC_A, FXC_B, FXC_C );
      } else if (is_uks) {
      // tgamma is not needed since it has different definitions than gamma
      // gamma  = nabla rho * nabla rho, but tgamma = nabla trho * nabla rho, not both trho
      lwd->eval_uvvar_mgga_uks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
        dbasis_z_eval, lbasis_eval, zmat, nbe, zmat_z, nbe, 
        mmat_x, mmat_y, mmat_z, nbe, mmat_x_z, mmat_y_z, mmat_z_z, nbe, 
        tden_eval, tdden_x_eval, tdden_y_eval, tdden_z_eval, gamma, ttau, tlapl);
      lwd->eval_tmat_mgga_vxc_uks( npts, vgamma, v2rho2, v2rhogamma, v2rholapl, v2rhotau, v2gamma2, 
        v2gammalapl, v2gammatau, v2lapl2, v2lapltau, v2tau2, tden_eval, tdden_x_eval, 
        tdden_y_eval, tdden_z_eval, ttau, dden_x_eval, dden_y_eval, dden_z_eval, FXC_A, FXC_B, FXC_C );
      }
    } else if ( func.is_gga() ) {
      if(is_rks) {
        lwd->eval_uvvar_gga_rks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, zmat, nbe, tden_eval, tdden_x_eval, tdden_y_eval, tdden_z_eval,
          gamma );
        lwd->eval_tmat_gga_vxc_rks( npts, vgamma, v2rho2, v2rhogamma, v2gamma2, tden_eval, tdden_x_eval, 
          tdden_y_eval, tdden_z_eval, dden_x_eval, dden_y_eval, dden_z_eval, FXC_A, FXC_B );
      } else if(is_uks) {
      // tgamma is not needed since it has quite different definitions than gamma
      lwd->eval_uvvar_gga_uks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
        dbasis_z_eval, zmat, nbe, zmat_z, nbe, tden_eval, tdden_x_eval, 
        tdden_y_eval, tdden_z_eval, gamma ); 
      lwd->eval_tmat_gga_vxc_uks( npts, vgamma, v2rho2, v2rhogamma, v2gamma2, tden_eval, tdden_x_eval, 
        tdden_y_eval, tdden_z_eval, dden_x_eval, dden_y_eval, dden_z_eval, FXC_A, FXC_B );
      }
    } else {
      // LDA
      if(is_rks) {
        lwd->eval_uvvar_lda_rks( npts, nbe, basis_eval, zmat, nbe, tden_eval );
        lwd->eval_tmat_lda_vxc_rks( npts, v2rho2, tden_eval, FXC_A);
      } else if(is_uks) {
        lwd->eval_uvvar_lda_uks( npts, nbe, basis_eval, zmat, nbe, zmat_z, nbe,
          tden_eval );
        lwd->eval_tmat_lda_vxc_uks( npts, v2rho2, tden_eval, FXC_A);
      }
    }

    // Factor weights into XC results
    for( int32_t i = 0; i < npts; ++i ) {
      FXC_A[sds*i] *= weights[i];
      if(not is_rks) FXC_A[sds*i+1] *= weights[i];
    }
    if( func.is_gga() || func.is_mgga()){
      for( int32_t i = 0; i < npts; ++i ) {
        FXC_B[3*sds*i] *= weights[i];
        FXC_B[3*sds*i+1] *= weights[i];
        FXC_B[3*sds*i+2] *= weights[i];
        if(not is_rks) {
          FXC_B[3*sds*i+3] *= weights[i];
          FXC_B[3*sds*i+4] *= weights[i];
          FXC_B[3*sds*i+5] *= weights[i];
         }
      }
    }
    if( func.is_mgga() ){
      for( int32_t i = 0; i < npts; ++i) {
        FXC_C[sds*i] *= weights[i];
        if(not is_rks) FXC_C[sds*i+1] *= weights[i];
      }
    }

    // Scalar integrations
    double NEL_local = 0.0;
    for( int32_t i = 0; i < npts; ++i ) {
      const auto den = is_rks ? den_eval[i] : (den_eval[2*i] + den_eval[2*i+1]);
      NEL_local += weights[i] * den;
    }


    // Atomic updates
    #pragma omp atomic
    NEL_WORK += NEL_local;
    // Evaluate Z matrix for VXC
    if( func.is_mgga() ) {
      if(is_rks) {
        // Because we do not support Laplacian, so mgga will do the same operation as GGA
        lwd->eval_zmat_gga_vxc_rks_ts( npts, nbe, FXC_A, FXC_B, basis_eval, dbasis_x_eval,
                                dbasis_y_eval, dbasis_z_eval, zmat, nbe);
        lwd->eval_mmat_mgga_vxc_rks( npts, nbe, FXC_C, vlapl, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval,
                                     mmat_x, mmat_y, mmat_z, nbe);
      } else if (is_uks) {
        // Because we do not support Laplacian, so mgga will do the same operation as GGA
        lwd->eval_zmat_gga_vxc_uks_ts( npts, nbe, FXC_A, FXC_B, basis_eval, dbasis_x_eval,
                                dbasis_y_eval, dbasis_z_eval, zmat, nbe, zmat_z, nbe);
        lwd->eval_mmat_mgga_vxc_uks_ts( npts, nbe, FXC_C, vlapl, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval,
                                     mmat_x, mmat_y, mmat_z, nbe, mmat_x_z, mmat_y_z, mmat_z_z, nbe);
      }
    }
    else if( func.is_gga() ) {
      if(is_rks) {
        lwd->eval_zmat_gga_vxc_rks_ts( npts, nbe, FXC_A, FXC_B, basis_eval, dbasis_x_eval,
                                dbasis_y_eval, dbasis_z_eval, zmat, nbe);
      } else if(is_uks) {
        lwd->eval_zmat_gga_vxc_uks_ts( npts, nbe, FXC_A, FXC_B, basis_eval, dbasis_x_eval,
                                dbasis_y_eval, dbasis_z_eval, zmat, nbe, zmat_z, nbe);
      } 
       
    } else {
      if(is_rks) {
        lwd->eval_zmat_lda_vxc_rks( npts, nbe, FXC_A, basis_eval, zmat, nbe );
      } else if(is_uks) {
        lwd->eval_zmat_lda_vxc_uks_ts( npts, nbe, FXC_A, basis_eval, zmat, nbe, zmat_z, nbe );
      }
    }
     
    // Incremeta LT of VXC
    {

      // Increment VXC
      lwd->inc_vxc( mgga_dim_scal * npts, nbf, nbe, basis_eval, submat_map, zmat, nbe, FXCa, ldfxca, nbe_scr );
      if( not is_rks )
        lwd->inc_vxc( mgga_dim_scal * npts, nbf, nbe, basis_eval, submat_map, zmat_z, nbe, FXCb, ldfxcb, nbe_scr);
    }

  } // Loop over tasks

  } // End OpenMP region


  // Set scalar return values
  *N_EL = NEL_WORK;

    // Symmetrize VXC
  for( int32_t j = 0;   j < nbf; ++j ) 
    for( int32_t i = j+1; i < nbf; ++i ) 
      FXCa[ j + i*ldfxca ] = FXCa[ i + j*ldfxca ];
      
  if ( FXCz )
    for( int32_t j = 0;   j < nbf; ++j ) 
      for( int32_t i = j+1; i < nbf; ++i ) 
        FXCb[ j + i*ldfxcb ] = FXCb[ i + j*ldfxcb ];

  if( FXCz ) 
    // now convert to the final form of FXCs and FXCz
    for ( int32_t j = 0;   j < nbf; ++j ) 
      for( int32_t i = 0; i < nbf; ++i ) {
        value_type tmp_a = FXCa[ i + j*ldfxca ];
        value_type tmp_b = FXCb[ i + j*ldfxcb ];
        FXCs[ i + j*ldfxcs ] = 0.5 * ( tmp_a + tmp_b );
        FXCz[ i + j*ldfxcz ] = 0.5 * ( tmp_a - tmp_b );
      }
  
} 


  /// RKS FXC contraction
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
eval_fxc_contraction_( int64_t m, int64_t n, 
    const value_type* P, int64_t ldp, 
    const value_type* tP, int64_t ldtp,
    value_type* FXC, int64_t ldfxc,
    const IntegratorSettingsXC& ks_settings ){

    eval_fxc_contraction_( m, n, P, ldp, nullptr, 0, tP, ldtp, nullptr, 0,
      FXC, ldfxc, nullptr, 0, ks_settings );
}



} // namespace GauXC::detail
