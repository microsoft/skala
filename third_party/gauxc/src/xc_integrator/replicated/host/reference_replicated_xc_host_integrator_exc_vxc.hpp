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
 *  Generic implementation of EXC/VXC for RKS/UKS/GKS
 *  
 *  If passed pointers are null-y and the leading dimensions
 *  are zero, RKS/UKS are deduced. RKS/UKS drivers delegate
 *  to this function/
 */
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exc_vxc_( int64_t m, int64_t n, 
                 const value_type* Ps, int64_t ldps,
                 const value_type* Pz, int64_t ldpz,
                 const value_type* Py, int64_t ldpy,
                 const value_type* Px, int64_t ldpx,
                 value_type* VXCs, int64_t ldvxcs,
                 value_type* VXCz, int64_t ldvxcz,
                 value_type* VXCy, int64_t ldvxcy,
                 value_type* VXCx, int64_t ldvxcx,
                 value_type* EXC, const IntegratorSettingsXC& ks_settings ) {

  const auto& basis = this->load_balancer_->basis();

  // Check that P / VXC are sane
  const int64_t nbf = basis.nbf();
  if( m != n )
    GAUXC_GENERIC_EXCEPTION("P/VXC Must Be Square");
  if( m != nbf )
    GAUXC_GENERIC_EXCEPTION("P/VXC Must Have Same Dimension as Basis");

  if( ldps < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPS");
  if( ldpz and ldpz < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPZ");
  if( ldpy and ldpy < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPX");
  if( ldpx and ldpx < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPY");

  if( ldvxcs < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDVXCS");
  if( ldvxcz and ldvxcz < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDVXCZ");
  if( ldvxcy and ldvxcy < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDVXCX");
  if( ldvxcx and ldvxcx < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDVXCY");

  // Get Tasks
  auto& tasks = this->load_balancer_->get_tasks();

  // Temporary electron count to judge integrator accuracy
  value_type N_EL;
   
  // Compute Local contributions to EXC / VXC
  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
    exc_vxc_local_work_( basis, Ps, ldps, Pz, ldpz, Py, ldpy, Px, ldpx, 
                         VXCs, ldvxcs, VXCz, ldvxcz,
                         VXCy, ldvxcy, VXCx, ldvxcx, EXC, &N_EL, ks_settings,
                         tasks.begin(), tasks.end() );
  });


  // Reduce Results
  this->timer_.time_op("XCIntegrator.Allreduce", [&](){

    if( not this->reduction_driver_->takes_host_memory() )
      GAUXC_GENERIC_EXCEPTION("This Module Only Works With Host Reductions");

    this->reduction_driver_->allreduce_inplace( VXCs, nbf*nbf, ReductionOp::Sum );
    if(VXCz) this->reduction_driver_->allreduce_inplace( VXCz, nbf*nbf, ReductionOp::Sum );
    if(VXCy) this->reduction_driver_->allreduce_inplace( VXCy, nbf*nbf, ReductionOp::Sum ); 
    if(VXCx) this->reduction_driver_->allreduce_inplace( VXCx, nbf*nbf, ReductionOp::Sum );

    this->reduction_driver_->allreduce_inplace( EXC,   1    , ReductionOp::Sum );
    this->reduction_driver_->allreduce_inplace( &N_EL, 1    , ReductionOp::Sum );

  });

  
}


/// Generic implementation details of EXC/VXC local work - deduces RKS/UKS/GKS
/// based on null-y / zero parameters
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  exc_vxc_local_work_( const basis_type& basis, const value_type* Ps, int64_t ldps,
                       const value_type* Pz, int64_t ldpz,
                       const value_type* Py, int64_t ldpy,
                       const value_type* Px, int64_t ldpx,
                       value_type* VXCs, int64_t ldvxcs,
                       value_type* VXCz, int64_t ldvxcz,
                       value_type* VXCy, int64_t ldvxcy,
                       value_type* VXCx, int64_t ldvxcx,
                       value_type* EXC, value_type *N_EL, 
                       const IntegratorSettingsXC& settings,
                       task_iterator task_begin, task_iterator task_end) {

  const bool is_gks = (Pz != nullptr) and (Py != nullptr) and (Px != nullptr);
  const bool is_uks = (Pz != nullptr) and (Py == nullptr) and (Px == nullptr);
  const bool is_rks = not is_uks and not is_gks;
  if (not is_rks and not is_uks and not is_gks) {
    GAUXC_GENERIC_EXCEPTION("Must Be Either RKS, UKS, or GKS!");
  }

  const bool is_exc_only = (!VXCs) and (!VXCz) and (!VXCy) and (!VXCx);
  //if(is_exc_only) std::cout << "EXC ONLY" << std::endl;


  // Misc KS settings
  IntegratorSettingsKS ks_settings;
  if( auto* tmp = dynamic_cast<const IntegratorSettingsKS*>(&settings) ) {
    ks_settings = *tmp;
  }

  const double gks_dtol = ks_settings.gks_dtol;

  // Cast LWD to LocalHostWorkDriver
  auto* lwd = dynamic_cast<LocalHostWorkDriver*>(this->local_work_driver_.get());

  // Setup Aliases
  const auto& func  = *this->func_;
  const auto& mol   = this->load_balancer_->molecule();

  const bool needs_laplacian = func.needs_laplacian(); 
  
  if (func.is_mgga() and is_gks) {
    GAUXC_GENERIC_EXCEPTION("GKS Not Yet Implemented With MGGA Functionals!");
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
  
  if(VXCs)
  for( auto j = 0; j < nbf; ++j ) {
    for( auto i = 0; i < nbf; ++i ) {
      VXCs[i + j*ldvxcs] = 0.;
    }
  }

  if(VXCz) {
    for( auto j = 0; j < nbf; ++j ) {
      for( auto i = 0; i < nbf; ++i ) {
        VXCz[i + j*ldvxcz] = 0.;
      }
    }
  }

  if(VXCx and VXCy) {
    for( auto j = 0; j < nbf; ++j ) {
      for( auto i = 0; i < nbf; ++i ) {
        VXCy[i + j*ldvxcy] = 0.;
        VXCx[i + j*ldvxcx] = 0.;
      }
    }
  }
 
  double EXC_WORK = 0.0;
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
   
    const size_t spin_dim_scal = is_rks ? 1 : is_uks ? 2 : 4; // last case is_gks
    const size_t sds          = is_rks ? 1 : 2;
    const size_t gks_mod_KH = is_gks ? 6*npts : 0; // used to store H and H
    const size_t mgga_dim_scal = func.is_mgga() ? 4 : 1; // basis + d1basis

    // Things that every calc needs
    host_data.nbe_scr .resize(nbe  * nbe);
    host_data.zmat    .resize(npts * nbe * spin_dim_scal * mgga_dim_scal + gks_mod_KH); 
    host_data.eps     .resize(npts);
    host_data.vrho    .resize(npts * spin_dim_scal);

    // LDA data requirements
    if( func.is_lda() ){
      host_data.basis_eval .resize( npts * nbe );
      host_data.den_scr    .resize( npts * spin_dim_scal);
    }
     
    // GGA data requirements
    const size_t gga_dim_scal = is_rks ? 1 : 3;
    if( func.is_gga() ){
      host_data.basis_eval .resize( 4 * npts * nbe );
      host_data.den_scr    .resize( spin_dim_scal * 4 * npts );
      host_data.gamma      .resize( gga_dim_scal * npts );
      host_data.vgamma     .resize( gga_dim_scal * npts );
    }

    if( func.is_mgga() ){
      if ( needs_laplacian ) {
        host_data.basis_eval .resize( 11 * npts * nbe ); // basis + grad (3) + hess (6) + lapl 
        host_data.lapl       .resize( spin_dim_scal * npts );
        host_data.vlapl      .resize( spin_dim_scal * npts );
      } else {
        host_data.basis_eval .resize( 4 * npts * nbe ); // basis + grad (3)
      }

      host_data.den_scr    .resize( spin_dim_scal * 4 * npts );
      host_data.gamma      .resize( gga_dim_scal * npts );
      host_data.vgamma     .resize( gga_dim_scal * npts );
      host_data.tau        .resize( npts * spin_dim_scal );
      host_data.vtau       .resize( npts * spin_dim_scal );
    }

    // Alias/Partition out scratch memory
    auto* basis_eval = host_data.basis_eval.data();
    auto* den_eval   = host_data.den_scr.data();
    auto* nbe_scr    = host_data.nbe_scr.data();
    auto* zmat       = host_data.zmat.data();

    decltype(zmat) zmat_z = nullptr;
    decltype(zmat) zmat_x = nullptr;
    decltype(zmat) zmat_y = nullptr;
    if(!is_rks) {
      zmat_z = zmat + mgga_dim_scal * nbe * npts;
    }
    if(is_gks) {
      zmat_x = zmat_z + nbe * npts;
      zmat_y = zmat_x + nbe * npts;
    }
     
    auto* eps        = host_data.eps.data();
    auto* gamma      = host_data.gamma.data();
    auto* tau        = host_data.tau.data();
    auto* lapl       = host_data.lapl.data();
    auto* vrho       = host_data.vrho.data();
    auto* vgamma     = host_data.vgamma.data();
    auto* vtau       = host_data.vtau.data();
    auto* vlapl      = host_data.vlapl.data();


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
    value_type* K = nullptr;
    value_type* H = nullptr;
    if (is_gks) { K = zmat + npts * nbe * 4; }
    value_type* mmat_x      = nullptr;
    value_type* mmat_y      = nullptr;
    value_type* mmat_z      = nullptr;
    value_type* mmat_x_z    = nullptr;
    value_type* mmat_y_z    = nullptr;
    value_type* mmat_z_z    = nullptr;

    if( func.is_gga() ) {
      dbasis_x_eval = basis_eval    + npts * nbe;
      dbasis_y_eval = dbasis_x_eval + npts * nbe;
      dbasis_z_eval = dbasis_y_eval + npts * nbe;
      dden_x_eval   = den_eval    + spin_dim_scal * npts;
      dden_y_eval   = dden_x_eval + spin_dim_scal * npts;
      dden_z_eval   = dden_y_eval + spin_dim_scal * npts;
      if (is_gks) { H = K + 3*npts;}
    }

    if ( func.is_mgga() ) {
      dbasis_x_eval = basis_eval    + npts * nbe;
      dbasis_y_eval = dbasis_x_eval + npts * nbe;
      dbasis_z_eval = dbasis_y_eval + npts * nbe;
      dden_x_eval   = den_eval    + spin_dim_scal * npts;
      dden_y_eval   = dden_x_eval + spin_dim_scal * npts;
      dden_z_eval   = dden_y_eval + spin_dim_scal * npts;
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
     
    if(is_gks) {
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Py, ldpy, basis_eval, nbe,
        zmat_x, nbe, nbe_scr);
      lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, Px, ldpx, basis_eval, nbe,
        zmat_y, nbe, nbe_scr);
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
      } else if(is_gks) {
        lwd->eval_uvvar_gga_gks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, zmat, nbe, zmat_z, nbe, zmat_x, nbe, zmat_y, nbe, den_eval, dden_x_eval,
          dden_y_eval, dden_z_eval, gamma, K, H, gks_dtol );
      }
       
     } else {
      if(is_rks) {
        lwd->eval_uvvar_lda_rks( npts, nbe, basis_eval, zmat, nbe, den_eval );
      } else if(is_uks) {
        lwd->eval_uvvar_lda_uks( npts, nbe, basis_eval, zmat, nbe, zmat_z, nbe,
          den_eval );
      } else if(is_gks) {
        lwd->eval_uvvar_lda_gks( npts, nbe, basis_eval, zmat, nbe, zmat_z, nbe,
          zmat_x, nbe, zmat_y, nbe, den_eval, K, gks_dtol );
      }
     }
    
    // Evaluate XC functional
    if( func.is_mgga() )
      func.eval_exc_vxc( npts, den_eval, gamma, lapl, tau, eps, vrho, vgamma, vlapl, vtau);
    else if( func.is_gga() )
      func.eval_exc_vxc( npts, den_eval, gamma, eps, vrho, vgamma );
    else
      func.eval_exc_vxc( npts, den_eval, eps, vrho );

    // Factor weights into XC results
    for( int32_t i = 0; i < npts; ++i ) {
      eps[i]  *= weights[i];
      vrho[sds*i] *= weights[i];
      if(not is_rks) vrho[sds*i+1] *= weights[i];
    }
    if( func.is_gga() ){
      for( int32_t i = 0; i < npts; ++i ) {
         vgamma[gga_dim_scal*i] *= weights[i];
         if(not is_rks) {
           vgamma[gga_dim_scal*i+1] *= weights[i];
           vgamma[gga_dim_scal*i+2] *= weights[i];
         }
      }
    }

    if( func.is_mgga() ){
      for( int32_t i = 0; i < npts; ++i) {
        vtau[spin_dim_scal*i]  *= weights[i];
        vgamma[gga_dim_scal*i] *= weights[i];
        if(not is_rks) {
          vgamma[gga_dim_scal*i+1] *= weights[i];
          vgamma[gga_dim_scal*i+2] *= weights[i];
          vtau[spin_dim_scal*i+1]  *= weights[i];
        }

        // TODO: Add checks for Lapacian-dependent functionals
        if( needs_laplacian ) {
          vlapl[spin_dim_scal*i] *= weights[i];
          if(not is_rks) {
            vlapl[spin_dim_scal*i+1] *= weights[i];
          }
        }
      }
    }


    // Scalar integrations
    double NEL_local = 0.0;
    double EXC_local  = 0.0;
    for( int32_t i = 0; i < npts; ++i ) {
      const auto den = is_rks ? den_eval[i] : (den_eval[2*i] + den_eval[2*i+1]);
      NEL_local += weights[i] * den;
      EXC_local += eps[i]     * den;
    }

    // Atomic updates
    #pragma omp atomic
    EXC_WORK += EXC_local;
    #pragma omp atomic
    NEL_WORK += NEL_local;

    if(is_exc_only) continue;

    // Evaluate Z matrix for VXC
    if( func.is_mgga() ) {
      if(is_rks) {
        lwd->eval_zmat_mgga_vxc_rks( npts, nbe, vrho, vgamma, vlapl, basis_eval, dbasis_x_eval,
                                     dbasis_y_eval, dbasis_z_eval, lbasis_eval,
                                     dden_x_eval, dden_y_eval, dden_z_eval, zmat, nbe);
        lwd->eval_mmat_mgga_vxc_rks( npts, nbe, vtau, vlapl, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval,
                                     mmat_x, mmat_y, mmat_z, nbe);
      } else if (is_uks) {
        lwd->eval_zmat_mgga_vxc_uks( npts, nbe, vrho, vgamma, vlapl, basis_eval, dbasis_x_eval,
                                     dbasis_y_eval, dbasis_z_eval, lbasis_eval,
                                     dden_x_eval, dden_y_eval, dden_z_eval, zmat, nbe, zmat_z, nbe);
        lwd->eval_mmat_mgga_vxc_uks( npts, nbe, vtau, vlapl, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval,
                                     mmat_x, mmat_y, mmat_z, nbe, mmat_x_z, mmat_y_z, mmat_z_z, nbe);
      }
    }
    else if( func.is_gga() ) {
      if(is_rks) {
        lwd->eval_zmat_gga_vxc_rks( npts, nbe, vrho, vgamma, basis_eval, dbasis_x_eval,
                                dbasis_y_eval, dbasis_z_eval, dden_x_eval, dden_y_eval,
                                dden_z_eval, zmat, nbe);
      } else if(is_uks) {
        lwd->eval_zmat_gga_vxc_uks( npts, nbe, vrho, vgamma, basis_eval, dbasis_x_eval,
                                dbasis_y_eval, dbasis_z_eval, dden_x_eval, dden_y_eval,
                                dden_z_eval, zmat, nbe, zmat_z, nbe);
      } else if(is_gks) {
        lwd->eval_zmat_gga_vxc_gks( npts, nbe, vrho, vgamma, basis_eval, dbasis_x_eval,
                                dbasis_y_eval, dbasis_z_eval, dden_x_eval, dden_y_eval,
                                dden_z_eval, zmat, nbe, zmat_z, nbe, zmat_x, nbe, zmat_y, nbe,
                                K, H);
      }
       
    } else {
      if(is_rks) {
        lwd->eval_zmat_lda_vxc_rks( npts, nbe, vrho, basis_eval, zmat, nbe );
      } else if(is_uks) {
        lwd->eval_zmat_lda_vxc_uks( npts, nbe, vrho, basis_eval, zmat, nbe, zmat_z, nbe );
      } else if(is_gks) {
        lwd->eval_zmat_lda_vxc_gks( npts, nbe, vrho, basis_eval, zmat, nbe, zmat_z, nbe, 
                                    zmat_x, nbe, zmat_y, nbe, K);
      }
    }
    

     
    // Incremeta LT of VXC
    {

      // Increment VXC
      lwd->inc_vxc( mgga_dim_scal * npts, nbf, nbe, basis_eval, submat_map, zmat, nbe, VXCs, ldvxcs, nbe_scr );
      if(not is_rks) {
        lwd->inc_vxc( mgga_dim_scal * npts, nbf, nbe, basis_eval, submat_map, zmat_z, nbe,VXCz, ldvxcz, nbe_scr);
      }
      if(is_gks) {
        lwd->inc_vxc( npts, nbf, nbe, basis_eval, submat_map, zmat_x, nbe, VXCy, ldvxcy,
          nbe_scr);
        lwd->inc_vxc( npts, nbf, nbe, basis_eval, submat_map, zmat_y, nbe, VXCx, ldvxcx,
          nbe_scr);
      }
       
    }

  } // Loop over tasks

  } // End OpenMP region


  // Set scalar return values
  *EXC  = EXC_WORK;
  *N_EL = NEL_WORK;

  if(not is_exc_only) {
    // Symmetrize VXC
    for( int32_t j = 0;   j < nbf; ++j ) {
      for( int32_t i = j+1; i < nbf; ++i ) {
        VXCs[ j + i*ldvxcs ] = VXCs[ i + j*ldvxcs ];
      }
    }
    if(not is_rks) {
      for( int32_t j = 0;   j < nbf; ++j ) {
        for( int32_t i = j+1; i < nbf; ++i ) {
          VXCz[ j + i*ldvxcz ] = VXCz[ i + j*ldvxcz ];
        }
      }
    }
    if( is_gks) {
      for( int32_t j = 0;   j < nbf; ++j ) {
        for( int32_t i = j+1; i < nbf; ++i ) {
          VXCy[ j + i*ldvxcy ] = VXCy[ i + j*ldvxcy ];
          VXCx[ j + i*ldvxcx ] = VXCx[ i + j*ldvxcx ];
        }
      }
    }
  }

} 



/// RKS EXC/VXC driver - delegates to generic GKS impl
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exc_vxc_( int64_t m, int64_t n, 
                 const value_type* P, int64_t ldp,
                 value_type* VXC, int64_t ldvxc,
                 value_type* EXC, const IntegratorSettingsXC& ks_settings) {

  eval_exc_vxc_(m, n, P, ldp, nullptr, 0, nullptr, 0, nullptr, 0,
    VXC, ldvxc, nullptr, 0, nullptr, 0, nullptr, 0, EXC, ks_settings);

}


/// UKS EXC/VXC driver - delegates to generic GKS impl
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exc_vxc_( int64_t m, int64_t n, 
                 const value_type* Ps, int64_t ldps,
                 const value_type* Pz, int64_t ldpz,
                 value_type* VXCs, int64_t ldvxcs,
                 value_type* VXCz, int64_t ldvxcz,
                 value_type* EXC, const IntegratorSettingsXC& ks_settings) {

  eval_exc_vxc_(m, n, Ps, ldps, Pz, ldpz, nullptr, 0, nullptr, 0,
    VXCs, ldvxcs, VXCz, ldvxcz, nullptr, 0, nullptr, 0,
    EXC, ks_settings);

}

} // namespace GauXC::detail
