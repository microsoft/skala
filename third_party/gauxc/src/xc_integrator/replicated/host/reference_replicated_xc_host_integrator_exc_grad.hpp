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

template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exc_grad_( int64_t m, int64_t n, const value_type* P,
                 int64_t ldp, value_type* EXC_GRAD, const IntegratorSettingsXC& ks_settings ) { 
                 
                 
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
  this->load_balancer_->get_tasks();
                 
  // Compute Local contributions to EXC / VXC
  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
    exc_grad_local_work_( P, ldp, nullptr, 0, EXC_GRAD, ks_settings );
  });


  // Reduce Results
  this->timer_.time_op("XCIntegrator.Allreduce", [&](){

    if( not this->reduction_driver_->takes_host_memory() )
      GAUXC_GENERIC_EXCEPTION("This Module Only Works With Host Reductions");

    const int natoms = this->load_balancer_->molecule().natoms();
    this->reduction_driver_->allreduce_inplace( EXC_GRAD, 3*natoms, ReductionOp::Sum );
  });

}


template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exc_grad_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps, 
                  const value_type* Pz, int64_t ldpz, value_type* EXC_GRAD, const IntegratorSettingsXC& ks_settings ) { 
                 
                 
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
  this->load_balancer_->get_tasks();
                 
  // Compute Local contributions to EXC / VXC
  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
    exc_grad_local_work_( Ps, ldps, Pz, ldpz, EXC_GRAD, ks_settings );
  });


  // Reduce Results
  this->timer_.time_op("XCIntegrator.Allreduce", [&](){

    if( not this->reduction_driver_->takes_host_memory() )
      GAUXC_GENERIC_EXCEPTION("This Module Only Works With Host Reductions");

    const int natoms = this->load_balancer_->molecule().natoms();
    this->reduction_driver_->allreduce_inplace( EXC_GRAD, 3*natoms, ReductionOp::Sum );
  });

}

template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  exc_grad_local_work_( const value_type* Ps, int64_t ldps, const value_type* Pz, int64_t ldpz, value_type* EXC_GRAD, const IntegratorSettingsXC& settings ) {

  const bool is_uks = Pz != nullptr;
  const bool is_rks = not is_uks;

  // Cast LWD to LocalHostWorkDriver
  auto* lwd = dynamic_cast<LocalHostWorkDriver*>(this->local_work_driver_.get());

  // Setup Aliases
  const auto& func  = *this->func_;
  const auto& basis = this->load_balancer_->basis();
  const auto& mol   = this->load_balancer_->molecule();
  const auto& molmeta = this->load_balancer_->molmeta();

  // MGGA constants
  const bool needs_laplacian = func.needs_laplacian();
  if(needs_laplacian and is_uks) {
    GAUXC_GENERIC_EXCEPTION("UKS Gradients + Laplacian Dependent MGGAs is Not Yet Implemented");
  }

  // Misc KS settings
  IntegratorSettingsEXC_GRAD exc_grad_settings;
  if( auto* tmp = dynamic_cast<const IntegratorSettingsEXC_GRAD*>(&settings) ) {
    exc_grad_settings = *tmp;
  }

  // Get basis map
  BasisSetMap basis_map(basis,mol);

  const int32_t nbf = basis.nbf();
  const int32_t natoms = mol.natoms();

  // Sort tasks on size (XXX: maybe doesnt matter?)
  auto task_comparator = []( const XCTask& a, const XCTask& b ) {
    return (a.points.size() * a.bfn_screening.nbe) > (b.points.size() * b.bfn_screening.nbe);
  };

  auto& tasks = this->load_balancer_->get_tasks();
  std::sort( tasks.begin(), tasks.end(), task_comparator );


  // Check that Partition Weights have been calculated
  auto& lb_state = this->load_balancer_->state();
  if( not lb_state.modified_weights_are_stored ) {
    GAUXC_GENERIC_EXCEPTION("Weights Have Not Been Modified"); 
  }
  XCWeightAlg& weight_alg = lb_state.weight_alg;

  // Zero out integrands
  for( auto i = 0; i < 3*natoms; ++i ) {
    EXC_GRAD[i] = 0.;
  }

  // Loop over tasks
  const size_t ntasks = tasks.size();
  #pragma omp parallel
  {

  XCHostData<value_type> host_data; // Thread local host data

  #pragma omp for schedule(dynamic)
  for( size_t iT = 0; iT < ntasks; ++iT ) {

    // Alias current task
    auto& task = tasks[iT];

    // Get tasks constants
    const int32_t  npts    = task.points.size();
    const int32_t  nbe     = task.bfn_screening.nbe;
    const int32_t  nshells = task.bfn_screening.shell_list.size();
    const size_t spin_dim_scal = is_rks ? 1 : 2; // last case is_uks
    const size_t gga_dim_scal = is_rks ? 1 : 3;

    const auto* points      = task.points.data()->data();
    const auto* weights     = task.weights.data();
    const int32_t* shell_list = task.bfn_screening.shell_list.data();

    // Allocate enough memory for batch

    // Things that every calc needs
    host_data.nbe_scr .resize( nbe * nbe  );
    host_data.eps     .resize( npts );
    host_data.vrho    .resize( spin_dim_scal * npts );
    host_data.den_scr .resize( 4 * spin_dim_scal * npts );

    if( func.is_lda() ) {
      host_data.basis_eval .resize( 4 * npts * nbe );
      host_data.zmat       .resize( spin_dim_scal * npts * nbe );
    }

    if( func.is_gga() or func.is_mgga() ) {
      host_data.basis_eval .resize( 10 * npts * nbe );
      host_data.zmat       .resize( 4  * spin_dim_scal * npts * nbe );
      host_data.gamma      .resize( gga_dim_scal * npts );
      host_data.vgamma     .resize( gga_dim_scal * npts );
    }

    if( func.is_mgga() ) {
      host_data.tau .resize( spin_dim_scal * npts );
      host_data.vtau.resize( spin_dim_scal * npts );
      if ( needs_laplacian ) {
	host_data.basis_eval.resize( 24 * npts * nbe ); // 11 + lapl_grad(3) + der3(10)
	host_data.lapl .resize( spin_dim_scal * npts );
	host_data.vlapl.resize( spin_dim_scal * npts );
      }
    }

    // Alias/Partition out scratch memory
    auto* basis_eval = host_data.basis_eval.data();
    auto* den_eval   = host_data.den_scr.data();
    auto* nbe_scr    = host_data.nbe_scr.data();

    double* xNmat   = nullptr;
    double* xNmat_x = nullptr;
    double* xNmat_y = nullptr;
    double* xNmat_z = nullptr;
    double* xZmat   = nullptr;
    double* xZmat_x = nullptr;
    double* xZmat_y = nullptr;
    double* xZmat_z = nullptr;

    auto* eps        = host_data.eps.data();
    auto* gamma      = host_data.gamma.data();
    auto* vrho       = host_data.vrho.data();
    auto* vgamma     = host_data.vgamma.data();

    auto* tau        = host_data.tau.data();
    auto* lapl       = host_data.lapl.data();
    auto* vtau       = host_data.vtau.data();
    auto* vlapl      = host_data.vlapl.data();

    auto* dbasis_x_eval = basis_eval    + npts * nbe;
    auto* dbasis_y_eval = dbasis_x_eval + npts * nbe;
    auto* dbasis_z_eval = dbasis_y_eval + npts * nbe;
    auto* dden_x_eval   = den_eval    + spin_dim_scal * npts;
    auto* dden_y_eval   = dden_x_eval + spin_dim_scal * npts;
    auto* dden_z_eval   = dden_y_eval + spin_dim_scal * npts;
    

    xNmat   = host_data.zmat.data();
    if(func.is_lda()) {
      xZmat   = xNmat + npts*nbe;
    } else { 
      xNmat_x = xNmat   + npts*nbe;
      xNmat_y = xNmat_x + npts*nbe;
      xNmat_z = xNmat_y + npts*nbe;
      xZmat   = xNmat_z + npts*nbe;
      xZmat_x = xZmat   + npts*nbe;
      xZmat_y = xZmat_x + npts*nbe;
      xZmat_z = xZmat_y + npts*nbe;
    }

    value_type* d2basis_xx_eval = nullptr;
    value_type* d2basis_xy_eval = nullptr;
    value_type* d2basis_xz_eval = nullptr;
    value_type* d2basis_yy_eval = nullptr;
    value_type* d2basis_yz_eval = nullptr;
    value_type* d2basis_zz_eval = nullptr;
     
    value_type* lbasis_eval        = nullptr;
    value_type* d3basis_xxx_eval   = nullptr;
    value_type* d3basis_xxy_eval   = nullptr;
    value_type* d3basis_xxz_eval   = nullptr;
    value_type* d3basis_xyy_eval   = nullptr;
    value_type* d3basis_xyz_eval   = nullptr;
    value_type* d3basis_xzz_eval   = nullptr;
    value_type* d3basis_yyy_eval   = nullptr;
    value_type* d3basis_yyz_eval   = nullptr;
    value_type* d3basis_yzz_eval   = nullptr;
    value_type* d3basis_zzz_eval   = nullptr;
    value_type* dlgradbasis_x_eval = nullptr;
    value_type* dlgradbasis_y_eval = nullptr;
    value_type* dlgradbasis_z_eval = nullptr;

    if( func.is_gga() or func.is_mgga() ) {
      d2basis_xx_eval = dbasis_z_eval   + npts * nbe;
      d2basis_xy_eval = d2basis_xx_eval + npts * nbe;
      d2basis_xz_eval = d2basis_xy_eval + npts * nbe;
      d2basis_yy_eval = d2basis_xz_eval + npts * nbe;
      d2basis_yz_eval = d2basis_yy_eval + npts * nbe;
      d2basis_zz_eval = d2basis_yz_eval + npts * nbe;
    }

    if( needs_laplacian ) {
      lbasis_eval      = d2basis_zz_eval + npts * nbe;
      // TODO - this should not be needed once Gau2Grid 
      // can evaluate the laplacian gradients directly.
      d3basis_xxx_eval = lbasis_eval      + npts * nbe;
      d3basis_xxy_eval = d3basis_xxx_eval + npts * nbe;
      d3basis_xxz_eval = d3basis_xxy_eval + npts * nbe;
      d3basis_xyy_eval = d3basis_xxz_eval + npts * nbe;
      d3basis_xyz_eval = d3basis_xyy_eval + npts * nbe;
      d3basis_xzz_eval = d3basis_xyz_eval + npts * nbe;
      d3basis_yyy_eval = d3basis_xzz_eval + npts * nbe;
      d3basis_yyz_eval = d3basis_yyy_eval + npts * nbe;
      d3basis_yzz_eval = d3basis_yyz_eval + npts * nbe;
      d3basis_zzz_eval = d3basis_yzz_eval + npts * nbe;
      dlgradbasis_x_eval   = d3basis_zzz_eval + npts * nbe;
      dlgradbasis_y_eval   = dlgradbasis_x_eval   + npts * nbe;
      dlgradbasis_z_eval   = dlgradbasis_y_eval   + npts * nbe;
    }


    // Get the submatrix map for batch
    auto [submat_map, foo] = 
      gen_compressed_submat_map( basis_map, task.bfn_screening.shell_list, nbf, nbf );

    // Evaluate Collocation Gradient (+ Hessian)
    if( needs_laplacian ) {
      lwd->eval_collocation_der3( npts, nshells, nbe, points, basis, shell_list, 
        basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval, d2basis_xx_eval,
        d2basis_xy_eval, d2basis_xz_eval, d2basis_yy_eval, d2basis_yz_eval,
        d2basis_zz_eval, d3basis_xxx_eval, d3basis_xxy_eval, d3basis_xxz_eval,
	d3basis_xyy_eval, d3basis_xyz_eval, d3basis_xzz_eval, d3basis_yyy_eval,
	d3basis_yyz_eval, d3basis_yzz_eval, d3basis_zzz_eval);
    } else if( func.is_gga() or func.is_mgga() ) {
      lwd->eval_collocation_hessian( npts, nshells, nbe, points, basis, shell_list, 
        basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval, d2basis_xx_eval,
        d2basis_xy_eval, d2basis_xz_eval, d2basis_yy_eval, d2basis_yz_eval,
        d2basis_zz_eval );
    } else {
      lwd->eval_collocation_gradient( npts, nshells, nbe, points, basis, shell_list, 
        basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval );
    }


    // Evaluate X matrix (2 * P * B/Bx/By/Bz) -> store in Z
    // XXX: This assumes that bfn + gradients are contiguous in memory
    const auto xmat_fac = is_rks ? 2.0 : 1.0;
    const int  xmat_len = func.is_lda() ? 1 : 4;
    lwd->eval_xmat( xmat_len*npts, nbf, nbe, submat_map, xmat_fac, Ps, ldps, basis_eval, nbe,
                    xNmat, nbe, nbe_scr );
    if(is_uks) {
      lwd->eval_xmat( xmat_len*npts, nbf, nbe, submat_map, xmat_fac, Pz, ldpz, basis_eval, nbe,
                      xZmat, nbe, nbe_scr );
    }

    // Evaluate U and V variables
    if( func.is_mgga() ) {
      if ( needs_laplacian ) {
        blas::lacpy( 'A', nbe, npts, d2basis_xx_eval, nbe, lbasis_eval, nbe );
        blas::axpy( nbe * npts, 1., d2basis_yy_eval, 1, lbasis_eval, 1);
        blas::axpy( nbe * npts, 1., d2basis_zz_eval, 1, lbasis_eval, 1);

        // TODO - this should be done directly in Gau2Grid
	blas::lacpy( 'A', nbe, npts, d3basis_xxx_eval, nbe, dlgradbasis_x_eval, nbe );
        blas::axpy( nbe * npts, 1., d3basis_xyy_eval, 1, dlgradbasis_x_eval, 1);
        blas::axpy( nbe * npts, 1., d3basis_xzz_eval, 1, dlgradbasis_x_eval, 1);

	blas::lacpy( 'A', nbe, npts, d3basis_xxy_eval, nbe, dlgradbasis_y_eval, nbe );
        blas::axpy( nbe * npts, 1., d3basis_yyy_eval, 1, dlgradbasis_y_eval, 1);
        blas::axpy( nbe * npts, 1., d3basis_yzz_eval, 1, dlgradbasis_y_eval, 1);

	blas::lacpy( 'A', nbe, npts, d3basis_xxz_eval, nbe, dlgradbasis_z_eval, nbe );
        blas::axpy( nbe * npts, 1., d3basis_yyz_eval, 1, dlgradbasis_z_eval, 1);
        blas::axpy( nbe * npts, 1., d3basis_zzz_eval, 1, dlgradbasis_z_eval, 1);
      }
      if(is_rks)
        lwd->eval_uvvar_mgga_rks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, lbasis_eval, xNmat, nbe, xNmat_x, xNmat_y, xNmat_z, nbe, 
          den_eval, dden_x_eval, dden_y_eval, dden_z_eval, gamma, tau, lapl );
       else
         lwd->eval_uvvar_mgga_uks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
           dbasis_z_eval, lbasis_eval, xNmat, nbe, xZmat, nbe, xNmat_x, xNmat_y, xNmat_z, nbe, 
           xZmat_x, xZmat_y, xZmat_z, nbe, 
           den_eval, dden_x_eval, dden_y_eval, dden_z_eval, gamma, tau, lapl );
    } else if( func.is_gga() ) {
      if(is_rks)
        lwd->eval_uvvar_gga_rks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, xNmat, nbe, den_eval, dden_x_eval, dden_y_eval, dden_z_eval,
          gamma );
      else
        lwd->eval_uvvar_gga_uks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, xNmat, nbe, xZmat, nbe, den_eval, dden_x_eval, dden_y_eval, 
          dden_z_eval, gamma );
    } else {
      if(is_rks) lwd->eval_uvvar_lda_rks( npts, nbe, basis_eval, xNmat, nbe, den_eval );
      else       lwd->eval_uvvar_lda_uks( npts, nbe, basis_eval, xNmat, nbe, xZmat, nbe, den_eval );
    }
    

    // Evaluate XC functional
    if( func.is_mgga() )
      func.eval_exc_vxc( npts, den_eval, gamma, lapl, tau, eps, vrho, vgamma, vlapl, vtau );
    else if(func.is_gga() )
      func.eval_exc_vxc( npts, den_eval, gamma, eps, vrho, vgamma );
    else
      func.eval_exc_vxc( npts, den_eval, eps, vrho );

    if(exc_grad_settings.include_weight_derivatives){
      // grid weight contribution to exc grad
      for( int ipt = 0; ipt < npts; ++ipt ) {
        const auto den = is_rks ? den_eval[ipt] : (den_eval[2*ipt] + den_eval[2*ipt+1]);
        eps[ipt] *=  den * weights[ipt];
      }
      lwd->eval_weight_1st_deriv_contracted( weight_alg, mol, molmeta, 
        task, eps, EXC_GRAD);
    }


    // Increment EXC Gradient
    size_t bf_off = 0;
    for( auto ish = 0; ish < nshells; ++ish ) {
      const int sh_idx = shell_list[ish];
      const int sh_sz  = basis[sh_idx].size();
      const int iAt    = basis_map.shell_to_center( sh_idx );
      if(iAt == task.iParent and exc_grad_settings.include_weight_derivatives) {
        bf_off += sh_sz; // Increment basis offset
        continue;
      }

      double g_acc_x(0), g_acc_y(0), g_acc_z(0);
      for( int ibf = 0, mu = bf_off; ibf < sh_sz; ++ibf, ++mu )
      for( int ipt = 0; ipt < npts; ++ipt ) {

        const int32_t mu_i = mu + ipt*nbe;

        // LDA Contributions
        // vrhop is actually vrhon for RKS
        const double vrhop_ipt = weights[ipt] * vrho[spin_dim_scal * ipt];
        const double vrhom_ipt = is_uks ? weights[ipt] * vrho[spin_dim_scal * ipt + 1] : 0.0;

	const double xN = xNmat[mu_i]; // X = N * B
        const double xZ = is_uks ? xZmat[mu_i] : 0.0;

	const double dbx = dbasis_x_eval[mu_i]; // B_x
	const double dby = dbasis_y_eval[mu_i]; // B_y
	const double dbz = dbasis_z_eval[mu_i]; // B_z

        if(is_rks) {
          g_acc_x += vrhop_ipt * xN * dbx;
          g_acc_y += vrhop_ipt * xN * dby;
          g_acc_z += vrhop_ipt * xN * dbz;
        } else {
          const auto vrhon_ipt = vrhop_ipt + vrhom_ipt;
          const auto vrhoz_ipt = vrhop_ipt - vrhom_ipt;
          g_acc_x += 0.5 * vrhon_ipt * xN * dbx;
          g_acc_y += 0.5 * vrhon_ipt * xN * dby;
          g_acc_z += 0.5 * vrhon_ipt * xN * dbz;

          g_acc_x += 0.5 * vrhoz_ipt * xZ * dbx;
          g_acc_y += 0.5 * vrhoz_ipt * xZ * dby;
          g_acc_z += 0.5 * vrhoz_ipt * xZ * dbz;
        }


        if( func.is_gga() or func.is_mgga() ) {
          // GGA Contributions
          const double vgammapp_ipt = weights[ipt] * vgamma[gga_dim_scal * ipt + 0];
          const double vgammapm_ipt = is_uks ? weights[ipt] * vgamma[gga_dim_scal * ipt + 1] : 0.0;
          const double vgammamm_ipt = is_uks ? weights[ipt] * vgamma[gga_dim_scal * ipt + 2] : 0.0;

          const double ddenn_x = dden_x_eval[spin_dim_scal * ipt];
          const double ddenn_y = dden_y_eval[spin_dim_scal * ipt];
          const double ddenn_z = dden_z_eval[spin_dim_scal * ipt];
          const double ddenz_x = is_uks ? dden_x_eval[spin_dim_scal * ipt + 1] : 0.0;
          const double ddenz_y = is_uks ? dden_y_eval[spin_dim_scal * ipt + 1] : 0.0;
          const double ddenz_z = is_uks ? dden_z_eval[spin_dim_scal * ipt + 1] : 0.0;

          const double xNx = xNmat_x[mu_i]; // XN_x = N * B_x
          const double xNy = xNmat_y[mu_i]; // XN_y = N * B_y
          const double xNz = xNmat_z[mu_i]; // XN_z = N * B_z

          const double xZx = is_uks ? xZmat_x[mu_i] : 0.0;
          const double xZy = is_uks ? xZmat_y[mu_i] : 0.0;
          const double xZz = is_uks ? xZmat_z[mu_i] : 0.0;

          const double d2bxx = d2basis_xx_eval[mu_i]; // B^2_xx
          const double d2bxy = d2basis_xy_eval[mu_i]; // B^2_xy
          const double d2bxz = d2basis_xz_eval[mu_i]; // B^2_xz
          const double d2byy = d2basis_yy_eval[mu_i]; // B^2_yy
          const double d2byz = d2basis_yz_eval[mu_i]; // B^2_yz
          const double d2bzz = d2basis_zz_eval[mu_i]; // B^2_zz
      
          if(is_rks) {
            // sum_j B^2_{ij} * d_j n
            const auto d2_term_x = d2bxx * ddenn_x + d2bxy * ddenn_y + d2bxz * ddenn_z;
            const auto d2_term_y = d2bxy * ddenn_x + d2byy * ddenn_y + d2byz * ddenn_z;
            const auto d2_term_z = d2bxz * ddenn_x + d2byz * ddenn_y + d2bzz * ddenn_z;

            // sum_j (d_j n) * xN^j
            const double d11_xmat_term = ddenn_x * xNx + ddenn_y * xNy + ddenn_z * xNz;

            g_acc_x += 2 * vgammapp_ipt * ( xN * d2_term_x + dbx * d11_xmat_term );
            g_acc_y += 2 * vgammapp_ipt * ( xN * d2_term_y + dby * d11_xmat_term );
            g_acc_z += 2 * vgammapp_ipt * ( xN * d2_term_z + dbz * d11_xmat_term );
          } else {
            // sum_j B^2_{ij} * d_j n
            const auto d2n_term_x = d2bxx * ddenn_x + d2bxy * ddenn_y + d2bxz * ddenn_z;
            const auto d2n_term_y = d2bxy * ddenn_x + d2byy * ddenn_y + d2byz * ddenn_z;
            const auto d2n_term_z = d2bxz * ddenn_x + d2byz * ddenn_y + d2bzz * ddenn_z;

            // sum_j B^2_{ij} * d_j m_z
            const auto d2z_term_x = d2bxx * ddenz_x + d2bxy * ddenz_y + d2bxz * ddenz_z;
            const auto d2z_term_y = d2bxy * ddenz_x + d2byy * ddenz_y + d2byz * ddenz_z;
            const auto d2z_term_z = d2bxz * ddenz_x + d2byz * ddenz_y + d2bzz * ddenz_z;

            // sum_j (d_j n) * xN^j
            const double d11nn_xmat_term = ddenn_x * xNx + ddenn_y * xNy + ddenn_z * xNz;
            // sum_j (d_j n) * xZ^j
            const double d11nz_xmat_term = ddenn_x * xZx + ddenn_y * xZy + ddenn_z * xZz;
            // sum_j (d_j m_z) * xN^j
            const double d11zn_xmat_term = ddenz_x * xNx + ddenz_y * xNy + ddenz_z * xNz;
            // sum_j (d_j m_z) * xZ^j
            const double d11zz_xmat_term = ddenz_x * xZx + ddenz_y * xZy + ddenz_z * xZz;


            g_acc_x += 0.5 * (vgammapp_ipt + vgammapm_ipt + vgammamm_ipt) * (d2n_term_x * xN + d11nn_xmat_term * dbx);
            g_acc_x += 0.5 * (vgammapp_ipt                - vgammamm_ipt) * (d2z_term_x * xN + d11zn_xmat_term * dbx);
            g_acc_x += 0.5 * (vgammapp_ipt                - vgammamm_ipt) * (d2n_term_x * xZ + d11nz_xmat_term * dbx);
            g_acc_x += 0.5 * (vgammapp_ipt - vgammapm_ipt + vgammamm_ipt) * (d2z_term_x * xZ + d11zz_xmat_term * dbx);

            g_acc_y += 0.5 * (vgammapp_ipt + vgammapm_ipt + vgammamm_ipt) * (d2n_term_y * xN + d11nn_xmat_term * dby);
            g_acc_y += 0.5 * (vgammapp_ipt                - vgammamm_ipt) * (d2z_term_y * xN + d11zn_xmat_term * dby);
            g_acc_y += 0.5 * (vgammapp_ipt                - vgammamm_ipt) * (d2n_term_y * xZ + d11nz_xmat_term * dby);
            g_acc_y += 0.5 * (vgammapp_ipt - vgammapm_ipt + vgammamm_ipt) * (d2z_term_y * xZ + d11zz_xmat_term * dby);

            g_acc_z += 0.5 * (vgammapp_ipt + vgammapm_ipt + vgammamm_ipt) * (d2n_term_z * xN + d11nn_xmat_term * dbz);
            g_acc_z += 0.5 * (vgammapp_ipt                - vgammamm_ipt) * (d2z_term_z * xN + d11zn_xmat_term * dbz);
            g_acc_z += 0.5 * (vgammapp_ipt                - vgammamm_ipt) * (d2n_term_z * xZ + d11nz_xmat_term * dbz);
            g_acc_z += 0.5 * (vgammapp_ipt - vgammapm_ipt + vgammamm_ipt) * (d2z_term_z * xZ + d11zz_xmat_term * dbz);
            
          }

          if( func.is_mgga() ) {
            // vtaup is actually vtaun for RKS
            const double vtaup_ipt = 0.5 * weights[ipt] * vtau[spin_dim_scal * ipt + 0];
            const double vtaum_ipt = is_uks ? 0.5 * weights[ipt] * vtau[spin_dim_scal * ipt + 1] : 0.0;

            auto d2_term_x = d2bxx * xNx + d2bxy * xNy + d2bxz * xNz;
            auto d2_term_y = d2bxy * xNx + d2byy * xNy + d2byz * xNz;
            auto d2_term_z = d2bxz * xNx + d2byz * xNy + d2bzz * xNz;

            if(is_rks) {
              g_acc_x += vtaup_ipt * d2_term_x;
              g_acc_y += vtaup_ipt * d2_term_y;
              g_acc_z += vtaup_ipt * d2_term_z;
            } else {
              const auto vtaun_ipt = vtaup_ipt + vtaum_ipt;
              const auto vtauz_ipt = vtaup_ipt - vtaum_ipt;
              g_acc_x += 0.5 * vtaun_ipt * d2_term_x;
              g_acc_y += 0.5 * vtaun_ipt * d2_term_y;
              g_acc_z += 0.5 * vtaun_ipt * d2_term_z;

              d2_term_x = d2bxx * xZx + d2bxy * xZy + d2bxz * xZz;
              d2_term_y = d2bxy * xZx + d2byy * xZy + d2byz * xZz;
              d2_term_z = d2bxz * xZx + d2byz * xZy + d2bzz * xZz;

              g_acc_x += 0.5 * vtauz_ipt * d2_term_x;
              g_acc_y += 0.5 * vtauz_ipt * d2_term_y;
              g_acc_z += 0.5 * vtauz_ipt * d2_term_z;
            }

            if( needs_laplacian ) {
              const double vlapl_ipt = weights[ipt] * vlapl[ipt];
              const double lbf = lbasis_eval[mu_i];
              const double dlbx = dlgradbasis_x_eval[mu_i];
              const double dlby = dlgradbasis_y_eval[mu_i];
              const double dlbz = dlgradbasis_z_eval[mu_i];
              d2_term_x = xN * dlbx + xNx * lbf + 2.0*d2_term_x;
              d2_term_y = xN * dlby + xNy * lbf + 2.0*d2_term_y;
              d2_term_z = xN * dlbz + xNz * lbf + 2.0*d2_term_z;

              g_acc_x += vlapl_ipt * d2_term_x;
              g_acc_y += vlapl_ipt * d2_term_y;
              g_acc_z += vlapl_ipt * d2_term_z;
            }
          }
        }
      } // loop over bfns + grid points

      #pragma omp atomic
      EXC_GRAD[3*iAt + 0] += -2 * g_acc_x;
      #pragma omp atomic
      EXC_GRAD[3*iAt + 1] += -2 * g_acc_y;
      #pragma omp atomic
      EXC_GRAD[3*iAt + 2] += -2 * g_acc_z;

      if(exc_grad_settings.include_weight_derivatives){
        #pragma omp atomic
        EXC_GRAD[3*task.iParent + 0] -= -2 * g_acc_x;
        #pragma omp atomic
        EXC_GRAD[3*task.iParent + 1] -= -2 * g_acc_y;
        #pragma omp atomic
        EXC_GRAD[3*task.iParent + 2] -= -2 * g_acc_z;
      }

      bf_off += sh_sz; // Increment basis offset

    } // End loop over shells 

  } // End loop over tasks

  } // OpenMP Region

  
}

} // namespace GauXC::detail
