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
#include "local_host_work_driver_pimpl.hpp"
#include <stdexcept>

namespace GauXC {

LocalHostWorkDriver::LocalHostWorkDriver() : 
  pimpl_(nullptr) { }
LocalHostWorkDriver::LocalHostWorkDriver(pimpl_type&& ptr) :
  pimpl_( std::move(ptr) ){ }

LocalHostWorkDriver::~LocalHostWorkDriver() noexcept = default;

LocalHostWorkDriver::LocalHostWorkDriver( LocalHostWorkDriver&& other ) noexcept :
  pimpl_(std::move(other.pimpl_)) { }

#define throw_if_invalid_pimpl(ptr) \
  if(not ptr) GAUXC_PIMPL_NOT_INITIALIZED()





// Partition weights
void LocalHostWorkDriver::partition_weights( XCWeightAlg weight_alg, 
  const Molecule& mol, const MolMeta& meta, task_iterator task_begin, 
  task_iterator task_end ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->partition_weights(weight_alg, mol, meta, task_begin, task_end);

}

void LocalHostWorkDriver::eval_weight_1st_deriv_contracted( 
  XCWeightAlg weight_alg, const Molecule& mol, const MolMeta& meta, 
  const XCTask& task, const double* w_times_f, double* exc_grad_w ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_weight_1st_deriv_contracted(weight_alg, mol, meta, task, 
    w_times_f, exc_grad_w);

}


// Collocation
void LocalHostWorkDriver::eval_collocation( size_t npts, size_t nshells, size_t nbe, 
  const double* pts, const BasisSet<double>& basis, const int32_t* shell_list, 
  double* basis_eval ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_collocation(npts, nshells, nbe, pts, basis, shell_list, basis_eval);

}


// Collocation Gradient
void LocalHostWorkDriver::eval_collocation_gradient( size_t npts, size_t nshells, 
  size_t nbe, const double* pts, const BasisSet<double>& basis, 
  const int32_t* shell_list, double* basis_eval, double* dbasis_x_eval, 
  double* dbasis_y_eval, double* dbasis_z_eval) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_collocation_gradient(npts, nshells, nbe, pts, basis, shell_list, basis_eval,
    dbasis_x_eval, dbasis_y_eval, dbasis_z_eval);

}


// Collocation Hessian
void LocalHostWorkDriver::eval_collocation_hessian( size_t npts, size_t nshells, 
    size_t nbe, const double* pts, const BasisSet<double>& basis, 
    const int32_t* shell_list, double* basis_eval, double* dbasis_x_eval, 
    double* dbasis_y_eval, double* dbasis_z_eval, double* d2basis_xx_eval, 
    double* d2basis_xy_eval, double* d2basis_xz_eval, double* d2basis_yy_eval, 
    double* d2basis_yz_eval, double* d2basis_zz_eval ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_collocation_hessian(npts, nshells, nbe, pts, basis, shell_list, basis_eval,
    dbasis_x_eval, dbasis_y_eval, dbasis_z_eval, d2basis_xx_eval, d2basis_xy_eval,
    d2basis_xz_eval, d2basis_yy_eval, d2basis_yz_eval, d2basis_zz_eval);

}

// Collocation 3rd
void LocalHostWorkDriver::eval_collocation_der3( size_t npts, size_t nshells, size_t nbe, 
    const double* pts, const BasisSet<double>& basis, const int32_t* shell_list, 
    double* basis_eval, double* dbasis_x_eval, double* dbasis_y_eval, 
    double* dbasis_z_eval, double* d2basis_xx_eval, double* d2basis_xy_eval,
    double* d2basis_xz_eval, double* d2basis_yy_eval, double* d2basis_yz_eval,
    double* d2basis_zz_eval, double* d3basis_xxx_eval, double* d3basis_xxy_eval,
    double* d3basis_xxz_eval, double* d3basis_xyy_eval, double* d3basis_xyz_eval,
    double* d3basis_xzz_eval, double* d3basis_yyy_eval, double* d3basis_yyz_eval,
    double* d3basis_yzz_eval, double* d3basis_zzz_eval) {

   throw_if_invalid_pimpl(pimpl_);
   pimpl_->eval_collocation_der3(npts, nshells, nbe, pts, basis, shell_list, basis_eval,
    dbasis_x_eval, dbasis_y_eval, dbasis_z_eval, d2basis_xx_eval, d2basis_xy_eval,
    d2basis_xz_eval, d2basis_yy_eval, d2basis_yz_eval, d2basis_zz_eval,
    d3basis_xxx_eval, d3basis_xxy_eval, d3basis_xxz_eval, d3basis_xyy_eval, 
    d3basis_xyz_eval, d3basis_xzz_eval, d3basis_yyy_eval, d3basis_yyz_eval,
    d3basis_yzz_eval, d3basis_zzz_eval);
       
}


// X matrix (fac * P * B)
void LocalHostWorkDriver::eval_xmat( size_t npts, size_t nbf, size_t nbe, 
  const submat_map_t& submat_map, double fac, const double* P, size_t ldp, 
  const double* basis_eval, size_t ldb, double* X, size_t ldx, double* scr ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_xmat(npts, nbf, nbe, submat_map, fac, P, ldp, basis_eval, ldb, X, 
    ldx, scr);

}

void LocalHostWorkDriver::eval_exx_fmat( size_t npts, size_t nbf, size_t nbe_bra,
  size_t nbe_ket, const submat_map_t& submat_map_bra,
  const submat_map_t& submat_map_ket, const double* P, size_t ldp,
  const double* basis_eval, size_t ldb, double* F, size_t ldf,
  double* scr ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_exx_fmat(npts, nbf, nbe_bra, nbe_ket, submat_map_bra,
    submat_map_ket, P, ldp, basis_eval, ldb, F, ldf, scr ); 

}


// G Matrix G(mu,i) = w(i) * A(mu,nu,i) * X(mu,i)
void LocalHostWorkDriver::eval_exx_gmat( size_t npts, size_t nshells, 
  size_t nshell_pairs, size_t nbe, const double* points, const double* weights, 
  const BasisSet<double>& basis, const ShellPairCollection<double>& shpairs, 
  const BasisSetMap& basis_map, const int32_t* shell_list, 
  const std::pair<int32_t,int32_t>* shell_pair_list, 
  const double* X, size_t ldx, double* G, size_t ldg ) {;

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_exx_gmat(npts, nshells, nshell_pairs, nbe, points, weights, 
    basis, shpairs, basis_map, shell_list, shell_pair_list, X, ldx, G, ldg );

}

void LocalHostWorkDriver::inc_exx_k( size_t npts, size_t nbf, size_t nbe_bra, 
  size_t nbe_ket, const double* basis_eval, const submat_map_t& submat_map_bra, 
  const submat_map_t& submat_map_ket, const double* G, size_t ldg, double* K, 
  size_t ldk, double* scr ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->inc_exx_k(npts, nbf, nbe_bra, nbe_ket, basis_eval, submat_map_bra,
    submat_map_ket, G, ldg, K, ldk, scr );
}



// U/VVar LDA (density)
void LocalHostWorkDriver::eval_uvvar_lda_rks( size_t npts, size_t nbe, 
 const double* basis_eval, const double* X, size_t ldx, double* den_eval) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_uvvar_lda_rks(npts, nbe, basis_eval, X, ldx, den_eval);

}

void LocalHostWorkDriver::eval_uvvar_lda_uks( size_t npts, size_t nbe,
 const double* basis_eval, const double* Xs, size_t ldxs, const double* Xz,
 size_t ldxz, double* den_eval) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_uvvar_lda_uks(npts, nbe, basis_eval, Xs, ldxs, Xz, ldxz, den_eval);

}

void LocalHostWorkDriver::eval_uvvar_lda_gks( size_t npts, size_t nbe,
 const double* basis_eval, const double* Xs, size_t ldxs, const double* Xz,
 size_t ldxz, const double* Xx, size_t ldxx, const double* Xy, size_t ldxy,
 double* den_eval, double* K, const double dtol) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_uvvar_lda_gks(npts, nbe, basis_eval, Xs, ldxs, Xz, ldxz, Xx, ldxx, Xy, ldxy, den_eval, K, dtol);

}


// U/VVar GGA (density + grad, gamma)
void LocalHostWorkDriver::eval_uvvar_gga_rks( size_t npts, size_t nbe, 
  const double* basis_eval, const double* dbasis_x_eval, 
  const double *dbasis_y_eval, const double* dbasis_z_eval, const double* X, 
  size_t ldx, double* den_eval, double* dden_x_eval, double* dden_y_eval, 
  double* dden_z_eval, double* gamma ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_uvvar_gga_rks(npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
    dbasis_z_eval, X, ldx, den_eval, dden_x_eval, dden_y_eval, dden_z_eval,
    gamma);

}


void LocalHostWorkDriver::eval_uvvar_gga_uks( size_t npts, size_t nbe,
  const double* basis_eval, const double* dbasis_x_eval,
  const double *dbasis_y_eval, const double* dbasis_z_eval, const double* Xs,
  size_t ldxs, const double* Xz, size_t ldxz, double* den_eval, double* dden_x_eval, double* dden_y_eval,
  double* dden_z_eval, double* gamma ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_uvvar_gga_uks(npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
    dbasis_z_eval, Xs, ldxs, Xz, ldxz, den_eval, dden_x_eval, dden_y_eval, dden_z_eval,
    gamma);

}

void LocalHostWorkDriver::eval_uvvar_gga_gks( size_t npts, size_t nbe,
  const double* basis_eval, const double* dbasis_x_eval,
  const double *dbasis_y_eval, const double* dbasis_z_eval, const double* Xs,
  size_t ldxs, const double* Xz, size_t ldxz, const double* Xx, size_t ldxx,
  const double* Xy, size_t ldxy, double* den_eval, double* dden_x_eval, double* dden_y_eval,
  double* dden_z_eval, double* gamma, double* K, double* H, const double dtol ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_uvvar_gga_gks(npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
    dbasis_z_eval, Xs, ldxs, Xz, ldxz, Xx, ldxx, Xy, ldxy, den_eval, dden_x_eval, dden_y_eval, dden_z_eval,
    gamma, K, H, dtol);

}


// U/VVar MGGA(density, grad, gamma, tau, lapl)
void LocalHostWorkDriver::eval_uvvar_mgga_rks( size_t npts, size_t nbe,
  const double* basis_eval, const double* dbasis_x_eval, 
  const double* dbasis_y_eval, const double* dbasis_z_eval, const double* lbasis_eval,
  const double* X, size_t ldx, const double* mmat_x, const double* mmat_y, const double* mmat_z,
  size_t ldm, double* den_eval, double* dden_x_eval, double* dden_y_eval,
  double* dden_z_eval, double* gamma, double* tau, double* lapl ) {
  
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_uvvar_mgga_rks(npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
    dbasis_z_eval, lbasis_eval, X, ldx, mmat_x, mmat_y, mmat_z, ldm, den_eval, dden_x_eval, dden_y_eval, dden_z_eval,
    gamma, tau, lapl);
  
}


// U/VVar MGGA(density, grad, gamma, tau, lapl)
void LocalHostWorkDriver::eval_uvvar_mgga_uks( size_t npts, size_t nbe,
  const double* basis_eval, const double* dbasis_x_eval, 
  const double* dbasis_y_eval, const double* dbasis_z_eval, const double* lbasis_eval,
  const double* Xs, size_t ldxs, const double* Xz, size_t ldxz, 
  const double* mmat_xs, const double* mmat_ys, const double* mmat_zs, size_t ldms,
  const double* mmat_xz, const double* mmat_yz, const double* mmat_zz, size_t ldmz,
  double* den_eval, double* dden_x_eval, double* dden_y_eval,
  double* dden_z_eval, double* gamma, double* tau, double* lapl ) {
  
  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_uvvar_mgga_uks(npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
    dbasis_z_eval, lbasis_eval, Xs, ldxs, Xz, ldxz, mmat_xs, mmat_ys, mmat_zs, ldms, 
    mmat_xz, mmat_yz, mmat_zz, ldmz, den_eval, dden_x_eval, dden_y_eval, dden_z_eval,
    gamma, tau, lapl);
  
}

// Eval Z Matrix LDA VXC
void LocalHostWorkDriver::eval_zmat_lda_vxc_rks( size_t npts, size_t nbe, 
  const double* vrho, const double* basis_eval, double* Z, size_t ldz ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_zmat_lda_vxc_rks(npts, nbe, vrho, basis_eval, Z, ldz);

}

void LocalHostWorkDriver::eval_zmat_lda_vxc_uks( size_t npts, size_t nbe,
  const double* vrho, const double* basis_eval, double* Zs, size_t ldzs,
  double* Zz, size_t ldzz ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_zmat_lda_vxc_uks(npts, nbe, vrho, basis_eval, Zs, ldzs,
    Zz, ldzz);

}

void LocalHostWorkDriver::eval_zmat_lda_vxc_gks( size_t npts, size_t nbe,
  const double* vrho, const double* basis_eval, double* Zs, size_t ldzs,
  double* Zz, size_t ldzz,double* Zx, size_t ldzx, double* Zy, size_t ldzy, double* K ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_zmat_lda_vxc_gks(npts, nbe, vrho, basis_eval, Zs, ldzs,
    Zz, ldzz, Zx, ldzx, Zy, ldzy, K);


}


// Eval Z Matrix GGA VXC
void LocalHostWorkDriver::eval_zmat_gga_vxc_rks( size_t npts, size_t nbe, 
  const double* vrho, const double* vgamma, const double* basis_eval, 
  const double* dbasis_x_eval, const double* dbasis_y_eval, 
  const double* dbasis_z_eval, const double* dden_x_eval, 
  const double* dden_y_eval, const double* dden_z_eval, double* Z, size_t ldz ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_zmat_gga_vxc_rks(npts, nbe, vrho, vgamma, basis_eval, dbasis_x_eval,
    dbasis_y_eval, dbasis_z_eval, dden_x_eval, dden_y_eval, dden_z_eval,
    Z, ldz);

}

void LocalHostWorkDriver::eval_zmat_gga_vxc_uks( size_t npts, size_t nbe,
  const double* vrho, const double* vgamma, const double* basis_eval,
  const double* dbasis_x_eval, const double* dbasis_y_eval,
  const double* dbasis_z_eval, const double* dden_x_eval,
  const double* dden_y_eval, const double* dden_z_eval, double* Zs, size_t ldzs,
  double* Zz, size_t ldzz ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_zmat_gga_vxc_uks(npts, nbe, vrho, vgamma, basis_eval, dbasis_x_eval,
    dbasis_y_eval, dbasis_z_eval, dden_x_eval, dden_y_eval, dden_z_eval,
    Zs, ldzs, Zz, ldzz);

}

void LocalHostWorkDriver::eval_zmat_gga_vxc_gks( size_t npts, size_t nbe,
  const double* vrho, const double* vgamma, const double* basis_eval,
  const double* dbasis_x_eval, const double* dbasis_y_eval,
  const double* dbasis_z_eval, const double* dden_x_eval,
  const double* dden_y_eval, const double* dden_z_eval, double* Zs, size_t ldzs,
  double* Zz, size_t ldzz, double* Zx, size_t ldzx,double* Zy, size_t ldzy, double* K, double* H ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_zmat_gga_vxc_gks(npts, nbe, vrho, vgamma, basis_eval, dbasis_x_eval,
    dbasis_y_eval, dbasis_z_eval, dden_x_eval, dden_y_eval, dden_z_eval,
    Zs, ldzs, Zz, ldzz, Zx, ldzx, Zy, ldzy, K, H);

}

// Eval Z Matrix MGGA VXC
void LocalHostWorkDriver::eval_zmat_mgga_vxc_rks( size_t npts, size_t nbe, 
  const double* vrho, const double* vgamma, const double* vlapl,
  const double* basis_eval, 
  const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval,
  const double* lbasis_eval, const double* dden_x_eval, 
  const double* dden_y_eval, const double* dden_z_eval, double* Z, size_t ldz ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_zmat_mgga_vxc_rks(npts, nbe, vrho, vgamma, vlapl, basis_eval, dbasis_x_eval,
    dbasis_y_eval, dbasis_z_eval, lbasis_eval, dden_x_eval, dden_y_eval, dden_z_eval,
    Z, ldz);

}


// Eval Z Matrix MGGA VXC
void LocalHostWorkDriver::eval_zmat_mgga_vxc_uks( size_t npts, size_t nbe, 
  const double* vrho, const double* vgamma, const double* vlapl,
  const double* basis_eval, 
  const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval,
  const double* lbasis_eval, const double* dden_x_eval, 
  const double* dden_y_eval, const double* dden_z_eval, double* Zs, size_t ldzs,
  double* Zz, size_t ldzz) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_zmat_mgga_vxc_uks(npts, nbe, vrho, vgamma, vlapl, basis_eval, dbasis_x_eval,
    dbasis_y_eval, dbasis_z_eval, lbasis_eval, dden_x_eval, dden_y_eval, dden_z_eval,
    Zs, ldzs, Zz, ldzz);

}


// Eval M Matrix MGGA VXC
void LocalHostWorkDriver::eval_mmat_mgga_vxc_rks( size_t npts, size_t nbe, 
  const double* vtau, const double* vlapl,
  const double* dbasis_x_eval, const double* dbasis_y_eval, 
  const double* dbasis_z_eval, double* mmat_x, double* mmat_y, double* mmat_z, size_t ldm ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_mmat_mgga_vxc_rks(npts, nbe, vtau, vlapl, dbasis_x_eval,
    dbasis_y_eval, dbasis_z_eval, mmat_x, mmat_y, mmat_z, ldm);

}


// Eval M Matrix MGGA VXC
void LocalHostWorkDriver::eval_mmat_mgga_vxc_uks( size_t npts, size_t nbe, 
  const double* vtau, const double* vlapl,
  const double* dbasis_x_eval, const double* dbasis_y_eval, 
  const double* dbasis_z_eval, double* mmat_xs, double* mmat_ys, double* mmat_zs, size_t ldms,
  double* mmat_xz, double* mmat_yz, double* mmat_zz, size_t ldmz ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_mmat_mgga_vxc_uks(npts, nbe, vtau, vlapl, dbasis_x_eval,
    dbasis_y_eval, dbasis_z_eval, mmat_xs, mmat_ys, mmat_zs, ldms, mmat_xz, mmat_yz,
    mmat_zz, ldmz );

}

// Increment VXC by Z
void LocalHostWorkDriver::inc_vxc( size_t npts, size_t nbf, size_t nbe, 
  const double* basis_eval, const submat_map_t& submat_map, const double* Z, 
  size_t ldz, double* VXC, size_t ldvxc, double* scr ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->inc_vxc(npts, nbf, nbe, basis_eval, submat_map, Z, ldz, VXC, ldvxc, scr);

}


// eval_tmat LDA RKS
void LocalHostWorkDriver::eval_tmat_lda_vxc_rks( size_t npts, const double* v2rho2, const double* trho, double* A) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_tmat_lda_vxc_rks(npts, v2rho2, trho, A);

}

// eval_tmat GGA RKS
void LocalHostWorkDriver::eval_tmat_gga_vxc_rks( size_t npts, const double* vgamma, 
  const double* v2rho2, const double* v2rhogamma, const double* v2gamma2, 
  const double* tden_eval, const double* tdden_x_eval, const double* tdden_y_eval, const double* tdden_z_eval,
  const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* A, double* B ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_tmat_gga_vxc_rks(npts, vgamma, v2rho2, v2rhogamma, v2gamma2,
    tden_eval, tdden_x_eval, tdden_y_eval, tdden_z_eval, dden_x_eval, dden_y_eval,
    dden_z_eval, A, B);

}

// eval_tmat MGGA RKS
void LocalHostWorkDriver::eval_tmat_mgga_vxc_rks( size_t npts, const double* vgamma, 
  const double* v2rho2, const double* v2rhogamma, const double* v2rholapl, const double* v2rhotau, 
  const double* v2gamma2, const double* v2gammalapl, const double* v2gammatau,
  const double* v2lapl2, const double* v2lapltau, const double* v2tau2, 
  const double* tden_eval, const double* tdden_x_eval, const double* tdden_y_eval, const double* tdden_z_eval, const double* ttau, 
  const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* A, double* B, double* C) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_tmat_mgga_vxc_rks(npts, vgamma, v2rho2, v2rhogamma, v2rholapl, v2rhotau,
    v2gamma2, v2gammalapl, v2gammatau, v2lapl2, v2lapltau, v2tau2,
    tden_eval, tdden_x_eval, tdden_y_eval, tdden_z_eval, ttau, dden_x_eval,
    dden_y_eval, dden_z_eval, A, B, C);

}

void LocalHostWorkDriver::eval_tmat_lda_vxc_uks( size_t npts, const double* v2rho2, const double* trho, double* A) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_tmat_lda_vxc_uks(npts, v2rho2, trho, A);

}
void LocalHostWorkDriver::eval_tmat_gga_vxc_uks( size_t npts, const double* vgamma, 
  const double* v2rho2, const double* v2rhogamma, const double* v2gamma2, 
  const double* trho, const double* tdden_x_eval, const double* tdden_y_eval, const double* tdden_z_eval,
  const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* A, double* B ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_tmat_gga_vxc_uks(npts, vgamma, v2rho2, v2rhogamma, v2gamma2,
    trho, tdden_x_eval, tdden_y_eval, tdden_z_eval, dden_x_eval, dden_y_eval,
    dden_z_eval, A, B);

}
void LocalHostWorkDriver::eval_tmat_mgga_vxc_uks( size_t npts, const double* vgamma, 
  const double* v2rho2, const double* v2rhogamma, const double* v2rholapl, const double* v2rhotau, 
  const double* v2gamma2, const double* v2gammalapl, const double* v2gamma_tau,
  const double* v2lapl2, const double* v2tau_lapl, const double* v2tau2, 
  const double* trho, const double* tdden_x_eval, const double* tdden_y_eval, const double* tdden_z_eval, const double* ttau, 
  const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* A, double* B, double* C) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_tmat_mgga_vxc_uks(npts, vgamma, v2rho2, v2rhogamma, v2rholapl, v2rhotau,
    v2gamma2, v2gammalapl, v2gamma_tau, v2lapl2, v2tau_lapl, v2tau2,
    trho, tdden_x_eval, tdden_y_eval, tdden_z_eval, ttau, dden_x_eval,
    dden_y_eval, dden_z_eval, A, B, C);

}

void LocalHostWorkDriver::eval_zmat_lda_vxc_uks_ts( size_t npts, size_t nbe,
  const double* vrho, const double* basis_eval, double* Za, size_t ldza,
  double* Zb, size_t ldzb ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_zmat_lda_vxc_uks_ts(npts, nbe, vrho, basis_eval, Za, ldza,
    Zb, ldzb);

}

void LocalHostWorkDriver::eval_Bvec_gga_vxc_rks_ts( size_t npts, const double* vgamma, 
  const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* B ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_Bvec_gga_vxc_rks_ts(npts, vgamma, dden_x_eval, dden_y_eval,
    dden_z_eval, B);
}

void LocalHostWorkDriver::eval_Bvec_gga_vxc_uks_ts( size_t npts, const double* vgamma, 
  const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* B ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_Bvec_gga_vxc_uks_ts(npts, vgamma, dden_x_eval, dden_y_eval,
    dden_z_eval, B);
}
void LocalHostWorkDriver::eval_zmat_gga_vxc_rks_ts( size_t npts, size_t nbf, const double* A, const double* B, const double* basis_eval,
  const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval,
  double* Z, size_t ldz ){

    throw_if_invalid_pimpl(pimpl_);
    pimpl_->eval_zmat_gga_vxc_rks_ts(npts, nbf, A, B, basis_eval, dbasis_x_eval,
      dbasis_y_eval, dbasis_z_eval, Z, ldz);
}

void LocalHostWorkDriver::eval_zmat_gga_vxc_uks_ts( size_t npts, size_t nbf, const double* A, const double* B, const double* basis_eval,
  const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval,
  double* Za, size_t ldza, double* Zb, size_t ldzb ){

    throw_if_invalid_pimpl(pimpl_);
    pimpl_->eval_zmat_gga_vxc_uks_ts(npts, nbf, A, B, basis_eval, dbasis_x_eval,
      dbasis_y_eval, dbasis_z_eval, Za, ldza, Zb, ldzb);
}

void LocalHostWorkDriver::eval_zmat_mgga_vxc_uks_ts( size_t npts, size_t nbe, 
  const double* vrho, const double* vgamma, const double* vlapl,
  const double* basis_eval, 
  const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval,
  const double* lbasis_eval, const double* dden_x_eval, 
  const double* dden_y_eval, const double* dden_z_eval, double* Za, size_t ldza,
  double* Zb, size_t ldzb) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_zmat_mgga_vxc_uks_ts(npts, nbe, vrho, vgamma, vlapl, basis_eval, dbasis_x_eval,
    dbasis_y_eval, dbasis_z_eval, lbasis_eval, dden_x_eval, dden_y_eval, dden_z_eval,
    Za, ldza, Zb, ldzb);
}

void LocalHostWorkDriver::eval_zmat_gga_vxc_uks_ts( size_t npts, size_t nbe,
  const double* vrho, const double* vgamma, const double* basis_eval,
  const double* dbasis_x_eval, const double* dbasis_y_eval,
  const double* dbasis_z_eval, const double* dden_x_eval,
  const double* dden_y_eval, const double* dden_z_eval, double* Za, size_t ldza,
  double* Zb, size_t ldzb ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_zmat_gga_vxc_uks_ts(npts, nbe, vrho, vgamma, basis_eval, dbasis_x_eval,
    dbasis_y_eval, dbasis_z_eval, dden_x_eval, dden_y_eval, dden_z_eval,
    Za, ldza, Zb, ldzb);

}
void LocalHostWorkDriver::eval_mmat_mgga_vxc_uks_ts( size_t npts, size_t nbe, 
  const double* vtau, const double* vlapl,
  const double* dbasis_x_eval, const double* dbasis_y_eval, 
  const double* dbasis_z_eval, double* mmat_xs, double* mmat_ys, double* mmat_zs, size_t ldms,
  double* mmat_xz, double* mmat_yz, double* mmat_zz, size_t ldmz ) {

  throw_if_invalid_pimpl(pimpl_);
  pimpl_->eval_mmat_mgga_vxc_uks_ts(npts, nbe, vtau, vlapl, dbasis_x_eval,
    dbasis_y_eval, dbasis_z_eval, mmat_xs, mmat_ys, mmat_zs, ldms, mmat_xz, mmat_yz,
    mmat_zz, ldmz );

}



}
