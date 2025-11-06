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
#include "local_host_work_driver_pimpl.hpp"

namespace GauXC {

struct ReferenceLocalHostWorkDriver : public detail::LocalHostWorkDriverPIMPL {

  double *boys_table;
  
  using submat_map_t   = LocalHostWorkDriverPIMPL::submat_map_t;
  using task_container = LocalHostWorkDriverPIMPL::task_container;
  using tast_iterator  = LocalHostWorkDriverPIMPL::task_iterator;

  ReferenceLocalHostWorkDriver();

  virtual ~ReferenceLocalHostWorkDriver() noexcept;

  ReferenceLocalHostWorkDriver( const ReferenceLocalHostWorkDriver& )     = delete;
  ReferenceLocalHostWorkDriver( ReferenceLocalHostWorkDriver&& ) noexcept = delete;

  // Public APIs

  void partition_weights( XCWeightAlg weight_alg, const Molecule& mol, 
    const MolMeta& meta, task_iterator task_begin, task_iterator task_end ) override;

  void eval_weight_1st_deriv_contracted( XCWeightAlg weight_alg, const Molecule& mol, 
    const MolMeta& meta, const XCTask& task, const double* w_times_f, double* exc_grad_w ) override;

  void eval_collocation( size_t npts, size_t nshells, size_t nbe, 
    const double* pts, const BasisSet<double>& basis, const int32_t* shell_list, 
    double* basis_eval ) override;
  void eval_collocation_gradient( size_t npts, size_t nshells, size_t nbe, 
    const double* pts, const BasisSet<double>& basis, const int32_t* shell_list, 
    double* basis_eval, double* dbasis_x_eval, double* dbasis_y_eval, 
    double* dbasis_z_eval) override;
  void eval_collocation_hessian( size_t npts, size_t nshells, size_t nbe, 
    const double* pts, const BasisSet<double>& basis, const int32_t* shell_list, 
    double* basis_eval, double* dbasis_x_eval, double* dbasis_y_eval, 
    double* dbasis_z_eval, double* d2basis_xx_eval, double* d2basis_xy_eval,
    double* d2basis_xz_eval, double* d2basis_yy_eval, double* d2basis_yz_eval,
    double* d2basis_zz_eval ) override;
  void eval_collocation_der3( size_t npts, size_t nshells, size_t nbe,
    const double* pts, const BasisSet<double>& basis, const int32_t* shell_list, 
    double* basis_eval, double* dbasis_x_eval, double* dbasis_y_eval, 
    double* dbasis_z_eval, double* d2basis_xx_eval, double* d2basis_xy_eval,
    double* d2basis_xz_eval, double* d2basis_yy_eval, double* d2basis_yz_eval,
    double* d2basis_zz_eval, double* d3basis_xxx_eval, double* d3basis_xxy_eval,
    double* d3basis_xxz_eval, double* d3basis_xyy_eval, double* d3basis_xyz_eval,
    double* d3basis_xzz_eval, double* d3basis_yyy_eval, double* d3basis_yyz_eval,
    double* d3basis_yzz_eval, double* d3basis_zzz_eval) override;


  void eval_xmat( size_t npts, size_t nbf, size_t nbe, 
    const submat_map_t& submat_map, double fac, const double* P, size_t ldp, 
    const double* basis_eval, size_t ldb, double* X, size_t ldx, double* scr ) 
    override;

  void eval_exx_gmat( size_t npts, size_t nshells, size_t nshell_pairs,
    size_t nbe, const double* points, const double* weights, 
    const BasisSet<double>& basis, const ShellPairCollection<double>& shpairs, 
    const BasisSetMap& basis_map, const int32_t* shell_list, 
    const std::pair<int32_t,int32_t>* shell_pair_list, 
    const double* X, size_t ldx, double* G, size_t ldg ) override ;

  void eval_exx_fmat( size_t npts, size_t nbf, size_t nbe_bra,
    size_t nbe_ket, const submat_map_t& submat_map_bra,
    const submat_map_t& submat_map_ket, const double* P, size_t ldp,
    const double* basis_eval, size_t ldb, double* F, size_t ldf,
    double* scr ) override;

  void inc_exx_k( size_t npts, size_t nbf, size_t nbe_bra, size_t nbe_ket, 
    const double* basis_eval, const submat_map_t& submat_map_bra, 
    const submat_map_t& submat_map_ket, const double* G, size_t ldg, double* K, 
    size_t ldk, double* scr ) override;
    
  void eval_uvvar_lda_rks( size_t npts, size_t nbe, const double* basis_eval,
    const double* X, size_t ldx, double* den_eval) override;
  void eval_uvvar_lda_uks( size_t npts, size_t nbe, const double* basis_eval,
    const double* Xs, size_t ldxs, const double* Xz, size_t ldxz, 
    double* den_eval) override;
  void eval_uvvar_lda_gks( size_t npts, size_t nbe, const double* basis_eval,
    const double* Xs, size_t ldxs, const double* Xz, size_t ldxz,
    const double* Xx, size_t ldxx, const double* Xy, size_t ldxy,
    double* den_eval, double* K, const double dtol) override;

  void eval_uvvar_gga_rks( size_t npts, size_t nbe, const double* basis_eval,
    const double* dbasis_x_eval, const double *dbasis_y_eval, 
    const double* dbasis_z_eval, const double* X, size_t ldx, double* den_eval, 
    double* dden_x_eval, double* dden_y_eval, double* dden_z_eval, 
    double* gamma ) override;
  void eval_uvvar_gga_uks( size_t npts, size_t nbe, const double* basis_eval,
    const double* dbasis_x_eval, const double *dbasis_y_eval,
    const double* dbasis_z_eval, const double* Xs, size_t ldxs, 
    const double* Xz, size_t ldxz, double* den_eval,
    double* dden_x_eval, double* dden_y_eval, double* dden_z_eval,
    double* gamma ) override;
  void eval_uvvar_gga_gks( size_t npts, size_t nbe, const double* basis_eval,
    const double* dbasis_x_eavl, const double *dbasis_y_eval,
    const double* dbasis_z_eval, const double* Xs, size_t ldxs,
    const double* Xz, size_t ldxz, const double* Xx, size_t ldxx,
    const double* Xy, size_t ldxy, double* den_eval,
    double* dden_x_eval, double* dden_y_eval, double* dden_z_eval, double* gamma,
    double* K, double* H, const double dtol ) override;

  void eval_uvvar_mgga_rks( size_t npts, size_t nbe, const double* basis_eval,
    const double* dbasis_x_eval, const double* dbasis_y_eval, 
    const double* dbasis_z_eval, const double* lbasis_eval, 
    const double* X, size_t ldx, const double* mmat_x, const double* mmat_y,
    const double* mmat_z, size_t ldm, double* den_eval, 
    double* dden_x_eval, double* dden_y_eval, double* dden_z_eval, 
    double* gamma, double* tau, double* lapl ) override;
  void eval_uvvar_mgga_uks( size_t npts, size_t nbe, const double* basis_eval,
    const double* dbasis_x_eval, const double *dbasis_y_eval,
    const double* dbasis_z_eval, const double *lbasis_eval, 
    const double* Xs, size_t ldxs, 
    const double* Xz, size_t ldxz, 
    const double* mmat_xs, const double* mmat_ys, const double* mmat_zs, size_t ldms,
    const double* mmat_xz, const double* mmat_yz, const double* mmat_zz, size_t ldmz,
    double* den_eval,
    double* dden_x_eval, double* dden_y_eval, double* dden_z_eval,
    double* gamma, double* tau, double* lapl ) override;

  void eval_zmat_lda_vxc_rks( size_t npts, size_t nbe, const double* vrho, 
    const double* basis_eval, double* Z, size_t ldz ) override;
  void eval_zmat_lda_vxc_uks( size_t npts, size_t nbe, const double* vrho,
    const double* basis_eval, double* Zs, size_t ldzs, double* Zz, size_t ldzz ) override;
  void eval_zmat_lda_vxc_gks( size_t npts, size_t nbe, const double* vrho,
    const double* basis_eval, double* Zs, size_t ldzs, double* Zz, size_t ldzz,
    double* Zx, size_t ldzx,double* Zy, size_t ldzy, double *K ) override;

  void eval_zmat_gga_vxc_rks( size_t npts, size_t nbe, const double* vrho, 
    const double* vgamma, const double* basis_eval, const double* dbasis_x_eval,
    const double* dbasis_y_eval, const double* dbasis_z_eval, 
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval,
    double* Z, size_t ldz ) override;
  void eval_zmat_gga_vxc_uks( size_t npts, size_t nbe, const double* vrho,
    const double* vgamma, const double* basis_eval, const double* dbasis_x_eval,
    const double* dbasis_y_eval, const double* dbasis_z_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval,
    double* Zs, size_t ldzs, double* Zz, size_t ldzz ) override;
  void eval_zmat_gga_vxc_gks( size_t npts, size_t nbe, const double* vrho,
    const double* vgamma, const double* basis_eval, const double* dbasis_x_eval,
    const double* dbasis_y_eval, const double* dbasis_z_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval,
    double* Zs, size_t ldzs, double* Zz, size_t ldzz, double* Zx, size_t ldzx,
    double* Zy, size_t ldzy, double* K, double* H ) override;


  void eval_zmat_mgga_vxc_rks( size_t npts, size_t nbe, const double* vrho, 
    const double* vgamma, const double* vlapl, const double* basis_eval, const double* dbasis_x_eval,
    const double* dbasis_y_eval, const double* dbasis_z_eval, const double* lbasis_eval, 
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval,
    double* Z, size_t ldz ) override;
  void eval_zmat_mgga_vxc_uks( size_t npts, size_t nbe, const double* vrho,
    const double* vgamma, const double* vlapl, const double* basis_eval, const double* dbasis_x_eval,
    const double* dbasis_y_eval, const double* dbasis_z_eval, const double* lbasis_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval,
    double* Zs, size_t ldzs, double* Zz, size_t ldzz ) override;
  void eval_mmat_mgga_vxc_rks( size_t npts, size_t nbe, const double* vtau,
    const double* vlapl, const double* dbasis_x_eval, const double* dbasis_y_eval, 
    const double* dbasis_z_eval, double* mmat_x, double* mmat_y, double* mmat_z,
    size_t ldm ) override;
  void eval_mmat_mgga_vxc_uks( size_t npts, size_t nbe, const double* vtau,
    const double* vlapl, const double* dbasis_x_eval, const double* dbasis_y_eval, 
    const double* dbasis_z_eval, double* mmat_xs, double* mmat_ys, double* mmat_zs,
    size_t ldms, double* mmat_xz, double* mmat_yz, double* mmat_zz, size_t ldmz ) override;


  void inc_vxc( size_t npts, size_t nbf, size_t nbe, 
    const double* basis_eval, const submat_map_t& submat_map, const double* Z, 
    size_t ldz, double* VXC, size_t ldvxc, double* scr ) override;


  void eval_tmat_lda_vxc_rks( size_t npts, const double* v2rho2, const double* tden_eval, double* A) override;
  void eval_tmat_lda_vxc_uks( size_t npts, const double* v2rho2, const double* trho, double* tmat) override;
  
  void eval_tmat_gga_vxc_rks( size_t npts, const double* vgamma, 
    const double* v2rho2, const double* v2rhogamma, const double* v2gamma2, 
    const double* tden_eval, const double* tdden_x_eval, const double* tdden_y_eval, const double* tdden_z_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* A, double* B ) override;
  void eval_tmat_gga_vxc_uks( size_t npts, const double* vgamma, 
    const double* v2rho2, const double* v2rhogamma, const double* v2gamma2, 
    const double* trho, const double* tdden_x_eval, const double* tdden_y_eval, const double* tdden_z_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* A, double* B ) override;
  
  void eval_tmat_mgga_vxc_rks( size_t npts, const double* vgamma, 
    const double* v2rho2, const double* v2rhogamma, const double* v2rholapl, const double* v2rhotau, 
    const double* v2gamma2, const double* v2gammalapl, const double* v2gammatau,
    const double* v2lapl2, const double* v2lapltau, const double* v2tau2, 
    const double* tden_eval, const double* tdden_x_eval, const double* tdden_y_eval, const double* tdden_z_eval, const double* ttau, 
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* A, double* B, double* C) override;
  void eval_tmat_mgga_vxc_uks( size_t npts, const double* vgamma, 
    const double* v2rho2, const double* v2rhogamma, const double* v2rholapl, const double* v2rhotau, 
    const double* v2gamma2, const double* v2gammalapl, const double* v2gamma_tau,
    const double* v2lapl2, const double* v2tau_lapl, const double* v2tau2, 
    const double* trho, const double* tdden_x_eval, const double* tdden_y_eval, const double* tdden_z_eval, const double* ttau, 
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* A, double* B, double* C) override;

  void eval_zmat_lda_vxc_uks_ts( size_t npts, size_t nbe, const double* vrho,
    const double* basis_eval, double* Za, size_t ldza, double* Zb, size_t ldzb ) override;
  void eval_Bvec_gga_vxc_uks_ts( size_t npts, const double* vgamma, 
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* B ) override;
  void eval_zmat_gga_vxc_uks_ts( size_t npts, size_t nbf, const double* A, const double* B, const double* basis_eval,
    const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval, 
    double* Za, size_t ldza, double* Zb, size_t ldzb ) override;
  void eval_Bvec_gga_vxc_rks_ts( size_t npts, const double* vgamma, 
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* B ) override;
  void eval_zmat_gga_vxc_rks_ts( size_t npts, size_t nbf, const double* A, const double* B, const double* basis_eval,
    const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval, 
    double* Z, size_t ldz ) override;

  void eval_zmat_gga_vxc_uks_ts( size_t npts, size_t nbe, const double* vrho,
    const double* vgamma, const double* basis_eval, const double* dbasis_x_eval,
    const double* dbasis_y_eval, const double* dbasis_z_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval,
    double* Za, size_t ldza, double* Zb, size_t ldzb ) override;
  void eval_zmat_mgga_vxc_uks_ts( size_t npts, size_t nbe, const double* vrho,
    const double* vgamma, const double* vlapl,
    const double* basis_eval, const double* dbasis_x_eval, const double* dbasis_y_eval, 
    const double* dbasis_z_eval, const double* lbasis_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval,
    double* Za, size_t ldza, double* Zb, size_t ldzb ) override;
  void eval_mmat_mgga_vxc_uks_ts( size_t npts, size_t nbe, const double* vtau,
    const double* vlapl, const double* dbasis_x_eval, const double* dbasis_y_eval, 
    const double* dbasis_z_eval, double* mmat_xs, double* mmat_ys, double* mmat_zs,
    size_t ldms, double* mmat_xz, double* mmat_yz, double* mmat_zz, size_t ldmz ) override;
};


}
