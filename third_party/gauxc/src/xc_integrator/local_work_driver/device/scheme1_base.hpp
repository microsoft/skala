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
#include "device/local_device_work_driver_pimpl.hpp"
#include "device/scheme1_data_base.hpp"

namespace GauXC {

struct AoSScheme1Base : public detail::LocalDeviceWorkDriverPIMPL {

  // Device Common APIs (final overrides)
  void eval_collocation( XCDeviceData* ) override final;
  void eval_collocation_gradient( XCDeviceData* ) override final;
  void eval_collocation_hessian( XCDeviceData* ) override final;
  void eval_collocation_laplacian( XCDeviceData* ) override final;
  void eval_collocation_lapgrad( XCDeviceData* ) override final;

  void eval_uvars_lda( XCDeviceData*, integrator_ks_scheme ) override final;
  void eval_uvars_gga( XCDeviceData*, integrator_ks_scheme ) override final;
  void eval_uvars_mgga( XCDeviceData*, integrator_ks_scheme, bool ) override final;
  void eval_vvars_lda ( XCDeviceData*, density_id ) override final;
  void eval_vvars_gga ( XCDeviceData*, density_id ) override final;
  void eval_vvars_mgga( XCDeviceData*, density_id, bool ) override final;

  void eval_tmat_lda( XCDeviceData*, integrator_ks_scheme ) override final;
  void eval_tmat_gga( XCDeviceData*, integrator_ks_scheme ) override final;
  void eval_tmat_mgga( XCDeviceData*, integrator_ks_scheme, bool ) override final;
  void eval_vvars_lda_trial ( XCDeviceData*, density_id ) override final;
  void eval_vvars_gga_trial ( XCDeviceData*, density_id ) override final;
  void eval_vvars_mgga_trial( XCDeviceData*, density_id, bool ) override final;

  void eval_zmat_lda_vxc( XCDeviceData*, integrator_ks_scheme, density_id ) override final;
  void eval_zmat_gga_vxc( XCDeviceData*, integrator_ks_scheme, density_id ) override final;
  void eval_zmat_mgga_vxc( XCDeviceData*, integrator_ks_scheme, bool, density_id ) override final;
  void eval_mmat_mgga_vxc( XCDeviceData*, integrator_ks_scheme, bool, density_id ) override final;

  void eval_zmat_onedft( XCDeviceData*, integrator_term_tracker, density_id ) override final;
  void sz_to_ab_onedft( XCDeviceData*, size_t ) override final;
  
  void eval_zmat_lda_fxc( XCDeviceData*, density_id ) override final;
  void eval_zmat_gga_fxc( XCDeviceData*, density_id ) override final;
  void eval_zmat_mgga_fxc( XCDeviceData*, bool, density_id ) override final;
  void eval_mmat_mgga_fxc( XCDeviceData*, bool, density_id ) override final;

  void eval_kern_exc_vxc_lda( const functional_type&, XCDeviceData* ) override final;
  void eval_kern_exc_vxc_gga( const functional_type&, XCDeviceData* ) override final;
  void eval_kern_exc_vxc_mgga( const functional_type&, XCDeviceData* ) override final;
  void eval_kern_vxc_fxc_lda( const functional_type&, XCDeviceData* ) override final;
  void eval_kern_vxc_fxc_gga( const functional_type&, XCDeviceData* ) override final;
  void eval_kern_vxc_fxc_mgga( const functional_type&, XCDeviceData* ) override final;

  void inc_exc( XCDeviceData* ) override final;
  void inc_nel( XCDeviceData* ) override final;
  void inc_exc_grad_lda( XCDeviceData*, integrator_ks_scheme, bool  ) override final;
  void inc_exc_grad_gga( XCDeviceData*, integrator_ks_scheme, bool  ) override final;
  void inc_exc_grad_mgga( XCDeviceData*, integrator_ks_scheme , bool, bool ) override final;
  void symmetrize_vxc( XCDeviceData* , density_id) override final;
  void symmetrize_fxc( XCDeviceData* , density_id) override final;
  void symmetrize_exx_k( XCDeviceData* ) override final;
  //void eval_exx_gmat( XCDeviceData* ) override final;
  void eval_exx_gmat( XCDeviceData*, const BasisSetMap& ) override final;

  void eval_exx_ek_screening_bfn_stats( XCDeviceData* ) override final;
  void exx_ek_shellpair_collision( double eps_E, double eps_K, 
    XCDeviceData*, host_task_iterator, host_task_iterator,
    const ShellPairCollection<double>& ) override final;

  void save_xmat( XCDeviceData*, bool do_grad, density_id den ) override final;

  
  // Overridable APIs
  template<bool is_trial>
  void eval_xmat_impl(double fac, XCDeviceData*, bool do_grad, density_id );
  template<bool is_fxc>
  void inc_potential_impl(XCDeviceData*, density_id, bool do_m);
  virtual void eval_xmat( double fac, XCDeviceData*, bool , density_id ) override;
  virtual void eval_xmat_trial( double fac, XCDeviceData*, bool , density_id ) override;
  virtual void eval_exx_fmat( XCDeviceData* ) override;
  virtual void inc_vxc( XCDeviceData*, density_id, bool ) override;
  virtual void inc_fxc( XCDeviceData*, density_id, bool ) override;
  virtual void inc_exx_k( XCDeviceData* ) override;


  using Data = Scheme1DataBase;

  AoSScheme1Base();
  virtual ~AoSScheme1Base() noexcept;

  double* dev_boys_table = nullptr;
};

}
