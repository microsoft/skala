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
#include <gauxc/xc_integrator/local_work_driver.hpp>

#include <memory>
#include <gauxc/molmeta.hpp>
#include <gauxc/basisset.hpp>
#include <gauxc/basisset_map.hpp>
#include <gauxc/xc_task.hpp>

#include "device/xc_device_data.hpp"
#include <gauxc/runtime_environment/fwd.hpp>

namespace GauXC {
namespace detail {

struct LocalDeviceWorkDriverPIMPL;

}

/// Base class for local work drivers in Device execution spaces 
class LocalDeviceWorkDriver : public LocalWorkDriver {

  using pimpl_type = std::unique_ptr<detail::LocalDeviceWorkDriverPIMPL>;

public:

  using host_task_iterator = std::vector<XCTask>::iterator;

  /// Construct LocalDeviceWorkDriver instance in invalid state
  LocalDeviceWorkDriver();

  /** Construct LocalDeviceWorkDriver instance given implementation pointer
   *  @param[in] ptr Pointer to implementation
   */
  LocalDeviceWorkDriver( pimpl_type&& ptr );

  /// Destructor (default)
  ~LocalDeviceWorkDriver() noexcept;

  // Remove copy ctor
  LocalDeviceWorkDriver( const LocalDeviceWorkDriver& ) = delete;

  /** Construct LocalDeviceWorkDriver by transferring ownership
   *  @param[in] other LocalDeviceWorkDriver instance to take ownership
   */
  LocalDeviceWorkDriver( LocalDeviceWorkDriver&& other ) noexcept;


  // Public APIs

  void partition_weights( XCDeviceData* );
  void eval_weight_1st_deriv_contracted( XCDeviceData*, XCWeightAlg);

  void eval_collocation( XCDeviceData* );
  void eval_collocation_gradient( XCDeviceData* );
  void eval_collocation_hessian( XCDeviceData* );
  void eval_collocation_laplacian( XCDeviceData* );
  void eval_collocation_lapgrad( XCDeviceData* );
  void eval_xmat( double fac, XCDeviceData*, bool do_grad, density_id den );
  void eval_xmat_trial( double fac, XCDeviceData*, bool do_grad, density_id den );
  void save_xmat( XCDeviceData*, bool grad, density_id den );
  
  void eval_uvars_lda ( XCDeviceData*, integrator_ks_scheme ) ;
  void eval_uvars_gga ( XCDeviceData*, integrator_ks_scheme ) ;
  void eval_uvars_mgga( XCDeviceData*, integrator_ks_scheme, bool ) ;
  void eval_vvars_lda ( XCDeviceData*, density_id ) ;
  void eval_vvars_gga ( XCDeviceData*, density_id ) ;
  void eval_vvars_mgga( XCDeviceData*, density_id, bool ) ;

  void eval_tmat_lda ( XCDeviceData*, integrator_ks_scheme ) ;
  void eval_tmat_gga ( XCDeviceData*, integrator_ks_scheme ) ;
  void eval_tmat_mgga( XCDeviceData*, integrator_ks_scheme, bool ) ;
  void eval_vvars_lda_trial ( XCDeviceData*, density_id ) ;
  void eval_vvars_gga_trial ( XCDeviceData*, density_id ) ;
  void eval_vvars_mgga_trial( XCDeviceData*, density_id, bool ) ;


  void eval_kern_exc_vxc_lda( const functional_type&, XCDeviceData* );
  void eval_kern_exc_vxc_gga( const functional_type&, XCDeviceData* );
  void eval_kern_exc_vxc_mgga( const functional_type&, XCDeviceData* );

  void eval_kern_vxc_fxc_lda( const functional_type&, XCDeviceData* );
  void eval_kern_vxc_fxc_gga( const functional_type&, XCDeviceData* );
  void eval_kern_vxc_fxc_mgga( const functional_type&, XCDeviceData* );

  void eval_zmat_lda_vxc( XCDeviceData*, integrator_ks_scheme, density_id ) ;
  void eval_zmat_gga_vxc( XCDeviceData*, integrator_ks_scheme, density_id ) ;
  void eval_zmat_mgga_vxc( XCDeviceData*, integrator_ks_scheme, bool, density_id ) ;
  void eval_mmat_mgga_vxc( XCDeviceData*, integrator_ks_scheme, bool, density_id );

  void eval_zmat_onedft( XCDeviceData*, integrator_term_tracker, density_id );
  void sz_to_ab_onedft( XCDeviceData*, size_t );
  
  void eval_zmat_lda_fxc( XCDeviceData*, density_id ) ;
  void eval_zmat_gga_fxc( XCDeviceData*, density_id ) ;
  void eval_zmat_mgga_fxc( XCDeviceData*, bool, density_id ) ;
  void eval_mmat_mgga_fxc( XCDeviceData*, bool, density_id );

  void eval_exx_fmat( XCDeviceData* );
  void eval_exx_gmat( XCDeviceData*, const BasisSetMap& );

  void inc_exc( XCDeviceData* );
  void inc_nel( XCDeviceData* );
  void inc_vxc( XCDeviceData*, density_id, bool do_m = false );
  void inc_fxc( XCDeviceData*, density_id, bool do_m = false );
  void inc_exc_grad_lda( XCDeviceData*, integrator_ks_scheme, bool );
  void inc_exc_grad_gga( XCDeviceData*, integrator_ks_scheme, bool );
  void inc_exc_grad_mgga( XCDeviceData*, integrator_ks_scheme , bool, bool );
  void inc_exx_k( XCDeviceData* );

  void eval_exx_ek_screening_bfn_stats( XCDeviceData* );
  void exx_ek_shellpair_collision( double eps_E, double eps_K, XCDeviceData*, 
    host_task_iterator, host_task_iterator, const ShellPairCollection<double>& );

  void symmetrize_vxc( XCDeviceData*, density_id );
  void symmetrize_fxc( XCDeviceData*, density_id );
  void symmetrize_exx_k( XCDeviceData* );

  std::unique_ptr<XCDeviceData> create_device_data(const DeviceRuntimeEnvironment&);

private: 

  pimpl_type pimpl_; ///< Implementation

};

}

