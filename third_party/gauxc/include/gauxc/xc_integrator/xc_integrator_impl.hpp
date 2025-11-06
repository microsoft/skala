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

#include <gauxc/xc_integrator.hpp>

namespace GauXC  {
namespace detail {

/** Base class for XCIntegrator implementation */
template <typename MatrixType>
class XCIntegratorImpl {

public:

  using matrix_type    = MatrixType;
  using value_type     = typename matrix_type::value_type;
  using exc_vxc_type_rks   = typename XCIntegrator<MatrixType>::exc_vxc_type_rks;
  using exc_vxc_type_uks   = typename XCIntegrator<MatrixType>::exc_vxc_type_uks;
  using exc_vxc_type_gks   = typename XCIntegrator<MatrixType>::exc_vxc_type_gks;
  using exc_grad_type  = typename XCIntegrator<MatrixType>::exc_grad_type;
  using exx_type       = typename XCIntegrator<MatrixType>::exx_type;
  using fxc_contraction_type_rks   = typename XCIntegrator<MatrixType>::fxc_contraction_type_rks;
  using fxc_contraction_type_uks   = typename XCIntegrator<MatrixType>::fxc_contraction_type_uks;
  using dd_psi_type       = typename XCIntegrator<MatrixType>::dd_psi_type;
  using dd_psi_potential_type       = typename XCIntegrator<MatrixType>::dd_psi_potential_type;

protected:

  virtual value_type    integrate_den_( const MatrixType& P ) = 0;

  virtual value_type        eval_exc_     ( const MatrixType& P, const IntegratorSettingsXC& ks_settings ) = 0;
  virtual value_type        eval_exc_     ( const MatrixType& Ps, const MatrixType& Pz, const IntegratorSettingsXC& ks_settings ) = 0;
  virtual value_type        eval_exc_     ( const MatrixType& Ps, const MatrixType& Pz, const MatrixType& Py, const MatrixType& Px, const IntegratorSettingsXC& ks_settings ) = 0;

  virtual exc_vxc_type_rks  eval_exc_vxc_ ( const MatrixType& P, const IntegratorSettingsXC& ks_settings ) = 0;
  virtual exc_vxc_type_uks  eval_exc_vxc_ ( const MatrixType& Ps, const MatrixType& Pz, const IntegratorSettingsXC& ks_settings ) = 0;
  virtual exc_vxc_type_gks  eval_exc_vxc_ ( const MatrixType& Ps, const MatrixType& Pz, const MatrixType& Py, const MatrixType& Px, 
                                            const IntegratorSettingsXC& ks_settings ) = 0;
  virtual exc_vxc_type_uks eval_exc_vxc_onedft_  ( const MatrixType& Ps, const MatrixType& Pz, const IntegratorSettingsXC& ks_settings ) = 0;
  virtual exc_grad_type eval_exc_grad_( const MatrixType& P, const IntegratorSettingsXC& ks_settings ) = 0;
  virtual exc_grad_type eval_exc_grad_( const MatrixType& Ps, const MatrixType& Pz, const IntegratorSettingsXC& ks_settings ) = 0;
  virtual exx_type      eval_exx_     ( const MatrixType&     P, 
                                        const IntegratorSettingsEXX& settings ) = 0;
  virtual fxc_contraction_type_rks  eval_fxc_contraction_ ( const MatrixType& P,
    const MatrixType& tP, const IntegratorSettingsXC& ks_settings ) = 0;
  virtual fxc_contraction_type_uks  eval_fxc_contraction_ ( const MatrixType& Ps, const MatrixType& Pz, 
    const MatrixType& tPs, const MatrixType& tPz,  const IntegratorSettingsXC& ks_settings ) = 0;


  virtual dd_psi_type   eval_dd_psi_( const MatrixType& P, unsigned max_Ylm ) = 0;
  virtual dd_psi_potential_type   eval_dd_psi_potential_( const MatrixType& X, unsigned max_Ylm ) = 0;
  virtual const util::Timer& get_timings_() const = 0;
  virtual const LoadBalancer& get_load_balancer_() const = 0;
  virtual LoadBalancer& get_load_balancer_() = 0;
  
public:

  // Default all ctors as base is stateless

  XCIntegratorImpl()                                   = default;
  XCIntegratorImpl( const XCIntegratorImpl& )          = default;
  XCIntegratorImpl( XCIntegratorImpl&&      ) noexcept = default;
  virtual ~XCIntegratorImpl()                 noexcept = default;

  /** Integrate Density (approx N_EL)
   *
   *  @param[in] P The density matrix
   *  @returns Approx Tr[P*S]
   */
  value_type integrate_den( const MatrixType& P ) {
    return integrate_den_(P);
  }

  /** Integrate EXC for RKS
   *
   *  @param[in] P The alpha density matrix
   *  @returns Integrated EXC 
   */
  value_type eval_exc( const MatrixType& P, const IntegratorSettingsXC& ks_settings ) {
    return eval_exc_(P, ks_settings);
  }

  /** Integrate EXC for UKS
   *
   *  @param[in] P The alpha density matrix
   *  @returns Integrated EXC 
   */
  value_type eval_exc( const MatrixType& Ps, const MatrixType& Pz, const IntegratorSettingsXC& ks_settings ) {
    return eval_exc_(Ps, Pz, ks_settings);
  }

  /** Integrate EXC for GKS
   *
   *  @param[in] P The alpha density matrix
   *  @returns Integrated EXC 
   */
  value_type eval_exc( const MatrixType& Ps, const MatrixType& Pz, const MatrixType& Py, const MatrixType& Px,  const IntegratorSettingsXC& ks_settings ) {
    return eval_exc_(Ps, Pz, Py, Px, ks_settings);
  }

  /** Integrate EXC / VXC (Mean field terms) for RKS
   *
   *  @param[in] P The alpha density matrix
   *  @returns EXC / VXC in a combined structure
   */
  exc_vxc_type_rks eval_exc_vxc( const MatrixType& P, const IntegratorSettingsXC& ks_settings ) {
    return eval_exc_vxc_(P, ks_settings);
  }

  exc_vxc_type_uks eval_exc_vxc( const MatrixType& Ps, const MatrixType& Pz, const IntegratorSettingsXC& ks_settings ) {
    return eval_exc_vxc_(Ps, Pz, ks_settings);
  }

  exc_vxc_type_gks eval_exc_vxc( const MatrixType& Ps, const MatrixType& Pz, const MatrixType& Py, const MatrixType& Px, const IntegratorSettingsXC& ks_settings ) {
    return eval_exc_vxc_(Ps, Pz, Py, Px, ks_settings);
  }

  /** Integrate EXC / VXC (Mean field terms) for OneDFT models
  *
  *  @param[in] P The alpha density matrix
  *  @returns EXC / VXC in a combined structure
  */
  exc_vxc_type_uks eval_exc_vxc_onedft( const MatrixType& Ps, const MatrixType& Pz, const IntegratorSettingsXC& ks_settings ) {
    return eval_exc_vxc_onedft_(Ps, Pz, ks_settings);
  }

  /** Integrate EXC gradient for RKS
   *
   *  @param[in] P The alpha density matrix
   *  @returns EXC gradient
   */
  exc_grad_type eval_exc_grad( const MatrixType& P, const IntegratorSettingsXC& ks_settings ) {
    return eval_exc_grad_(P, ks_settings);
  }

  /** Integrate EXC gradient for UKS
   *
   *  @param[in] P The alpha density matrix
   *  @returns EXC gradient
   */
  exc_grad_type eval_exc_grad( const MatrixType& Ps, const MatrixType& Pz, const IntegratorSettingsXC& ks_settings ) {
    return eval_exc_grad_(Ps, Pz, ks_settings);
  }

  /** Integrate Exact Exchange for RHF
   *
   *  @param[in] P The alpha density matrix
   *  @returns Excact Exchange Matrix
   */
  exx_type eval_exx( const MatrixType& P, const IntegratorSettingsEXX& settings ) {
    return eval_exx_(P,settings);
  }

  
  /** Integrate FXC contraction for RKS
   * 
   * @param[in] P the alpha density matrix
   * @param[in] tP the alpha trial density matrix (contructed from purturbed MO coefficients)
   * @returns FXC contraction
   */
  fxc_contraction_type_rks eval_fxc_contraction( const MatrixType& P, const MatrixType& tP, const IntegratorSettingsXC& ks_settings ) {
    return eval_fxc_contraction_(P, tP, ks_settings);
  }

  /** Integrate FXC contraction for UKS
   *
   *  @param[in] Ps the scalar density matrix (Pa + Pb)
   *  @param[in] Pz the Z density matrix (Pa - Pb)
   *  @param[in] tPs the trial scalar density matrices (contructed from purturbed MO coefficients)
   *  @param[in] tPz the trial Z density matrices      (contructed from purturbed MO coefficients)
   *  @returns FXC contraction
   */
  fxc_contraction_type_uks eval_fxc_contraction( const MatrixType& Ps, const MatrixType& Pz, 
    const MatrixType& tPs, const MatrixType& tPz, const IntegratorSettingsXC& ks_settings ) {
    return eval_fxc_contraction_(Ps, Pz, tPs, tPz, ks_settings);
  }

  /** Evaluate Psi vector for ddX
   *
   *  @param[in] P        The density matrix
   *  @param[in] max_Ylm  The max "l" degree for Ylm
   *  @returns   The atomic contributions to the SH projection of the density onto the DD domains
   */   
  dd_psi_type eval_dd_psi( const MatrixType& P, unsigned max_Ylm ) {
    return eval_dd_psi_(P,max_Ylm);
  }

  /** Evaluate Psi Potential for ddX
   *
   *  @param[in] X        The local ASC coefficients, (nharmonics, atom) array in column-major ordering.
   *  @param[in] max_Ylm  The max "l" degree for Ylm
   *  @returns   fock contributions
   */   
  dd_psi_potential_type eval_dd_psi_potential( const MatrixType& X, unsigned max_Ylm ) {
    return eval_dd_psi_potential_(X,max_Ylm);
  }

  /** Get internal timers
   *
   *  @returns Timer instance for internal timings
   */
  const util::Timer& get_timings() const {
    return get_timings_();
  }


  const LoadBalancer& load_balancer() const {
    return get_load_balancer_();
  }
  LoadBalancer& load_balancer() {
    return get_load_balancer_();
  }
};

}
}
