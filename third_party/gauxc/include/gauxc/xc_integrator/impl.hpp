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

#include <gauxc/xc_integrator/xc_integrator_impl.hpp>

// Implementations of XCIntegrator public API

namespace GauXC {

template <typename MatrixType>
XCIntegrator<MatrixType>::XCIntegrator( std::unique_ptr<pimpl_type>&& pimpl ) :
  pimpl_( std::move( pimpl ) ) { }


template <typename MatrixType>
XCIntegrator<MatrixType>::~XCIntegrator() noexcept = default;

template <typename MatrixType>
XCIntegrator<MatrixType>::XCIntegrator(XCIntegrator&&) noexcept = default;

template <typename MatrixType>
typename XCIntegrator<MatrixType>::value_type
  XCIntegrator<MatrixType>::integrate_den( const MatrixType& P ) {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->integrate_den(P);
};

template <typename MatrixType>
typename XCIntegrator<MatrixType>::value_type
  XCIntegrator<MatrixType>::eval_exc( const MatrixType& P, const IntegratorSettingsXC& ks_settings ) {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->eval_exc(P, ks_settings);
}

template <typename MatrixType>
typename XCIntegrator<MatrixType>::value_type
  XCIntegrator<MatrixType>::eval_exc( const MatrixType& Ps, const MatrixType& Pz, const IntegratorSettingsXC& ks_settings ) {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->eval_exc(Ps, Pz, ks_settings);
}

template <typename MatrixType>
typename XCIntegrator<MatrixType>::value_type
  XCIntegrator<MatrixType>::eval_exc( const MatrixType& Ps, const MatrixType& Pz, const MatrixType& Py, const MatrixType& Px, const IntegratorSettingsXC& ks_settings ) {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->eval_exc(Ps, Pz, Py, Px, ks_settings);
}

template <typename MatrixType>
typename XCIntegrator<MatrixType>::exc_vxc_type_rks
  XCIntegrator<MatrixType>::eval_exc_vxc( const MatrixType& P, const IntegratorSettingsXC& ks_settings ) {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->eval_exc_vxc(P, ks_settings);
};

template <typename MatrixType>
typename XCIntegrator<MatrixType>::exc_vxc_type_uks
  XCIntegrator<MatrixType>::eval_exc_vxc( const MatrixType& Ps, const MatrixType& Pz, const IntegratorSettingsXC& ks_settings ) {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->eval_exc_vxc(Ps, Pz, ks_settings);
};

template <typename MatrixType>
typename XCIntegrator<MatrixType>::exc_vxc_type_gks
  XCIntegrator<MatrixType>::eval_exc_vxc( const MatrixType& Ps, const MatrixType& Pz, const MatrixType& Py, const MatrixType& Px, 
                                          const IntegratorSettingsXC& ks_settings ) {
      if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
        return pimpl_->eval_exc_vxc(Ps, Pz, Py, Px, ks_settings);
  };

template <typename MatrixType>
typename XCIntegrator<MatrixType>::exc_vxc_type_uks
  XCIntegrator<MatrixType>::eval_exc_vxc_onedft( const MatrixType& Ps, const MatrixType& Pz, const IntegratorSettingsXC& ks_settings ) {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->eval_exc_vxc_onedft(Ps, Pz, ks_settings);
};

template <typename MatrixType>
typename XCIntegrator<MatrixType>::exc_grad_type
  XCIntegrator<MatrixType>::eval_exc_grad( const MatrixType& P, const IntegratorSettingsXC& ks_settings ) {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->eval_exc_grad(P, ks_settings);
};

template <typename MatrixType>
typename XCIntegrator<MatrixType>::exc_grad_type
  XCIntegrator<MatrixType>::eval_exc_grad( const MatrixType& Ps, const MatrixType& Pz, const IntegratorSettingsXC& ks_settings ) {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->eval_exc_grad(Ps, Pz, ks_settings);
};

template <typename MatrixType>
typename XCIntegrator<MatrixType>::exx_type
  XCIntegrator<MatrixType>::eval_exx( const MatrixType&     P,
                                      const IntegratorSettingsEXX& settings ) {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->eval_exx(P,settings);
};

template <typename MatrixType>
typename XCIntegrator<MatrixType>::fxc_contraction_type_rks
  XCIntegrator<MatrixType>::eval_fxc_contraction( const MatrixType& P, const MatrixType& tP, 
                                               const IntegratorSettingsXC& ks_settings ) { 
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->eval_fxc_contraction(P, tP, ks_settings);
};

template <typename MatrixType>
typename XCIntegrator<MatrixType>::fxc_contraction_type_uks
  XCIntegrator<MatrixType>::eval_fxc_contraction( const MatrixType& Ps, const MatrixType& Pz, 
                           const MatrixType& tPs, const MatrixType& tPz, const IntegratorSettingsXC& ks_settings ) { 
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->eval_fxc_contraction(Ps, Pz, tPs, tPz, ks_settings);
};

template <typename MatrixType>
typename XCIntegrator<MatrixType>::dd_psi_type
  XCIntegrator<MatrixType>::eval_dd_psi(const MatrixType& P, unsigned max_Ylm) {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->eval_dd_psi(P, max_Ylm);
}

template <typename MatrixType>
typename XCIntegrator<MatrixType>::dd_psi_potential_type
  XCIntegrator<MatrixType>::eval_dd_psi_potential(const MatrixType& X, unsigned max_Ylm) {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->eval_dd_psi_potential(X, max_Ylm);
}


template <typename MatrixType>
const util::Timer& XCIntegrator<MatrixType>::get_timings() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->get_timings();
}

template <typename MatrixType>
const LoadBalancer& XCIntegrator<MatrixType>::load_balancer() const {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->load_balancer();
}

template <typename MatrixType>
LoadBalancer& XCIntegrator<MatrixType>::load_balancer() {
  if( not pimpl_ ) GAUXC_PIMPL_NOT_INITIALIZED();
  return pimpl_->load_balancer();
}
 
}
