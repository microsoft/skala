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
#include <gauxc/types.hpp>
#include "device/device_queue.hpp"

namespace GauXC {

void eval_kern_exc_vxc_lda( const functional_type& func, size_t npts,
  const double* rho, double* eps, double* vrho, device_queue queue );
void eval_kern_exc_vxc_gga( const functional_type& func, size_t npts,
  const double* rho, const double* gamma, double* eps, double* vrho,
  double* vgamma, device_queue queue );
void eval_kern_exc_vxc_mgga( const functional_type& func, size_t npts,
  const double* rho, const double* gamma, const double* tau, const double* lapl,
  double* eps, double* vrho, double* vgamma, double* vtau, double* vlapl,
  device_queue queue );

void eval_kern_vxc_fxc_lda( const functional_type& func, size_t npts,
  const double* rho, double* vrho, double* v2rho2, device_queue queue );
void eval_kern_vxc_fxc_gga( const functional_type& func, size_t npts,
  const double* rho, const double* gamma, double* vrho, double* vgamma,
  double* v2rho2, double* v2rhogamma, double* v2gamma2, device_queue queue );
void eval_kern_vxc_fxc_mgga( const functional_type& func, size_t npts,
  const double* rho, const double* gamma, const double* lapl, const double* tau,
  double* vrho, double* vgamma, double* vlapl, double* vtau,
  double* v2rho2, double* v2rhogamma, double* v2rholapl, double* v2rhotau,
  double* v2gamma2, double* v2gammalapl, double* v2gammatau, double* v2lapl2,
  double* v2lapltau, double* v2tau2, device_queue queue );

}
