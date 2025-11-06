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
#include "incore_replicated_xc_device_integrator.hpp"
#include <gauxc/util/misc.hpp>
#include <gauxc/util/unused.hpp>

namespace GauXC::detail {

  template <typename ValueType>
  void IncoreReplicatedXCDeviceIntegrator<ValueType>::
    eval_dd_psi_( int64_t m, int64_t n, const value_type* P,
                  int64_t ldp, unsigned max_Ylm, value_type* ddPsi, int64_t ldPsi ) {
      GAUXC_GENERIC_EXCEPTION("Device DD-PSI NYI");
      util::unused(m,n,P,ldp,max_Ylm,ddPsi,ldPsi);
  }
  
  template <typename ValueType>
  void IncoreReplicatedXCDeviceIntegrator<ValueType>::
    eval_dd_psi_potential_( int64_t m, int64_t n, const value_type* X,
                   unsigned max_Ylm, value_type* Vddx ) {
      GAUXC_GENERIC_EXCEPTION("Device DD-PHIX NYI");
      util::unused(m,n,X,max_Ylm,Vddx);
  }

}
