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
#include "shell_batched_replicated_xc_integrator.hpp"
#include <gauxc/util/misc.hpp>
#include <gauxc/util/unused.hpp>

namespace GauXC  {
namespace detail {

template <typename BaseIntegratorType, typename IncoreIntegratorType>
void ShellBatchedReplicatedXCIntegrator<BaseIntegratorType, IncoreIntegratorType>::
  eval_exc_grad_( int64_t m, int64_t n, const value_type* P, int64_t ldp, value_type* EXC_GRAD, const IntegratorSettingsXC& settings ) { 
                 
  GAUXC_GENERIC_EXCEPTION("ShellBatched exc_grad NYI" );                 
  util::unused(m,n,P,ldp,EXC_GRAD);
}

template <typename BaseIntegratorType, typename IncoreIntegratorType>
void ShellBatchedReplicatedXCIntegrator<BaseIntegratorType, IncoreIntegratorType>::
  eval_exc_grad_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps, 
                  const value_type* Pz, int64_t lpdz, value_type* EXC_GRAD, const IntegratorSettingsXC& settings ) { 
                 
  GAUXC_GENERIC_EXCEPTION("ShellBatched exc_grad NYI" );                 
  util::unused(m,n,Ps,ldps,Pz,lpdz,EXC_GRAD);
}

}
}
