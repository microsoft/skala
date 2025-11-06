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
  eval_fxc_contraction_( int64_t m, int64_t n, 
                        const value_type* P, int64_t ldp,  
                        const value_type* tP, int64_t ldtp,
                        value_type* FXC, int64_t ldfxc,
                        const IntegratorSettingsXC& ks_settings ) {
  GAUXC_GENERIC_EXCEPTION("ShellBatched FXC contraction NYI");            
  util::unused(m,n,P,ldp,tP,ldtp,FXC,ldfxc,ks_settings);

}

template <typename BaseIntegratorType, typename IncoreIntegratorType>
void ShellBatchedReplicatedXCIntegrator<BaseIntegratorType, IncoreIntegratorType>::
  eval_fxc_contraction_( int64_t m, int64_t n, 
                        const value_type* Ps, int64_t ldps,
                        const value_type* Pz, int64_t ldpz,
                        const value_type* tPs, int64_t ldtps,
                        const value_type* tPz, int64_t ldtpz,
                        value_type* FXCs, int64_t ldfxcs,
                        value_type* FXCz, int64_t ldfxcz,
                        const IntegratorSettingsXC& ks_settings ) {
  GAUXC_GENERIC_EXCEPTION("ShellBatched FXC contraction NYI");            
  util::unused(m,n,Ps,ldps,Pz,ldpz,tPs,ldtps,tPz,ldtpz,
                 FXCs,ldfxcs,FXCz,ldfxcz);

}


}
}
