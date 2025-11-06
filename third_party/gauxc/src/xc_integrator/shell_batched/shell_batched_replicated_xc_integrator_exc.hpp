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
#ifdef GAUXC_HAS_DEVICE
#include "device/local_device_work_driver.hpp"
#include "device/xc_device_aos_data.hpp"
#endif
#include "integrator_util/integrator_common.hpp"
#include "host/util.hpp"
#include <gauxc/util/misc.hpp>
#include <gauxc/util/unused.hpp>

#include <stdexcept>
#include <fstream>
#include <queue>
#include <mutex>
#include <future>
#include <set>

namespace GauXC  {
namespace detail {


template <typename BaseIntegratorType, typename IncoreIntegratorType>
void ShellBatchedReplicatedXCIntegrator<BaseIntegratorType, IncoreIntegratorType>::
  eval_exc_( int64_t m, int64_t n, 
             const value_type* Ps, int64_t ldps,
             const value_type* Pz, int64_t ldpz,
             const value_type* Py, int64_t ldpy,
             const value_type* Px, int64_t ldpx,
             value_type* EXC, const IntegratorSettingsXC& /*ks_settings*/) {


  const auto& basis = this->load_balancer_->basis();

  // Check that P / VXC are sane
  const int64_t nbf = basis.nbf();
  if( m != n )
    GAUXC_GENERIC_EXCEPTION("P/VXC Must Be Square");
  if( m != nbf )
    GAUXC_GENERIC_EXCEPTION("P/VXC Must Have Same Dimension as Basis");

  if( ldps < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPS");
  if( ldpz and ldpz < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPZ");
  if( ldpy and ldpy < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPX");
  if( ldpx and ldpx < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPY");

  // Get Tasks
  auto& tasks = this->load_balancer_->get_tasks();

  #ifdef GAUXC_HAS_DEVICE
  // Allocate Device memory
  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get() );
  auto rt  = detail::as_device_runtime(this->load_balancer_->runtime());
  if constexpr (IncoreIntegratorType::is_device) {
    device_data_ptr_ = 
      this->timer_.time_op("XCIntegrator.DeviceAlloc",
        [&](){ return lwd->create_device_data(rt); });
  }
  #endif

  // Generate incore integrator instance, transfer ownership of LWD
  incore_integrator_type incore_integrator( this->func_, this->load_balancer_,
    this->release_local_work_driver(), this->reduction_driver_ );

  // Temporary electron count to judge integrator accuracy
  value_type N_EL;

  // Compute local contributions to EXC/VXC
  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
    exc_vxc_local_work_( basis, Ps, ldps, Pz, ldpz, Py, ldpy, Px, ldpx,
      nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, EXC, 
      &N_EL, tasks.begin(), tasks.end(), incore_integrator );
  });

  // Release ownership of LWD back to this integrator instance
  this->local_work_driver_ = std::move( incore_integrator.release_local_work_driver() );


  // Reduce Results
  this->timer_.time_op("XCIntegrator.Allreduce", [&](){
    if( not this->reduction_driver_->takes_host_memory() )
      GAUXC_GENERIC_EXCEPTION("This Module Only Works With Host Reductions");

    this->reduction_driver_->allreduce_inplace( EXC,   1    , ReductionOp::Sum );
    this->reduction_driver_->allreduce_inplace( &N_EL, 1    , ReductionOp::Sum );
  });

  #ifdef GAUXC_HAS_DEVICE
  device_data_ptr_.reset();
  #endif

}



template <typename BaseIntegratorType, typename IncoreIntegratorType>
void ShellBatchedReplicatedXCIntegrator<BaseIntegratorType, IncoreIntegratorType>::
  eval_exc_( int64_t m, int64_t n, 
             const value_type* Ps, int64_t ldps,
             const value_type* Pz, int64_t ldpz,
             value_type* EXC, const IntegratorSettingsXC& ks_settings) {

  eval_exc_(m, n, Ps, ldps, Pz, ldpz, nullptr, 0, nullptr, 0,
    EXC, ks_settings);

}

template <typename BaseIntegratorType, typename IncoreIntegratorType>
void ShellBatchedReplicatedXCIntegrator<BaseIntegratorType, IncoreIntegratorType>::
  eval_exc_( int64_t m, int64_t n, 
             const value_type* P, int64_t ldp,
             value_type* EXC, const IntegratorSettingsXC& ks_settings) {

  eval_exc_(m, n, P, ldp, nullptr, 0, nullptr, 0, nullptr, 0,
    EXC, ks_settings);

}


}
}

