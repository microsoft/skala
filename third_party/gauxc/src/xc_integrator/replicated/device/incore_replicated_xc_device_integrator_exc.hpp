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
#include "incore_replicated_xc_device_integrator.hpp"
#include "device/local_device_work_driver.hpp"
#include "device/xc_device_aos_data.hpp"
#include <fstream>
#include <gauxc/exceptions.hpp>
#include <gauxc/util/unused.hpp>

namespace GauXC  {
namespace detail {

template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  eval_exc_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps, 
             const value_type* Pz, int64_t ldpz,
             const value_type* Py, int64_t ldpy,
             const value_type* Px, int64_t ldpx,
             value_type* EXC, const IntegratorSettingsXC& settings ) {


  const auto& basis = this->load_balancer_->basis();

  // Check that P / VXC are sane
  const int64_t nbf = basis.nbf();
  if( m != n ) 
    GAUXC_GENERIC_EXCEPTION("P/VXC Must Be Square");
  if( m != nbf ) 
    GAUXC_GENERIC_EXCEPTION("P/VXC Must Have Same Dimension as Basis");
  if( ldps < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDP");


  // Get Tasks
  auto& tasks = this->load_balancer_->get_tasks();

  // Allocate Device memory
  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get() );
  auto rt  = detail::as_device_runtime(this->load_balancer_->runtime());
  auto device_data_ptr = lwd->create_device_data(rt);

  GAUXC_MPI_CODE( MPI_Barrier(rt.comm());) 

  // Temporary electron count to judge integrator accuracy
  value_type N_EL;

  // Compute local contributions to EXC/VXC and retrieve
  // data from device 
  this->timer_.time_op("XCIntegrator.LocalWork_EXC", [&](){
    exc_vxc_local_work_( basis, Ps, ldps, Pz, ldpz, Py, ldpy, Px, ldpx,
        // Passing nullptr for VXCs disables VXC entirely
        nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, EXC, &N_EL,
       tasks.begin(), tasks.end(), *device_data_ptr);
  });

  GAUXC_MPI_CODE(
  this->timer_.time_op("XCIntegrator.ImbalanceWait_EXC",[&](){
    MPI_Barrier(this->load_balancer_->runtime().comm());
  });  
  )

  // Reduce Results in host mem
  this->timer_.time_op("XCIntegrator.Allreduce_EXC", [&](){
    this->reduction_driver_->allreduce_inplace( EXC,   1    , ReductionOp::Sum );
    this->reduction_driver_->allreduce_inplace( &N_EL, 1    , ReductionOp::Sum );
  });

}



template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  eval_exc_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps, 
             const value_type* Pz, int64_t ldpz,
             value_type* EXC, const IntegratorSettingsXC& settings ) {

  eval_exc_(m, n, Ps, ldps, Pz, ldpz, nullptr, 0, nullptr, 0, EXC, settings);

}

template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  eval_exc_( int64_t m, int64_t n, const value_type* P, int64_t ldp, 
             value_type* EXC, const IntegratorSettingsXC& settings ) {

  eval_exc_(m, n, P, ldp, nullptr, 0, nullptr, 0, nullptr, 0, EXC, settings);

}


}
}

