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
  eval_exc_vxc_( int64_t m, int64_t n, 
                 const value_type* Ps, int64_t ldps,
                 const value_type* Pz, int64_t ldpz,
                 const value_type* Py, int64_t ldpy,
                 const value_type* Px, int64_t ldpx,
                 value_type* VXCs, int64_t ldvxcs,
                 value_type* VXCz, int64_t ldvxcz,
                 value_type* VXCy, int64_t ldvxcy,
                 value_type* VXCx, int64_t ldvxcx,
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

  if( ldvxcs < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDVXCS");
  if( ldvxcz and ldvxcz < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDVXCZ");
  if( ldvxcy and ldvxcy < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDVXCX");
  if( ldvxcx and ldvxcx < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDVXCY");


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
      VXCs, ldvxcs, VXCz, ldvxcz, VXCy, ldvxcy, VXCx, ldvxcx, EXC, 
      &N_EL, tasks.begin(), tasks.end(), incore_integrator );
  });

  // Release ownership of LWD back to this integrator instance
  this->local_work_driver_ = std::move( incore_integrator.release_local_work_driver() );


  // Reduce Results
  this->timer_.time_op("XCIntegrator.Allreduce", [&](){
    if( not this->reduction_driver_->takes_host_memory() )
      GAUXC_GENERIC_EXCEPTION("This Module Only Works With Host Reductions");

    this->reduction_driver_->allreduce_inplace( VXCs, nbf*nbf, ReductionOp::Sum );
    if(VXCz) this->reduction_driver_->allreduce_inplace( VXCz, nbf*nbf, ReductionOp::Sum );
    if(VXCy) this->reduction_driver_->allreduce_inplace( VXCy, nbf*nbf, ReductionOp::Sum ); 
    if(VXCx) this->reduction_driver_->allreduce_inplace( VXCx, nbf*nbf, ReductionOp::Sum );
    this->reduction_driver_->allreduce_inplace( EXC,   1    , ReductionOp::Sum );
    this->reduction_driver_->allreduce_inplace( &N_EL, 1    , ReductionOp::Sum );
  });

  #ifdef GAUXC_HAS_DEVICE
  device_data_ptr_.reset();
  #endif

}



template <typename BaseIntegratorType, typename IncoreIntegratorType>
void ShellBatchedReplicatedXCIntegrator<BaseIntegratorType, IncoreIntegratorType>::
  eval_exc_vxc_( int64_t m, int64_t n, 
                 const value_type* Ps, int64_t ldps,
                 const value_type* Pz, int64_t ldpz,
                 value_type* VXCs, int64_t ldvxcs,
                 value_type* VXCz, int64_t ldvxcz,
                 value_type* EXC, const IntegratorSettingsXC& ks_settings) {

  eval_exc_vxc_(m, n, Ps, ldps, Pz, ldpz, nullptr, 0, nullptr, 0,
    VXCs, ldvxcs, VXCz, ldvxcz, nullptr, 0, nullptr, 0,
    EXC, ks_settings);

}

template <typename BaseIntegratorType, typename IncoreIntegratorType>
void ShellBatchedReplicatedXCIntegrator<BaseIntegratorType, IncoreIntegratorType>::
  eval_exc_vxc_( int64_t m, int64_t n, 
                 const value_type* P, int64_t ldp,
                 value_type* VXC, int64_t ldvxc,
                 value_type* EXC, const IntegratorSettingsXC& ks_settings) {

  eval_exc_vxc_(m, n, P, ldp, nullptr, 0, nullptr, 0, nullptr, 0,
    VXC, ldvxc, nullptr, 0, nullptr, 0, nullptr, 0, EXC, ks_settings);

}

template <typename BaseIntegratorType, typename IncoreIntegratorType>
void ShellBatchedReplicatedXCIntegrator<BaseIntegratorType, IncoreIntegratorType>::
  exc_vxc_local_work_( const basis_type& basis, 
                       const value_type* Ps, int64_t ldps,
                       const value_type* Pz, int64_t ldpz,
                       const value_type* Py, int64_t ldpy,
                       const value_type* Px, int64_t ldpx,
                       value_type* VXCs, int64_t ldvxcs,
                       value_type* VXCz, int64_t ldvxcz,
                       value_type* VXCy, int64_t ldvxcy,
                       value_type* VXCx, int64_t ldvxcx,
                       value_type* EXC, value_type *N_EL, 
                       host_task_iterator task_begin, host_task_iterator task_end,
                       incore_integrator_type& incore_integrator ) {

  //incore_integrator.exc_vxc_local_work( basis, P, ldp, VXC, ldvxc, EXC, N_EL, task_begin, task_end, device_data );
  //return;


  const auto     nbf = basis.nbf();
  const uint32_t nbf_threshold = 8000;
  const auto&    mol = this->load_balancer_->molecule();
  // Zero out integrands on host
  this->timer_.time_op("XCIntegrator.ZeroHost", [&](){
    *EXC  = 0.;
    *N_EL = 0.;
    if(VXCs)
    for( auto j = 0; j < nbf; ++j )
    for( auto i = 0; i < nbf; ++i ) {
      VXCs[i + j*ldvxcs] = 0.;
    }
    if(VXCz)
    for( auto j = 0; j < nbf; ++j )
    for( auto i = 0; i < nbf; ++i ) {
      VXCz[i + j*ldvxcz] = 0.;
    }
    if(VXCy)
    for( auto j = 0; j < nbf; ++j )
    for( auto i = 0; i < nbf; ++i ) {
      VXCy[i + j*ldvxcy] = 0.;
    }
    if(VXCx)
    for( auto j = 0; j < nbf; ++j )
    for( auto i = 0; i < nbf; ++i ) {
      VXCx[i + j*ldvxcx] = 0.;
    }
  });


  // Task queue
  std::queue< incore_task_data > incore_task_data_queue;

  // Task queue modification mutex
  std::mutex queue_mod_ex;

  // Lambda for the execution of incore tasks on the device
  auto execute_incore_task = [&]() {

    // Early return if there is no task to execute
    if( incore_task_data_queue.empty() ) return;

    incore_task_data next_task;
    {
      std::lock_guard<std::mutex> lock(queue_mod_ex);

      // Move the next task into local scope and remove
      // from queue
      next_task = std::move( incore_task_data_queue.front() );
      incore_task_data_queue.pop();
    }

    // Execute task
    execute_task_batch( next_task, basis, mol, Ps, ldps, Pz, ldpz,
      Py, ldpy, Px, ldpx, VXCs, ldvxcs, VXCz, ldvxcz, VXCy, ldvxcy,
      VXCx, ldvxcx, EXC, N_EL, incore_integrator );
  };


  // Setup future to track execution of currently running
  // device task
  std::future<void> task_future;

  auto task_it = task_begin;
  while( task_it != task_end ) {

    // Generate and enqueue task
    incore_task_data_queue.emplace(
      generate_incore_task( nbf_threshold, basis, task_it, task_end )
    );

    // Update iterator for next task generation
    task_it = incore_task_data_queue.back().task_end;

    if( not task_future.valid() ) {
      // No device task to wait on
      task_future = std::async( std::launch::async, execute_incore_task );
    } else {
      // Check the status of current device task
      auto status = task_future.wait_for( std::chrono::milliseconds(5) );
      if( status == std::future_status::ready ) {
        // If the status is ready - execute the next task in queue
        task_future.get();
        task_future = std::async( std::launch::async, execute_incore_task ); 
      }
    }

  } // Loop until all tasks have been enqued 

  // TODO: Try to merge remaining tasks appropriately

  // Execute remaining tasks sequentially
  if( task_future.valid() ) {
    task_future.wait();
    task_future.get(); // Propagate trailing exceptions if present
  }
  while( not incore_task_data_queue.empty() ) {
    execute_incore_task();
  }
}



template <typename BaseIntegratorType, typename IncoreIntegratorType>
void ShellBatchedReplicatedXCIntegrator<BaseIntegratorType, IncoreIntegratorType>::
  execute_task_batch( incore_task_data& task, const basis_type& basis, const Molecule& mol, 
                      const value_type* Ps, int64_t ldps,
                      const value_type* Pz, int64_t ldpz,
                      const value_type* Py, int64_t ldpy,
                      const value_type* Px, int64_t ldpx,
                      value_type* VXCs, int64_t ldvxcs,
                      value_type* VXCz, int64_t ldvxcz,
                      value_type* VXCy, int64_t ldvxcy,
                      value_type* VXCx, int64_t ldvxcx,
                      value_type* EXC, value_type *N_EL, 
                      incore_integrator_type& incore_integrator ) {


  // Alias information
  auto task_begin  = task.task_begin;
  auto task_end    = task.task_end;
  auto& union_shell_list = task.shell_list;


  // Extract subbasis
  BasisSet<double> basis_subset; basis_subset.reserve(union_shell_list.size());
  this->timer_.time_op_accumulate("XCIntegrator.CopySubBasis",[&]() {
    for( auto i : union_shell_list ) {
      basis_subset.emplace_back( basis.at(i) );
    }
  });

  // Setup basis maps
  BasisSetMap basis_map( basis, mol );

  //const size_t nshells = basis_subset.nshells();
  const size_t nbe     = basis_subset.nbf();
  //std::cout << "TASK_UNION HAS:"   << std::endl
  //          << "  NSHELLS    = " <<  nshells << std::endl
  //          << "  NBE        = " <<  nbe     << std::endl;

  // Recalculate shell_list based on subbasis
  this->timer_.time_op_accumulate("XCIntegrator.RecalcShellList",[&]() {
    for( auto _it = task_begin; _it != task_end; ++_it ) {
      auto union_list_idx = 0;
      auto& cur_shell_list = _it->bfn_screening.shell_list;
      for( auto j = 0ul; j < cur_shell_list.size(); ++j ) {
        while( union_shell_list[union_list_idx] != cur_shell_list[j] )
          union_list_idx++;
        cur_shell_list[j] = union_list_idx;
      }
    }
  } );


  // Allocate host temporaries
  double EXC_tmp, NEL_tmp;
  std::vector<double> Ps_submat_host(nbe*nbe); 
  double* Ps_submat   = Ps_submat_host.data();
  std::vector<double> VXCs_submat_host(VXCs ? nbe*nbe : 0); 
  double* VXCs_submat = VXCs ? VXCs_submat_host.data() : nullptr;

  std::vector<double> Pz_submat_host, Py_submat_host, Px_submat_host;
  std::vector<double> VXCz_submat_host, VXCy_submat_host, VXCx_submat_host;
  double *Pz_submat = nullptr, *Py_submat = nullptr, *Px_submat = nullptr;
  double *VXCz_submat = nullptr, *VXCy_submat = nullptr , *VXCx_submat = nullptr;

  if(Pz) {
    Pz_submat_host.resize(nbe*nbe);
    Pz_submat = Pz_submat_host.data();
    if(VXCz) {
      VXCz_submat_host.resize(nbe*nbe, 0.0);
      VXCz_submat = VXCz_submat_host.data();
    }
  }

  if(Py) {
    Py_submat_host.resize(nbe*nbe);
    Py_submat = Py_submat_host.data();
    if(VXCy) {
      VXCy_submat_host.resize(nbe*nbe, 0.0);
      VXCy_submat = VXCy_submat_host.data();
    }
  }

  if(Px) {
    Px_submat_host.resize(nbe*nbe);
    Px_submat = Px_submat_host.data();
    if(VXCx) {
      VXCx_submat_host.resize(nbe*nbe, 0.0);
      VXCx_submat = VXCx_submat_host.data();
    }
  }


  // Extract subdensity
  std::vector<std::array<int32_t,3>> union_submat_cut;
  std::vector<int32_t> foo;
  std::tie(union_submat_cut,foo) = 
    gen_compressed_submat_map( basis_map, union_shell_list, 
      basis.nbf(), basis.nbf() );

  this->timer_.time_op_accumulate("XCIntegrator.ExtractSubDensity",[&]() {
    detail::submat_set( basis.nbf(), basis.nbf(), nbe, nbe, Ps, ldps, 
                        Ps_submat, nbe, union_submat_cut );
    if(Pz)
    detail::submat_set( basis.nbf(), basis.nbf(), nbe, nbe, Pz, ldpz, 
                        Pz_submat, nbe, union_submat_cut );

    if(Py)
    detail::submat_set( basis.nbf(), basis.nbf(), nbe, nbe, Py, ldpy, 
                        Py_submat, nbe, union_submat_cut );

    if(Px)
    detail::submat_set( basis.nbf(), basis.nbf(), nbe, nbe, Px, ldpx, 
                        Px_submat, nbe, union_submat_cut );
  } );


  // Process selected task batch
#ifdef GAUXC_HAS_DEVICE
  if constexpr (IncoreIntegratorType::is_device) {

    incore_integrator.exc_vxc_local_work( basis_subset, Ps_submat, nbe, 
      Pz_submat, nbe, Py_submat, nbe, Px_submat, nbe, VXCs_submat, nbe,
      VXCz_submat, nbe, VXCy_submat, nbe, VXCx_submat, nbe,
      &EXC_tmp, &NEL_tmp, task_begin, task_end, *device_data_ptr_ );
  } else if constexpr (not IncoreIntegratorType::is_device) {
#endif
    incore_integrator.exc_vxc_local_work( basis_subset, Ps_submat, nbe, 
      Pz_submat, nbe, Py_submat, nbe, Px_submat, nbe, VXCs_submat, nbe,
      VXCz_submat, nbe, VXCy_submat, nbe, VXCx_submat, nbe,
      &EXC_tmp, &NEL_tmp, IntegratorSettingsKS{}, task_begin, task_end );
#ifdef GAUXC_HAS_DEVICE
  }
#endif


  // Update full quantities
  *EXC += EXC_tmp;
  *N_EL += NEL_tmp;
  this->timer_.time_op_accumulate("XCIntegrator.IncrementSubPotential",[&]() {
    if(VXCs)
    detail::inc_by_submat( basis.nbf(), basis.nbf(), nbe, nbe, VXCs, ldvxcs, 
                           VXCs_submat, nbe, union_submat_cut );

    if(VXCz)
    detail::inc_by_submat( basis.nbf(), basis.nbf(), nbe, nbe, VXCz, ldvxcz, 
                           VXCz_submat, nbe, union_submat_cut );

    if(VXCy)
    detail::inc_by_submat( basis.nbf(), basis.nbf(), nbe, nbe, VXCy, ldvxcy, 
                           VXCy_submat, nbe, union_submat_cut );

    if(VXCx)
    detail::inc_by_submat( basis.nbf(), basis.nbf(), nbe, nbe, VXCx, ldvxcx, 
                           VXCx_submat, nbe, union_submat_cut );
  });


  // Reset shell_list to be wrt full basis
  this->timer_.time_op_accumulate("XCIntegrator.ResetShellList",[&]() {
    for( auto _it = task_begin; _it != task_end; ++_it ) 
    for( auto j = 0ul; j < _it->bfn_screening.shell_list.size();  ++j  ) {
      _it->bfn_screening.shell_list[j] = union_shell_list[_it->bfn_screening.shell_list[j]];
    }
  });

}

}
}

