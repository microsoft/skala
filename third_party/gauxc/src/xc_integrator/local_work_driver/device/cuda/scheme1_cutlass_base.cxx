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
#include <stdexcept>

#include "scheme1_cutlass_base.hpp"
#include "device/common/pack_submat.hpp"
#include "device/common/inc_potential.hpp"
#include "device/common/device_blas.hpp"

#include "device/cuda/kernels/cutlass_wrapper.hpp"

namespace GauXC {

// Common implementation for eval_xmat and eval_xmat_trial
template<bool is_trial>
void AoSScheme1CUTLASSBase::eval_xmat_impl(double fac, XCDeviceData* _data, bool do_grad, density_id den_id) {
  auto* data = dynamic_cast<AoSScheme1CUTLASSBase::Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();

  // Pack density matrix 
  const auto nbf = data->global_dims.nbf;
  const auto submat_block_size = data->get_submat_chunk_size( nbf, 0 );
  auto static_stack  = data->static_stack;
  auto aos_stack     = data->aos_stack;
  
  double* dmat_ptr;
  if constexpr (is_trial) {
    dmat_ptr = static_stack.tden_selector(den_id);
    // now screened trial density matrix is stored in aos_stack.device_tasks[itask].nbe_scr
  } else {
    dmat_ptr = static_stack.den_selector(den_id);
  }
  
  sym_pack_submat( ntasks, aos_stack.device_tasks, dmat_ptr, 
    nbf, submat_block_size, data->device_backend_->queue() );

  auto cutlass_stack = data->cutlass_stack;
  double** dmat_array;
  if constexpr (is_trial) {
    dmat_array = cutlass_stack.tdmat_array(den_id);
  } else {
    dmat_array = cutlass_stack.dmat_array(den_id);
  }
  cutlass_gemm(
    cutlass_stack.problem_sizes_device,
    data->problem_sizes_host.data(),
    ntasks,
    cutlass_stack.bf_array_device, dmat_array,
    cutlass_stack.zmat_array_device, cutlass_stack.zmat_array_device,
    cutlass_stack.ld64_bf_array_device, cutlass_stack.ld64_dmat_array_device,
    cutlass_stack.ld64_zmat_array_device, cutlass_stack.ld64_zmat_array_device,
    fac, 0.0,
    data->device_backend_->queue()
  );

  if(do_grad) {
    cutlass_gemm(
      cutlass_stack.problem_sizes_device,
      data->problem_sizes_host.data(),
      ntasks,
      cutlass_stack.bfx_array_device, dmat_array,
      cutlass_stack.xmat_x_array_device, cutlass_stack.xmat_x_array_device,
      cutlass_stack.ld64_bf_array_device, cutlass_stack.ld64_dmat_array_device,
      cutlass_stack.ld64_zmat_array_device, cutlass_stack.ld64_zmat_array_device,
      fac, 0.0,
      data->device_backend_->queue()
    );
    cutlass_gemm(
      cutlass_stack.problem_sizes_device,
      data->problem_sizes_host.data(),
      ntasks,
      cutlass_stack.bfy_array_device, dmat_array,
      cutlass_stack.xmat_y_array_device, cutlass_stack.xmat_y_array_device,
      cutlass_stack.ld64_bf_array_device, cutlass_stack.ld64_dmat_array_device,
      cutlass_stack.ld64_zmat_array_device, cutlass_stack.ld64_zmat_array_device,
      fac, 0.0,
      data->device_backend_->queue()
    );
    cutlass_gemm(
      cutlass_stack.problem_sizes_device,
      data->problem_sizes_host.data(),
      ntasks,
      cutlass_stack.bfz_array_device, dmat_array,
      cutlass_stack.xmat_z_array_device, cutlass_stack.xmat_z_array_device,
      cutlass_stack.ld64_bf_array_device, cutlass_stack.ld64_dmat_array_device,
      cutlass_stack.ld64_zmat_array_device, cutlass_stack.ld64_zmat_array_device,
      fac, 0.0,
      data->device_backend_->queue()
    );
  }
}

void AoSScheme1CUTLASSBase::eval_xmat(double fac, XCDeviceData* _data, bool do_grad, density_id den_id ) {
  eval_xmat_impl<false>(fac, _data, do_grad, den_id);
}

void AoSScheme1CUTLASSBase::eval_xmat_trial(double fac, XCDeviceData* _data, bool do_grad, density_id den_id ) {
  eval_xmat_impl<true>(fac, _data, do_grad, den_id);
}


// Common implementation for inc_vxc and inc_fxc
template<bool is_fxc>
void AoSScheme1CUTLASSBase::inc_potential_impl(XCDeviceData* _data, density_id den_id, bool do_m) {
  auto* data = dynamic_cast<AoSScheme1CUTLASSBase::Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();

  auto cutlass_stack = data->cutlass_stack;
  cutlass_syr2k(
    cutlass_stack.syr2k_sizes_device,
    data->syr2k_sizes_host.data(),
    ntasks,
    cutlass_stack.bf_array_device, cutlass_stack.zmat_array_device,
    cutlass_stack.vmat_array_device, cutlass_stack.vmat_array_device,
    cutlass_stack.ld64_bf_array_device, cutlass_stack.ld64_zmat_array_device,
    cutlass_stack.ld64_vmat_array_device, cutlass_stack.ld64_vmat_array_device,
    1.0, 0.0,
    data->device_backend_->queue()
  );
  if(do_m) {
    cutlass_syr2k(
      cutlass_stack.syr2k_sizes_device,
      data->syr2k_sizes_host.data(),
      ntasks,
      cutlass_stack.bfx_array_device, cutlass_stack.xmat_x_array_device,
      cutlass_stack.vmat_array_device, cutlass_stack.vmat_array_device,
      cutlass_stack.ld64_bf_array_device, cutlass_stack.ld64_zmat_array_device,
      cutlass_stack.ld64_vmat_array_device, cutlass_stack.ld64_vmat_array_device,
      1.0, 1.0,
      data->device_backend_->queue()
    );
    cutlass_syr2k(
      cutlass_stack.syr2k_sizes_device,
      data->syr2k_sizes_host.data(),
      ntasks,
      cutlass_stack.bfy_array_device, cutlass_stack.xmat_y_array_device,
      cutlass_stack.vmat_array_device, cutlass_stack.vmat_array_device,
      cutlass_stack.ld64_bf_array_device, cutlass_stack.ld64_zmat_array_device,
      cutlass_stack.ld64_vmat_array_device, cutlass_stack.ld64_vmat_array_device,
      1.0, 1.0,
      data->device_backend_->queue()
    );
    cutlass_syr2k(
      cutlass_stack.syr2k_sizes_device,
      data->syr2k_sizes_host.data(),
      ntasks,
      cutlass_stack.bfz_array_device, cutlass_stack.xmat_z_array_device,
      cutlass_stack.vmat_array_device, cutlass_stack.vmat_array_device,
      cutlass_stack.ld64_bf_array_device, cutlass_stack.ld64_zmat_array_device,
      cutlass_stack.ld64_vmat_array_device, cutlass_stack.ld64_vmat_array_device,
      1.0, 1.0,
      data->device_backend_->queue()
    );
  }

  // Increment global VXC/FXC
  const auto nbf = data->global_dims.nbf;
  const auto submat_block_size = data->get_submat_chunk_size( nbf, 0 );
  auto static_stack  = data->static_stack;
  auto aos_stack     = data->aos_stack;
  
  double* potential_ptr;
  if constexpr (is_fxc) {
    potential_ptr = static_stack.fxc_selector(den_id);
    // cutlass_stack.vmat_array_device points to aos_stack.device_tasks[itask].nbe_scr
  } else {
    potential_ptr = static_stack.vxc_selector(den_id);
  }
  
  sym_task_inc_potential( ntasks, aos_stack.device_tasks, potential_ptr, nbf, 
    submat_block_size, data->device_backend_->queue() );
}

void AoSScheme1CUTLASSBase::inc_vxc( XCDeviceData* _data, density_id den_id, bool do_m ) {
  inc_potential_impl<false>(_data, den_id, do_m);
}

void AoSScheme1CUTLASSBase::inc_fxc( XCDeviceData* _data, density_id den_id, bool do_m ) {
  inc_potential_impl<true>(_data, den_id, do_m);
}

}
