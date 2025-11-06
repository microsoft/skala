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
#include "device/common/uvvars.hpp"
#include "hip_extensions.hpp"
#include "device_specific/hip_device_constants.hpp"
#include <gauxc/util/div_ceil.hpp>
#include "device_specific/hip_util.hpp"

namespace GauXC {


__global__ void eval_uvars_lda_kernel( size_t           ntasks,
                                       XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];

  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;

  auto* den_eval_device   = task.den;

  const auto* basis_eval_device = task.bf;

  const auto* den_basis_prod_device = task.zmat;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  double den_reg = 0.;

  if( tid_x < nbf and tid_y < npts ) {

    const double* bf_col   = basis_eval_device     + tid_x*npts;
    const double* db_col   = den_basis_prod_device + tid_x*npts;

    den_reg = bf_col[ tid_y ]   * db_col[ tid_y ];

  }

  // Warp blocks are stored col major
  den_reg = 2 * hip::warp_reduce_sum<hip::warp_size>( den_reg );


  if( threadIdx.x == 0 and tid_y < npts ) {
    atomicAdd( den_eval_device   + tid_y, den_reg );
  }


}


void eval_uvvars_lda( size_t ntasks, int32_t nbf_max, int32_t npts_max,
  XCDeviceTask* device_tasks, device_queue queue ) {

  hipStream_t stream = queue.queue_as<util::hip_stream>();

  dim3 threads(hip::warp_size, hip::max_warps_per_thread_block, 1);
  dim3 blocks( util::div_ceil( nbf_max , threads.x ),
               util::div_ceil( npts_max , threads.y ),
               ntasks );

  hipLaunchKernelGGL(eval_uvars_lda_kernel, dim3(blocks), dim3(threads), 0, 
    stream, ntasks, device_tasks );

}













__global__ void eval_uvars_gga_kernel( size_t           ntasks,
                                       XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];

  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;

  auto* den_eval_device   = task.den;
  auto* den_x_eval_device = task.ddenx;
  auto* den_y_eval_device = task.ddeny;
  auto* den_z_eval_device = task.ddenz;

  const auto* basis_eval_device = task.bf;
  const auto* dbasis_x_eval_device = task.dbfx;
  const auto* dbasis_y_eval_device = task.dbfy;
  const auto* dbasis_z_eval_device = task.dbfz;

  const auto* den_basis_prod_device = task.zmat;

  // We always launch enough blocks to cover npts, so blocks aren't doing multiple results
  double den_reg = 0.;
  double dx_reg = 0.;
  double dy_reg = 0.;
  double dz_reg = 0.;

  // Have each thread accumulate its own reduction result into a register.
  // There's no real _need_ for LDS because the reductions are small and
  // therefore can be done without sharing.
  for( int ibf = 0; ibf < nbf; ibf++ ) {

    for( int  ipt = blockIdx.x * blockDim.x + threadIdx.x; ipt < npts; ipt += blockDim.x * gridDim.x ) {

      const double* bf_col   = basis_eval_device     + ibf*npts;
      const double* bf_x_col = dbasis_x_eval_device  + ibf*npts;
      const double* bf_y_col = dbasis_y_eval_device  + ibf*npts;
      const double* bf_z_col = dbasis_z_eval_device  + ibf*npts;
      const double* db_col   = den_basis_prod_device + ibf*npts;

      den_reg += 2 * bf_col[ ipt ]   * db_col[ ipt ];
      dx_reg += 4 * bf_x_col[ ipt ] * db_col[ ipt ];
      dy_reg += 4 * bf_y_col[ ipt ] * db_col[ ipt ];
      dz_reg += 4 * bf_z_col[ ipt ] * db_col[ ipt ];
    }
  }


  for( int  ipt = blockIdx.x * blockDim.x + threadIdx.x; ipt < npts; ipt += blockDim.x * gridDim.x ) {
    den_eval_device   [ipt] = den_reg;
    den_x_eval_device [ipt] = dx_reg ;
    den_y_eval_device [ipt] = dy_reg ;
    den_z_eval_device [ipt] = dz_reg ;
  }

}


__global__ void eval_vvars_gga_kernel( 
  size_t        npts,
  const double* den_x_eval_device,
  const double* den_y_eval_device,
  const double* den_z_eval_device,
        double* gamma_eval_device
) {

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if( tid < npts ) {

    const double dx = den_x_eval_device[ tid ];
    const double dy = den_y_eval_device[ tid ];
    const double dz = den_z_eval_device[ tid ];

    gamma_eval_device[tid] = dx*dx + dy*dy + dz*dz;

  }

}




void eval_uvvars_gga( size_t ntasks, size_t npts_total, int32_t nbf_max, 
  int32_t npts_max, XCDeviceTask* device_tasks, const double* denx, 
  const double* deny, const double* denz, double* gamma, device_queue queue ) {

  hipStream_t stream = queue.queue_as<util::hip_stream>();

  // U Variables
  {
  dim3 threads(hip::max_threads_per_thread_block, 1, 1);
  dim3 blocks( util::div_ceil( npts_max , threads.x ),
               1,
               ntasks );

  hipLaunchKernelGGL(eval_uvars_gga_kernel, dim3(blocks), dim3(threads), 0, 
    stream, ntasks, device_tasks );
  }

  // V Variables
  dim3 threads( hip::max_threads_per_thread_block );
  dim3 blocks( util::div_ceil( npts_total, threads.x ) );
  hipLaunchKernelGGL(eval_vvars_gga_kernel, blocks, threads, 0, stream,
    npts_total, denx, deny, denz, gamma);

}









}
