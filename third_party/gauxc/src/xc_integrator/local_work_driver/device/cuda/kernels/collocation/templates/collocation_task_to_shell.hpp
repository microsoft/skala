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
#include "collocation_device_constants.hpp"
#include "device/xc_device_task.hpp"
#include "device_specific/cuda_device_constants.hpp"
#include <cassert>

namespace GauXC {

$py(do_grad = 'gradient' in type)\

__global__ __launch_bounds__(512,2) void collocation_device_task_to_shell_kernel_$(type)_$(L)(
  uint32_t                        ntask,
  XCDeviceTask*      __restrict__ device_tasks
  const Shell<double>*            shells_device,
) {


  // Storage for shell data
  __shared__ double  alpha[detail::shell_nprim_max], coeff[detail::shell_nprim_max];

  // Storage for points
  __shared__ double points_x[ cuda::warp_size ];
  __shared__ double points_y[ cuda::warp_size ];
  __shared__ double points_z[ cuda::warp_size ];

  for( int itask = blockIdx.z; itask < ntask; itask += gridDim.z ) {

    auto* task_ptr = device_tasks + itask;

    const uint32_t nshells = task_ptr->nshells;
    const uint32_t npts    = task_ptr->npts;

    const auto* __restrict__ pts_x_device  = task.points_x;
    const auto* __restrict__ pts_y_device  = task.points_y;
    const auto* __restrict__ pts_z_device  = task.points_z;
    const auto* __restrict__ mask_device = task.shell_list;
    const auto* __restrict__ offs_device = task.shell_offs;

    auto* __restrict__ eval_device    = task.bf;
    auto* __restrict__ deval_device_x = task.dbfx;
    auto* __restrict__ deval_device_y = task.dbfy;
    auto* __restrict__ deval_device_z = task.dbfz;

    // Loop over batches of points in task
    for( uint32_t ipt_st = 0; ipt_st < npts; ipt_st += cuda::warp_size ) {

      // Load a batch of points into shared memory 
      {
      uint32_t nleft = min( npts-ipt_st, cuda::warp_size );
      uint32_t idx = ipt_st + (threadId.x % nleft); // Wrap around to avoid warp divergence
      points_x[threadIdx.x] = pts_x_devlce[ idx ]; 
      points_y[threadIdx.x] = pts_y_devlce[ idx ]; 
      points_z[threadIdx.x] = pts_z_devlce[ idx ]; 
      }


    } // Loop over batches of points

  } // Loop over tasks

} // end kernel

} // namespace GauXC
