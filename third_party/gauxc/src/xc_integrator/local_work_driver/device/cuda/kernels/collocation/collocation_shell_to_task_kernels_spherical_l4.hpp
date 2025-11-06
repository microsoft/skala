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
#include "device/common/shell_to_task.hpp"
#include <cassert>

namespace GauXC {


__global__ __launch_bounds__(512,2) void collocation_device_shell_to_task_kernel_spherical_4(
  uint32_t                        nshell,
  ShellToTaskDevice* __restrict__ shell_to_task,
  XCDeviceTask*      __restrict__ device_tasks
) {


  __shared__ double alpha[16][detail::shell_nprim_max + 1]; 
  __shared__ double coeff[16][detail::shell_nprim_max + 1];
  double* my_alpha = alpha[threadIdx.x/32];
  double* my_coeff = coeff[threadIdx.x/32];

  for( auto ish = blockIdx.z; ish < nshell; ish += gridDim.z ) {
  const uint32_t ntasks      = shell_to_task[ish].ntask;
  const auto shell           = shell_to_task[ish].shell_device;
  const auto task_idx        = shell_to_task[ish].task_idx_device;
  const auto task_shell_offs = shell_to_task[ish].task_shell_offs_device;


  // Load Shell Data into registers / SM
  const uint32_t nprim = shell->nprim();
  const double3 O  = *reinterpret_cast<const double3*>(shell->O_data());

  const int global_warp_id = (threadIdx.x + blockIdx.x*blockDim.x) / cuda::warp_size;
  const int nwarp_global   = max((blockDim.x*gridDim.x) / cuda::warp_size,1);

  // Read in coeffs/exps into SM on first warp
  {
    auto* coeff_gm = shell->coeff_data();
    auto* alpha_gm = shell->alpha_data();
    static_assert( detail::shell_nprim_max == cuda::warp_size );
    const int warp_rank = threadIdx.x % cuda::warp_size;
    my_alpha[warp_rank] = alpha_gm[warp_rank];
    my_coeff[warp_rank] = coeff_gm[warp_rank];
  }

  // Loop over tasks assigned to shells
  // Place each task on a different warp + schedule across blocks
  for( int itask = global_warp_id; itask < ntasks; itask += nwarp_global ) {

    const auto*              task   = device_tasks + task_idx[itask];
    const auto* __restrict__ points_x = task->points_x;
    const auto* __restrict__ points_y = task->points_y;
    const auto* __restrict__ points_z = task->points_z;
    const uint32_t           npts   = task->npts;
    const size_t             shoff  = task_shell_offs[itask] * npts;

    auto* __restrict__ basis_eval = task->bf + shoff;

    // Loop over points in task
    // Assign each point to separate thread within the warp
    #pragma unroll 1
    for( int ipt = threadIdx.x % cuda::warp_size; ipt < npts; ipt += cuda::warp_size ) {
      //const double3 point = points[ipt];
      double3 point;
      point.x = points_x[ipt];
      point.y = points_y[ipt];
      point.z = points_z[ipt];


      const auto x = point.x - O.x;
      const auto y = point.y - O.y;
      const auto z = point.z - O.z;
      const auto rsq = x*x + y*y + z*z;

      // Evaluate radial part of bfn
      double radial_eval = 0.;

      #pragma unroll 1
      for( uint32_t i = 0; i < nprim; ++i ) {
        const auto a = my_alpha[i];
        const auto e = my_coeff[i] * std::exp( - a * rsq );

        radial_eval += e;
      }


      // Common Subexpressions
      const auto x0 = 0.5*radial_eval*x*y; 
      const auto x1 = x*x; 
      const auto x2 = y*y; 
      const auto x3 = -x2; 
      const auto x4 = 0.25*radial_eval; 
      const auto x5 = x4*z; 
      const auto x6 = x5*y; 
      const auto x7 = 3.0*x1; 
      const auto x8 = z*z; 
      const auto x9 = 3.0*x2; 
      const auto x10 = -x7 + 4.0*x8 - x9; 
      const auto x11 = 0.125*radial_eval; 
      const auto x12 = x*x*x*x; 
      const auto x13 = y*y*y*y; 
      const auto x14 = 6.0*x1*x2; 
      const auto x15 = x1*x8; 
      const auto x16 = x2*x8; 
      const auto x17 = x*x5; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = sqrt_35*x0*(x1 + x3);
      basis_eval[ipt + 1*npts] = sqrt_70*x6*(x3 + x7);
      basis_eval[ipt + 2*npts] = -sqrt_5*x0*(x1 + x2 - 6.0*x8);
      basis_eval[ipt + 3*npts] = sqrt_10*x10*x6;
      basis_eval[ipt + 4*npts] = x11*(3.0*x12 + 3.0*x13 + x14 - 24.0*x15 - 24.0*x16 + 8.0*(z*z*z*z));
      basis_eval[ipt + 5*npts] = sqrt_10*x10*x17;
      basis_eval[ipt + 6*npts] = -sqrt_5*x4*(x12 - x13 - 6.0*x15 + 6.0*x16);
      basis_eval[ipt + 7*npts] = sqrt_70*x17*(x1 - x9);
      basis_eval[ipt + 8*npts] = sqrt_35*x11*(x12 + x13 - x14);


    







#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = sqrt_35*x0*(x1 + x3);
      ang_eval_1 = sqrt_70*x6*(x3 + x7);
      ang_eval_2 = -sqrt_5*x0*(x1 + x2 - 6.0*x8);
      ang_eval_3 = sqrt_10*x10*x6;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = x11*(3.0*x12 + 3.0*x13 + x14 - 24.0*x15 - 24.0*x16 + 8.0*(z*z*z*z));
      ang_eval_1 = sqrt_10*x10*x17;
      ang_eval_2 = -sqrt_5*x4*(x12 - x13 - 6.0*x15 + 6.0*x16);
      ang_eval_3 = sqrt_70*x17*(x1 - x9);
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = sqrt_35*x11*(x12 + x13 - x14);
      basis_eval[ipt + 8*npts] = ang_eval_0;


#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
