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


__global__ __launch_bounds__(256,2) void collocation_device_shell_to_task_kernel_cartesian_gradient_2(
  uint32_t                        nshell,
  ShellToTaskDevice* __restrict__ shell_to_task,
  XCDeviceTask*      __restrict__ device_tasks
) {


  __shared__ double alpha[8][detail::shell_nprim_max + 1]; 
  __shared__ double coeff[8][detail::shell_nprim_max + 1];
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
    auto* __restrict__ basis_x_eval = task->dbfx + shoff;
    auto* __restrict__ basis_y_eval = task->dbfy + shoff;
    auto* __restrict__ basis_z_eval = task->dbfz + shoff;

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
      double radial_eval_alpha = 0.;

      #pragma unroll 1
      for( uint32_t i = 0; i < nprim; ++i ) {
        const auto a = my_alpha[i];
        const auto e = my_coeff[i] * std::exp( - a * rsq );

        radial_eval += e;
        radial_eval_alpha += a * e;
      }

      radial_eval_alpha *= -2;

      // Common Subexpressions
      const auto x0 = x*x; 
      const auto x1 = radial_eval*x; 
      const auto x2 = y*y; 
      const auto x3 = radial_eval*y; 
      const auto x4 = z*z; 
      const auto x5 = radial_eval + radial_eval_alpha*x0; 
      const auto x6 = radial_eval_alpha*x; 
      const auto x7 = x6*y*z; 
      const auto x8 = radial_eval_alpha*y; 
      const auto x9 = radial_eval + radial_eval_alpha*x2; 
      const auto x10 = radial_eval_alpha*z; 
      const auto x11 = radial_eval + radial_eval_alpha*x4; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x0;
      basis_eval[ipt + 1*npts] = x1*y;
      basis_eval[ipt + 2*npts] = x1*z;
      basis_eval[ipt + 3*npts] = radial_eval*x2;
      basis_eval[ipt + 4*npts] = x3*z;
      basis_eval[ipt + 5*npts] = radial_eval*x4;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = radial_eval_alpha*(x*x*x) + 2.0*x1;
      basis_x_eval[ipt + 1*npts] = x5*y;
      basis_x_eval[ipt + 2*npts] = x5*z;
      basis_x_eval[ipt + 3*npts] = x2*x6;
      basis_x_eval[ipt + 4*npts] = x7;
      basis_x_eval[ipt + 5*npts] = x4*x6;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*x8;
      basis_y_eval[ipt + 1*npts] = x*x9;
      basis_y_eval[ipt + 2*npts] = x7;
      basis_y_eval[ipt + 3*npts] = radial_eval_alpha*(y*y*y) + 2.0*x3;
      basis_y_eval[ipt + 4*npts] = x9*z;
      basis_y_eval[ipt + 5*npts] = x4*x8;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x0*x10;
      basis_z_eval[ipt + 1*npts] = x7;
      basis_z_eval[ipt + 2*npts] = x*x11;
      basis_z_eval[ipt + 3*npts] = x10*x2;
      basis_z_eval[ipt + 4*npts] = x11*y;
      basis_z_eval[ipt + 5*npts] = z*(2.0*radial_eval + radial_eval_alpha*(z*z));







#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x0;
      ang_eval_1 = x1*y;
      ang_eval_2 = x1*z;
      ang_eval_3 = radial_eval*x2;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = x3*z;
      ang_eval_1 = radial_eval*x4;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = radial_eval_alpha*(x*x*x) + 2.0*x1;
      dang_eval_y_0 = x0*x8;
      dang_eval_z_0 = x0*x10;
      dang_eval_x_1 = x5*y;
      dang_eval_y_1 = x*x9;
      dang_eval_z_1 = x7;
      dang_eval_x_2 = x5*z;
      dang_eval_y_2 = x7;
      dang_eval_z_2 = x*x11;
      dang_eval_x_3 = x2*x6;
      dang_eval_y_3 = radial_eval_alpha*(y*y*y) + 2.0*x3;
      dang_eval_z_3 = x10*x2;
      basis_x_eval[ipt + 0*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 0*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 0*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 1*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 1*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 1*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 2*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 2*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 2*npts] = dang_eval_z_2;
      basis_x_eval[ipt + 3*npts] = dang_eval_x_3;
      basis_y_eval[ipt + 3*npts] = dang_eval_y_3;
      basis_z_eval[ipt + 3*npts] = dang_eval_z_3;

      dang_eval_x_0 = x7;
      dang_eval_y_0 = x9*z;
      dang_eval_z_0 = x11*y;
      dang_eval_x_1 = x4*x6;
      dang_eval_y_1 = x4*x8;
      dang_eval_z_1 = z*(2.0*radial_eval + radial_eval_alpha*(z*z));
      basis_x_eval[ipt + 4*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 4*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 4*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 5*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 5*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 5*npts] = dang_eval_z_1;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
