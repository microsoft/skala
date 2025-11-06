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


__global__ __launch_bounds__(256,2) void collocation_device_shell_to_task_kernel_cartesian_gradient_3(
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
      const auto x0 = x*x*x; 
      const auto x1 = radial_eval*y; 
      const auto x2 = x*x; 
      const auto x3 = radial_eval*z; 
      const auto x4 = radial_eval*x; 
      const auto x5 = y*y; 
      const auto x6 = x*z; 
      const auto x7 = z*z; 
      const auto x8 = y*y*y; 
      const auto x9 = z*z*z; 
      const auto x10 = 3.0*radial_eval; 
      const auto x11 = radial_eval_alpha*x0 + 2.0*x4; 
      const auto x12 = radial_eval*x5; 
      const auto x13 = radial_eval_alpha*x2*x5; 
      const auto x14 = y*z; 
      const auto x15 = radial_eval*x7; 
      const auto x16 = radial_eval_alpha*x2*x7; 
      const auto x17 = radial_eval_alpha*x; 
      const auto x18 = x17*x5*z; 
      const auto x19 = x17*x7*y; 
      const auto x20 = radial_eval_alpha*y; 
      const auto x21 = radial_eval*x2; 
      const auto x22 = radial_eval_alpha*x14*x2; 
      const auto x23 = radial_eval_alpha*x8 + 2.0*x1; 
      const auto x24 = radial_eval_alpha*x5*x7; 
      const auto x25 = radial_eval_alpha*z; 
      const auto x26 = radial_eval_alpha*x9 + 2.0*x3; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x0;
      basis_eval[ipt + 1*npts] = x1*x2;
      basis_eval[ipt + 2*npts] = x2*x3;
      basis_eval[ipt + 3*npts] = x4*x5;
      basis_eval[ipt + 4*npts] = x1*x6;
      basis_eval[ipt + 5*npts] = x4*x7;
      basis_eval[ipt + 6*npts] = radial_eval*x8;
      basis_eval[ipt + 7*npts] = x3*x5;
      basis_eval[ipt + 8*npts] = x1*x7;
      basis_eval[ipt + 9*npts] = radial_eval*x9;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = radial_eval_alpha*(x*x*x*x) + x10*x2;
      basis_x_eval[ipt + 1*npts] = x11*y;
      basis_x_eval[ipt + 2*npts] = x11*z;
      basis_x_eval[ipt + 3*npts] = x12 + x13;
      basis_x_eval[ipt + 4*npts] = x14*(radial_eval + radial_eval_alpha*x2);
      basis_x_eval[ipt + 5*npts] = x15 + x16;
      basis_x_eval[ipt + 6*npts] = x17*x8;
      basis_x_eval[ipt + 7*npts] = x18;
      basis_x_eval[ipt + 8*npts] = x19;
      basis_x_eval[ipt + 9*npts] = x17*x9;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*x20;
      basis_y_eval[ipt + 1*npts] = x13 + x21;
      basis_y_eval[ipt + 2*npts] = x22;
      basis_y_eval[ipt + 3*npts] = x*x23;
      basis_y_eval[ipt + 4*npts] = x6*(radial_eval + radial_eval_alpha*x5);
      basis_y_eval[ipt + 5*npts] = x19;
      basis_y_eval[ipt + 6*npts] = radial_eval_alpha*(y*y*y*y) + x10*x5;
      basis_y_eval[ipt + 7*npts] = x23*z;
      basis_y_eval[ipt + 8*npts] = x15 + x24;
      basis_y_eval[ipt + 9*npts] = x20*x9;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x0*x25;
      basis_z_eval[ipt + 1*npts] = x22;
      basis_z_eval[ipt + 2*npts] = x16 + x21;
      basis_z_eval[ipt + 3*npts] = x18;
      basis_z_eval[ipt + 4*npts] = x*y*(radial_eval + radial_eval_alpha*x7);
      basis_z_eval[ipt + 5*npts] = x*x26;
      basis_z_eval[ipt + 6*npts] = x25*x8;
      basis_z_eval[ipt + 7*npts] = x12 + x24;
      basis_z_eval[ipt + 8*npts] = x26*y;
      basis_z_eval[ipt + 9*npts] = radial_eval_alpha*(z*z*z*z) + x10*x7;







#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x0;
      ang_eval_1 = x1*x2;
      ang_eval_2 = x2*x3;
      ang_eval_3 = x4*x5;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = x1*x6;
      ang_eval_1 = x4*x7;
      ang_eval_2 = radial_eval*x8;
      ang_eval_3 = x3*x5;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = x1*x7;
      ang_eval_1 = radial_eval*x9;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = radial_eval_alpha*(x*x*x*x) + x10*x2;
      dang_eval_y_0 = x0*x20;
      dang_eval_z_0 = x0*x25;
      dang_eval_x_1 = x11*y;
      dang_eval_y_1 = x13 + x21;
      dang_eval_z_1 = x22;
      dang_eval_x_2 = x11*z;
      dang_eval_y_2 = x22;
      dang_eval_z_2 = x16 + x21;
      dang_eval_x_3 = x12 + x13;
      dang_eval_y_3 = x*x23;
      dang_eval_z_3 = x18;
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

      dang_eval_x_0 = x14*(radial_eval + radial_eval_alpha*x2);
      dang_eval_y_0 = x6*(radial_eval + radial_eval_alpha*x5);
      dang_eval_z_0 = x*y*(radial_eval + radial_eval_alpha*x7);
      dang_eval_x_1 = x15 + x16;
      dang_eval_y_1 = x19;
      dang_eval_z_1 = x*x26;
      dang_eval_x_2 = x17*x8;
      dang_eval_y_2 = radial_eval_alpha*(y*y*y*y) + x10*x5;
      dang_eval_z_2 = x25*x8;
      dang_eval_x_3 = x18;
      dang_eval_y_3 = x23*z;
      dang_eval_z_3 = x12 + x24;
      basis_x_eval[ipt + 4*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 4*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 4*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 5*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 5*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 5*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 6*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 6*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 6*npts] = dang_eval_z_2;
      basis_x_eval[ipt + 7*npts] = dang_eval_x_3;
      basis_y_eval[ipt + 7*npts] = dang_eval_y_3;
      basis_z_eval[ipt + 7*npts] = dang_eval_z_3;

      dang_eval_x_0 = x19;
      dang_eval_y_0 = x15 + x24;
      dang_eval_z_0 = x26*y;
      dang_eval_x_1 = x17*x9;
      dang_eval_y_1 = x20*x9;
      dang_eval_z_1 = radial_eval_alpha*(z*z*z*z) + x10*x7;
      basis_x_eval[ipt + 8*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 8*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 8*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 9*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 9*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 9*npts] = dang_eval_z_1;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
