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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_cartesian_hessian_2(
  uint32_t                        nshell,
  ShellToTaskDevice* __restrict__ shell_to_task,
  XCDeviceTask*      __restrict__ device_tasks
) {


  __shared__ double alpha[4][detail::shell_nprim_max + 1]; 
  __shared__ double coeff[4][detail::shell_nprim_max + 1];
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
    auto* __restrict__ basis_xx_eval = task->d2bfxx + shoff;
    auto* __restrict__ basis_xy_eval = task->d2bfxy + shoff;
    auto* __restrict__ basis_xz_eval = task->d2bfxz + shoff;
    auto* __restrict__ basis_yy_eval = task->d2bfyy + shoff;
    auto* __restrict__ basis_yz_eval = task->d2bfyz + shoff;
    auto* __restrict__ basis_zz_eval = task->d2bfzz + shoff;

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
      double radial_eval_alpha_squared = 0.;

      #pragma unroll 1
      for( uint32_t i = 0; i < nprim; ++i ) {
        const auto a = my_alpha[i];
        const auto e = my_coeff[i] * std::exp( - a * rsq );

        radial_eval += e;
        radial_eval_alpha += a * e;
        radial_eval_alpha_squared += a * a * e;
      }

      radial_eval_alpha *= -2;
      radial_eval_alpha_squared *= 4;

      // Common Subexpressions
      const auto x0 = x*x; 
      const auto x1 = x*y; 
      const auto x2 = x*z; 
      const auto x3 = y*y; 
      const auto x4 = y*z; 
      const auto x5 = z*z; 
      const auto x6 = 2.0*radial_eval; 
      const auto x7 = x*x*x; 
      const auto x8 = radial_eval + radial_eval_alpha*x0; 
      const auto x9 = radial_eval_alpha*x; 
      const auto x10 = x4*x9; 
      const auto x11 = radial_eval_alpha*y; 
      const auto x12 = radial_eval_alpha*x3; 
      const auto x13 = radial_eval + x12; 
      const auto x14 = y*y*y; 
      const auto x15 = radial_eval_alpha*z; 
      const auto x16 = radial_eval_alpha*x5; 
      const auto x17 = radial_eval + x16; 
      const auto x18 = z*z*z; 
      const auto x19 = 4.0*radial_eval_alpha; 
      const auto x20 = radial_eval_alpha_squared*x0; 
      const auto x21 = radial_eval_alpha + x20; 
      const auto x22 = x0*x19 + x0*x21 + x6; 
      const auto x23 = 3.0*radial_eval_alpha; 
      const auto x24 = x20 + x23; 
      const auto x25 = x21*x3; 
      const auto x26 = x21*x4; 
      const auto x27 = x21*x5; 
      const auto x28 = radial_eval_alpha_squared*x7 + 2.0*x9; 
      const auto x29 = radial_eval_alpha_squared*x14 + 2.0*x11; 
      const auto x30 = radial_eval_alpha_squared*x3; 
      const auto x31 = radial_eval_alpha + x30; 
      const auto x32 = x2*x31; 
      const auto x33 = radial_eval_alpha_squared*x5; 
      const auto x34 = radial_eval_alpha + x33; 
      const auto x35 = x1*x34; 
      const auto x36 = radial_eval_alpha_squared*x18 + 2.0*x15; 
      const auto x37 = x0*x31; 
      const auto x38 = x23 + x30; 
      const auto x39 = x19*x3 + x3*x31 + x6; 
      const auto x40 = x31*x5; 
      const auto x41 = x0*x34; 
      const auto x42 = x23 + x33; 
      const auto x43 = x3*x34; 
      const auto x44 = x19*x5 + x34*x5 + x6; 
      const auto x45 = 7.0*radial_eval_alpha + x20 + x30 + x33; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x0;
      basis_eval[ipt + 1*npts] = radial_eval*x1;
      basis_eval[ipt + 2*npts] = radial_eval*x2;
      basis_eval[ipt + 3*npts] = radial_eval*x3;
      basis_eval[ipt + 4*npts] = radial_eval*x4;
      basis_eval[ipt + 5*npts] = radial_eval*x5;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = radial_eval_alpha*x7 + x*x6;
      basis_x_eval[ipt + 1*npts] = x8*y;
      basis_x_eval[ipt + 2*npts] = x8*z;
      basis_x_eval[ipt + 3*npts] = x3*x9;
      basis_x_eval[ipt + 4*npts] = x10;
      basis_x_eval[ipt + 5*npts] = x5*x9;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*x11;
      basis_y_eval[ipt + 1*npts] = x*x13;
      basis_y_eval[ipt + 2*npts] = x10;
      basis_y_eval[ipt + 3*npts] = radial_eval_alpha*x14 + x6*y;
      basis_y_eval[ipt + 4*npts] = x13*z;
      basis_y_eval[ipt + 5*npts] = x11*x5;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x0*x15;
      basis_z_eval[ipt + 1*npts] = x10;
      basis_z_eval[ipt + 2*npts] = x*x17;
      basis_z_eval[ipt + 3*npts] = x15*x3;
      basis_z_eval[ipt + 4*npts] = x17*y;
      basis_z_eval[ipt + 5*npts] = radial_eval_alpha*x18 + x6*z;

      // Evaluate second derivative of bfn wrt xx
      basis_xx_eval[ipt + 0*npts] = x22;
      basis_xx_eval[ipt + 1*npts] = x1*x24;
      basis_xx_eval[ipt + 2*npts] = x2*x24;
      basis_xx_eval[ipt + 3*npts] = x25;
      basis_xx_eval[ipt + 4*npts] = x26;
      basis_xx_eval[ipt + 5*npts] = x27;

      // Evaluate second derivative of bfn wrt xy
      basis_xy_eval[ipt + 0*npts] = x28*y;
      basis_xy_eval[ipt + 1*npts] = radial_eval_alpha_squared*x0*x3 + x12 + x8;
      basis_xy_eval[ipt + 2*npts] = x26;
      basis_xy_eval[ipt + 3*npts] = x*x29;
      basis_xy_eval[ipt + 4*npts] = x32;
      basis_xy_eval[ipt + 5*npts] = radial_eval_alpha_squared*x1*x5;

      // Evaluate second derivative of bfn wrt xz
      basis_xz_eval[ipt + 0*npts] = x28*z;
      basis_xz_eval[ipt + 1*npts] = x26;
      basis_xz_eval[ipt + 2*npts] = radial_eval_alpha_squared*x0*x5 + x16 + x8;
      basis_xz_eval[ipt + 3*npts] = radial_eval_alpha_squared*x2*x3;
      basis_xz_eval[ipt + 4*npts] = x35;
      basis_xz_eval[ipt + 5*npts] = x*x36;

      // Evaluate second derivative of bfn wrt yy
      basis_yy_eval[ipt + 0*npts] = x37;
      basis_yy_eval[ipt + 1*npts] = x1*x38;
      basis_yy_eval[ipt + 2*npts] = x32;
      basis_yy_eval[ipt + 3*npts] = x39;
      basis_yy_eval[ipt + 4*npts] = x38*x4;
      basis_yy_eval[ipt + 5*npts] = x40;

      // Evaluate second derivative of bfn wrt yz
      basis_yz_eval[ipt + 0*npts] = radial_eval_alpha_squared*x0*x4;
      basis_yz_eval[ipt + 1*npts] = x32;
      basis_yz_eval[ipt + 2*npts] = x35;
      basis_yz_eval[ipt + 3*npts] = x29*z;
      basis_yz_eval[ipt + 4*npts] = radial_eval_alpha_squared*x3*x5 + x13 + x16;
      basis_yz_eval[ipt + 5*npts] = x36*y;

      // Evaluate second derivative of bfn wrt zz
      basis_zz_eval[ipt + 0*npts] = x41;
      basis_zz_eval[ipt + 1*npts] = x35;
      basis_zz_eval[ipt + 2*npts] = x2*x42;
      basis_zz_eval[ipt + 3*npts] = x43;
      basis_zz_eval[ipt + 4*npts] = x4*x42;
      basis_zz_eval[ipt + 5*npts] = x44;






#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x0;
      ang_eval_1 = radial_eval*x1;
      ang_eval_2 = radial_eval*x2;
      ang_eval_3 = radial_eval*x3;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x4;
      ang_eval_1 = radial_eval*x5;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = radial_eval_alpha*x7 + x*x6;
      dang_eval_y_0 = x0*x11;
      dang_eval_z_0 = x0*x15;
      dang_eval_x_1 = x8*y;
      dang_eval_y_1 = x*x13;
      dang_eval_z_1 = x10;
      dang_eval_x_2 = x8*z;
      dang_eval_y_2 = x10;
      dang_eval_z_2 = x*x17;
      dang_eval_x_3 = x3*x9;
      dang_eval_y_3 = radial_eval_alpha*x14 + x6*y;
      dang_eval_z_3 = x15*x3;
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

      dang_eval_x_0 = x10;
      dang_eval_y_0 = x13*z;
      dang_eval_z_0 = x17*y;
      dang_eval_x_1 = x5*x9;
      dang_eval_y_1 = x11*x5;
      dang_eval_z_1 = radial_eval_alpha*x18 + x6*z;
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
