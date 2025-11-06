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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_cartesian_lapgrad_1(
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
    auto* __restrict__ basis_lapl_eval = task->d2bflapl + shoff;
    auto* __restrict__ basis_lapl_x_eval = task->d3bflapl_x + shoff;
    auto* __restrict__ basis_lapl_y_eval = task->d3bflapl_y + shoff;
    auto* __restrict__ basis_lapl_z_eval = task->d3bflapl_z + shoff;

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
      double radial_eval_alpha_cubed = 0.;

      #pragma unroll 1
      for( uint32_t i = 0; i < nprim; ++i ) {
        const auto a = my_alpha[i];
        const auto e = my_coeff[i] * std::exp( - a * rsq );

        radial_eval += e;
        radial_eval_alpha += a * e;
        radial_eval_alpha_squared += a * a * e;
        radial_eval_alpha_cubed += a * a * a * e;
      }

      radial_eval_alpha *= -2;
      radial_eval_alpha_squared *= 4;
      radial_eval_alpha_cubed *= -8;

      // Common Subexpressions
      const auto x0 = x*x; 
      const auto x1 = x0; 
      const auto x2 = radial_eval_alpha*x; 
      const auto x3 = x2*y; 
      const auto x4 = x2*z; 
      const auto x5 = y*y; 
      const auto x6 = x5; 
      const auto x7 = y*z; 
      const auto x8 = radial_eval_alpha*x7; 
      const auto x9 = z*z; 
      const auto x10 = x9; 
      const auto x11 = 3.0*radial_eval_alpha; 
      const auto x12 = radial_eval_alpha_squared*x1; 
      const auto x13 = radial_eval_alpha + x12; 
      const auto x14 = x13*y; 
      const auto x15 = x13*z; 
      const auto x16 = radial_eval_alpha_squared*x6; 
      const auto x17 = radial_eval_alpha + x16; 
      const auto x18 = x*x17; 
      const auto x19 = radial_eval_alpha_squared*x*x7; 
      const auto x20 = radial_eval_alpha_squared*x10; 
      const auto x21 = radial_eval_alpha + x20; 
      const auto x22 = x*x21; 
      const auto x23 = x17*z; 
      const auto x24 = x21*y; 
      const auto x25 = 5.0*radial_eval_alpha; 
      const auto x26 = x16 + x20 + x25; 
      const auto x27 = x12 + x26; 
      const auto x28 = 3.0*radial_eval_alpha_squared; 
      const auto x29 = radial_eval_alpha_cubed*(x*x*x); 
      const auto x30 = radial_eval_alpha_cubed*x6 + radial_eval_alpha_squared; 
      const auto x31 = radial_eval_alpha_cubed*x10 + radial_eval_alpha_squared; 
      const auto x32 = 5.0*radial_eval_alpha_squared; 
      const auto x33 = x*x30 + x*x31 + x*x32 + x29; 
      const auto x34 = radial_eval_alpha_cubed*(y*y*y); 
      const auto x35 = radial_eval_alpha_cubed*x1 + radial_eval_alpha_squared; 
      const auto x36 = x31*y + x32*y + x34 + x35*y; 
      const auto x37 = x12 + x25; 
      const auto x38 = radial_eval_alpha_cubed*(z*z*z); 
      const auto x39 = x30*z + x32*z + x35*z + x38; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x;
      basis_eval[ipt + 1*npts] = radial_eval*y;
      basis_eval[ipt + 2*npts] = radial_eval*z;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = radial_eval + radial_eval_alpha*x1;
      basis_x_eval[ipt + 1*npts] = x3;
      basis_x_eval[ipt + 2*npts] = x4;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x3;
      basis_y_eval[ipt + 1*npts] = radial_eval + radial_eval_alpha*x6;
      basis_y_eval[ipt + 2*npts] = x8;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x4;
      basis_z_eval[ipt + 1*npts] = x8;
      basis_z_eval[ipt + 2*npts] = radial_eval + radial_eval_alpha*x10;

      // Evaluate second derivative of bfn wrt xx
      basis_xx_eval[ipt + 0*npts] = x*(x11 + x12);
      basis_xx_eval[ipt + 1*npts] = x14;
      basis_xx_eval[ipt + 2*npts] = x15;

      // Evaluate second derivative of bfn wrt xy
      basis_xy_eval[ipt + 0*npts] = x14;
      basis_xy_eval[ipt + 1*npts] = x18;
      basis_xy_eval[ipt + 2*npts] = x19;

      // Evaluate second derivative of bfn wrt xz
      basis_xz_eval[ipt + 0*npts] = x15;
      basis_xz_eval[ipt + 1*npts] = x19;
      basis_xz_eval[ipt + 2*npts] = x22;

      // Evaluate second derivative of bfn wrt yy
      basis_yy_eval[ipt + 0*npts] = x18;
      basis_yy_eval[ipt + 1*npts] = y*(x11 + x16);
      basis_yy_eval[ipt + 2*npts] = x23;

      // Evaluate second derivative of bfn wrt yz
      basis_yz_eval[ipt + 0*npts] = x19;
      basis_yz_eval[ipt + 1*npts] = x23;
      basis_yz_eval[ipt + 2*npts] = x24;

      // Evaluate second derivative of bfn wrt zz
      basis_zz_eval[ipt + 0*npts] = x22;
      basis_zz_eval[ipt + 1*npts] = x24;
      basis_zz_eval[ipt + 2*npts] = z*(x11 + x20);

      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x*x27;
      basis_lapl_eval[ipt + 1*npts] = x27*y;
      basis_lapl_eval[ipt + 2*npts] = x27*z;

      // Evaluate Laplacian gradient of bfn (dx)
      basis_lapl_x_eval[ipt + 0*npts] = x*(x*x28 + x29) + x0*x30 + x0*x31 + x1*x28 + x26;
      basis_lapl_x_eval[ipt + 1*npts] = x33*y;
      basis_lapl_x_eval[ipt + 2*npts] = x33*z;
      // Evaluate Laplacian gradient of bfn (dy)
      basis_lapl_y_eval[ipt + 0*npts] = x*x36;
      basis_lapl_y_eval[ipt + 1*npts] = x20 + x28*x6 + x31*x5 + x35*x5 + x37 + y*(x28*y + x34);
      basis_lapl_y_eval[ipt + 2*npts] = x36*z;
      // Evaluate Laplacian gradient of bfn (dz)
      basis_lapl_z_eval[ipt + 0*npts] = x*x39;
      basis_lapl_z_eval[ipt + 1*npts] = x39*y;
      basis_lapl_z_eval[ipt + 2*npts] = x10*x28 + x16 + x30*x9 + x35*x9 + x37 + z*(x28*z + x38);




#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;


      ang_eval_0 = radial_eval*x;
      ang_eval_1 = radial_eval*y;
      ang_eval_2 = radial_eval*z;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;

      dang_eval_x_0 = radial_eval + radial_eval_alpha*x1;
      dang_eval_y_0 = x3;
      dang_eval_z_0 = x4;
      dang_eval_x_1 = x3;
      dang_eval_y_1 = radial_eval + radial_eval_alpha*x6;
      dang_eval_z_1 = x8;
      dang_eval_x_2 = x4;
      dang_eval_y_2 = x8;
      dang_eval_z_2 = radial_eval + radial_eval_alpha*x10;
      basis_x_eval[ipt + 0*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 0*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 0*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 1*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 1*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 1*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 2*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 2*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 2*npts] = dang_eval_z_2;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
