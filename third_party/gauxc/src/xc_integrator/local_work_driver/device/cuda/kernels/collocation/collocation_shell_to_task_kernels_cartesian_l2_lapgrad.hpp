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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_cartesian_lapgrad_2(
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
      const auto x2 = x*y; 
      const auto x3 = x*z; 
      const auto x4 = y*y; 
      const auto x5 = x4; 
      const auto x6 = y*z; 
      const auto x7 = z*z; 
      const auto x8 = x7; 
      const auto x9 = 2.0*radial_eval; 
      const auto x10 = x*x*x; 
      const auto x11 = radial_eval + radial_eval_alpha*x1; 
      const auto x12 = radial_eval_alpha*x; 
      const auto x13 = x12*x6; 
      const auto x14 = radial_eval_alpha*y; 
      const auto x15 = radial_eval_alpha*x5; 
      const auto x16 = radial_eval + x15; 
      const auto x17 = y*y*y; 
      const auto x18 = radial_eval_alpha*z; 
      const auto x19 = radial_eval_alpha*x8; 
      const auto x20 = radial_eval + x19; 
      const auto x21 = z*z*z; 
      const auto x22 = 4.0*radial_eval_alpha; 
      const auto x23 = radial_eval_alpha_squared*x1; 
      const auto x24 = radial_eval_alpha + x23; 
      const auto x25 = x1*x22 + x1*x24 + x9; 
      const auto x26 = 3.0*radial_eval_alpha; 
      const auto x27 = x23 + x26; 
      const auto x28 = x24*x5; 
      const auto x29 = x24*x6; 
      const auto x30 = x24*x8; 
      const auto x31 = 2.0*x12; 
      const auto x32 = radial_eval_alpha_squared*x10 + x31; 
      const auto x33 = 2.0*x14; 
      const auto x34 = radial_eval_alpha_squared*x17 + x33; 
      const auto x35 = radial_eval_alpha_squared*x5; 
      const auto x36 = radial_eval_alpha + x35; 
      const auto x37 = x3*x36; 
      const auto x38 = radial_eval_alpha_squared*x8; 
      const auto x39 = radial_eval_alpha + x38; 
      const auto x40 = x2*x39; 
      const auto x41 = 2.0*x18; 
      const auto x42 = radial_eval_alpha_squared*x21 + x41; 
      const auto x43 = x1*x36; 
      const auto x44 = x26 + x35; 
      const auto x45 = x22*x5 + x36*x5 + x9; 
      const auto x46 = x36*x8; 
      const auto x47 = x1*x39; 
      const auto x48 = x26 + x38; 
      const auto x49 = x39*x5; 
      const auto x50 = x22*x8 + x39*x8 + x9; 
      const auto x51 = x35 + x38; 
      const auto x52 = 7.0*radial_eval_alpha + x23 + x51; 
      const auto x53 = 2.0*x; 
      const auto x54 = radial_eval_alpha_cubed*x5 + radial_eval_alpha_squared; 
      const auto x55 = x1*x54; 
      const auto x56 = radial_eval_alpha_cubed*x8 + radial_eval_alpha_squared; 
      const auto x57 = x1*x56; 
      const auto x58 = radial_eval_alpha_squared*x; 
      const auto x59 = radial_eval_alpha_cubed*x10; 
      const auto x60 = 3.0*x58 + x59; 
      const auto x61 = 2.0*radial_eval_alpha_squared; 
      const auto x62 = x*x60 + x0*x54 + x0*x56 + x1*x61 + x22 + 3.0*x24 + x51; 
      const auto x63 = 4.0*x58; 
      const auto x64 = x5*x54; 
      const auto x65 = x5*x56; 
      const auto x66 = x54*x8; 
      const auto x67 = x56*x8; 
      const auto x68 = radial_eval_alpha_squared*y; 
      const auto x69 = 4.0*x68; 
      const auto x70 = radial_eval_alpha_cubed*x1 + radial_eval_alpha_squared; 
      const auto x71 = x1*x70; 
      const auto x72 = radial_eval_alpha_cubed*x17; 
      const auto x73 = 3.0*x68 + x72; 
      const auto x74 = x22 + x23; 
      const auto x75 = 3.0*x36 + x38 + x4*x56 + x4*x70 + x5*x61 + x73*y + x74; 
      const auto x76 = 2.0*y; 
      const auto x77 = x5*x70; 
      const auto x78 = x70*x8; 
      const auto x79 = radial_eval_alpha_squared*z; 
      const auto x80 = 4.0*x79; 
      const auto x81 = radial_eval_alpha_cubed*x21; 
      const auto x82 = 3.0*x79 + x81; 
      const auto x83 = x35 + 3.0*x39 + x54*x7 + x61*x8 + x7*x70 + x74 + x82*z; 
      const auto x84 = 2.0*z; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x1;
      basis_eval[ipt + 1*npts] = radial_eval*x2;
      basis_eval[ipt + 2*npts] = radial_eval*x3;
      basis_eval[ipt + 3*npts] = radial_eval*x5;
      basis_eval[ipt + 4*npts] = radial_eval*x6;
      basis_eval[ipt + 5*npts] = radial_eval*x8;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = radial_eval_alpha*x10 + x*x9;
      basis_x_eval[ipt + 1*npts] = x11*y;
      basis_x_eval[ipt + 2*npts] = x11*z;
      basis_x_eval[ipt + 3*npts] = x12*x5;
      basis_x_eval[ipt + 4*npts] = x13;
      basis_x_eval[ipt + 5*npts] = x12*x8;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x1*x14;
      basis_y_eval[ipt + 1*npts] = x*x16;
      basis_y_eval[ipt + 2*npts] = x13;
      basis_y_eval[ipt + 3*npts] = radial_eval_alpha*x17 + x9*y;
      basis_y_eval[ipt + 4*npts] = x16*z;
      basis_y_eval[ipt + 5*npts] = x14*x8;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x1*x18;
      basis_z_eval[ipt + 1*npts] = x13;
      basis_z_eval[ipt + 2*npts] = x*x20;
      basis_z_eval[ipt + 3*npts] = x18*x5;
      basis_z_eval[ipt + 4*npts] = x20*y;
      basis_z_eval[ipt + 5*npts] = radial_eval_alpha*x21 + x9*z;

      // Evaluate second derivative of bfn wrt xx
      basis_xx_eval[ipt + 0*npts] = x25;
      basis_xx_eval[ipt + 1*npts] = x2*x27;
      basis_xx_eval[ipt + 2*npts] = x27*x3;
      basis_xx_eval[ipt + 3*npts] = x28;
      basis_xx_eval[ipt + 4*npts] = x29;
      basis_xx_eval[ipt + 5*npts] = x30;

      // Evaluate second derivative of bfn wrt xy
      basis_xy_eval[ipt + 0*npts] = x32*y;
      basis_xy_eval[ipt + 1*npts] = radial_eval_alpha_squared*x1*x5 + x11 + x15;
      basis_xy_eval[ipt + 2*npts] = x29;
      basis_xy_eval[ipt + 3*npts] = x*x34;
      basis_xy_eval[ipt + 4*npts] = x37;
      basis_xy_eval[ipt + 5*npts] = radial_eval_alpha_squared*x2*x8;

      // Evaluate second derivative of bfn wrt xz
      basis_xz_eval[ipt + 0*npts] = x32*z;
      basis_xz_eval[ipt + 1*npts] = x29;
      basis_xz_eval[ipt + 2*npts] = radial_eval_alpha_squared*x1*x8 + x11 + x19;
      basis_xz_eval[ipt + 3*npts] = radial_eval_alpha_squared*x3*x5;
      basis_xz_eval[ipt + 4*npts] = x40;
      basis_xz_eval[ipt + 5*npts] = x*x42;

      // Evaluate second derivative of bfn wrt yy
      basis_yy_eval[ipt + 0*npts] = x43;
      basis_yy_eval[ipt + 1*npts] = x2*x44;
      basis_yy_eval[ipt + 2*npts] = x37;
      basis_yy_eval[ipt + 3*npts] = x45;
      basis_yy_eval[ipt + 4*npts] = x44*x6;
      basis_yy_eval[ipt + 5*npts] = x46;

      // Evaluate second derivative of bfn wrt yz
      basis_yz_eval[ipt + 0*npts] = radial_eval_alpha_squared*x1*x6;
      basis_yz_eval[ipt + 1*npts] = x37;
      basis_yz_eval[ipt + 2*npts] = x40;
      basis_yz_eval[ipt + 3*npts] = x34*z;
      basis_yz_eval[ipt + 4*npts] = radial_eval_alpha_squared*x5*x8 + x16 + x19;
      basis_yz_eval[ipt + 5*npts] = x42*y;

      // Evaluate second derivative of bfn wrt zz
      basis_zz_eval[ipt + 0*npts] = x47;
      basis_zz_eval[ipt + 1*npts] = x40;
      basis_zz_eval[ipt + 2*npts] = x3*x48;
      basis_zz_eval[ipt + 3*npts] = x49;
      basis_zz_eval[ipt + 4*npts] = x48*x6;
      basis_zz_eval[ipt + 5*npts] = x50;

      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x25 + x43 + x47;
      basis_lapl_eval[ipt + 1*npts] = x2*x52;
      basis_lapl_eval[ipt + 2*npts] = x3*x52;
      basis_lapl_eval[ipt + 3*npts] = x28 + x45 + x49;
      basis_lapl_eval[ipt + 4*npts] = x52*x6;
      basis_lapl_eval[ipt + 5*npts] = x30 + x46 + x50;

      // Evaluate Laplacian gradient of bfn (dx)
      basis_lapl_x_eval[ipt + 0*npts] = 6.0*x*x24 + x*x55 + x*x57 + x1*x60 + 6.0*x12 + x36*x53 + x39*x53;
      basis_lapl_x_eval[ipt + 1*npts] = x62*y;
      basis_lapl_x_eval[ipt + 2*npts] = x62*z;
      basis_lapl_x_eval[ipt + 3*npts] = x*x64 + x*x65 + x31 + x5*x60 + x5*x63;
      basis_lapl_x_eval[ipt + 4*npts] = x6*(x*x54 + x*x56 + 7.0*x58 + x59);
      basis_lapl_x_eval[ipt + 5*npts] = x*x66 + x*x67 + x31 + x60*x8 + x63*x8;
      // Evaluate Laplacian gradient of bfn (dy)
      basis_lapl_y_eval[ipt + 0*npts] = x1*x69 + x1*x73 + x33 + x57*y + x71*y;
      basis_lapl_y_eval[ipt + 1*npts] = x*x75;
      basis_lapl_y_eval[ipt + 2*npts] = x3*(x56*y + 7.0*x68 + x70*y + x72);
      basis_lapl_y_eval[ipt + 3*npts] = 6.0*x14 + x24*x76 + 6.0*x36*y + x39*x76 + x5*x73 + x65*y + x77*y;
      basis_lapl_y_eval[ipt + 4*npts] = x75*z;
      basis_lapl_y_eval[ipt + 5*npts] = x33 + x67*y + x69*x8 + x73*x8 + x78*y;
      // Evaluate Laplacian gradient of bfn (dz)
      basis_lapl_z_eval[ipt + 0*npts] = x1*x80 + x1*x82 + x41 + x55*z + x71*z;
      basis_lapl_z_eval[ipt + 1*npts] = x2*(x54*z + x70*z + 7.0*x79 + x81);
      basis_lapl_z_eval[ipt + 2*npts] = x*x83;
      basis_lapl_z_eval[ipt + 3*npts] = x41 + x5*x80 + x5*x82 + x64*z + x77*z;
      basis_lapl_z_eval[ipt + 4*npts] = x83*y;
      basis_lapl_z_eval[ipt + 5*npts] = 6.0*x18 + x24*x84 + x36*x84 + 6.0*x39*z + x66*z + x78*z + x8*x82;




#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x1;
      ang_eval_1 = radial_eval*x2;
      ang_eval_2 = radial_eval*x3;
      ang_eval_3 = radial_eval*x5;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x6;
      ang_eval_1 = radial_eval*x8;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = radial_eval_alpha*x10 + x*x9;
      dang_eval_y_0 = x1*x14;
      dang_eval_z_0 = x1*x18;
      dang_eval_x_1 = x11*y;
      dang_eval_y_1 = x*x16;
      dang_eval_z_1 = x13;
      dang_eval_x_2 = x11*z;
      dang_eval_y_2 = x13;
      dang_eval_z_2 = x*x20;
      dang_eval_x_3 = x12*x5;
      dang_eval_y_3 = radial_eval_alpha*x17 + x9*y;
      dang_eval_z_3 = x18*x5;
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

      dang_eval_x_0 = x13;
      dang_eval_y_0 = x16*z;
      dang_eval_z_0 = x20*y;
      dang_eval_x_1 = x12*x8;
      dang_eval_y_1 = x14*x8;
      dang_eval_z_1 = radial_eval_alpha*x21 + x9*z;
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
