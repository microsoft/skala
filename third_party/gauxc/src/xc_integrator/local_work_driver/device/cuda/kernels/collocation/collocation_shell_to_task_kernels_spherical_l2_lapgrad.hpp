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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_spherical_lapgrad_2(
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
      const auto x0 = sqrt_3*y; 
      const auto x1 = x*x0; 
      const auto x2 = x0*z; 
      const auto x3 = 0.5*radial_eval; 
      const auto x4 = x*x; 
      const auto x5 = x4; 
      const auto x6 = y*y; 
      const auto x7 = x6; 
      const auto x8 = z*z; 
      const auto x9 = x8; 
      const auto x10 = -x5 - x7 + 2.0*x9; 
      const auto x11 = sqrt_3*z; 
      const auto x12 = x*x11; 
      const auto x13 = x5 - x7; 
      const auto x14 = radial_eval + radial_eval_alpha*x5; 
      const auto x15 = radial_eval_alpha*x1*z; 
      const auto x16 = 0.5*x; 
      const auto x17 = 2.0*radial_eval; 
      const auto x18 = -x17; 
      const auto x19 = radial_eval_alpha*x10; 
      const auto x20 = x18 + x19; 
      const auto x21 = radial_eval_alpha*x13; 
      const auto x22 = sqrt_3*x; 
      const auto x23 = radial_eval_alpha*x7; 
      const auto x24 = radial_eval + x23; 
      const auto x25 = 0.5*y; 
      const auto x26 = radial_eval_alpha*x9; 
      const auto x27 = radial_eval + x26; 
      const auto x28 = 0.5*z; 
      const auto x29 = 4.0*radial_eval; 
      const auto x30 = 3.0*radial_eval_alpha; 
      const auto x31 = radial_eval_alpha_squared*x5; 
      const auto x32 = x30 + x31; 
      const auto x33 = radial_eval_alpha + x31; 
      const auto x34 = x2*x33; 
      const auto x35 = 4.0*radial_eval_alpha; 
      const auto x36 = x35*x5; 
      const auto x37 = x17 + x36; 
      const auto x38 = 0.5*sqrt_3; 
      const auto x39 = x13*x33; 
      const auto x40 = radial_eval_alpha_squared*x7; 
      const auto x41 = radial_eval_alpha + x40; 
      const auto x42 = x12*x41; 
      const auto x43 = radial_eval_alpha_squared*x10; 
      const auto x44 = radial_eval_alpha_squared*x9; 
      const auto x45 = radial_eval_alpha + x44; 
      const auto x46 = x1*x45; 
      const auto x47 = 2.0*radial_eval_alpha; 
      const auto x48 = x43 + x47; 
      const auto x49 = radial_eval_alpha_squared*x13; 
      const auto x50 = x30 + x40; 
      const auto x51 = x35*x7; 
      const auto x52 = x17 + x51; 
      const auto x53 = x30 + x44; 
      const auto x54 = 8.0*radial_eval_alpha; 
      const auto x55 = x10*x45 + x54*x9; 
      const auto x56 = x13*x45; 
      const auto x57 = x40 + x44; 
      const auto x58 = 7.0*radial_eval_alpha + x31 + x57; 
      const auto x59 = -x51; 
      const auto x60 = radial_eval_alpha_squared*x; 
      const auto x61 = radial_eval_alpha_cubed*(x*x*x); 
      const auto x62 = 3.0*x60 + x61; 
      const auto x63 = radial_eval_alpha_cubed*x7 + radial_eval_alpha_squared; 
      const auto x64 = radial_eval_alpha_cubed*x9 + radial_eval_alpha_squared; 
      const auto x65 = 2.0*radial_eval_alpha_squared; 
      const auto x66 = x*x62 + 3.0*x33 + x35 + x4*x63 + x4*x64 + x5*x65 + x57; 
      const auto x67 = 4.0*x60*x7; 
      const auto x68 = 2.0*x; 
      const auto x69 = 6.0*x*x33 + x*x35 + x41*x68 + x45*x68; 
      const auto x70 = x13*x63; 
      const auto x71 = x13*x64; 
      const auto x72 = radial_eval_alpha_squared*y; 
      const auto x73 = radial_eval_alpha_cubed*(y*y*y); 
      const auto x74 = 3.0*x72 + x73; 
      const auto x75 = radial_eval_alpha_cubed*x5 + radial_eval_alpha_squared; 
      const auto x76 = x31 + x35; 
      const auto x77 = 3.0*x41 + x44 + x6*x64 + x6*x75 + x65*x7 + x74*y + x76; 
      const auto x78 = x35*y; 
      const auto x79 = 4.0*x5*x72; 
      const auto x80 = 2.0*y; 
      const auto x81 = x33*x80; 
      const auto x82 = 6.0*x41*y; 
      const auto x83 = x45*x80; 
      const auto x84 = x13*x75; 
      const auto x85 = radial_eval_alpha_squared*z; 
      const auto x86 = radial_eval_alpha_cubed*(z*z*z); 
      const auto x87 = 3.0*x85 + x86; 
      const auto x88 = x40 + 3.0*x45 + x63*x8 + x65*x9 + x75*x8 + x76 + x87*z; 
      const auto x89 = 4.0*z; 
      const auto x90 = radial_eval_alpha_squared*x89; 
      const auto x91 = x5*x90; 
      const auto x92 = -x7*x90; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x1;
      basis_eval[ipt + 1*npts] = radial_eval*x2;
      basis_eval[ipt + 2*npts] = x10*x3;
      basis_eval[ipt + 3*npts] = radial_eval*x12;
      basis_eval[ipt + 4*npts] = sqrt_3*x13*x3;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = x0*x14;
      basis_x_eval[ipt + 1*npts] = x15;
      basis_x_eval[ipt + 2*npts] = x16*x20;
      basis_x_eval[ipt + 3*npts] = x11*x14;
      basis_x_eval[ipt + 4*npts] = sqrt_3*x16*(x17 + x21);

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x22*x24;
      basis_y_eval[ipt + 1*npts] = x11*x24;
      basis_y_eval[ipt + 2*npts] = x20*x25;
      basis_y_eval[ipt + 3*npts] = x15;
      basis_y_eval[ipt + 4*npts] = 0.5*x0*(x18 + x21);

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x15;
      basis_z_eval[ipt + 1*npts] = x0*x27;
      basis_z_eval[ipt + 2*npts] = x28*(x19 + x29);
      basis_z_eval[ipt + 3*npts] = x22*x27;
      basis_z_eval[ipt + 4*npts] = 0.5*radial_eval_alpha*x11*x13;

      // Evaluate second derivative of bfn wrt xx
      basis_xx_eval[ipt + 0*npts] = x1*x32;
      basis_xx_eval[ipt + 1*npts] = x34;
      basis_xx_eval[ipt + 2*npts] = 0.5*x10*x33 - 0.5*x37;
      basis_xx_eval[ipt + 3*npts] = x12*x32;
      basis_xx_eval[ipt + 4*npts] = x38*(x37 + x39);

      // Evaluate second derivative of bfn wrt xy
      basis_xy_eval[ipt + 0*npts] = sqrt_3*(radial_eval_alpha_squared*x5*x7 + x14 + x23);
      basis_xy_eval[ipt + 1*npts] = x42;
      basis_xy_eval[ipt + 2*npts] = x16*y*(-x35 + x43);
      basis_xy_eval[ipt + 3*npts] = x34;
      basis_xy_eval[ipt + 4*npts] = radial_eval_alpha_squared*x0*x13*x16;

      // Evaluate second derivative of bfn wrt xz
      basis_xz_eval[ipt + 0*npts] = x34;
      basis_xz_eval[ipt + 1*npts] = x46;
      basis_xz_eval[ipt + 2*npts] = x16*x48*z;
      basis_xz_eval[ipt + 3*npts] = sqrt_3*(radial_eval_alpha_squared*x5*x9 + x14 + x26);
      basis_xz_eval[ipt + 4*npts] = x11*x16*(x47 + x49);

      // Evaluate second derivative of bfn wrt yy
      basis_yy_eval[ipt + 0*npts] = x1*x50;
      basis_yy_eval[ipt + 1*npts] = x2*x50;
      basis_yy_eval[ipt + 2*npts] = 0.5*x10*x41 - 0.5*x52;
      basis_yy_eval[ipt + 3*npts] = x42;
      basis_yy_eval[ipt + 4*npts] = x38*(x13*x41 - x52);

      // Evaluate second derivative of bfn wrt yz
      basis_yz_eval[ipt + 0*npts] = x42;
      basis_yz_eval[ipt + 1*npts] = sqrt_3*(radial_eval_alpha_squared*x7*x9 + x24 + x26);
      basis_yz_eval[ipt + 2*npts] = x25*x48*z;
      basis_yz_eval[ipt + 3*npts] = x46;
      basis_yz_eval[ipt + 4*npts] = x0*x28*(-x47 + x49);

      // Evaluate second derivative of bfn wrt zz
      basis_zz_eval[ipt + 0*npts] = x46;
      basis_zz_eval[ipt + 1*npts] = x2*x53;
      basis_zz_eval[ipt + 2*npts] = 0.5*x29 + 0.5*x55;
      basis_zz_eval[ipt + 3*npts] = x12*x53;
      basis_zz_eval[ipt + 4*npts] = x38*x56;

      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x1*x58;
      basis_lapl_eval[ipt + 1*npts] = x2*x58;
      basis_lapl_eval[ipt + 2*npts] = 0.5*x10*x33 + 0.5*x10*x41 - 0.5*x36 + 0.5*x55 + 0.5*x59;
      basis_lapl_eval[ipt + 3*npts] = x12*x58;
      basis_lapl_eval[ipt + 4*npts] = x38*(x13*x41 + x36 + x39 + x56 + x59);

      // Evaluate Laplacian gradient of bfn (dx)
      basis_lapl_x_eval[ipt + 0*npts] = x0*x66;
      basis_lapl_x_eval[ipt + 1*npts] = x2*(x*x63 + x*x64 + 7.0*x60 + x61);
      basis_lapl_x_eval[ipt + 2*npts] = 4.0*radial_eval_alpha_squared*x*x9 + 0.5*x*x10*x63 + 0.5*x*x10*x64 + 0.5*x10*x62 - 0.5*x67 - 0.5*x69;
      basis_lapl_x_eval[ipt + 3*npts] = x11*x66;
      basis_lapl_x_eval[ipt + 4*npts] = x38*(x*x70 + x*x71 + x13*x62 - x67 + x69);
      // Evaluate Laplacian gradient of bfn (dy)
      basis_lapl_y_eval[ipt + 0*npts] = x22*x77;
      basis_lapl_y_eval[ipt + 1*npts] = x11*x77;
      basis_lapl_y_eval[ipt + 2*npts] = 4.0*radial_eval_alpha_squared*x9*y + 0.5*x10*x64*y + 0.5*x10*x74 + 0.5*x10*x75*y - 0.5*x78 - 0.5*x79 - 0.5*x81 - 0.5*x82 - 0.5*x83;
      basis_lapl_y_eval[ipt + 3*npts] = x12*(x64*y + 7.0*x72 + x73 + x75*y);
      basis_lapl_y_eval[ipt + 4*npts] = x38*(x13*x74 + x71*y - x78 + x79 - x81 - x82 - x83 + x84*y);
      // Evaluate Laplacian gradient of bfn (dz)
      basis_lapl_z_eval[ipt + 0*npts] = x1*(x63*z + x75*z + 7.0*x85 + x86);
      basis_lapl_z_eval[ipt + 1*npts] = x0*x88;
      basis_lapl_z_eval[ipt + 2*npts] = 0.5*x10*x63*z + 0.5*x10*x75*z + 0.5*x10*x87 + 0.5*x33*x89 + 0.5*x41*x89 + 6.0*x45*z + 0.5*x54*z - 0.5*x91 + 0.5*x92;
      basis_lapl_z_eval[ipt + 3*npts] = x22*x88;
      basis_lapl_z_eval[ipt + 4*npts] = x38*(x13*x87 + x70*z + x84*z + x91 + x92);




#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x1;
      ang_eval_1 = radial_eval*x2;
      ang_eval_2 = x10*x3;
      ang_eval_3 = radial_eval*x12;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = sqrt_3*x13*x3;
      basis_eval[ipt + 4*npts] = ang_eval_0;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = x0*x14;
      dang_eval_y_0 = x22*x24;
      dang_eval_z_0 = x15;
      dang_eval_x_1 = x15;
      dang_eval_y_1 = x11*x24;
      dang_eval_z_1 = x0*x27;
      dang_eval_x_2 = x16*x20;
      dang_eval_y_2 = x20*x25;
      dang_eval_z_2 = x28*(x19 + x29);
      dang_eval_x_3 = x11*x14;
      dang_eval_y_3 = x15;
      dang_eval_z_3 = x22*x27;
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

      dang_eval_x_0 = sqrt_3*x16*(x17 + x21);
      dang_eval_y_0 = 0.5*x0*(x18 + x21);
      dang_eval_z_0 = 0.5*radial_eval_alpha*x11*x13;
      basis_x_eval[ipt + 4*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 4*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 4*npts] = dang_eval_z_0;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
