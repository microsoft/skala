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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_spherical_laplacian_3(
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
    auto* __restrict__ basis_lapl_eval = task->d2bflapl + shoff;

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
      const auto x0 = 0.25*sqrt_10; 
      const auto x1 = x0*y; 
      const auto x2 = x*x; 
      const auto x3 = 3.0*x2; 
      const auto x4 = y*y; 
      const auto x5 = -x4; 
      const auto x6 = x3 + x5; 
      const auto x7 = sqrt_15*z; 
      const auto x8 = x7*y; 
      const auto x9 = x*x8; 
      const auto x10 = 0.25*sqrt_6; 
      const auto x11 = x10*y; 
      const auto x12 = z*z; 
      const auto x13 = -4.0*x12; 
      const auto x14 = x13 + x4; 
      const auto x15 = -x14 - x2; 
      const auto x16 = 0.5*z; 
      const auto x17 = 3.0*x4; 
      const auto x18 = -2.0*x12; 
      const auto x19 = -x17 - x18 - x3; 
      const auto x20 = x*x10; 
      const auto x21 = 0.5*sqrt_15; 
      const auto x22 = x21*z; 
      const auto x23 = x2 + x5; 
      const auto x24 = x*x0; 
      const auto x25 = -x17; 
      const auto x26 = x2 + x25; 
      const auto x27 = x*x1; 
      const auto x28 = 6.0*radial_eval; 
      const auto x29 = radial_eval + radial_eval_alpha*x2; 
      const auto x30 = x*x11; 
      const auto x31 = 2.0*radial_eval; 
      const auto x32 = -x31; 
      const auto x33 = radial_eval_alpha*x15; 
      const auto x34 = x30*(x32 + x33); 
      const auto x35 = x*x16; 
      const auto x36 = -x28; 
      const auto x37 = radial_eval_alpha*x19 + x36; 
      const auto x38 = -x14 - x3; 
      const auto x39 = x15*x2; 
      const auto x40 = x*x22; 
      const auto x41 = radial_eval_alpha*x23; 
      const auto x42 = x31 + x41; 
      const auto x43 = x25 + x3; 
      const auto x44 = radial_eval*x43; 
      const auto x45 = x2*x26; 
      const auto x46 = x4*x6; 
      const auto x47 = radial_eval_alpha*x4; 
      const auto x48 = radial_eval + x47; 
      const auto x49 = -x13 - x17 - x2; 
      const auto x50 = x15*x4; 
      const auto x51 = x32 + x41; 
      const auto x52 = radial_eval_alpha*z; 
      const auto x53 = sqrt_15*y; 
      const auto x54 = radial_eval_alpha*x12; 
      const auto x55 = 8.0*radial_eval; 
      const auto x56 = x33 + x55; 
      const auto x57 = -x18 - x2 - x4; 
      const auto x58 = x12*x19; 
      const auto x59 = x12*x23; 
      const auto x60 = radial_eval_alpha_squared*x2; 
      const auto x61 = radial_eval_alpha + x60; 
      const auto x62 = x6*x61; 
      const auto x63 = 12.0*radial_eval_alpha; 
      const auto x64 = x2*x63; 
      const auto x65 = x28 + x64; 
      const auto x66 = 3.0*radial_eval_alpha; 
      const auto x67 = 4.0*radial_eval_alpha; 
      const auto x68 = x2*x67; 
      const auto x69 = x31 + x68; 
      const auto x70 = x15*x61; 
      const auto x71 = 2.0*radial_eval_alpha; 
      const auto x72 = x38*x71 + x70; 
      const auto x73 = x23*x61; 
      const auto x74 = x43*x71; 
      const auto x75 = x26*x61 + x74; 
      const auto x76 = 6.0*radial_eval_alpha; 
      const auto x77 = radial_eval_alpha*x43; 
      const auto x78 = radial_eval_alpha_squared*x46 + x77; 
      const auto x79 = radial_eval_alpha*x49 + radial_eval_alpha_squared*x50; 
      const auto x80 = radial_eval_alpha*x38 + radial_eval_alpha_squared*x39; 
      const auto x81 = radial_eval_alpha_squared*x45 + x77; 
      const auto x82 = x27*z; 
      const auto x83 = x30*z*(radial_eval_alpha_squared*x15 + x76); 
      const auto x84 = radial_eval_alpha_squared*x58 - x12*x76 + x36 + x57*x66; 
      const auto x85 = x10*z; 
      const auto x86 = 8.0*radial_eval_alpha; 
      const auto x87 = x12*x71; 
      const auto x88 = radial_eval_alpha_squared*x59; 
      const auto x89 = x0*z; 
      const auto x90 = radial_eval_alpha_squared*x4; 
      const auto x91 = radial_eval_alpha + x90; 
      const auto x92 = x6*x91 + x74; 
      const auto x93 = x15*x91; 
      const auto x94 = x49*x71 + x93; 
      const auto x95 = x4*x63; 
      const auto x96 = x28 + x95; 
      const auto x97 = x4*x67; 
      const auto x98 = x31 + x97; 
      const auto x99 = radial_eval_alpha_squared*x12; 
      const auto x100 = radial_eval_alpha + x99; 
      const auto x101 = x100*x6; 
      const auto x102 = 16.0*radial_eval_alpha*x12 + x100*x15; 
      const auto x103 = x102 + x55; 
      const auto x104 = x100*x19 + x57*x76; 
      const auto x105 = x23*(x100 + x71); 
      const auto x106 = x100*x26; 
      const auto x107 = -x95; 
      const auto x108 = -x97; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x1*x6;
      basis_eval[ipt + 1*npts] = radial_eval*x9;
      basis_eval[ipt + 2*npts] = radial_eval*x11*x15;
      basis_eval[ipt + 3*npts] = radial_eval*x16*x19;
      basis_eval[ipt + 4*npts] = radial_eval*x15*x20;
      basis_eval[ipt + 5*npts] = radial_eval*x22*x23;
      basis_eval[ipt + 6*npts] = radial_eval*x24*x26;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = x27*(radial_eval_alpha*x6 + x28);
      basis_x_eval[ipt + 1*npts] = x29*x8;
      basis_x_eval[ipt + 2*npts] = x34;
      basis_x_eval[ipt + 3*npts] = x35*x37;
      basis_x_eval[ipt + 4*npts] = x10*(radial_eval*x38 + radial_eval_alpha*x39);
      basis_x_eval[ipt + 5*npts] = x40*x42;
      basis_x_eval[ipt + 6*npts] = x0*(radial_eval_alpha*x45 + x44);

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*(radial_eval_alpha*x46 + x44);
      basis_y_eval[ipt + 1*npts] = x*x48*x7;
      basis_y_eval[ipt + 2*npts] = x10*(radial_eval*x49 + radial_eval_alpha*x50);
      basis_y_eval[ipt + 3*npts] = x16*x37*y;
      basis_y_eval[ipt + 4*npts] = x34;
      basis_y_eval[ipt + 5*npts] = x22*x51*y;
      basis_y_eval[ipt + 6*npts] = x27*(radial_eval_alpha*x26 + x36);

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x1*x52*x6;
      basis_z_eval[ipt + 1*npts] = x*x53*(radial_eval + x54);
      basis_z_eval[ipt + 2*npts] = x11*x56*z;
      basis_z_eval[ipt + 3*npts] = 1.5*radial_eval*x57 + 0.5*radial_eval_alpha*x58;
      basis_z_eval[ipt + 4*npts] = x20*x56*z;
      basis_z_eval[ipt + 5*npts] = x21*(radial_eval*x23 + radial_eval_alpha*x59);
      basis_z_eval[ipt + 6*npts] = x24*x26*x52;


      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x1*(x101 + x62 + x64 + x92);
      basis_lapl_eval[ipt + 1*npts] = x9*(9.0*radial_eval_alpha + x60 + x90 + x99);
      basis_lapl_eval[ipt + 2*npts] = x11*(x102 - x68 + x70 + x94);
      basis_lapl_eval[ipt + 3*npts] = x16*(x104 + x107 + x19*x61 + x19*x91 - x64);
      basis_lapl_eval[ipt + 4*npts] = x20*(x102 + x108 + x72 + x93);
      basis_lapl_eval[ipt + 5*npts] = x22*(x105 + x108 + x23*x91 + x68 + x73);
      basis_lapl_eval[ipt + 6*npts] = x24*(x106 + x107 + x26*x91 + x75);





#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x1*x6;
      ang_eval_1 = radial_eval*x9;
      ang_eval_2 = radial_eval*x11*x15;
      ang_eval_3 = radial_eval*x16*x19;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x15*x20;
      ang_eval_1 = radial_eval*x22*x23;
      ang_eval_2 = radial_eval*x24*x26;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = x27*(radial_eval_alpha*x6 + x28);
      dang_eval_y_0 = x0*(radial_eval_alpha*x46 + x44);
      dang_eval_z_0 = x1*x52*x6;
      dang_eval_x_1 = x29*x8;
      dang_eval_y_1 = x*x48*x7;
      dang_eval_z_1 = x*x53*(radial_eval + x54);
      dang_eval_x_2 = x34;
      dang_eval_y_2 = x10*(radial_eval*x49 + radial_eval_alpha*x50);
      dang_eval_z_2 = x11*x56*z;
      dang_eval_x_3 = x35*x37;
      dang_eval_y_3 = x16*x37*y;
      dang_eval_z_3 = 1.5*radial_eval*x57 + 0.5*radial_eval_alpha*x58;
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

      dang_eval_x_0 = x10*(radial_eval*x38 + radial_eval_alpha*x39);
      dang_eval_y_0 = x34;
      dang_eval_z_0 = x20*x56*z;
      dang_eval_x_1 = x40*x42;
      dang_eval_y_1 = x22*x51*y;
      dang_eval_z_1 = x21*(radial_eval*x23 + radial_eval_alpha*x59);
      dang_eval_x_2 = x0*(radial_eval_alpha*x45 + x44);
      dang_eval_y_2 = x27*(radial_eval_alpha*x26 + x36);
      dang_eval_z_2 = x24*x26*x52;
      basis_x_eval[ipt + 4*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 4*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 4*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 5*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 5*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 5*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 6*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 6*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 6*npts] = dang_eval_z_2;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
