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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_cartesian_laplacian_3(
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
      const auto x10 = x*x*x*x; 
      const auto x11 = 3.0*radial_eval; 
      const auto x12 = radial_eval_alpha*x0 + 2.0*x4; 
      const auto x13 = radial_eval*x5; 
      const auto x14 = x2*x5; 
      const auto x15 = radial_eval_alpha*x14; 
      const auto x16 = y*z; 
      const auto x17 = radial_eval_alpha*x2; 
      const auto x18 = radial_eval + x17; 
      const auto x19 = radial_eval*x7; 
      const auto x20 = x2*x7; 
      const auto x21 = radial_eval_alpha*x20; 
      const auto x22 = radial_eval_alpha*x; 
      const auto x23 = x22*x5*z; 
      const auto x24 = x22*x7*y; 
      const auto x25 = radial_eval_alpha*y; 
      const auto x26 = radial_eval*x2; 
      const auto x27 = radial_eval_alpha*x16*x2; 
      const auto x28 = radial_eval_alpha*x8 + 2.0*x1; 
      const auto x29 = radial_eval_alpha*x5; 
      const auto x30 = radial_eval + x29; 
      const auto x31 = y*y*y*y; 
      const auto x32 = x5*x7; 
      const auto x33 = radial_eval_alpha*x32; 
      const auto x34 = radial_eval_alpha*z; 
      const auto x35 = x*y; 
      const auto x36 = radial_eval_alpha*x7; 
      const auto x37 = radial_eval_alpha*x9 + 2.0*x3; 
      const auto x38 = z*z*z*z; 
      const auto x39 = 6.0*radial_eval_alpha; 
      const auto x40 = radial_eval_alpha_squared*x2; 
      const auto x41 = radial_eval_alpha + x40; 
      const auto x42 = x0*x39 + x0*x41 + 6.0*x4; 
      const auto x43 = 4.0*radial_eval_alpha; 
      const auto x44 = 2.0*radial_eval; 
      const auto x45 = x2*x41 + x44; 
      const auto x46 = x2*x43 + x45; 
      const auto x47 = 2.0*radial_eval_alpha; 
      const auto x48 = x47*x5; 
      const auto x49 = x41*x5; 
      const auto x50 = x*x16; 
      const auto x51 = 3.0*radial_eval_alpha; 
      const auto x52 = x47*x7; 
      const auto x53 = x41*x7; 
      const auto x54 = x41*x8; 
      const auto x55 = x41*x9; 
      const auto x56 = radial_eval_alpha_squared*x10 + x2*x51; 
      const auto x57 = 2.0*x22; 
      const auto x58 = x16*(radial_eval_alpha_squared*x0 + x57); 
      const auto x59 = 2.0*x25; 
      const auto x60 = radial_eval_alpha_squared*x14; 
      const auto x61 = x29 + x60; 
      const auto x62 = radial_eval_alpha_squared*x20; 
      const auto x63 = x36 + x62; 
      const auto x64 = radial_eval_alpha_squared*x31 + x5*x51; 
      const auto x65 = x6*(radial_eval_alpha_squared*x8 + x59); 
      const auto x66 = radial_eval_alpha_squared*x32; 
      const auto x67 = x36 + x66; 
      const auto x68 = 2.0*x34; 
      const auto x69 = x35*(radial_eval_alpha_squared*x9 + x68); 
      const auto x70 = radial_eval_alpha_squared*x38 + x51*x7; 
      const auto x71 = radial_eval_alpha_squared*x5; 
      const auto x72 = radial_eval_alpha + x71; 
      const auto x73 = x0*x72; 
      const auto x74 = x2*x47; 
      const auto x75 = x2*x72; 
      const auto x76 = x44 + x5*x72; 
      const auto x77 = x43*x5 + x76; 
      const auto x78 = x7*x72; 
      const auto x79 = 6.0*x1 + x39*x8 + x72*x8; 
      const auto x80 = x72*x9; 
      const auto x81 = radial_eval_alpha_squared*x7; 
      const auto x82 = radial_eval_alpha + x81; 
      const auto x83 = x0*x82; 
      const auto x84 = x2*x82; 
      const auto x85 = x5*x82; 
      const auto x86 = x44 + x7*x82; 
      const auto x87 = x43*x7 + x86; 
      const auto x88 = x8*x82; 
      const auto x89 = 6.0*x3 + x39*x9 + x82*x9; 
      const auto x90 = x2*x39 + x45 + x75 + x84; 
      const auto x91 = x39*x5 + x49 + x76 + x85; 
      const auto x92 = x39*x7 + x53 + x78 + x86; 


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
      basis_x_eval[ipt + 0*npts] = radial_eval_alpha*x10 + x11*x2;
      basis_x_eval[ipt + 1*npts] = x12*y;
      basis_x_eval[ipt + 2*npts] = x12*z;
      basis_x_eval[ipt + 3*npts] = x13 + x15;
      basis_x_eval[ipt + 4*npts] = x16*x18;
      basis_x_eval[ipt + 5*npts] = x19 + x21;
      basis_x_eval[ipt + 6*npts] = x22*x8;
      basis_x_eval[ipt + 7*npts] = x23;
      basis_x_eval[ipt + 8*npts] = x24;
      basis_x_eval[ipt + 9*npts] = x22*x9;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*x25;
      basis_y_eval[ipt + 1*npts] = x15 + x26;
      basis_y_eval[ipt + 2*npts] = x27;
      basis_y_eval[ipt + 3*npts] = x*x28;
      basis_y_eval[ipt + 4*npts] = x30*x6;
      basis_y_eval[ipt + 5*npts] = x24;
      basis_y_eval[ipt + 6*npts] = radial_eval_alpha*x31 + x11*x5;
      basis_y_eval[ipt + 7*npts] = x28*z;
      basis_y_eval[ipt + 8*npts] = x19 + x33;
      basis_y_eval[ipt + 9*npts] = x25*x9;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x0*x34;
      basis_z_eval[ipt + 1*npts] = x27;
      basis_z_eval[ipt + 2*npts] = x21 + x26;
      basis_z_eval[ipt + 3*npts] = x23;
      basis_z_eval[ipt + 4*npts] = x35*(radial_eval + x36);
      basis_z_eval[ipt + 5*npts] = x*x37;
      basis_z_eval[ipt + 6*npts] = x34*x8;
      basis_z_eval[ipt + 7*npts] = x13 + x33;
      basis_z_eval[ipt + 8*npts] = x37*y;
      basis_z_eval[ipt + 9*npts] = radial_eval_alpha*x38 + x11*x7;


      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x42 + x73 + x83;
      basis_lapl_eval[ipt + 1*npts] = x90*y;
      basis_lapl_eval[ipt + 2*npts] = x90*z;
      basis_lapl_eval[ipt + 3*npts] = x*x91;
      basis_lapl_eval[ipt + 4*npts] = x50*(9.0*radial_eval_alpha + x40 + x71 + x81);
      basis_lapl_eval[ipt + 5*npts] = x*x92;
      basis_lapl_eval[ipt + 6*npts] = x54 + x79 + x88;
      basis_lapl_eval[ipt + 7*npts] = x91*z;
      basis_lapl_eval[ipt + 8*npts] = x92*y;
      basis_lapl_eval[ipt + 9*npts] = x55 + x80 + x89;





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

      dang_eval_x_0 = radial_eval_alpha*x10 + x11*x2;
      dang_eval_y_0 = x0*x25;
      dang_eval_z_0 = x0*x34;
      dang_eval_x_1 = x12*y;
      dang_eval_y_1 = x15 + x26;
      dang_eval_z_1 = x27;
      dang_eval_x_2 = x12*z;
      dang_eval_y_2 = x27;
      dang_eval_z_2 = x21 + x26;
      dang_eval_x_3 = x13 + x15;
      dang_eval_y_3 = x*x28;
      dang_eval_z_3 = x23;
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

      dang_eval_x_0 = x16*x18;
      dang_eval_y_0 = x30*x6;
      dang_eval_z_0 = x35*(radial_eval + x36);
      dang_eval_x_1 = x19 + x21;
      dang_eval_y_1 = x24;
      dang_eval_z_1 = x*x37;
      dang_eval_x_2 = x22*x8;
      dang_eval_y_2 = radial_eval_alpha*x31 + x11*x5;
      dang_eval_z_2 = x34*x8;
      dang_eval_x_3 = x23;
      dang_eval_y_3 = x28*z;
      dang_eval_z_3 = x13 + x33;
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

      dang_eval_x_0 = x24;
      dang_eval_y_0 = x19 + x33;
      dang_eval_z_0 = x37*y;
      dang_eval_x_1 = x22*x9;
      dang_eval_y_1 = x25*x9;
      dang_eval_z_1 = radial_eval_alpha*x38 + x11*x7;
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
