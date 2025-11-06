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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_cartesian_gradient_4(
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
      const auto x0 = x*x*x*x; 
      const auto x1 = radial_eval*y; 
      const auto x2 = x*x*x; 
      const auto x3 = radial_eval*z; 
      const auto x4 = x*x; 
      const auto x5 = y*y; 
      const auto x6 = x4*x5; 
      const auto x7 = z*z; 
      const auto x8 = x4*x7; 
      const auto x9 = radial_eval*x; 
      const auto x10 = y*y*y; 
      const auto x11 = z*z*z; 
      const auto x12 = y*y*y*y; 
      const auto x13 = x5*x7; 
      const auto x14 = z*z*z*z; 
      const auto x15 = 4.0*radial_eval; 
      const auto x16 = 3.0*radial_eval; 
      const auto x17 = radial_eval_alpha*x0 + x16*x4; 
      const auto x18 = 2.0*x9; 
      const auto x19 = radial_eval_alpha*x2*x5; 
      const auto x20 = y*z; 
      const auto x21 = radial_eval_alpha*x2*x7; 
      const auto x22 = radial_eval*x10; 
      const auto x23 = radial_eval_alpha*x10*x4; 
      const auto x24 = radial_eval*x5; 
      const auto x25 = radial_eval_alpha*x6; 
      const auto x26 = radial_eval*x7; 
      const auto x27 = radial_eval_alpha*x8; 
      const auto x28 = radial_eval*x11; 
      const auto x29 = radial_eval_alpha*x11*x4; 
      const auto x30 = radial_eval_alpha*x; 
      const auto x31 = x10*x30*z; 
      const auto x32 = x11*x30*y; 
      const auto x33 = radial_eval_alpha*y; 
      const auto x34 = radial_eval*x2; 
      const auto x35 = radial_eval_alpha*x2*x20; 
      const auto x36 = 2.0*x1; 
      const auto x37 = radial_eval*x4; 
      const auto x38 = radial_eval_alpha*x12 + x16*x5; 
      const auto x39 = radial_eval_alpha*x13; 
      const auto x40 = radial_eval_alpha*x10*x7; 
      const auto x41 = radial_eval_alpha*x11*x5; 
      const auto x42 = radial_eval_alpha*z; 
      const auto x43 = 2.0*x3; 
      const auto x44 = radial_eval_alpha*x14 + x16*x7; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x0;
      basis_eval[ipt + 1*npts] = x1*x2;
      basis_eval[ipt + 2*npts] = x2*x3;
      basis_eval[ipt + 3*npts] = radial_eval*x6;
      basis_eval[ipt + 4*npts] = x1*x4*z;
      basis_eval[ipt + 5*npts] = radial_eval*x8;
      basis_eval[ipt + 6*npts] = x10*x9;
      basis_eval[ipt + 7*npts] = x*x3*x5;
      basis_eval[ipt + 8*npts] = x*x1*x7;
      basis_eval[ipt + 9*npts] = x11*x9;
      basis_eval[ipt + 10*npts] = radial_eval*x12;
      basis_eval[ipt + 11*npts] = x10*x3;
      basis_eval[ipt + 12*npts] = radial_eval*x13;
      basis_eval[ipt + 13*npts] = x1*x11;
      basis_eval[ipt + 14*npts] = radial_eval*x14;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = radial_eval_alpha*(x*x*x*x*x) + x15*x2;
      basis_x_eval[ipt + 1*npts] = x17*y;
      basis_x_eval[ipt + 2*npts] = x17*z;
      basis_x_eval[ipt + 3*npts] = x18*x5 + x19;
      basis_x_eval[ipt + 4*npts] = x20*(radial_eval_alpha*x2 + x18);
      basis_x_eval[ipt + 5*npts] = x18*x7 + x21;
      basis_x_eval[ipt + 6*npts] = x22 + x23;
      basis_x_eval[ipt + 7*npts] = z*(x24 + x25);
      basis_x_eval[ipt + 8*npts] = y*(x26 + x27);
      basis_x_eval[ipt + 9*npts] = x28 + x29;
      basis_x_eval[ipt + 10*npts] = x12*x30;
      basis_x_eval[ipt + 11*npts] = x31;
      basis_x_eval[ipt + 12*npts] = x13*x30;
      basis_x_eval[ipt + 13*npts] = x32;
      basis_x_eval[ipt + 14*npts] = x14*x30;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*x33;
      basis_y_eval[ipt + 1*npts] = x19 + x34;
      basis_y_eval[ipt + 2*npts] = x35;
      basis_y_eval[ipt + 3*npts] = x23 + x36*x4;
      basis_y_eval[ipt + 4*npts] = z*(x25 + x37);
      basis_y_eval[ipt + 5*npts] = x33*x8;
      basis_y_eval[ipt + 6*npts] = x*x38;
      basis_y_eval[ipt + 7*npts] = x*z*(radial_eval_alpha*x10 + x36);
      basis_y_eval[ipt + 8*npts] = x*(x26 + x39);
      basis_y_eval[ipt + 9*npts] = x32;
      basis_y_eval[ipt + 10*npts] = radial_eval_alpha*(y*y*y*y*y) + x10*x15;
      basis_y_eval[ipt + 11*npts] = x38*z;
      basis_y_eval[ipt + 12*npts] = x36*x7 + x40;
      basis_y_eval[ipt + 13*npts] = x28 + x41;
      basis_y_eval[ipt + 14*npts] = x14*x33;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x0*x42;
      basis_z_eval[ipt + 1*npts] = x35;
      basis_z_eval[ipt + 2*npts] = x21 + x34;
      basis_z_eval[ipt + 3*npts] = x42*x6;
      basis_z_eval[ipt + 4*npts] = y*(x27 + x37);
      basis_z_eval[ipt + 5*npts] = x29 + x4*x43;
      basis_z_eval[ipt + 6*npts] = x31;
      basis_z_eval[ipt + 7*npts] = x*(x24 + x39);
      basis_z_eval[ipt + 8*npts] = x*y*(radial_eval_alpha*x11 + x43);
      basis_z_eval[ipt + 9*npts] = x*x44;
      basis_z_eval[ipt + 10*npts] = x12*x42;
      basis_z_eval[ipt + 11*npts] = x22 + x40;
      basis_z_eval[ipt + 12*npts] = x41 + x43*x5;
      basis_z_eval[ipt + 13*npts] = x44*y;
      basis_z_eval[ipt + 14*npts] = radial_eval_alpha*(z*z*z*z*z) + x11*x15;







#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x0;
      ang_eval_1 = x1*x2;
      ang_eval_2 = x2*x3;
      ang_eval_3 = radial_eval*x6;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = x1*x4*z;
      ang_eval_1 = radial_eval*x8;
      ang_eval_2 = x10*x9;
      ang_eval_3 = x*x3*x5;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = x*x1*x7;
      ang_eval_1 = x11*x9;
      ang_eval_2 = radial_eval*x12;
      ang_eval_3 = x10*x3;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;
      basis_eval[ipt + 11*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x13;
      ang_eval_1 = x1*x11;
      ang_eval_2 = radial_eval*x14;
      basis_eval[ipt + 12*npts] = ang_eval_0;
      basis_eval[ipt + 13*npts] = ang_eval_1;
      basis_eval[ipt + 14*npts] = ang_eval_2;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = radial_eval_alpha*(x*x*x*x*x) + x15*x2;
      dang_eval_y_0 = x0*x33;
      dang_eval_z_0 = x0*x42;
      dang_eval_x_1 = x17*y;
      dang_eval_y_1 = x19 + x34;
      dang_eval_z_1 = x35;
      dang_eval_x_2 = x17*z;
      dang_eval_y_2 = x35;
      dang_eval_z_2 = x21 + x34;
      dang_eval_x_3 = x18*x5 + x19;
      dang_eval_y_3 = x23 + x36*x4;
      dang_eval_z_3 = x42*x6;
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

      dang_eval_x_0 = x20*(radial_eval_alpha*x2 + x18);
      dang_eval_y_0 = z*(x25 + x37);
      dang_eval_z_0 = y*(x27 + x37);
      dang_eval_x_1 = x18*x7 + x21;
      dang_eval_y_1 = x33*x8;
      dang_eval_z_1 = x29 + x4*x43;
      dang_eval_x_2 = x22 + x23;
      dang_eval_y_2 = x*x38;
      dang_eval_z_2 = x31;
      dang_eval_x_3 = z*(x24 + x25);
      dang_eval_y_3 = x*z*(radial_eval_alpha*x10 + x36);
      dang_eval_z_3 = x*(x24 + x39);
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

      dang_eval_x_0 = y*(x26 + x27);
      dang_eval_y_0 = x*(x26 + x39);
      dang_eval_z_0 = x*y*(radial_eval_alpha*x11 + x43);
      dang_eval_x_1 = x28 + x29;
      dang_eval_y_1 = x32;
      dang_eval_z_1 = x*x44;
      dang_eval_x_2 = x12*x30;
      dang_eval_y_2 = radial_eval_alpha*(y*y*y*y*y) + x10*x15;
      dang_eval_z_2 = x12*x42;
      dang_eval_x_3 = x31;
      dang_eval_y_3 = x38*z;
      dang_eval_z_3 = x22 + x40;
      basis_x_eval[ipt + 8*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 8*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 8*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 9*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 9*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 9*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 10*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 10*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 10*npts] = dang_eval_z_2;
      basis_x_eval[ipt + 11*npts] = dang_eval_x_3;
      basis_y_eval[ipt + 11*npts] = dang_eval_y_3;
      basis_z_eval[ipt + 11*npts] = dang_eval_z_3;

      dang_eval_x_0 = x13*x30;
      dang_eval_y_0 = x36*x7 + x40;
      dang_eval_z_0 = x41 + x43*x5;
      dang_eval_x_1 = x32;
      dang_eval_y_1 = x28 + x41;
      dang_eval_z_1 = x44*y;
      dang_eval_x_2 = x14*x30;
      dang_eval_y_2 = x14*x33;
      dang_eval_z_2 = radial_eval_alpha*(z*z*z*z*z) + x11*x15;
      basis_x_eval[ipt + 12*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 12*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 12*npts] = dang_eval_z_0;
      basis_x_eval[ipt + 13*npts] = dang_eval_x_1;
      basis_y_eval[ipt + 13*npts] = dang_eval_y_1;
      basis_z_eval[ipt + 13*npts] = dang_eval_z_1;
      basis_x_eval[ipt + 14*npts] = dang_eval_x_2;
      basis_y_eval[ipt + 14*npts] = dang_eval_y_2;
      basis_z_eval[ipt + 14*npts] = dang_eval_z_2;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
