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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_spherical_gradient_4(
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
      const auto x0 = 0.5*y; 
      const auto x1 = sqrt_35*x0; 
      const auto x2 = radial_eval*x; 
      const auto x3 = x*x; 
      const auto x4 = y*y; 
      const auto x5 = -x4; 
      const auto x6 = x3 + x5; 
      const auto x7 = 0.25*z; 
      const auto x8 = sqrt_70*x7; 
      const auto x9 = radial_eval*y; 
      const auto x10 = 3.0*x3; 
      const auto x11 = x10 + x5; 
      const auto x12 = sqrt_5*x0; 
      const auto x13 = z*z; 
      const auto x14 = -6.0*x13; 
      const auto x15 = x14 + x4; 
      const auto x16 = -x15 - x3; 
      const auto x17 = sqrt_10*x7; 
      const auto x18 = -4.0*x13; 
      const auto x19 = 3.0*x4; 
      const auto x20 = x18 + x19; 
      const auto x21 = -x10 - x20; 
      const auto x22 = 0.125*radial_eval; 
      const auto x23 = x*x*x*x; 
      const auto x24 = y*y*y*y; 
      const auto x25 = 6.0*x3*x4; 
      const auto x26 = x13*x3; 
      const auto x27 = x13*x4; 
      const auto x28 = 3.0*x23 + 3.0*x24 + x25 - 24.0*x26 - 24.0*x27 + 8.0*(z*z*z*z); 
      const auto x29 = 0.25*sqrt_5; 
      const auto x30 = -x23 + x24 + 6.0*x26 - 6.0*x27; 
      const auto x31 = -x19; 
      const auto x32 = x3 + x31; 
      const auto x33 = x23 + x24 - x25; 
      const auto x34 = radial_eval*x11; 
      const auto x35 = x*y; 
      const auto x36 = x35*x8; 
      const auto x37 = 6.0*radial_eval; 
      const auto x38 = -x37; 
      const auto x39 = x17*x35*(radial_eval_alpha*x21 + x38); 
      const auto x40 = 12.0*radial_eval; 
      const auto x41 = x*x*x; 
      const auto x42 = radial_eval_alpha*x; 
      const auto x43 = 4.0*radial_eval; 
      const auto x44 = 3.0*x; 
      const auto x45 = radial_eval*(x10 + x31); 
      const auto x46 = 0.125*sqrt_35; 
      const auto x47 = 0.5*x; 
      const auto x48 = radial_eval*x32; 
      const auto x49 = y*y*y; 
      const auto x50 = radial_eval_alpha*y; 
      const auto x51 = 3.0*y; 
      const auto x52 = 0.25*y; 
      const auto x53 = -radial_eval*(x10 - 12.0*x13 + x19) + radial_eval_alpha*x13*x21; 
      const auto x54 = 3.0*z; 
      const auto x55 = radial_eval_alpha*z; 
      const auto x56 = 0.25*x; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = x1*x2*x6;
      basis_eval[ipt + 1*npts] = x11*x8*x9;
      basis_eval[ipt + 2*npts] = x12*x16*x2;
      basis_eval[ipt + 3*npts] = x17*x21*x9;
      basis_eval[ipt + 4*npts] = x22*x28;
      basis_eval[ipt + 5*npts] = x17*x2*x21;
      basis_eval[ipt + 6*npts] = radial_eval*x29*x30;
      basis_eval[ipt + 7*npts] = x2*x32*x8;
      basis_eval[ipt + 8*npts] = sqrt_35*x22*x33;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = x1*(radial_eval_alpha*x3*x6 + x34);
      basis_x_eval[ipt + 1*npts] = x36*(radial_eval_alpha*x11 + x37);
      basis_x_eval[ipt + 2*npts] = -x12*(radial_eval*(x10 + x15) - radial_eval_alpha*x16*x3);
      basis_x_eval[ipt + 3*npts] = x39;
      basis_x_eval[ipt + 4*npts] = 0.125*x28*x42 + 0.125*x40*(-4.0*x*x13 + x*x4 + x41);
      basis_x_eval[ipt + 5*npts] = -x17*(radial_eval*(x20 + 9.0*x3) - radial_eval_alpha*x21*x3);
      basis_x_eval[ipt + 6*npts] = x29*(x30*x42 + x43*(x13*x44 - x41));
      basis_x_eval[ipt + 7*npts] = x8*(radial_eval_alpha*x3*x32 + x45);
      basis_x_eval[ipt + 8*npts] = x46*(x33*x42 - x43*(x4*x44 - x41));

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = sqrt_35*x47*(radial_eval_alpha*x4*x6 + x48);
      basis_y_eval[ipt + 1*npts] = x8*(radial_eval_alpha*x11*x4 + x45);
      basis_y_eval[ipt + 2*npts] = -sqrt_5*x47*(radial_eval*(x14 + x19 + x3) - radial_eval_alpha*x16*x4);
      basis_y_eval[ipt + 3*npts] = -x17*(radial_eval*(x10 + x18 + 9.0*x4) - radial_eval_alpha*x21*x4);
      basis_y_eval[ipt + 4*npts] = 0.125*x28*x50 + 0.125*x40*(-4.0*x13*y + x3*y + x49);
      basis_y_eval[ipt + 5*npts] = x39;
      basis_y_eval[ipt + 6*npts] = x29*(x30*x50 - x43*(x13*x51 - x49));
      basis_y_eval[ipt + 7*npts] = x36*(radial_eval_alpha*x32 + x38);
      basis_y_eval[ipt + 8*npts] = x46*(x33*x50 - x43*(x3*x51 - x49));

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x1*x42*x6*z;
      basis_z_eval[ipt + 1*npts] = sqrt_70*x52*(radial_eval_alpha*x11*x13 + x34);
      basis_z_eval[ipt + 2*npts] = x*x12*z*(radial_eval_alpha*x16 + x40);
      basis_z_eval[ipt + 3*npts] = sqrt_10*x52*x53;
      basis_z_eval[ipt + 4*npts] = -2.0*radial_eval*(x3*x54 + x4*x54 - 2.0*z*z*z) + 0.125*x28*x55;
      basis_z_eval[ipt + 5*npts] = sqrt_10*x53*x56;
      basis_z_eval[ipt + 6*npts] = x29*z*(radial_eval_alpha*x30 + x40*x6);
      basis_z_eval[ipt + 7*npts] = sqrt_70*x56*(radial_eval_alpha*x13*x32 + x48);
      basis_z_eval[ipt + 8*npts] = x33*x46*x55;







#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = x1*x2*x6;
      ang_eval_1 = x11*x8*x9;
      ang_eval_2 = x12*x16*x2;
      ang_eval_3 = x17*x21*x9;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = x22*x28;
      ang_eval_1 = x17*x2*x21;
      ang_eval_2 = radial_eval*x29*x30;
      ang_eval_3 = x2*x32*x8;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = sqrt_35*x22*x33;
      basis_eval[ipt + 8*npts] = ang_eval_0;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = x1*(radial_eval_alpha*x3*x6 + x34);
      dang_eval_y_0 = sqrt_35*x47*(radial_eval_alpha*x4*x6 + x48);
      dang_eval_z_0 = x1*x42*x6*z;
      dang_eval_x_1 = x36*(radial_eval_alpha*x11 + x37);
      dang_eval_y_1 = x8*(radial_eval_alpha*x11*x4 + x45);
      dang_eval_z_1 = sqrt_70*x52*(radial_eval_alpha*x11*x13 + x34);
      dang_eval_x_2 = -x12*(radial_eval*(x10 + x15) - radial_eval_alpha*x16*x3);
      dang_eval_y_2 = -sqrt_5*x47*(radial_eval*(x14 + x19 + x3) - radial_eval_alpha*x16*x4);
      dang_eval_z_2 = x*x12*z*(radial_eval_alpha*x16 + x40);
      dang_eval_x_3 = x39;
      dang_eval_y_3 = -x17*(radial_eval*(x10 + x18 + 9.0*x4) - radial_eval_alpha*x21*x4);
      dang_eval_z_3 = sqrt_10*x52*x53;
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

      dang_eval_x_0 = 0.125*x28*x42 + 0.125*x40*(-4.0*x*x13 + x*x4 + x41);
      dang_eval_y_0 = 0.125*x28*x50 + 0.125*x40*(-4.0*x13*y + x3*y + x49);
      dang_eval_z_0 = -2.0*radial_eval*(x3*x54 + x4*x54 - 2.0*z*z*z) + 0.125*x28*x55;
      dang_eval_x_1 = -x17*(radial_eval*(x20 + 9.0*x3) - radial_eval_alpha*x21*x3);
      dang_eval_y_1 = x39;
      dang_eval_z_1 = sqrt_10*x53*x56;
      dang_eval_x_2 = x29*(x30*x42 + x43*(x13*x44 - x41));
      dang_eval_y_2 = x29*(x30*x50 - x43*(x13*x51 - x49));
      dang_eval_z_2 = x29*z*(radial_eval_alpha*x30 + x40*x6);
      dang_eval_x_3 = x8*(radial_eval_alpha*x3*x32 + x45);
      dang_eval_y_3 = x36*(radial_eval_alpha*x32 + x38);
      dang_eval_z_3 = sqrt_70*x56*(radial_eval_alpha*x13*x32 + x48);
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

      dang_eval_x_0 = x46*(x33*x42 - x43*(x4*x44 - x41));
      dang_eval_y_0 = x46*(x33*x50 - x43*(x3*x51 - x49));
      dang_eval_z_0 = x33*x46*x55;
      basis_x_eval[ipt + 8*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 8*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 8*npts] = dang_eval_z_0;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
