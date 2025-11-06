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


__global__ __launch_bounds__(256,2) void collocation_device_shell_to_task_kernel_spherical_gradient_3(
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
      const auto x0 = 0.25*sqrt_10; 
      const auto x1 = radial_eval*y; 
      const auto x2 = x*x; 
      const auto x3 = 3.0*x2; 
      const auto x4 = y*y; 
      const auto x5 = -x4; 
      const auto x6 = x3 + x5; 
      const auto x7 = sqrt_15*z; 
      const auto x8 = x7*y; 
      const auto x9 = radial_eval*x; 
      const auto x10 = 0.25*sqrt_6; 
      const auto x11 = z*z; 
      const auto x12 = -4.0*x11; 
      const auto x13 = x12 + x4; 
      const auto x14 = -x13 - x2; 
      const auto x15 = 0.5*z; 
      const auto x16 = 3.0*x4; 
      const auto x17 = -2.0*x11; 
      const auto x18 = -x16 - x17 - x3; 
      const auto x19 = 0.5*sqrt_15; 
      const auto x20 = x19*z; 
      const auto x21 = x2 + x5; 
      const auto x22 = -x16; 
      const auto x23 = x2 + x22; 
      const auto x24 = x*y; 
      const auto x25 = x0*x24; 
      const auto x26 = 6.0*radial_eval; 
      const auto x27 = 2.0*radial_eval; 
      const auto x28 = -x27; 
      const auto x29 = radial_eval_alpha*x14; 
      const auto x30 = x10*x24*(x28 + x29); 
      const auto x31 = -x26; 
      const auto x32 = radial_eval_alpha*x18 + x31; 
      const auto x33 = radial_eval_alpha*x21; 
      const auto x34 = radial_eval*(x22 + x3); 
      const auto x35 = radial_eval_alpha*x0*z; 
      const auto x36 = x10*z; 
      const auto x37 = 8.0*radial_eval + x29; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = x0*x1*x6;
      basis_eval[ipt + 1*npts] = x8*x9;
      basis_eval[ipt + 2*npts] = x1*x10*x14;
      basis_eval[ipt + 3*npts] = radial_eval*x15*x18;
      basis_eval[ipt + 4*npts] = x10*x14*x9;
      basis_eval[ipt + 5*npts] = radial_eval*x20*x21;
      basis_eval[ipt + 6*npts] = x0*x23*x9;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = x25*(radial_eval_alpha*x6 + x26);
      basis_x_eval[ipt + 1*npts] = x8*(radial_eval + radial_eval_alpha*x2);
      basis_x_eval[ipt + 2*npts] = x30;
      basis_x_eval[ipt + 3*npts] = x*x15*x32;
      basis_x_eval[ipt + 4*npts] = -x10*(radial_eval*(x13 + x3) - radial_eval_alpha*x14*x2);
      basis_x_eval[ipt + 5*npts] = x*x20*(x27 + x33);
      basis_x_eval[ipt + 6*npts] = x0*(radial_eval_alpha*x2*x23 + x34);

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*(radial_eval_alpha*x4*x6 + x34);
      basis_y_eval[ipt + 1*npts] = x*x7*(radial_eval + radial_eval_alpha*x4);
      basis_y_eval[ipt + 2*npts] = -x10*(radial_eval*(x12 + x16 + x2) - radial_eval_alpha*x14*x4);
      basis_y_eval[ipt + 3*npts] = x15*x32*y;
      basis_y_eval[ipt + 4*npts] = x30;
      basis_y_eval[ipt + 5*npts] = x20*y*(x28 + x33);
      basis_y_eval[ipt + 6*npts] = x25*(radial_eval_alpha*x23 + x31);

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x35*x6*y;
      basis_z_eval[ipt + 1*npts] = sqrt_15*x24*(radial_eval + radial_eval_alpha*x11);
      basis_z_eval[ipt + 2*npts] = x36*x37*y;
      basis_z_eval[ipt + 3*npts] = -1.5*radial_eval*(x17 + x2 + x4) + 0.5*radial_eval_alpha*x11*x18;
      basis_z_eval[ipt + 4*npts] = x*x36*x37;
      basis_z_eval[ipt + 5*npts] = x19*x21*(radial_eval + radial_eval_alpha*x11);
      basis_z_eval[ipt + 6*npts] = x*x23*x35;







#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = x0*x1*x6;
      ang_eval_1 = x8*x9;
      ang_eval_2 = x1*x10*x14;
      ang_eval_3 = radial_eval*x15*x18;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = x10*x14*x9;
      ang_eval_1 = radial_eval*x20*x21;
      ang_eval_2 = x0*x23*x9;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = x25*(radial_eval_alpha*x6 + x26);
      dang_eval_y_0 = x0*(radial_eval_alpha*x4*x6 + x34);
      dang_eval_z_0 = x35*x6*y;
      dang_eval_x_1 = x8*(radial_eval + radial_eval_alpha*x2);
      dang_eval_y_1 = x*x7*(radial_eval + radial_eval_alpha*x4);
      dang_eval_z_1 = sqrt_15*x24*(radial_eval + radial_eval_alpha*x11);
      dang_eval_x_2 = x30;
      dang_eval_y_2 = -x10*(radial_eval*(x12 + x16 + x2) - radial_eval_alpha*x14*x4);
      dang_eval_z_2 = x36*x37*y;
      dang_eval_x_3 = x*x15*x32;
      dang_eval_y_3 = x15*x32*y;
      dang_eval_z_3 = -1.5*radial_eval*(x17 + x2 + x4) + 0.5*radial_eval_alpha*x11*x18;
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

      dang_eval_x_0 = -x10*(radial_eval*(x13 + x3) - radial_eval_alpha*x14*x2);
      dang_eval_y_0 = x30;
      dang_eval_z_0 = x*x36*x37;
      dang_eval_x_1 = x*x20*(x27 + x33);
      dang_eval_y_1 = x20*y*(x28 + x33);
      dang_eval_z_1 = x19*x21*(radial_eval + radial_eval_alpha*x11);
      dang_eval_x_2 = x0*(radial_eval_alpha*x2*x23 + x34);
      dang_eval_y_2 = x25*(radial_eval_alpha*x23 + x31);
      dang_eval_z_2 = x*x23*x35;
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
