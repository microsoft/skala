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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_spherical_laplacian_2(
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
      const auto x0 = sqrt_3*y; 
      const auto x1 = x*x0; 
      const auto x2 = x0*z; 
      const auto x3 = 0.5*radial_eval; 
      const auto x4 = x*x; 
      const auto x5 = y*y; 
      const auto x6 = z*z; 
      const auto x7 = -x4 - x5 + 2.0*x6; 
      const auto x8 = sqrt_3*z; 
      const auto x9 = x*x8; 
      const auto x10 = x4 - x5; 
      const auto x11 = radial_eval + radial_eval_alpha*x4; 
      const auto x12 = radial_eval_alpha*x1*z; 
      const auto x13 = 0.5*x; 
      const auto x14 = 2.0*radial_eval; 
      const auto x15 = -x14; 
      const auto x16 = radial_eval_alpha*x7; 
      const auto x17 = x15 + x16; 
      const auto x18 = radial_eval_alpha*x10; 
      const auto x19 = sqrt_3*x; 
      const auto x20 = radial_eval_alpha*x5; 
      const auto x21 = radial_eval + x20; 
      const auto x22 = 0.5*y; 
      const auto x23 = radial_eval_alpha*x6; 
      const auto x24 = radial_eval + x23; 
      const auto x25 = 0.5*z; 
      const auto x26 = 4.0*radial_eval; 
      const auto x27 = 3.0*radial_eval_alpha; 
      const auto x28 = radial_eval_alpha_squared*x4; 
      const auto x29 = x27 + x28; 
      const auto x30 = radial_eval_alpha + x28; 
      const auto x31 = x2*x30; 
      const auto x32 = 4.0*radial_eval_alpha; 
      const auto x33 = x32*x4; 
      const auto x34 = x14 + x33; 
      const auto x35 = 0.5*sqrt_3; 
      const auto x36 = x10*x30; 
      const auto x37 = radial_eval_alpha_squared*x5; 
      const auto x38 = radial_eval_alpha + x37; 
      const auto x39 = x38*x9; 
      const auto x40 = radial_eval_alpha_squared*x7; 
      const auto x41 = radial_eval_alpha_squared*x6; 
      const auto x42 = radial_eval_alpha + x41; 
      const auto x43 = x1*x42; 
      const auto x44 = 2.0*radial_eval_alpha; 
      const auto x45 = x40 + x44; 
      const auto x46 = radial_eval_alpha_squared*x10; 
      const auto x47 = x27 + x37; 
      const auto x48 = x32*x5; 
      const auto x49 = x14 + x48; 
      const auto x50 = x27 + x41; 
      const auto x51 = 8.0*radial_eval_alpha*x6 + x42*x7; 
      const auto x52 = x10*x42; 
      const auto x53 = 7.0*radial_eval_alpha + x28 + x37 + x41; 
      const auto x54 = -x48; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x1;
      basis_eval[ipt + 1*npts] = radial_eval*x2;
      basis_eval[ipt + 2*npts] = x3*x7;
      basis_eval[ipt + 3*npts] = radial_eval*x9;
      basis_eval[ipt + 4*npts] = sqrt_3*x10*x3;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = x0*x11;
      basis_x_eval[ipt + 1*npts] = x12;
      basis_x_eval[ipt + 2*npts] = x13*x17;
      basis_x_eval[ipt + 3*npts] = x11*x8;
      basis_x_eval[ipt + 4*npts] = sqrt_3*x13*(x14 + x18);

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x19*x21;
      basis_y_eval[ipt + 1*npts] = x21*x8;
      basis_y_eval[ipt + 2*npts] = x17*x22;
      basis_y_eval[ipt + 3*npts] = x12;
      basis_y_eval[ipt + 4*npts] = 0.5*x0*(x15 + x18);

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x12;
      basis_z_eval[ipt + 1*npts] = x0*x24;
      basis_z_eval[ipt + 2*npts] = x25*(x16 + x26);
      basis_z_eval[ipt + 3*npts] = x19*x24;
      basis_z_eval[ipt + 4*npts] = 0.5*radial_eval_alpha*x10*x8;


      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x1*x53;
      basis_lapl_eval[ipt + 1*npts] = x2*x53;
      basis_lapl_eval[ipt + 2*npts] = 0.5*x30*x7 - 0.5*x33 + 0.5*x38*x7 + 0.5*x51 + 0.5*x54;
      basis_lapl_eval[ipt + 3*npts] = x53*x9;
      basis_lapl_eval[ipt + 4*npts] = x35*(x10*x38 + x33 + x36 + x52 + x54);





#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x1;
      ang_eval_1 = radial_eval*x2;
      ang_eval_2 = x3*x7;
      ang_eval_3 = radial_eval*x9;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = sqrt_3*x10*x3;
      basis_eval[ipt + 4*npts] = ang_eval_0;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = x0*x11;
      dang_eval_y_0 = x19*x21;
      dang_eval_z_0 = x12;
      dang_eval_x_1 = x12;
      dang_eval_y_1 = x21*x8;
      dang_eval_z_1 = x0*x24;
      dang_eval_x_2 = x13*x17;
      dang_eval_y_2 = x17*x22;
      dang_eval_z_2 = x25*(x16 + x26);
      dang_eval_x_3 = x11*x8;
      dang_eval_y_3 = x12;
      dang_eval_z_3 = x19*x24;
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

      dang_eval_x_0 = sqrt_3*x13*(x14 + x18);
      dang_eval_y_0 = 0.5*x0*(x15 + x18);
      dang_eval_z_0 = 0.5*radial_eval_alpha*x10*x8;
      basis_x_eval[ipt + 4*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 4*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 4*npts] = dang_eval_z_0;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
