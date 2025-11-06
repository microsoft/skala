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


__global__ __launch_bounds__(256,2) void collocation_device_shell_to_task_kernel_cartesian_lapgrad_0(
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
      const auto x1 = radial_eval_alpha_squared*x0; 
      const auto x2 = radial_eval_alpha_squared*x; 
      const auto x3 = y*y; 
      const auto x4 = radial_eval_alpha_squared*x3; 
      const auto x5 = radial_eval_alpha_squared*y; 
      const auto x6 = z*z; 
      const auto x7 = radial_eval_alpha_squared*x6; 
      const auto x8 = radial_eval_alpha_cubed*x; 
      const auto x9 = radial_eval_alpha_cubed*y; 
      const auto x10 = radial_eval_alpha_cubed*z; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = radial_eval_alpha*x;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = radial_eval_alpha*y;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = radial_eval_alpha*z;

      // Evaluate second derivative of bfn wrt xx
      basis_xx_eval[ipt + 0*npts] = radial_eval_alpha + x1;

      // Evaluate second derivative of bfn wrt xy
      basis_xy_eval[ipt + 0*npts] = x2*y;

      // Evaluate second derivative of bfn wrt xz
      basis_xz_eval[ipt + 0*npts] = x2*z;

      // Evaluate second derivative of bfn wrt yy
      basis_yy_eval[ipt + 0*npts] = radial_eval_alpha + x4;

      // Evaluate second derivative of bfn wrt yz
      basis_yz_eval[ipt + 0*npts] = x5*z;

      // Evaluate second derivative of bfn wrt zz
      basis_zz_eval[ipt + 0*npts] = radial_eval_alpha + x7;

      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = 3.0*radial_eval_alpha + x1 + x4 + x7;

      // Evaluate Laplacian gradient of bfn (dx)
      basis_lapl_x_eval[ipt + 0*npts] = radial_eval_alpha_cubed*(x*x*x) + 5.0*x2 + x3*x8 + x6*x8;
      // Evaluate Laplacian gradient of bfn (dy)
      basis_lapl_y_eval[ipt + 0*npts] = radial_eval_alpha_cubed*(y*y*y) + x0*x9 + 5.0*x5 + x6*x9;
      // Evaluate Laplacian gradient of bfn (dz)
      basis_lapl_z_eval[ipt + 0*npts] = radial_eval_alpha_cubed*(z*z*z) + 5.0*radial_eval_alpha_squared*z + x0*x10 + x10*x3;




#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;


      ang_eval_0 = radial_eval;
      basis_eval[ipt + 0*npts] = ang_eval_0;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;

      dang_eval_x_0 = radial_eval_alpha*x;
      dang_eval_y_0 = radial_eval_alpha*y;
      dang_eval_z_0 = radial_eval_alpha*z;
      basis_x_eval[ipt + 0*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 0*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 0*npts] = dang_eval_z_0;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
