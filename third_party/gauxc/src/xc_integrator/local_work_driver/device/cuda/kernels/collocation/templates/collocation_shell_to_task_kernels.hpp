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

$py(do_grad = 'gradient' in type or 'hessian' in type or 'lapl' in type or 'lapgrad' in type)\
$py(do_hess = 'hessian' in type or 'lapgrad' in type)\
$py(do_lapl = 'lapl' in type or 'lapgrad' in type)\
$py(do_lapl_grad = 'lapgrad' in type)\

__global__ __launch_bounds__($(nt),2) void collocation_device_shell_to_task_kernel_$(type)_$(L)(
  uint32_t                        nshell,
  ShellToTaskDevice* __restrict__ shell_to_task,
  XCDeviceTask*      __restrict__ device_tasks
) {


  __shared__ double alpha[$(nt//32)][detail::shell_nprim_max + 1]; 
  __shared__ double coeff[$(nt//32)][detail::shell_nprim_max + 1];
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
$if( do_grad )\
    auto* __restrict__ basis_x_eval = task->dbfx + shoff;
    auto* __restrict__ basis_y_eval = task->dbfy + shoff;
    auto* __restrict__ basis_z_eval = task->dbfz + shoff;
$endif\
$if( do_hess )\
    auto* __restrict__ basis_xx_eval = task->d2bfxx + shoff;
    auto* __restrict__ basis_xy_eval = task->d2bfxy + shoff;
    auto* __restrict__ basis_xz_eval = task->d2bfxz + shoff;
    auto* __restrict__ basis_yy_eval = task->d2bfyy + shoff;
    auto* __restrict__ basis_yz_eval = task->d2bfyz + shoff;
    auto* __restrict__ basis_zz_eval = task->d2bfzz + shoff;
$endif\
$if( do_lapl )\
    auto* __restrict__ basis_lapl_eval = task->d2bflapl + shoff;
$endif\
$if( do_lapl_grad )\
    auto* __restrict__ basis_lapl_x_eval = task->d3bflapl_x + shoff;
    auto* __restrict__ basis_lapl_y_eval = task->d3bflapl_y + shoff;
    auto* __restrict__ basis_lapl_z_eval = task->d3bflapl_z + shoff;
$endif\

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
$if( do_grad )\
      double radial_eval_alpha = 0.;
$endif\
$if( do_hess or do_lapl)\
      double radial_eval_alpha_squared = 0.;
$endif\
$if( do_lapl_grad)\
      double radial_eval_alpha_cubed = 0.;
$endif\

      #pragma unroll 1
      for( uint32_t i = 0; i < nprim; ++i ) {
        const auto a = my_alpha[i];
        const auto e = my_coeff[i] * std::exp( - a * rsq );

        radial_eval += e;
$if( do_grad )\
        radial_eval_alpha += a * e;
$endif\
$if( do_hess or do_lapl)\
        radial_eval_alpha_squared += a * a * e;
$endif\
$if( do_lapl_grad)\
        radial_eval_alpha_cubed += a * a * a * e;
$endif\
      }

$if( do_grad )\
      radial_eval_alpha *= -2;
$endif\
$if( do_hess or do_lapl)\
      radial_eval_alpha_squared *= 4;
$endif\
$if( do_lapl_grad )\
      radial_eval_alpha_cubed *= -8;
$endif\

      // Common Subexpressions
$for( i in range(len(common_lines)) )\
      const auto $(common_lines[i][0]) = $(common_lines[i][1]); 
$endfor

      // Evaluate basis function
$for( j in range(len(eval_lines)) )\
      basis_eval[ipt + $(j)*npts] = $(eval_lines[j]);
$endfor

    
$if(do_grad)\
      // Evaluate first derivative of bfn wrt x
$for( j in range(len(eval_lines_dx)) )\
      basis_x_eval[ipt + $(j)*npts] = $(eval_lines_dx[j]);
$endfor\

      // Evaluate first derivative of bfn wrt y
$for( j in range(len(eval_lines_dy)) )\
      basis_y_eval[ipt + $(j)*npts] = $(eval_lines_dy[j]);
$endfor\

      // Evaluate first derivative of bfn wrt z
$for( j in range(len(eval_lines_dz)) )\
      basis_z_eval[ipt + $(j)*npts] = $(eval_lines_dz[j]);
$endfor\
$endif\

$if(do_hess)\
      // Evaluate second derivative of bfn wrt xx
$for( j in range(len(eval_lines_dxx)) )\
      basis_xx_eval[ipt + $(j)*npts] = $(eval_lines_dxx[j]);
$endfor\

      // Evaluate second derivative of bfn wrt xy
$for( j in range(len(eval_lines_dxy)) )\
      basis_xy_eval[ipt + $(j)*npts] = $(eval_lines_dxy[j]);
$endfor\

      // Evaluate second derivative of bfn wrt xz
$for( j in range(len(eval_lines_dxz)) )\
      basis_xz_eval[ipt + $(j)*npts] = $(eval_lines_dxz[j]);
$endfor\

      // Evaluate second derivative of bfn wrt yy
$for( j in range(len(eval_lines_dyy)) )\
      basis_yy_eval[ipt + $(j)*npts] = $(eval_lines_dyy[j]);
$endfor\

      // Evaluate second derivative of bfn wrt yz
$for( j in range(len(eval_lines_dyz)) )\
      basis_yz_eval[ipt + $(j)*npts] = $(eval_lines_dyz[j]);
$endfor\

      // Evaluate second derivative of bfn wrt zz
$for( j in range(len(eval_lines_dzz)) )\
      basis_zz_eval[ipt + $(j)*npts] = $(eval_lines_dzz[j]);
$endfor\
$endif\

$if(do_lapl)\
      // Evaluate Laplacian of bfn 
$for( j in range(len(eval_lines_lapl)) )\
      basis_lapl_eval[ipt + $(j)*npts] = $(eval_lines_lapl[j]);
$endfor\
$endif\

$if(do_lapl_grad)\
      // Evaluate Laplacian gradient of bfn (dx)
$for( j in range(len(eval_lines_lapl_x)) )\
      basis_lapl_x_eval[ipt + $(j)*npts] = $(eval_lines_lapl_x[j]);
$endfor\
      // Evaluate Laplacian gradient of bfn (dy)
$for( j in range(len(eval_lines_lapl_y)) )\
      basis_lapl_y_eval[ipt + $(j)*npts] = $(eval_lines_lapl_y[j]);
$endfor\
      // Evaluate Laplacian gradient of bfn (dz)
$for( j in range(len(eval_lines_lapl_z)) )\
      basis_lapl_z_eval[ipt + $(j)*npts] = $(eval_lines_lapl_z[j]);
$endfor\
$endif\




#if 0
      // Evaluate the angular part of bfn

$py(unroll_max = min(len(eval_lines),4))

$for( i in range(unroll_max) )\
      double ang_eval_$(i);
$endfor\

$py(unroll_loop_ceil = len(eval_lines)//unroll_max)
$py(idx_st = unroll_loop_ceil*unroll_max)\
$for( i in range(unroll_loop_ceil) )\
$for( j in range(unroll_max) )\
      ang_eval_$(j) = $(eval_lines[i*unroll_max + j]);
$endfor\
$for( j in range(unroll_max) )\
      basis_eval[ipt + $(i*unroll_max + j)*npts] = ang_eval_$(j);
$endfor\

$endfor\
$if( len(eval_lines)%unroll_max )\
$for( j in range(len(eval_lines)%unroll_max) )\
      ang_eval_$(j) = $(eval_lines[idx_st + j]);
$endfor\
$for( j in range(len(eval_lines)%unroll_max) )\
      basis_eval[ipt + $(idx_st + j)*npts] = ang_eval_$(j);
$endfor\

$endif\

$if(do_grad)\
$for( i in range(unroll_max) )\
      double dang_eval_x_$(i), dang_eval_y_$(i), dang_eval_z_$(i);
$endfor\

$for( i in range(unroll_loop_ceil) )\
$for( j in range(unroll_max) )\
      dang_eval_x_$(j) = $(eval_lines_dx[i*unroll_max + j]);
      dang_eval_y_$(j) = $(eval_lines_dy[i*unroll_max + j]);
      dang_eval_z_$(j) = $(eval_lines_dz[i*unroll_max + j]);
$endfor\
$for( j in range(unroll_max) )\
      basis_x_eval[ipt + $(i*unroll_max + j)*npts] = dang_eval_x_$(j);
      basis_y_eval[ipt + $(i*unroll_max + j)*npts] = dang_eval_y_$(j);
      basis_z_eval[ipt + $(i*unroll_max + j)*npts] = dang_eval_z_$(j);
$endfor\

$endfor\
$if( len(eval_lines)%unroll_max )\
$for( j in range(len(eval_lines)%unroll_max) )\
      dang_eval_x_$(j) = $(eval_lines_dx[idx_st + j]);
      dang_eval_y_$(j) = $(eval_lines_dy[idx_st + j]);
      dang_eval_z_$(j) = $(eval_lines_dz[idx_st + j]);
$endfor\
$for( j in range(len(eval_lines)%unroll_max) )\
      basis_x_eval[ipt + $(idx_st + j)*npts] = dang_eval_x_$(j);
      basis_y_eval[ipt + $(idx_st + j)*npts] = dang_eval_y_$(j);
      basis_z_eval[ipt + $(idx_st + j)*npts] = dang_eval_z_$(j);
$endfor\

$endif\
$endif\
#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
