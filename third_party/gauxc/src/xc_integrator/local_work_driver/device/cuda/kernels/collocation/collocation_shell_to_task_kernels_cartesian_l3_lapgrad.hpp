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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_cartesian_lapgrad_3(
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
      const auto x0 = x*x*x; 
      const auto x1 = radial_eval*y; 
      const auto x2 = x*x; 
      const auto x3 = x2; 
      const auto x4 = radial_eval*z; 
      const auto x5 = radial_eval*x; 
      const auto x6 = y*y; 
      const auto x7 = x6; 
      const auto x8 = x*z; 
      const auto x9 = z*z; 
      const auto x10 = x9; 
      const auto x11 = y*y*y; 
      const auto x12 = z*z*z; 
      const auto x13 = x*x*x*x; 
      const auto x14 = 3.0*radial_eval; 
      const auto x15 = radial_eval_alpha*x0 + 2.0*x5; 
      const auto x16 = radial_eval*x7; 
      const auto x17 = x3*x7; 
      const auto x18 = radial_eval_alpha*x17; 
      const auto x19 = y*z; 
      const auto x20 = radial_eval_alpha*x3; 
      const auto x21 = radial_eval + x20; 
      const auto x22 = radial_eval*x10; 
      const auto x23 = x10*x3; 
      const auto x24 = radial_eval_alpha*x23; 
      const auto x25 = radial_eval_alpha*x; 
      const auto x26 = x25*x7*z; 
      const auto x27 = x10*x25*y; 
      const auto x28 = radial_eval_alpha*y; 
      const auto x29 = radial_eval*x3; 
      const auto x30 = radial_eval_alpha*x19*x3; 
      const auto x31 = radial_eval_alpha*x11 + 2.0*x1; 
      const auto x32 = radial_eval_alpha*x7; 
      const auto x33 = radial_eval + x32; 
      const auto x34 = y*y*y*y; 
      const auto x35 = x10*x7; 
      const auto x36 = radial_eval_alpha*x35; 
      const auto x37 = radial_eval_alpha*z; 
      const auto x38 = x*y; 
      const auto x39 = radial_eval_alpha*x10; 
      const auto x40 = radial_eval_alpha*x12 + 2.0*x4; 
      const auto x41 = z*z*z*z; 
      const auto x42 = 6.0*radial_eval_alpha; 
      const auto x43 = radial_eval_alpha_squared*x3; 
      const auto x44 = radial_eval_alpha + x43; 
      const auto x45 = x0*x42 + x0*x44 + 6.0*x5; 
      const auto x46 = 4.0*radial_eval_alpha; 
      const auto x47 = 2.0*radial_eval; 
      const auto x48 = x3*x44; 
      const auto x49 = x47 + x48; 
      const auto x50 = x3*x46 + x49; 
      const auto x51 = 2.0*radial_eval_alpha; 
      const auto x52 = x51*x7; 
      const auto x53 = x44*x7; 
      const auto x54 = x*x19; 
      const auto x55 = 3.0*radial_eval_alpha; 
      const auto x56 = x10*x51; 
      const auto x57 = x10*x44; 
      const auto x58 = x11*x44; 
      const auto x59 = x12*x44; 
      const auto x60 = radial_eval_alpha_squared*x13 + x3*x55; 
      const auto x61 = 2.0*x25; 
      const auto x62 = x19*(radial_eval_alpha_squared*x0 + x61); 
      const auto x63 = 2.0*x28; 
      const auto x64 = radial_eval_alpha_squared*x17; 
      const auto x65 = x32 + x64; 
      const auto x66 = radial_eval_alpha_squared*x23; 
      const auto x67 = x39 + x66; 
      const auto x68 = radial_eval_alpha_squared*x34 + x55*x7; 
      const auto x69 = x8*(radial_eval_alpha_squared*x11 + x63); 
      const auto x70 = radial_eval_alpha_squared*x35; 
      const auto x71 = x39 + x70; 
      const auto x72 = 2.0*x37; 
      const auto x73 = x38*(radial_eval_alpha_squared*x12 + x72); 
      const auto x74 = radial_eval_alpha_squared*x41 + x10*x55; 
      const auto x75 = radial_eval_alpha_squared*x7; 
      const auto x76 = radial_eval_alpha + x75; 
      const auto x77 = x0*x76; 
      const auto x78 = x3*x51; 
      const auto x79 = x3*x76; 
      const auto x80 = x7*x76; 
      const auto x81 = x47 + x80; 
      const auto x82 = x46*x7 + x81; 
      const auto x83 = x10*x76; 
      const auto x84 = 6.0*x1 + x11*x42 + x11*x76; 
      const auto x85 = x12*x76; 
      const auto x86 = radial_eval_alpha_squared*x10; 
      const auto x87 = radial_eval_alpha + x86; 
      const auto x88 = x0*x87; 
      const auto x89 = x3*x87; 
      const auto x90 = x7*x87; 
      const auto x91 = x10*x87; 
      const auto x92 = x47 + x91; 
      const auto x93 = x10*x46 + x92; 
      const auto x94 = x11*x87; 
      const auto x95 = x12*x42 + x12*x87 + 6.0*x4; 
      const auto x96 = x3*x42 + x49 + x79 + x89; 
      const auto x97 = x42*x7 + x53 + x81 + x90; 
      const auto x98 = x75 + x86; 
      const auto x99 = x10*x42 + x57 + x83 + x92; 
      const auto x100 = 6.0*radial_eval; 
      const auto x101 = 18.0*radial_eval_alpha; 
      const auto x102 = 3.0*x79; 
      const auto x103 = 3.0*x89; 
      const auto x104 = radial_eval_alpha_cubed*x7 + radial_eval_alpha_squared; 
      const auto x105 = x0*x104; 
      const auto x106 = radial_eval_alpha_cubed*x10 + radial_eval_alpha_squared; 
      const auto x107 = x0*x106; 
      const auto x108 = 3.0*radial_eval_alpha_squared; 
      const auto x109 = radial_eval_alpha_cubed*x0 + x*x108; 
      const auto x110 = 2.0*radial_eval_alpha_squared; 
      const auto x111 = 6.0*x; 
      const auto x112 = 2.0*x; 
      const auto x113 = x104*x3; 
      const auto x114 = x106*x3; 
      const auto x115 = x*x113 + x*x114 + x0*x110 + x109*x3 + x111*x44 + x112*x76 + x112*x87 + 10.0*x25; 
      const auto x116 = 3.0*x53; 
      const auto x117 = x109*x7; 
      const auto x118 = x104*x7; 
      const auto x119 = x106*x7; 
      const auto x120 = 4.0*radial_eval_alpha_squared; 
      const auto x121 = x120*x17; 
      const auto x122 = 3.0*x57; 
      const auto x123 = x10*x109; 
      const auto x124 = x10*x104; 
      const auto x125 = x10*x106; 
      const auto x126 = x120*x23; 
      const auto x127 = 6.0*y; 
      const auto x128 = x127*x25; 
      const auto x129 = radial_eval_alpha_squared*x111; 
      const auto x130 = x104*x11; 
      const auto x131 = x106*x11; 
      const auto x132 = 6.0*z; 
      const auto x133 = x132*x25; 
      const auto x134 = x104*x12; 
      const auto x135 = x106*x12; 
      const auto x136 = radial_eval_alpha_squared*x127; 
      const auto x137 = radial_eval_alpha_cubed*x3 + radial_eval_alpha_squared; 
      const auto x138 = x0*x137; 
      const auto x139 = radial_eval_alpha_cubed*x11 + x108*y; 
      const auto x140 = x139*x3; 
      const auto x141 = x137*x3; 
      const auto x142 = 2.0*y; 
      const auto x143 = x137*x7; 
      const auto x144 = x11*x110 + x119*y + x127*x76 + x139*x7 + x142*x44 + x142*x87 + x143*y + 10.0*x28; 
      const auto x145 = x42 + x43; 
      const auto x146 = x10*x137; 
      const auto x147 = x10*x139; 
      const auto x148 = 3.0*x90; 
      const auto x149 = x11*x137; 
      const auto x150 = 3.0*x83; 
      const auto x151 = x120*x35; 
      const auto x152 = x19*x42; 
      const auto x153 = x12*x137; 
      const auto x154 = radial_eval_alpha_squared*x132; 
      const auto x155 = radial_eval_alpha_cubed*x12 + x108*z; 
      const auto x156 = x155*x3; 
      const auto x157 = x155*x7; 
      const auto x158 = 2.0*z; 
      const auto x159 = x10*x155 + x110*x12 + x124*z + x132*x87 + x146*z + x158*x44 + x158*x76 + 10.0*x37; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x0;
      basis_eval[ipt + 1*npts] = x1*x3;
      basis_eval[ipt + 2*npts] = x3*x4;
      basis_eval[ipt + 3*npts] = x5*x7;
      basis_eval[ipt + 4*npts] = x1*x8;
      basis_eval[ipt + 5*npts] = x10*x5;
      basis_eval[ipt + 6*npts] = radial_eval*x11;
      basis_eval[ipt + 7*npts] = x4*x7;
      basis_eval[ipt + 8*npts] = x1*x10;
      basis_eval[ipt + 9*npts] = radial_eval*x12;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = radial_eval_alpha*x13 + x14*x3;
      basis_x_eval[ipt + 1*npts] = x15*y;
      basis_x_eval[ipt + 2*npts] = x15*z;
      basis_x_eval[ipt + 3*npts] = x16 + x18;
      basis_x_eval[ipt + 4*npts] = x19*x21;
      basis_x_eval[ipt + 5*npts] = x22 + x24;
      basis_x_eval[ipt + 6*npts] = x11*x25;
      basis_x_eval[ipt + 7*npts] = x26;
      basis_x_eval[ipt + 8*npts] = x27;
      basis_x_eval[ipt + 9*npts] = x12*x25;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*x28;
      basis_y_eval[ipt + 1*npts] = x18 + x29;
      basis_y_eval[ipt + 2*npts] = x30;
      basis_y_eval[ipt + 3*npts] = x*x31;
      basis_y_eval[ipt + 4*npts] = x33*x8;
      basis_y_eval[ipt + 5*npts] = x27;
      basis_y_eval[ipt + 6*npts] = radial_eval_alpha*x34 + x14*x7;
      basis_y_eval[ipt + 7*npts] = x31*z;
      basis_y_eval[ipt + 8*npts] = x22 + x36;
      basis_y_eval[ipt + 9*npts] = x12*x28;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x0*x37;
      basis_z_eval[ipt + 1*npts] = x30;
      basis_z_eval[ipt + 2*npts] = x24 + x29;
      basis_z_eval[ipt + 3*npts] = x26;
      basis_z_eval[ipt + 4*npts] = x38*(radial_eval + x39);
      basis_z_eval[ipt + 5*npts] = x*x40;
      basis_z_eval[ipt + 6*npts] = x11*x37;
      basis_z_eval[ipt + 7*npts] = x16 + x36;
      basis_z_eval[ipt + 8*npts] = x40*y;
      basis_z_eval[ipt + 9*npts] = radial_eval_alpha*x41 + x10*x14;

      // Evaluate second derivative of bfn wrt xx
      basis_xx_eval[ipt + 0*npts] = x45;
      basis_xx_eval[ipt + 1*npts] = x50*y;
      basis_xx_eval[ipt + 2*npts] = x50*z;
      basis_xx_eval[ipt + 3*npts] = x*(x52 + x53);
      basis_xx_eval[ipt + 4*npts] = x54*(x43 + x55);
      basis_xx_eval[ipt + 5*npts] = x*(x56 + x57);
      basis_xx_eval[ipt + 6*npts] = x58;
      basis_xx_eval[ipt + 7*npts] = x53*z;
      basis_xx_eval[ipt + 8*npts] = x57*y;
      basis_xx_eval[ipt + 9*npts] = x59;

      // Evaluate second derivative of bfn wrt xy
      basis_xy_eval[ipt + 0*npts] = x60*y;
      basis_xy_eval[ipt + 1*npts] = radial_eval_alpha_squared*x0*x7 + x15 + x61*x7;
      basis_xy_eval[ipt + 2*npts] = x62;
      basis_xy_eval[ipt + 3*npts] = radial_eval_alpha_squared*x11*x3 + x3*x63 + x31;
      basis_xy_eval[ipt + 4*npts] = z*(x21 + x65);
      basis_xy_eval[ipt + 5*npts] = x67*y;
      basis_xy_eval[ipt + 6*npts] = x*x68;
      basis_xy_eval[ipt + 7*npts] = x69;
      basis_xy_eval[ipt + 8*npts] = x*x71;
      basis_xy_eval[ipt + 9*npts] = radial_eval_alpha_squared*x12*x38;

      // Evaluate second derivative of bfn wrt xz
      basis_xz_eval[ipt + 0*npts] = x60*z;
      basis_xz_eval[ipt + 1*npts] = x62;
      basis_xz_eval[ipt + 2*npts] = radial_eval_alpha_squared*x0*x10 + x10*x61 + x15;
      basis_xz_eval[ipt + 3*npts] = x65*z;
      basis_xz_eval[ipt + 4*npts] = y*(x21 + x67);
      basis_xz_eval[ipt + 5*npts] = radial_eval_alpha_squared*x12*x3 + x3*x72 + x40;
      basis_xz_eval[ipt + 6*npts] = radial_eval_alpha_squared*x11*x8;
      basis_xz_eval[ipt + 7*npts] = x*(x32 + x70);
      basis_xz_eval[ipt + 8*npts] = x73;
      basis_xz_eval[ipt + 9*npts] = x*x74;

      // Evaluate second derivative of bfn wrt yy
      basis_yy_eval[ipt + 0*npts] = x77;
      basis_yy_eval[ipt + 1*npts] = y*(x78 + x79);
      basis_yy_eval[ipt + 2*npts] = x79*z;
      basis_yy_eval[ipt + 3*npts] = x*x82;
      basis_yy_eval[ipt + 4*npts] = x54*(x55 + x75);
      basis_yy_eval[ipt + 5*npts] = x*x83;
      basis_yy_eval[ipt + 6*npts] = x84;
      basis_yy_eval[ipt + 7*npts] = x82*z;
      basis_yy_eval[ipt + 8*npts] = y*(x56 + x83);
      basis_yy_eval[ipt + 9*npts] = x85;

      // Evaluate second derivative of bfn wrt yz
      basis_yz_eval[ipt + 0*npts] = radial_eval_alpha_squared*x0*x19;
      basis_yz_eval[ipt + 1*npts] = z*(x20 + x64);
      basis_yz_eval[ipt + 2*npts] = y*(x20 + x66);
      basis_yz_eval[ipt + 3*npts] = x69;
      basis_yz_eval[ipt + 4*npts] = x*(x33 + x71);
      basis_yz_eval[ipt + 5*npts] = x73;
      basis_yz_eval[ipt + 6*npts] = x68*z;
      basis_yz_eval[ipt + 7*npts] = radial_eval_alpha_squared*x10*x11 + x10*x63 + x31;
      basis_yz_eval[ipt + 8*npts] = radial_eval_alpha_squared*x12*x7 + x40 + x7*x72;
      basis_yz_eval[ipt + 9*npts] = x74*y;

      // Evaluate second derivative of bfn wrt zz
      basis_zz_eval[ipt + 0*npts] = x88;
      basis_zz_eval[ipt + 1*npts] = x89*y;
      basis_zz_eval[ipt + 2*npts] = z*(x78 + x89);
      basis_zz_eval[ipt + 3*npts] = x*x90;
      basis_zz_eval[ipt + 4*npts] = x54*(x55 + x86);
      basis_zz_eval[ipt + 5*npts] = x*x93;
      basis_zz_eval[ipt + 6*npts] = x94;
      basis_zz_eval[ipt + 7*npts] = z*(x52 + x90);
      basis_zz_eval[ipt + 8*npts] = x93*y;
      basis_zz_eval[ipt + 9*npts] = x95;

      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x45 + x77 + x88;
      basis_lapl_eval[ipt + 1*npts] = x96*y;
      basis_lapl_eval[ipt + 2*npts] = x96*z;
      basis_lapl_eval[ipt + 3*npts] = x*x97;
      basis_lapl_eval[ipt + 4*npts] = x54*(9.0*radial_eval_alpha + x43 + x98);
      basis_lapl_eval[ipt + 5*npts] = x*x99;
      basis_lapl_eval[ipt + 6*npts] = x58 + x84 + x94;
      basis_lapl_eval[ipt + 7*npts] = x97*z;
      basis_lapl_eval[ipt + 8*npts] = x99*y;
      basis_lapl_eval[ipt + 9*npts] = x59 + x85 + x95;

      // Evaluate Laplacian gradient of bfn (dx)
      basis_lapl_x_eval[ipt + 0*npts] = x*x105 + x*x107 + x0*x109 + x100 + x101*x3 + x102 + x103 + 9.0*x48;
      basis_lapl_x_eval[ipt + 1*npts] = x115*y;
      basis_lapl_x_eval[ipt + 2*npts] = x115*z;
      basis_lapl_x_eval[ipt + 3*npts] = x*x117 + x116 + x118*x2 + x119*x2 + x121 + x78 + x82 + x90;
      basis_lapl_x_eval[ipt + 4*npts] = x19*(x*x109 + x104*x2 + x106*x2 + x120*x3 + x42 + 3.0*x44 + x98);
      basis_lapl_x_eval[ipt + 5*npts] = x*x123 + x122 + x124*x2 + x125*x2 + x126 + x78 + x83 + x93;
      basis_lapl_x_eval[ipt + 6*npts] = x*x130 + x*x131 + x109*x11 + x11*x129 + x128;
      basis_lapl_x_eval[ipt + 7*npts] = z*(x*x118 + x*x119 + x117 + x129*x7 + x61);
      basis_lapl_x_eval[ipt + 8*npts] = y*(x*x124 + x*x125 + x10*x129 + x123 + x61);
      basis_lapl_x_eval[ipt + 9*npts] = x*x134 + x*x135 + x109*x12 + x12*x129 + x133;
      // Evaluate Laplacian gradient of bfn (dy)
      basis_lapl_y_eval[ipt + 0*npts] = x0*x136 + x0*x139 + x107*y + x128 + x138*y;
      basis_lapl_y_eval[ipt + 1*npts] = x102 + x114*x6 + x121 + x140*y + x141*x6 + x50 + x52 + x89;
      basis_lapl_y_eval[ipt + 2*npts] = z*(x114*y + x136*x3 + x140 + x141*y + x63);
      basis_lapl_y_eval[ipt + 3*npts] = x*x144;
      basis_lapl_y_eval[ipt + 4*npts] = x8*(x106*x6 + x120*x7 + x137*x6 + x139*y + x145 + 3.0*x76 + x86);
      basis_lapl_y_eval[ipt + 5*npts] = x*(x10*x136 + x125*y + x146*y + x147 + x63);
      basis_lapl_y_eval[ipt + 6*npts] = x100 + x101*x7 + x11*x139 + x116 + x131*y + x148 + x149*y + 9.0*x80;
      basis_lapl_y_eval[ipt + 7*npts] = x144*z;
      basis_lapl_y_eval[ipt + 8*npts] = x125*x6 + x146*x6 + x147*y + x150 + x151 + x52 + x57 + x93;
      basis_lapl_y_eval[ipt + 9*npts] = x12*x136 + x12*x139 + x135*y + x152 + x153*y;
      // Evaluate Laplacian gradient of bfn (dz)
      basis_lapl_z_eval[ipt + 0*npts] = x0*x154 + x0*x155 + x105*z + x133 + x138*z;
      basis_lapl_z_eval[ipt + 1*npts] = y*(x113*z + x141*z + x154*x3 + x156 + x72);
      basis_lapl_z_eval[ipt + 2*npts] = x103 + x113*x9 + x126 + x141*x9 + x156*z + x50 + x56 + x79;
      basis_lapl_z_eval[ipt + 3*npts] = x*(x118*z + x143*z + x154*x7 + x157 + x72);
      basis_lapl_z_eval[ipt + 4*npts] = x38*(x10*x120 + x104*x9 + x137*x9 + x145 + x155*z + x75 + 3.0*x87);
      basis_lapl_z_eval[ipt + 5*npts] = x*x159;
      basis_lapl_z_eval[ipt + 6*npts] = x11*x154 + x11*x155 + x130*z + x149*z + x152;
      basis_lapl_z_eval[ipt + 7*npts] = x118*x9 + x143*x9 + x148 + x151 + x157*z + x53 + x56 + x82;
      basis_lapl_z_eval[ipt + 8*npts] = x159*y;
      basis_lapl_z_eval[ipt + 9*npts] = x10*x101 + x100 + x12*x155 + x122 + x134*z + x150 + x153*z + 9.0*x91;




#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x0;
      ang_eval_1 = x1*x3;
      ang_eval_2 = x3*x4;
      ang_eval_3 = x5*x7;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = x1*x8;
      ang_eval_1 = x10*x5;
      ang_eval_2 = radial_eval*x11;
      ang_eval_3 = x4*x7;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = x1*x10;
      ang_eval_1 = radial_eval*x12;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = radial_eval_alpha*x13 + x14*x3;
      dang_eval_y_0 = x0*x28;
      dang_eval_z_0 = x0*x37;
      dang_eval_x_1 = x15*y;
      dang_eval_y_1 = x18 + x29;
      dang_eval_z_1 = x30;
      dang_eval_x_2 = x15*z;
      dang_eval_y_2 = x30;
      dang_eval_z_2 = x24 + x29;
      dang_eval_x_3 = x16 + x18;
      dang_eval_y_3 = x*x31;
      dang_eval_z_3 = x26;
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

      dang_eval_x_0 = x19*x21;
      dang_eval_y_0 = x33*x8;
      dang_eval_z_0 = x38*(radial_eval + x39);
      dang_eval_x_1 = x22 + x24;
      dang_eval_y_1 = x27;
      dang_eval_z_1 = x*x40;
      dang_eval_x_2 = x11*x25;
      dang_eval_y_2 = radial_eval_alpha*x34 + x14*x7;
      dang_eval_z_2 = x11*x37;
      dang_eval_x_3 = x26;
      dang_eval_y_3 = x31*z;
      dang_eval_z_3 = x16 + x36;
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

      dang_eval_x_0 = x27;
      dang_eval_y_0 = x22 + x36;
      dang_eval_z_0 = x40*y;
      dang_eval_x_1 = x12*x25;
      dang_eval_y_1 = x12*x28;
      dang_eval_z_1 = radial_eval_alpha*x41 + x10*x14;
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
