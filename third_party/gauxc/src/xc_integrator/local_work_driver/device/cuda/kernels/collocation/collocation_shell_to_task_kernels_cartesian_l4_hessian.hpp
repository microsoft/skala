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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_cartesian_hessian_4(
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
      const auto x0 = x*x*x*x; 
      const auto x1 = radial_eval*y; 
      const auto x2 = x*x*x; 
      const auto x3 = radial_eval*z; 
      const auto x4 = x*x; 
      const auto x5 = y*y; 
      const auto x6 = x4*x5; 
      const auto x7 = x1*z; 
      const auto x8 = z*z; 
      const auto x9 = x4*x8; 
      const auto x10 = radial_eval*x; 
      const auto x11 = y*y*y; 
      const auto x12 = x*x3; 
      const auto x13 = x*x1; 
      const auto x14 = z*z*z; 
      const auto x15 = y*y*y*y; 
      const auto x16 = x5*x8; 
      const auto x17 = z*z*z*z; 
      const auto x18 = x*x*x*x*x; 
      const auto x19 = 4.0*radial_eval; 
      const auto x20 = 3.0*radial_eval; 
      const auto x21 = radial_eval_alpha*x0 + x20*x4; 
      const auto x22 = 2.0*x10; 
      const auto x23 = x2*x5; 
      const auto x24 = radial_eval_alpha*x23; 
      const auto x25 = y*z; 
      const auto x26 = radial_eval_alpha*x2; 
      const auto x27 = x22 + x26; 
      const auto x28 = x2*x8; 
      const auto x29 = radial_eval_alpha*x28; 
      const auto x30 = radial_eval*x11; 
      const auto x31 = x11*x4; 
      const auto x32 = radial_eval_alpha*x31; 
      const auto x33 = radial_eval*x5; 
      const auto x34 = radial_eval_alpha*x6; 
      const auto x35 = x33 + x34; 
      const auto x36 = radial_eval*x8; 
      const auto x37 = radial_eval_alpha*x9; 
      const auto x38 = x36 + x37; 
      const auto x39 = radial_eval*x14; 
      const auto x40 = x14*x4; 
      const auto x41 = radial_eval_alpha*x40; 
      const auto x42 = radial_eval_alpha*x; 
      const auto x43 = x11*x42*z; 
      const auto x44 = x14*x42*y; 
      const auto x45 = radial_eval_alpha*y; 
      const auto x46 = radial_eval*x2; 
      const auto x47 = radial_eval_alpha*x2*x25; 
      const auto x48 = 2.0*x1; 
      const auto x49 = radial_eval*x4; 
      const auto x50 = x34 + x49; 
      const auto x51 = radial_eval_alpha*x15 + x20*x5; 
      const auto x52 = x*z; 
      const auto x53 = radial_eval_alpha*x11; 
      const auto x54 = x48 + x53; 
      const auto x55 = radial_eval_alpha*x16; 
      const auto x56 = y*y*y*y*y; 
      const auto x57 = x11*x8; 
      const auto x58 = radial_eval_alpha*x57; 
      const auto x59 = x14*x5; 
      const auto x60 = radial_eval_alpha*x59; 
      const auto x61 = radial_eval_alpha*z; 
      const auto x62 = 2.0*x3; 
      const auto x63 = x*y; 
      const auto x64 = radial_eval_alpha*x14; 
      const auto x65 = x62 + x64; 
      const auto x66 = radial_eval_alpha*x17 + x20*x8; 
      const auto x67 = z*z*z*z*z; 
      const auto x68 = 12.0*radial_eval; 
      const auto x69 = 8.0*radial_eval_alpha; 
      const auto x70 = radial_eval_alpha + radial_eval_alpha_squared*x4; 
      const auto x71 = x0*x69 + x0*x70 + x4*x68; 
      const auto x72 = 6.0*radial_eval_alpha; 
      const auto x73 = 6.0*x10 + x2*x70; 
      const auto x74 = x2*x72 + x73; 
      const auto x75 = 4.0*radial_eval_alpha; 
      const auto x76 = x6*x75; 
      const auto x77 = 2.0*radial_eval; 
      const auto x78 = x5*x77; 
      const auto x79 = x4*x5*x70 + x78; 
      const auto x80 = x4*x70 + x77; 
      const auto x81 = x75*x9; 
      const auto x82 = x77*x8; 
      const auto x83 = x4*x70*x8 + x82; 
      const auto x84 = 2.0*radial_eval_alpha; 
      const auto x85 = x11*x84; 
      const auto x86 = x11*x70; 
      const auto x87 = x5*x84; 
      const auto x88 = x5*x70; 
      const auto x89 = x8*x84; 
      const auto x90 = x70*x8; 
      const auto x91 = x14*x84; 
      const auto x92 = x14*x70; 
      const auto x93 = x15*x70; 
      const auto x94 = x5*x70*x8; 
      const auto x95 = x17*x70; 
      const auto x96 = radial_eval_alpha_squared*x18 + x2*x75; 
      const auto x97 = 3.0*radial_eval_alpha; 
      const auto x98 = x6*x97; 
      const auto x99 = x25*(radial_eval_alpha_squared*x0 + x4*x97); 
      const auto x100 = 2.0*x42; 
      const auto x101 = 2.0*x45; 
      const auto x102 = radial_eval_alpha_squared*x23; 
      const auto x103 = x100*x5 + x102; 
      const auto x104 = radial_eval_alpha_squared*x28; 
      const auto x105 = x100*x8 + x104; 
      const auto x106 = radial_eval_alpha_squared*x31; 
      const auto x107 = x101*x4 + x106; 
      const auto x108 = radial_eval_alpha_squared*x4*x5*x8; 
      const auto x109 = x108 + x55; 
      const auto x110 = radial_eval_alpha_squared*x40; 
      const auto x111 = radial_eval_alpha_squared*x56 + x11*x75; 
      const auto x112 = x52*(radial_eval_alpha_squared*x15 + x5*x97); 
      const auto x113 = radial_eval_alpha_squared*x57; 
      const auto x114 = x101*x8 + x113; 
      const auto x115 = radial_eval_alpha_squared*x59; 
      const auto x116 = x9*x97; 
      const auto x117 = 2.0*x61; 
      const auto x118 = x110 + x117*x4; 
      const auto x119 = x115 + x117*x5; 
      const auto x120 = x63*(radial_eval_alpha_squared*x17 + x8*x97); 
      const auto x121 = radial_eval_alpha_squared*x67 + x14*x75; 
      const auto x122 = radial_eval_alpha + radial_eval_alpha_squared*x5; 
      const auto x123 = x0*x122; 
      const auto x124 = x2*x84; 
      const auto x125 = x122*x2; 
      const auto x126 = x4*x77; 
      const auto x127 = x122*x4*x5 + x126; 
      const auto x128 = x4*x84; 
      const auto x129 = x122*x4; 
      const auto x130 = x122*x4*x8; 
      const auto x131 = 6.0*x1 + x11*x122; 
      const auto x132 = x11*x72 + x131; 
      const auto x133 = x122*x5 + x77; 
      const auto x134 = x122*x8; 
      const auto x135 = x122*x14; 
      const auto x136 = x122*x15 + x15*x69 + x5*x68; 
      const auto x137 = x16*x75; 
      const auto x138 = x122*x5*x8 + x82; 
      const auto x139 = x122*x17; 
      const auto x140 = x16*x97; 
      const auto x141 = radial_eval_alpha + radial_eval_alpha_squared*x8; 
      const auto x142 = x0*x141; 
      const auto x143 = x141*x2; 
      const auto x144 = x141*x4*x5; 
      const auto x145 = x141*x4; 
      const auto x146 = x126 + x141*x4*x8; 
      const auto x147 = x11*x141; 
      const auto x148 = x141*x5; 
      const auto x149 = x141*x8 + x77; 
      const auto x150 = x14*x141 + 6.0*x3; 
      const auto x151 = x14*x72 + x150; 
      const auto x152 = x141*x15; 
      const auto x153 = x141*x5*x8 + x78; 
      const auto x154 = x141*x17 + x17*x69 + x68*x8; 
      const auto x155 = x125 + x143 + x2*x69 + x73; 
      const auto x156 = x11*x69 + x131 + x147 + x86; 
      const auto x157 = x135 + x14*x69 + x150 + x92; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x0;
      basis_eval[ipt + 1*npts] = x1*x2;
      basis_eval[ipt + 2*npts] = x2*x3;
      basis_eval[ipt + 3*npts] = radial_eval*x6;
      basis_eval[ipt + 4*npts] = x4*x7;
      basis_eval[ipt + 5*npts] = radial_eval*x9;
      basis_eval[ipt + 6*npts] = x10*x11;
      basis_eval[ipt + 7*npts] = x12*x5;
      basis_eval[ipt + 8*npts] = x13*x8;
      basis_eval[ipt + 9*npts] = x10*x14;
      basis_eval[ipt + 10*npts] = radial_eval*x15;
      basis_eval[ipt + 11*npts] = x11*x3;
      basis_eval[ipt + 12*npts] = radial_eval*x16;
      basis_eval[ipt + 13*npts] = x1*x14;
      basis_eval[ipt + 14*npts] = radial_eval*x17;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = radial_eval_alpha*x18 + x19*x2;
      basis_x_eval[ipt + 1*npts] = x21*y;
      basis_x_eval[ipt + 2*npts] = x21*z;
      basis_x_eval[ipt + 3*npts] = x22*x5 + x24;
      basis_x_eval[ipt + 4*npts] = x25*x27;
      basis_x_eval[ipt + 5*npts] = x22*x8 + x29;
      basis_x_eval[ipt + 6*npts] = x30 + x32;
      basis_x_eval[ipt + 7*npts] = x35*z;
      basis_x_eval[ipt + 8*npts] = x38*y;
      basis_x_eval[ipt + 9*npts] = x39 + x41;
      basis_x_eval[ipt + 10*npts] = x15*x42;
      basis_x_eval[ipt + 11*npts] = x43;
      basis_x_eval[ipt + 12*npts] = x16*x42;
      basis_x_eval[ipt + 13*npts] = x44;
      basis_x_eval[ipt + 14*npts] = x17*x42;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*x45;
      basis_y_eval[ipt + 1*npts] = x24 + x46;
      basis_y_eval[ipt + 2*npts] = x47;
      basis_y_eval[ipt + 3*npts] = x32 + x4*x48;
      basis_y_eval[ipt + 4*npts] = x50*z;
      basis_y_eval[ipt + 5*npts] = x45*x9;
      basis_y_eval[ipt + 6*npts] = x*x51;
      basis_y_eval[ipt + 7*npts] = x52*x54;
      basis_y_eval[ipt + 8*npts] = x*(x36 + x55);
      basis_y_eval[ipt + 9*npts] = x44;
      basis_y_eval[ipt + 10*npts] = radial_eval_alpha*x56 + x11*x19;
      basis_y_eval[ipt + 11*npts] = x51*z;
      basis_y_eval[ipt + 12*npts] = x48*x8 + x58;
      basis_y_eval[ipt + 13*npts] = x39 + x60;
      basis_y_eval[ipt + 14*npts] = x17*x45;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x0*x61;
      basis_z_eval[ipt + 1*npts] = x47;
      basis_z_eval[ipt + 2*npts] = x29 + x46;
      basis_z_eval[ipt + 3*npts] = x6*x61;
      basis_z_eval[ipt + 4*npts] = y*(x37 + x49);
      basis_z_eval[ipt + 5*npts] = x4*x62 + x41;
      basis_z_eval[ipt + 6*npts] = x43;
      basis_z_eval[ipt + 7*npts] = x*(x33 + x55);
      basis_z_eval[ipt + 8*npts] = x63*x65;
      basis_z_eval[ipt + 9*npts] = x*x66;
      basis_z_eval[ipt + 10*npts] = x15*x61;
      basis_z_eval[ipt + 11*npts] = x30 + x58;
      basis_z_eval[ipt + 12*npts] = x5*x62 + x60;
      basis_z_eval[ipt + 13*npts] = x66*y;
      basis_z_eval[ipt + 14*npts] = radial_eval_alpha*x67 + x14*x19;

      // Evaluate second derivative of bfn wrt xx
      basis_xx_eval[ipt + 0*npts] = x71;
      basis_xx_eval[ipt + 1*npts] = x74*y;
      basis_xx_eval[ipt + 2*npts] = x74*z;
      basis_xx_eval[ipt + 3*npts] = x76 + x79;
      basis_xx_eval[ipt + 4*npts] = x25*(x4*x75 + x80);
      basis_xx_eval[ipt + 5*npts] = x81 + x83;
      basis_xx_eval[ipt + 6*npts] = x*(x85 + x86);
      basis_xx_eval[ipt + 7*npts] = x52*(x87 + x88);
      basis_xx_eval[ipt + 8*npts] = x63*(x89 + x90);
      basis_xx_eval[ipt + 9*npts] = x*(x91 + x92);
      basis_xx_eval[ipt + 10*npts] = x93;
      basis_xx_eval[ipt + 11*npts] = x86*z;
      basis_xx_eval[ipt + 12*npts] = x94;
      basis_xx_eval[ipt + 13*npts] = x92*y;
      basis_xx_eval[ipt + 14*npts] = x95;

      // Evaluate second derivative of bfn wrt xy
      basis_xy_eval[ipt + 0*npts] = x96*y;
      basis_xy_eval[ipt + 1*npts] = radial_eval_alpha_squared*x0*x5 + x21 + x98;
      basis_xy_eval[ipt + 2*npts] = x99;
      basis_xy_eval[ipt + 3*npts] = radial_eval_alpha_squared*x11*x2 + x100*x11 + x101*x2 + 4.0*x13;
      basis_xy_eval[ipt + 4*npts] = z*(x103 + x27);
      basis_xy_eval[ipt + 5*npts] = x105*y;
      basis_xy_eval[ipt + 6*npts] = radial_eval_alpha_squared*x15*x4 + x51 + x98;
      basis_xy_eval[ipt + 7*npts] = z*(x107 + x54);
      basis_xy_eval[ipt + 8*npts] = x109 + x38;
      basis_xy_eval[ipt + 9*npts] = y*(x110 + x64);
      basis_xy_eval[ipt + 10*npts] = x*x111;
      basis_xy_eval[ipt + 11*npts] = x112;
      basis_xy_eval[ipt + 12*npts] = x*x114;
      basis_xy_eval[ipt + 13*npts] = x*(x115 + x64);
      basis_xy_eval[ipt + 14*npts] = radial_eval_alpha_squared*x17*x63;

      // Evaluate second derivative of bfn wrt xz
      basis_xz_eval[ipt + 0*npts] = x96*z;
      basis_xz_eval[ipt + 1*npts] = x99;
      basis_xz_eval[ipt + 2*npts] = radial_eval_alpha_squared*x0*x8 + x116 + x21;
      basis_xz_eval[ipt + 3*npts] = x103*z;
      basis_xz_eval[ipt + 4*npts] = y*(x105 + x27);
      basis_xz_eval[ipt + 5*npts] = radial_eval_alpha_squared*x14*x2 + x100*x14 + x117*x2 + 4.0*x12;
      basis_xz_eval[ipt + 6*npts] = z*(x106 + x53);
      basis_xz_eval[ipt + 7*npts] = x109 + x35;
      basis_xz_eval[ipt + 8*npts] = y*(x118 + x65);
      basis_xz_eval[ipt + 9*npts] = radial_eval_alpha_squared*x17*x4 + x116 + x66;
      basis_xz_eval[ipt + 10*npts] = radial_eval_alpha_squared*x15*x52;
      basis_xz_eval[ipt + 11*npts] = x*(x113 + x53);
      basis_xz_eval[ipt + 12*npts] = x*x119;
      basis_xz_eval[ipt + 13*npts] = x120;
      basis_xz_eval[ipt + 14*npts] = x*x121;

      // Evaluate second derivative of bfn wrt yy
      basis_yy_eval[ipt + 0*npts] = x123;
      basis_yy_eval[ipt + 1*npts] = y*(x124 + x125);
      basis_yy_eval[ipt + 2*npts] = x125*z;
      basis_yy_eval[ipt + 3*npts] = x127 + x76;
      basis_yy_eval[ipt + 4*npts] = x25*(x128 + x129);
      basis_yy_eval[ipt + 5*npts] = x130;
      basis_yy_eval[ipt + 6*npts] = x*x132;
      basis_yy_eval[ipt + 7*npts] = x52*(x133 + x5*x75);
      basis_yy_eval[ipt + 8*npts] = x63*(x134 + x89);
      basis_yy_eval[ipt + 9*npts] = x*x135;
      basis_yy_eval[ipt + 10*npts] = x136;
      basis_yy_eval[ipt + 11*npts] = x132*z;
      basis_yy_eval[ipt + 12*npts] = x137 + x138;
      basis_yy_eval[ipt + 13*npts] = y*(x135 + x91);
      basis_yy_eval[ipt + 14*npts] = x139;

      // Evaluate second derivative of bfn wrt yz
      basis_yz_eval[ipt + 0*npts] = radial_eval_alpha_squared*x0*x25;
      basis_yz_eval[ipt + 1*npts] = z*(x102 + x26);
      basis_yz_eval[ipt + 2*npts] = y*(x104 + x26);
      basis_yz_eval[ipt + 3*npts] = x107*z;
      basis_yz_eval[ipt + 4*npts] = x108 + x37 + x50;
      basis_yz_eval[ipt + 5*npts] = x118*y;
      basis_yz_eval[ipt + 6*npts] = x112;
      basis_yz_eval[ipt + 7*npts] = x*(x114 + x54);
      basis_yz_eval[ipt + 8*npts] = x*(x119 + x65);
      basis_yz_eval[ipt + 9*npts] = x120;
      basis_yz_eval[ipt + 10*npts] = x111*z;
      basis_yz_eval[ipt + 11*npts] = radial_eval_alpha_squared*x15*x8 + x140 + x51;
      basis_yz_eval[ipt + 12*npts] = radial_eval_alpha_squared*x11*x14 + x101*x14 + x11*x117 + 4.0*x7;
      basis_yz_eval[ipt + 13*npts] = radial_eval_alpha_squared*x17*x5 + x140 + x66;
      basis_yz_eval[ipt + 14*npts] = x121*y;

      // Evaluate second derivative of bfn wrt zz
      basis_zz_eval[ipt + 0*npts] = x142;
      basis_zz_eval[ipt + 1*npts] = x143*y;
      basis_zz_eval[ipt + 2*npts] = z*(x124 + x143);
      basis_zz_eval[ipt + 3*npts] = x144;
      basis_zz_eval[ipt + 4*npts] = x25*(x128 + x145);
      basis_zz_eval[ipt + 5*npts] = x146 + x81;
      basis_zz_eval[ipt + 6*npts] = x*x147;
      basis_zz_eval[ipt + 7*npts] = x52*(x148 + x87);
      basis_zz_eval[ipt + 8*npts] = x63*(x149 + x75*x8);
      basis_zz_eval[ipt + 9*npts] = x*x151;
      basis_zz_eval[ipt + 10*npts] = x152;
      basis_zz_eval[ipt + 11*npts] = z*(x147 + x85);
      basis_zz_eval[ipt + 12*npts] = x137 + x153;
      basis_zz_eval[ipt + 13*npts] = x151*y;
      basis_zz_eval[ipt + 14*npts] = x154;






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

      ang_eval_0 = x4*x7;
      ang_eval_1 = radial_eval*x9;
      ang_eval_2 = x10*x11;
      ang_eval_3 = x12*x5;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = x13*x8;
      ang_eval_1 = x10*x14;
      ang_eval_2 = radial_eval*x15;
      ang_eval_3 = x11*x3;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;
      basis_eval[ipt + 11*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x16;
      ang_eval_1 = x1*x14;
      ang_eval_2 = radial_eval*x17;
      basis_eval[ipt + 12*npts] = ang_eval_0;
      basis_eval[ipt + 13*npts] = ang_eval_1;
      basis_eval[ipt + 14*npts] = ang_eval_2;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = radial_eval_alpha*x18 + x19*x2;
      dang_eval_y_0 = x0*x45;
      dang_eval_z_0 = x0*x61;
      dang_eval_x_1 = x21*y;
      dang_eval_y_1 = x24 + x46;
      dang_eval_z_1 = x47;
      dang_eval_x_2 = x21*z;
      dang_eval_y_2 = x47;
      dang_eval_z_2 = x29 + x46;
      dang_eval_x_3 = x22*x5 + x24;
      dang_eval_y_3 = x32 + x4*x48;
      dang_eval_z_3 = x6*x61;
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

      dang_eval_x_0 = x25*x27;
      dang_eval_y_0 = x50*z;
      dang_eval_z_0 = y*(x37 + x49);
      dang_eval_x_1 = x22*x8 + x29;
      dang_eval_y_1 = x45*x9;
      dang_eval_z_1 = x4*x62 + x41;
      dang_eval_x_2 = x30 + x32;
      dang_eval_y_2 = x*x51;
      dang_eval_z_2 = x43;
      dang_eval_x_3 = x35*z;
      dang_eval_y_3 = x52*x54;
      dang_eval_z_3 = x*(x33 + x55);
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

      dang_eval_x_0 = x38*y;
      dang_eval_y_0 = x*(x36 + x55);
      dang_eval_z_0 = x63*x65;
      dang_eval_x_1 = x39 + x41;
      dang_eval_y_1 = x44;
      dang_eval_z_1 = x*x66;
      dang_eval_x_2 = x15*x42;
      dang_eval_y_2 = radial_eval_alpha*x56 + x11*x19;
      dang_eval_z_2 = x15*x61;
      dang_eval_x_3 = x43;
      dang_eval_y_3 = x51*z;
      dang_eval_z_3 = x30 + x58;
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

      dang_eval_x_0 = x16*x42;
      dang_eval_y_0 = x48*x8 + x58;
      dang_eval_z_0 = x5*x62 + x60;
      dang_eval_x_1 = x44;
      dang_eval_y_1 = x39 + x60;
      dang_eval_z_1 = x66*y;
      dang_eval_x_2 = x17*x42;
      dang_eval_y_2 = x17*x45;
      dang_eval_z_2 = radial_eval_alpha*x67 + x14*x19;
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
