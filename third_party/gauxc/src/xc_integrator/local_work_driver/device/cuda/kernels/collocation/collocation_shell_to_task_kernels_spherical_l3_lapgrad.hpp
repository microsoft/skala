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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_spherical_lapgrad_3(
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
      const auto x0 = 0.25*sqrt_10; 
      const auto x1 = x0*y; 
      const auto x2 = x*x; 
      const auto x3 = x2; 
      const auto x4 = 3.0*x3; 
      const auto x5 = y*y; 
      const auto x6 = x5; 
      const auto x7 = -x6; 
      const auto x8 = x4 + x7; 
      const auto x9 = sqrt_15*z; 
      const auto x10 = x9*y; 
      const auto x11 = x*x10; 
      const auto x12 = 0.25*sqrt_6; 
      const auto x13 = x12*y; 
      const auto x14 = z*z; 
      const auto x15 = x14; 
      const auto x16 = -4.0*x15; 
      const auto x17 = x16 + x6; 
      const auto x18 = -x17 - x3; 
      const auto x19 = 0.5*z; 
      const auto x20 = 3.0*x6; 
      const auto x21 = -2.0*x15; 
      const auto x22 = -x20 - x21 - x4; 
      const auto x23 = x*x12; 
      const auto x24 = 0.5*sqrt_15; 
      const auto x25 = x24*z; 
      const auto x26 = x3 + x7; 
      const auto x27 = x*x0; 
      const auto x28 = -x20; 
      const auto x29 = x28 + x3; 
      const auto x30 = x*x1; 
      const auto x31 = 6.0*radial_eval; 
      const auto x32 = radial_eval + radial_eval_alpha*x3; 
      const auto x33 = x*x13; 
      const auto x34 = 2.0*radial_eval; 
      const auto x35 = -x34; 
      const auto x36 = radial_eval_alpha*x18; 
      const auto x37 = x33*(x35 + x36); 
      const auto x38 = x*x19; 
      const auto x39 = -x31; 
      const auto x40 = radial_eval_alpha*x22 + x39; 
      const auto x41 = -x17 - x4; 
      const auto x42 = x18*x3; 
      const auto x43 = x*x25; 
      const auto x44 = radial_eval_alpha*x26; 
      const auto x45 = x34 + x44; 
      const auto x46 = x28 + x4; 
      const auto x47 = radial_eval*x46; 
      const auto x48 = x29*x3; 
      const auto x49 = x6*x8; 
      const auto x50 = x*x9; 
      const auto x51 = radial_eval_alpha*x6; 
      const auto x52 = radial_eval + x51; 
      const auto x53 = -x16 - x20 - x3; 
      const auto x54 = x18*x6; 
      const auto x55 = x35 + x44; 
      const auto x56 = radial_eval_alpha*z; 
      const auto x57 = sqrt_15*y; 
      const auto x58 = x*x57; 
      const auto x59 = radial_eval_alpha*x15; 
      const auto x60 = 8.0*radial_eval; 
      const auto x61 = x36 + x60; 
      const auto x62 = -x21 - x3 - x6; 
      const auto x63 = x15*x22; 
      const auto x64 = x15*x26; 
      const auto x65 = radial_eval_alpha_squared*x3; 
      const auto x66 = radial_eval_alpha + x65; 
      const auto x67 = x66*x8; 
      const auto x68 = 12.0*radial_eval_alpha; 
      const auto x69 = x3*x68; 
      const auto x70 = x31 + x69; 
      const auto x71 = 3.0*radial_eval_alpha; 
      const auto x72 = 4.0*radial_eval_alpha; 
      const auto x73 = x3*x72; 
      const auto x74 = x34 + x73; 
      const auto x75 = x18*x66; 
      const auto x76 = 2.0*radial_eval_alpha; 
      const auto x77 = x41*x76 + x75; 
      const auto x78 = x26*x66; 
      const auto x79 = x46*x76; 
      const auto x80 = x29*x66 + x79; 
      const auto x81 = 6.0*radial_eval_alpha; 
      const auto x82 = radial_eval_alpha*x46; 
      const auto x83 = radial_eval_alpha_squared*x49 + x82; 
      const auto x84 = x3*x6; 
      const auto x85 = radial_eval_alpha*x53 + radial_eval_alpha_squared*x54; 
      const auto x86 = radial_eval_alpha*x41 + radial_eval_alpha_squared*x42; 
      const auto x87 = radial_eval_alpha_squared*x48 + x82; 
      const auto x88 = x30*z; 
      const auto x89 = x15*x3; 
      const auto x90 = x33*z*(radial_eval_alpha_squared*x18 + x81); 
      const auto x91 = radial_eval_alpha_squared*x63 - x15*x81 + x39 + x62*x71; 
      const auto x92 = x12*z; 
      const auto x93 = 8.0*radial_eval_alpha; 
      const auto x94 = x15*x76; 
      const auto x95 = radial_eval_alpha_squared*x64; 
      const auto x96 = x0*z; 
      const auto x97 = radial_eval_alpha_squared*x6; 
      const auto x98 = radial_eval_alpha + x97; 
      const auto x99 = x79 + x8*x98; 
      const auto x100 = x18*x98; 
      const auto x101 = x100 + x53*x76; 
      const auto x102 = x6*x68; 
      const auto x103 = x102 + x31; 
      const auto x104 = x6*x72; 
      const auto x105 = x104 + x34; 
      const auto x106 = x15*x6; 
      const auto x107 = radial_eval_alpha_squared*x15; 
      const auto x108 = radial_eval_alpha + x107; 
      const auto x109 = x108*x8; 
      const auto x110 = 16.0*radial_eval_alpha*x15; 
      const auto x111 = x108*x18 + x110; 
      const auto x112 = x111 + x60; 
      const auto x113 = x108*x22 + x62*x81; 
      const auto x114 = x108*x26; 
      const auto x115 = x114 + x26*x76; 
      const auto x116 = x108*x29; 
      const auto x117 = x107 + x97; 
      const auto x118 = -x73; 
      const auto x119 = -x102; 
      const auto x120 = -x69; 
      const auto x121 = x119 + x120; 
      const auto x122 = -x104; 
      const auto x123 = x122 + x26*x98 + x73 + x78; 
      const auto x124 = 3.0*radial_eval_alpha_squared; 
      const auto x125 = x*(radial_eval_alpha_cubed*(x*x) + x124); 
      const auto x126 = radial_eval_alpha_cubed*x6 + radial_eval_alpha_squared; 
      const auto x127 = x126*x8; 
      const auto x128 = radial_eval_alpha_cubed*x15 + radial_eval_alpha_squared; 
      const auto x129 = x128*x8; 
      const auto x130 = 2.0*x; 
      const auto x131 = radial_eval_alpha_squared*x130; 
      const auto x132 = 6.0*x; 
      const auto x133 = 24.0*radial_eval_alpha; 
      const auto x134 = x*x133 + 18.0*x*x66 + x108*x132 + x132*x98; 
      const auto x135 = 4.0*radial_eval_alpha_squared; 
      const auto x136 = x*x93; 
      const auto x137 = 16.0*radial_eval_alpha_squared; 
      const auto x138 = x132*x66; 
      const auto x139 = x130*x98; 
      const auto x140 = x108*x130; 
      const auto x141 = x126*x18; 
      const auto x142 = x128*x18; 
      const auto x143 = x125*x18; 
      const auto x144 = 12.0*radial_eval_alpha_squared; 
      const auto x145 = x110 - x135*x84; 
      const auto x146 = x126*x26; 
      const auto x147 = x128*x26; 
      const auto x148 = x46*x98; 
      const auto x149 = x46*x66; 
      const auto x150 = x126*x29; 
      const auto x151 = x128*x29; 
      const auto x152 = x144*x84; 
      const auto x153 = x108*x46 + x119 + x69; 
      const auto x154 = y*(radial_eval_alpha_cubed*(y*y) + x124); 
      const auto x155 = radial_eval_alpha_cubed*x3 + radial_eval_alpha_squared; 
      const auto x156 = x155*x8; 
      const auto x157 = x65 + x81; 
      const auto x158 = x154*x18; 
      const auto x159 = x155*x18; 
      const auto x160 = x133*y; 
      const auto x161 = 6.0*y; 
      const auto x162 = x161*x66; 
      const auto x163 = 18.0*x98*y; 
      const auto x164 = x108*x161; 
      const auto x165 = 2.0*y; 
      const auto x166 = radial_eval_alpha_squared*x165; 
      const auto x167 = -x108*x165 - x161*x98 - x165*x66 - x93*y; 
      const auto x168 = x155*x26; 
      const auto x169 = x155*x29; 
      const auto x170 = x144*z; 
      const auto x171 = 2.0*radial_eval_alpha_squared*z; 
      const auto x172 = x171*x46; 
      const auto x173 = z*(radial_eval_alpha_cubed*(z*z) + x124); 
      const auto x174 = x135*z; 
      const auto x175 = 8.0*z; 
      const auto x176 = 24.0*x108*z + x141*z + x159*z + x173*x18 + x175*x66 + x175*x98 + 32.0*x56; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x1*x8;
      basis_eval[ipt + 1*npts] = radial_eval*x11;
      basis_eval[ipt + 2*npts] = radial_eval*x13*x18;
      basis_eval[ipt + 3*npts] = radial_eval*x19*x22;
      basis_eval[ipt + 4*npts] = radial_eval*x18*x23;
      basis_eval[ipt + 5*npts] = radial_eval*x25*x26;
      basis_eval[ipt + 6*npts] = radial_eval*x27*x29;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = x30*(radial_eval_alpha*x8 + x31);
      basis_x_eval[ipt + 1*npts] = x10*x32;
      basis_x_eval[ipt + 2*npts] = x37;
      basis_x_eval[ipt + 3*npts] = x38*x40;
      basis_x_eval[ipt + 4*npts] = x12*(radial_eval*x41 + radial_eval_alpha*x42);
      basis_x_eval[ipt + 5*npts] = x43*x45;
      basis_x_eval[ipt + 6*npts] = x0*(radial_eval_alpha*x48 + x47);

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*(radial_eval_alpha*x49 + x47);
      basis_y_eval[ipt + 1*npts] = x50*x52;
      basis_y_eval[ipt + 2*npts] = x12*(radial_eval*x53 + radial_eval_alpha*x54);
      basis_y_eval[ipt + 3*npts] = x19*x40*y;
      basis_y_eval[ipt + 4*npts] = x37;
      basis_y_eval[ipt + 5*npts] = x25*x55*y;
      basis_y_eval[ipt + 6*npts] = x30*(radial_eval_alpha*x29 + x39);

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x1*x56*x8;
      basis_z_eval[ipt + 1*npts] = x58*(radial_eval + x59);
      basis_z_eval[ipt + 2*npts] = x13*x61*z;
      basis_z_eval[ipt + 3*npts] = 1.5*radial_eval*x62 + 0.5*radial_eval_alpha*x63;
      basis_z_eval[ipt + 4*npts] = x23*x61*z;
      basis_z_eval[ipt + 5*npts] = x24*(radial_eval*x26 + radial_eval_alpha*x64);
      basis_z_eval[ipt + 6*npts] = x27*x29*x56;

      // Evaluate second derivative of bfn wrt xx
      basis_xx_eval[ipt + 0*npts] = x1*(x67 + x70);
      basis_xx_eval[ipt + 1*npts] = x11*(x65 + x71);
      basis_xx_eval[ipt + 2*npts] = x13*(x18*x66 - x74);
      basis_xx_eval[ipt + 3*npts] = x19*(x22*x66 - x70);
      basis_xx_eval[ipt + 4*npts] = x23*(x39 + x77);
      basis_xx_eval[ipt + 5*npts] = x25*(x74 + x78);
      basis_xx_eval[ipt + 6*npts] = x27*(x31 + x80);

      // Evaluate second derivative of bfn wrt xy
      basis_xy_eval[ipt + 0*npts] = x27*(x31 + x6*x81 + x83);
      basis_xy_eval[ipt + 1*npts] = x9*(radial_eval_alpha_squared*x84 + x32 + x51);
      basis_xy_eval[ipt + 2*npts] = x23*(x35 - x6*x76 + x85);
      basis_xy_eval[ipt + 3*npts] = x38*y*(radial_eval_alpha_squared*x22 - x68);
      basis_xy_eval[ipt + 4*npts] = x13*(-x3*x76 + x35 + x86);
      basis_xy_eval[ipt + 5*npts] = radial_eval_alpha_squared*x26*x43*y;
      basis_xy_eval[ipt + 6*npts] = x1*(-x3*x81 + x39 + x87);

      // Evaluate second derivative of bfn wrt xz
      basis_xz_eval[ipt + 0*npts] = x88*(radial_eval_alpha_squared*x8 + x81);
      basis_xz_eval[ipt + 1*npts] = x57*(radial_eval_alpha_squared*x89 + x32 + x59);
      basis_xz_eval[ipt + 2*npts] = x90;
      basis_xz_eval[ipt + 3*npts] = 0.5*x*x91;
      basis_xz_eval[ipt + 4*npts] = x92*(x3*x93 + x60 + x86);
      basis_xz_eval[ipt + 5*npts] = x*x24*(x45 + x94 + x95);
      basis_xz_eval[ipt + 6*npts] = x87*x96;

      // Evaluate second derivative of bfn wrt yy
      basis_yy_eval[ipt + 0*npts] = x1*(x39 + x99);
      basis_yy_eval[ipt + 1*npts] = x11*(x71 + x97);
      basis_yy_eval[ipt + 2*npts] = x13*(x101 + x39);
      basis_yy_eval[ipt + 3*npts] = x19*(-x103 + x22*x98);
      basis_yy_eval[ipt + 4*npts] = x23*(-x105 + x18*x98);
      basis_yy_eval[ipt + 5*npts] = x25*(-x105 + x26*x98);
      basis_yy_eval[ipt + 6*npts] = x27*(-x103 + x29*x98);

      // Evaluate second derivative of bfn wrt yz
      basis_yz_eval[ipt + 0*npts] = x83*x96;
      basis_yz_eval[ipt + 1*npts] = sqrt_15*x*(radial_eval_alpha_squared*x106 + x52 + x59);
      basis_yz_eval[ipt + 2*npts] = x92*(x6*x93 + x60 + x85);
      basis_yz_eval[ipt + 3*npts] = 0.5*x91*y;
      basis_yz_eval[ipt + 4*npts] = x90;
      basis_yz_eval[ipt + 5*npts] = x24*y*(x55 - x94 + x95);
      basis_yz_eval[ipt + 6*npts] = x88*(radial_eval_alpha_squared*x29 - x81);

      // Evaluate second derivative of bfn wrt zz
      basis_zz_eval[ipt + 0*npts] = x1*x109;
      basis_zz_eval[ipt + 1*npts] = x11*(x107 + x71);
      basis_zz_eval[ipt + 2*npts] = x112*x13;
      basis_zz_eval[ipt + 3*npts] = x19*(12.0*radial_eval + x113);
      basis_zz_eval[ipt + 4*npts] = x112*x23;
      basis_zz_eval[ipt + 5*npts] = x115*x25;
      basis_zz_eval[ipt + 6*npts] = x116*x27;

      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x1*(x109 + x67 + x69 + x99);
      basis_lapl_eval[ipt + 1*npts] = x11*(9.0*radial_eval_alpha + x117 + x65);
      basis_lapl_eval[ipt + 2*npts] = x13*(x101 + x111 + x118 + x75);
      basis_lapl_eval[ipt + 3*npts] = x19*(x113 + x121 + x22*x66 + x22*x98);
      basis_lapl_eval[ipt + 4*npts] = x23*(x100 + x111 + x122 + x77);
      basis_lapl_eval[ipt + 5*npts] = x25*(x115 + x123);
      basis_lapl_eval[ipt + 6*npts] = x27*(x116 + x119 + x29*x98 + x80);

      // Evaluate Laplacian gradient of bfn (dx)
      basis_lapl_x_eval[ipt + 0*npts] = x1*(x*x127 + x*x129 + x125*x8 + x131*x46 + x134);
      basis_lapl_x_eval[ipt + 1*npts] = x10*(x*x125 + x117 + x126*x2 + x128*x2 + x135*x3 + 3.0*x66 + x81);
      basis_lapl_x_eval[ipt + 2*npts] = x13*(x*x137*x15 + x*x141 + x*x142 + x131*x53 - x136 - x138 - x139 - x140 + x143);
      basis_lapl_x_eval[ipt + 3*npts] = x19*(6.0*radial_eval_alpha_squared*x*x62 + x*x126*x22 + x*x128*x22 - x*x144*x6 + x125*x22 - x134);
      basis_lapl_x_eval[ipt + 4*npts] = x12*(x*x143 + x108*x41 + x120 + x122 + x137*x89 + x141*x2 + x142*x2 + x145 + 3.0*x41*x66 + x41*x98);
      basis_lapl_x_eval[ipt + 5*npts] = x25*(-x*x135*x6 + x*x146 + x*x147 + x125*x26 + x131*x26 + x136 + x138 + x139 + x140);
      basis_lapl_x_eval[ipt + 6*npts] = x0*(x*x125*x29 + x148 + 3.0*x149 + x150*x2 + x151*x2 - x152 + x153);
      // Evaluate Laplacian gradient of bfn (dy)
      basis_lapl_y_eval[ipt + 0*npts] = x0*(x129*x5 + 3.0*x148 + x149 + x152 + x153 + x154*x8*y + x156*x5);
      basis_lapl_y_eval[ipt + 1*npts] = x50*(x107 + x128*x5 + x135*x6 + x154*y + x155*x5 + x157 + 3.0*x98);
      basis_lapl_y_eval[ipt + 2*npts] = x12*(x106*x137 + x108*x53 + x118 + x119 + x142*x5 + x145 + x158*y + x159*x5 + x53*x66 + 3.0*x53*x98);
      basis_lapl_y_eval[ipt + 3*npts] = -x19*(-6.0*radial_eval_alpha_squared*x62*y - x128*x22*y + x144*x3*y - x154*x22 - x155*x22*y + x160 + x162 + x163 + x164);
      basis_lapl_y_eval[ipt + 4*npts] = x23*(x137*x15*y + x142*y + x158 + x159*y + x166*x41 + x167);
      basis_lapl_y_eval[ipt + 5*npts] = x25*(x135*x3*y + x147*y + x154*x26 + x166*x26 + x167 + x168*y);
      basis_lapl_y_eval[ipt + 6*npts] = x27*(x151*y + x154*x29 - x160 - x162 - x163 - x164 + x166*x46 + x169*y);
      // Evaluate Laplacian gradient of bfn (dz)
      basis_lapl_z_eval[ipt + 0*npts] = x1*(x127*z + x156*z + x170*x3 + x172 + x173*x8);
      basis_lapl_z_eval[ipt + 1*npts] = x58*(3.0*x108 + x126*x14 + x135*x15 + x14*x155 + x157 + x173*z + x97);
      basis_lapl_z_eval[ipt + 2*npts] = x13*(x171*x53 - x174*x3 + x176);
      basis_lapl_z_eval[ipt + 3*npts] = -0.5*x106*x144 + 4.5*x108*x62 + 0.5*x121 + 0.5*x126*x14*x22 + 0.5*x133*x15 + 0.5*x14*x155*x22 - 0.5*x144*x89 + 0.5*x173*x22*z + 1.5*x62*x66 + 1.5*x62*x98;
      basis_lapl_z_eval[ipt + 4*npts] = x23*(x171*x41 - x174*x6 + x176);
      basis_lapl_z_eval[ipt + 5*npts] = x24*(-x106*x135 + 3.0*x114 + x123 + x135*x89 + x14*x146 + x14*x168 + x173*x26*z);
      basis_lapl_z_eval[ipt + 6*npts] = x27*(x150*z + x169*z - x170*x6 + x172 + x173*x29);




#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x1*x8;
      ang_eval_1 = radial_eval*x11;
      ang_eval_2 = radial_eval*x13*x18;
      ang_eval_3 = radial_eval*x19*x22;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x18*x23;
      ang_eval_1 = radial_eval*x25*x26;
      ang_eval_2 = radial_eval*x27*x29;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = x30*(radial_eval_alpha*x8 + x31);
      dang_eval_y_0 = x0*(radial_eval_alpha*x49 + x47);
      dang_eval_z_0 = x1*x56*x8;
      dang_eval_x_1 = x10*x32;
      dang_eval_y_1 = x50*x52;
      dang_eval_z_1 = x58*(radial_eval + x59);
      dang_eval_x_2 = x37;
      dang_eval_y_2 = x12*(radial_eval*x53 + radial_eval_alpha*x54);
      dang_eval_z_2 = x13*x61*z;
      dang_eval_x_3 = x38*x40;
      dang_eval_y_3 = x19*x40*y;
      dang_eval_z_3 = 1.5*radial_eval*x62 + 0.5*radial_eval_alpha*x63;
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

      dang_eval_x_0 = x12*(radial_eval*x41 + radial_eval_alpha*x42);
      dang_eval_y_0 = x37;
      dang_eval_z_0 = x23*x61*z;
      dang_eval_x_1 = x43*x45;
      dang_eval_y_1 = x25*x55*y;
      dang_eval_z_1 = x24*(radial_eval*x26 + radial_eval_alpha*x64);
      dang_eval_x_2 = x0*(radial_eval_alpha*x48 + x47);
      dang_eval_y_2 = x30*(radial_eval_alpha*x29 + x39);
      dang_eval_z_2 = x27*x29*x56;
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
