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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_cartesian_lapgrad_4(
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
      const auto x0 = x*x*x*x; 
      const auto x1 = radial_eval*y; 
      const auto x2 = x*x*x; 
      const auto x3 = radial_eval*z; 
      const auto x4 = x*x; 
      const auto x5 = x4; 
      const auto x6 = y*y; 
      const auto x7 = x6; 
      const auto x8 = x5*x7; 
      const auto x9 = x1*z; 
      const auto x10 = z*z; 
      const auto x11 = x10; 
      const auto x12 = x11*x5; 
      const auto x13 = radial_eval*x; 
      const auto x14 = y*y*y; 
      const auto x15 = x*x3; 
      const auto x16 = x*x1; 
      const auto x17 = z*z*z; 
      const auto x18 = y*y*y*y; 
      const auto x19 = x11*x7; 
      const auto x20 = z*z*z*z; 
      const auto x21 = x*x*x*x*x; 
      const auto x22 = 4.0*radial_eval; 
      const auto x23 = 3.0*radial_eval; 
      const auto x24 = radial_eval_alpha*x0 + x23*x5; 
      const auto x25 = 2.0*x13; 
      const auto x26 = x2*x7; 
      const auto x27 = radial_eval_alpha*x26; 
      const auto x28 = y*z; 
      const auto x29 = radial_eval_alpha*x2; 
      const auto x30 = x25 + x29; 
      const auto x31 = x11*x2; 
      const auto x32 = radial_eval_alpha*x31; 
      const auto x33 = radial_eval*x14; 
      const auto x34 = x14*x5; 
      const auto x35 = radial_eval_alpha*x34; 
      const auto x36 = radial_eval*x7; 
      const auto x37 = radial_eval_alpha*x8; 
      const auto x38 = x36 + x37; 
      const auto x39 = radial_eval*x11; 
      const auto x40 = radial_eval_alpha*x12; 
      const auto x41 = x39 + x40; 
      const auto x42 = radial_eval*x17; 
      const auto x43 = x17*x5; 
      const auto x44 = radial_eval_alpha*x43; 
      const auto x45 = radial_eval_alpha*x; 
      const auto x46 = x14*x45*z; 
      const auto x47 = x17*x45*y; 
      const auto x48 = radial_eval_alpha*y; 
      const auto x49 = radial_eval*x2; 
      const auto x50 = radial_eval_alpha*x2*x28; 
      const auto x51 = 2.0*x1; 
      const auto x52 = radial_eval*x5; 
      const auto x53 = x37 + x52; 
      const auto x54 = radial_eval_alpha*x18 + x23*x7; 
      const auto x55 = x*z; 
      const auto x56 = radial_eval_alpha*x14; 
      const auto x57 = x51 + x56; 
      const auto x58 = radial_eval_alpha*x19; 
      const auto x59 = y*y*y*y*y; 
      const auto x60 = x11*x14; 
      const auto x61 = radial_eval_alpha*x60; 
      const auto x62 = x17*x7; 
      const auto x63 = radial_eval_alpha*x62; 
      const auto x64 = radial_eval_alpha*z; 
      const auto x65 = 2.0*x3; 
      const auto x66 = x*y; 
      const auto x67 = radial_eval_alpha*x17; 
      const auto x68 = x65 + x67; 
      const auto x69 = radial_eval_alpha*x20 + x11*x23; 
      const auto x70 = z*z*z*z*z; 
      const auto x71 = 12.0*radial_eval; 
      const auto x72 = 8.0*radial_eval_alpha; 
      const auto x73 = radial_eval_alpha + radial_eval_alpha_squared*x5; 
      const auto x74 = x0*x72 + x0*x73 + x5*x71; 
      const auto x75 = 6.0*radial_eval_alpha; 
      const auto x76 = x2*x73; 
      const auto x77 = 6.0*x13 + x76; 
      const auto x78 = x2*x75 + x77; 
      const auto x79 = 4.0*radial_eval_alpha; 
      const auto x80 = x79*x8; 
      const auto x81 = 2.0*radial_eval; 
      const auto x82 = x7*x81; 
      const auto x83 = x5*x7*x73 + x82; 
      const auto x84 = x5*x73; 
      const auto x85 = x81 + x84; 
      const auto x86 = x12*x79; 
      const auto x87 = x11*x81; 
      const auto x88 = x11*x5*x73 + x87; 
      const auto x89 = 2.0*radial_eval_alpha; 
      const auto x90 = x14*x89; 
      const auto x91 = x14*x73; 
      const auto x92 = x7*x89; 
      const auto x93 = x7*x73; 
      const auto x94 = x11*x89; 
      const auto x95 = x11*x73; 
      const auto x96 = x17*x89; 
      const auto x97 = x17*x73; 
      const auto x98 = x18*x73; 
      const auto x99 = x11*x7*x73; 
      const auto x100 = x20*x73; 
      const auto x101 = radial_eval_alpha_squared*x21 + x2*x79; 
      const auto x102 = 3.0*radial_eval_alpha; 
      const auto x103 = x102*x8; 
      const auto x104 = x28*(radial_eval_alpha_squared*x0 + x102*x5); 
      const auto x105 = 2.0*x45; 
      const auto x106 = 2.0*x48; 
      const auto x107 = x105*x7; 
      const auto x108 = radial_eval_alpha_squared*x26; 
      const auto x109 = x107 + x108; 
      const auto x110 = x105*x11; 
      const auto x111 = radial_eval_alpha_squared*x31; 
      const auto x112 = x110 + x111; 
      const auto x113 = x106*x5; 
      const auto x114 = radial_eval_alpha_squared*x34; 
      const auto x115 = x113 + x114; 
      const auto x116 = radial_eval_alpha_squared*x11*x5*x7; 
      const auto x117 = x116 + x58; 
      const auto x118 = radial_eval_alpha_squared*x43; 
      const auto x119 = radial_eval_alpha_squared*x59 + x14*x79; 
      const auto x120 = x55*(radial_eval_alpha_squared*x18 + x102*x7); 
      const auto x121 = x106*x11; 
      const auto x122 = radial_eval_alpha_squared*x60; 
      const auto x123 = x121 + x122; 
      const auto x124 = radial_eval_alpha_squared*x62; 
      const auto x125 = x102*x12; 
      const auto x126 = 2.0*x64; 
      const auto x127 = x126*x5; 
      const auto x128 = x118 + x127; 
      const auto x129 = x126*x7; 
      const auto x130 = x124 + x129; 
      const auto x131 = x66*(radial_eval_alpha_squared*x20 + x102*x11); 
      const auto x132 = radial_eval_alpha_squared*x70 + x17*x79; 
      const auto x133 = radial_eval_alpha + radial_eval_alpha_squared*x7; 
      const auto x134 = x0*x133; 
      const auto x135 = x2*x89; 
      const auto x136 = x133*x2; 
      const auto x137 = x5*x81; 
      const auto x138 = x133*x5*x7 + x137; 
      const auto x139 = x5*x89; 
      const auto x140 = x133*x5; 
      const auto x141 = x11*x133*x5; 
      const auto x142 = x133*x14; 
      const auto x143 = 6.0*x1 + x142; 
      const auto x144 = x14*x75 + x143; 
      const auto x145 = x133*x7; 
      const auto x146 = x145 + x81; 
      const auto x147 = x11*x133; 
      const auto x148 = x133*x17; 
      const auto x149 = x133*x18 + x18*x72 + x7*x71; 
      const auto x150 = x19*x79; 
      const auto x151 = x11*x133*x7 + x87; 
      const auto x152 = x133*x20; 
      const auto x153 = x102*x19; 
      const auto x154 = radial_eval_alpha + radial_eval_alpha_squared*x11; 
      const auto x155 = x0*x154; 
      const auto x156 = x154*x2; 
      const auto x157 = x154*x5*x7; 
      const auto x158 = x154*x5; 
      const auto x159 = x11*x154*x5 + x137; 
      const auto x160 = x14*x154; 
      const auto x161 = x154*x7; 
      const auto x162 = x11*x154; 
      const auto x163 = x162 + x81; 
      const auto x164 = x154*x17; 
      const auto x165 = x164 + 6.0*x3; 
      const auto x166 = x165 + x17*x75; 
      const auto x167 = x154*x18; 
      const auto x168 = x11*x154*x7 + x82; 
      const auto x169 = x11*x71 + x154*x20 + x20*x72; 
      const auto x170 = x136 + x156 + x2*x72 + x77; 
      const auto x171 = x158 + x85; 
      const auto x172 = x14*x72 + x143 + x160 + x91; 
      const auto x173 = x146 + x161; 
      const auto x174 = x147 + x163; 
      const auto x175 = x148 + x165 + x17*x72 + x97; 
      const auto x176 = 36.0*radial_eval_alpha; 
      const auto x177 = radial_eval_alpha_cubed*x7 + radial_eval_alpha_squared; 
      const auto x178 = x0*x177; 
      const auto x179 = radial_eval_alpha_cubed*x11 + radial_eval_alpha_squared; 
      const auto x180 = x0*x179; 
      const auto x181 = radial_eval_alpha_squared*x; 
      const auto x182 = radial_eval_alpha_cubed*x2 + 3.0*x181; 
      const auto x183 = 6.0*radial_eval; 
      const auto x184 = 24.0*radial_eval_alpha; 
      const auto x185 = 2.0*radial_eval_alpha_squared; 
      const auto x186 = 3.0*x140; 
      const auto x187 = 3.0*x158; 
      const auto x188 = x177*x2; 
      const auto x189 = x179*x2; 
      const auto x190 = x*x188 + x*x189 + x0*x185 + x182*x2 + x183 + x184*x5 + x186 + x187 + 9.0*x84; 
      const auto x191 = 2.0*x; 
      const auto x192 = 4.0*radial_eval_alpha_squared; 
      const auto x193 = 6.0*x; 
      const auto x194 = 14.0*x45; 
      const auto x195 = x177*x5*x7; 
      const auto x196 = x179*x5*x7; 
      const auto x197 = 4.0*x13 + x135; 
      const auto x198 = x177*x5; 
      const auto x199 = x179*x5; 
      const auto x200 = x11*x177*x5; 
      const auto x201 = x11*x179*x5; 
      const auto x202 = x14*x182; 
      const auto x203 = x14*x177; 
      const auto x204 = x14*x179; 
      const auto x205 = 6.0*x48; 
      const auto x206 = 6.0*radial_eval_alpha_squared; 
      const auto x207 = 3.0*x93; 
      const auto x208 = x7*x75; 
      const auto x209 = x177*x7; 
      const auto x210 = x179*x7; 
      const auto x211 = x206*x8; 
      const auto x212 = 3.0*x95; 
      const auto x213 = x11*x75; 
      const auto x214 = x11*x177; 
      const auto x215 = x11*x179; 
      const auto x216 = x12*x206; 
      const auto x217 = x17*x182; 
      const auto x218 = x17*x177; 
      const auto x219 = x17*x179; 
      const auto x220 = 6.0*x64; 
      const auto x221 = 12.0*x45; 
      const auto x222 = 8.0*x181; 
      const auto x223 = x177*x18; 
      const auto x224 = x179*x18; 
      const auto x225 = 6.0*y; 
      const auto x226 = x225*x45; 
      const auto x227 = x11*x177*x7; 
      const auto x228 = x11*x179*x7; 
      const auto x229 = 6.0*z; 
      const auto x230 = x229*x45; 
      const auto x231 = x177*x20; 
      const auto x232 = x179*x20; 
      const auto x233 = 12.0*x48; 
      const auto x234 = radial_eval_alpha_squared*y; 
      const auto x235 = 8.0*x234; 
      const auto x236 = radial_eval_alpha_cubed*x5 + radial_eval_alpha_squared; 
      const auto x237 = x0*x236; 
      const auto x238 = radial_eval_alpha_cubed*x14 + 3.0*x234; 
      const auto x239 = x2*x238; 
      const auto x240 = x2*x236; 
      const auto x241 = 6.0*x45; 
      const auto x242 = 2.0*y; 
      const auto x243 = 14.0*x48; 
      const auto x244 = x236*x5*x7; 
      const auto x245 = 4.0*x1 + x90; 
      const auto x246 = x5*x75; 
      const auto x247 = x236*x5; 
      const auto x248 = x11*x236*x5; 
      const auto x249 = 3.0*x161; 
      const auto x250 = x14*x236; 
      const auto x251 = x14*x238 + 9.0*x145 + x18*x185 + x183 + x184*x7 + x204*y + x207 + x249 + x250*y; 
      const auto x252 = x236*x7; 
      const auto x253 = 3.0*x147; 
      const auto x254 = x11*x236; 
      const auto x255 = x19*x206; 
      const auto x256 = x28*x75; 
      const auto x257 = x17*x236; 
      const auto x258 = x17*x238; 
      const auto x259 = x18*x236; 
      const auto x260 = x11*x236*x7; 
      const auto x261 = x20*x236; 
      const auto x262 = 12.0*x64; 
      const auto x263 = radial_eval_alpha_squared*z; 
      const auto x264 = 8.0*x263; 
      const auto x265 = radial_eval_alpha_cubed*x17 + 3.0*x263; 
      const auto x266 = x2*x265; 
      const auto x267 = 2.0*z; 
      const auto x268 = 14.0*x64; 
      const auto x269 = 4.0*x3 + x96; 
      const auto x270 = x14*x265; 
      const auto x271 = x11*x184 + 9.0*x162 + x17*x265 + x183 + x185*x20 + x212 + x218*z + x253 + x257*z; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x0;
      basis_eval[ipt + 1*npts] = x1*x2;
      basis_eval[ipt + 2*npts] = x2*x3;
      basis_eval[ipt + 3*npts] = radial_eval*x8;
      basis_eval[ipt + 4*npts] = x5*x9;
      basis_eval[ipt + 5*npts] = radial_eval*x12;
      basis_eval[ipt + 6*npts] = x13*x14;
      basis_eval[ipt + 7*npts] = x15*x7;
      basis_eval[ipt + 8*npts] = x11*x16;
      basis_eval[ipt + 9*npts] = x13*x17;
      basis_eval[ipt + 10*npts] = radial_eval*x18;
      basis_eval[ipt + 11*npts] = x14*x3;
      basis_eval[ipt + 12*npts] = radial_eval*x19;
      basis_eval[ipt + 13*npts] = x1*x17;
      basis_eval[ipt + 14*npts] = radial_eval*x20;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = radial_eval_alpha*x21 + x2*x22;
      basis_x_eval[ipt + 1*npts] = x24*y;
      basis_x_eval[ipt + 2*npts] = x24*z;
      basis_x_eval[ipt + 3*npts] = x25*x7 + x27;
      basis_x_eval[ipt + 4*npts] = x28*x30;
      basis_x_eval[ipt + 5*npts] = x11*x25 + x32;
      basis_x_eval[ipt + 6*npts] = x33 + x35;
      basis_x_eval[ipt + 7*npts] = x38*z;
      basis_x_eval[ipt + 8*npts] = x41*y;
      basis_x_eval[ipt + 9*npts] = x42 + x44;
      basis_x_eval[ipt + 10*npts] = x18*x45;
      basis_x_eval[ipt + 11*npts] = x46;
      basis_x_eval[ipt + 12*npts] = x19*x45;
      basis_x_eval[ipt + 13*npts] = x47;
      basis_x_eval[ipt + 14*npts] = x20*x45;

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x0*x48;
      basis_y_eval[ipt + 1*npts] = x27 + x49;
      basis_y_eval[ipt + 2*npts] = x50;
      basis_y_eval[ipt + 3*npts] = x35 + x5*x51;
      basis_y_eval[ipt + 4*npts] = x53*z;
      basis_y_eval[ipt + 5*npts] = x12*x48;
      basis_y_eval[ipt + 6*npts] = x*x54;
      basis_y_eval[ipt + 7*npts] = x55*x57;
      basis_y_eval[ipt + 8*npts] = x*(x39 + x58);
      basis_y_eval[ipt + 9*npts] = x47;
      basis_y_eval[ipt + 10*npts] = radial_eval_alpha*x59 + x14*x22;
      basis_y_eval[ipt + 11*npts] = x54*z;
      basis_y_eval[ipt + 12*npts] = x11*x51 + x61;
      basis_y_eval[ipt + 13*npts] = x42 + x63;
      basis_y_eval[ipt + 14*npts] = x20*x48;

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x0*x64;
      basis_z_eval[ipt + 1*npts] = x50;
      basis_z_eval[ipt + 2*npts] = x32 + x49;
      basis_z_eval[ipt + 3*npts] = x64*x8;
      basis_z_eval[ipt + 4*npts] = y*(x40 + x52);
      basis_z_eval[ipt + 5*npts] = x44 + x5*x65;
      basis_z_eval[ipt + 6*npts] = x46;
      basis_z_eval[ipt + 7*npts] = x*(x36 + x58);
      basis_z_eval[ipt + 8*npts] = x66*x68;
      basis_z_eval[ipt + 9*npts] = x*x69;
      basis_z_eval[ipt + 10*npts] = x18*x64;
      basis_z_eval[ipt + 11*npts] = x33 + x61;
      basis_z_eval[ipt + 12*npts] = x63 + x65*x7;
      basis_z_eval[ipt + 13*npts] = x69*y;
      basis_z_eval[ipt + 14*npts] = radial_eval_alpha*x70 + x17*x22;

      // Evaluate second derivative of bfn wrt xx
      basis_xx_eval[ipt + 0*npts] = x74;
      basis_xx_eval[ipt + 1*npts] = x78*y;
      basis_xx_eval[ipt + 2*npts] = x78*z;
      basis_xx_eval[ipt + 3*npts] = x80 + x83;
      basis_xx_eval[ipt + 4*npts] = x28*(x5*x79 + x85);
      basis_xx_eval[ipt + 5*npts] = x86 + x88;
      basis_xx_eval[ipt + 6*npts] = x*(x90 + x91);
      basis_xx_eval[ipt + 7*npts] = x55*(x92 + x93);
      basis_xx_eval[ipt + 8*npts] = x66*(x94 + x95);
      basis_xx_eval[ipt + 9*npts] = x*(x96 + x97);
      basis_xx_eval[ipt + 10*npts] = x98;
      basis_xx_eval[ipt + 11*npts] = x91*z;
      basis_xx_eval[ipt + 12*npts] = x99;
      basis_xx_eval[ipt + 13*npts] = x97*y;
      basis_xx_eval[ipt + 14*npts] = x100;

      // Evaluate second derivative of bfn wrt xy
      basis_xy_eval[ipt + 0*npts] = x101*y;
      basis_xy_eval[ipt + 1*npts] = radial_eval_alpha_squared*x0*x7 + x103 + x24;
      basis_xy_eval[ipt + 2*npts] = x104;
      basis_xy_eval[ipt + 3*npts] = radial_eval_alpha_squared*x14*x2 + x105*x14 + x106*x2 + 4.0*x16;
      basis_xy_eval[ipt + 4*npts] = z*(x109 + x30);
      basis_xy_eval[ipt + 5*npts] = x112*y;
      basis_xy_eval[ipt + 6*npts] = radial_eval_alpha_squared*x18*x5 + x103 + x54;
      basis_xy_eval[ipt + 7*npts] = z*(x115 + x57);
      basis_xy_eval[ipt + 8*npts] = x117 + x41;
      basis_xy_eval[ipt + 9*npts] = y*(x118 + x67);
      basis_xy_eval[ipt + 10*npts] = x*x119;
      basis_xy_eval[ipt + 11*npts] = x120;
      basis_xy_eval[ipt + 12*npts] = x*x123;
      basis_xy_eval[ipt + 13*npts] = x*(x124 + x67);
      basis_xy_eval[ipt + 14*npts] = radial_eval_alpha_squared*x20*x66;

      // Evaluate second derivative of bfn wrt xz
      basis_xz_eval[ipt + 0*npts] = x101*z;
      basis_xz_eval[ipt + 1*npts] = x104;
      basis_xz_eval[ipt + 2*npts] = radial_eval_alpha_squared*x0*x11 + x125 + x24;
      basis_xz_eval[ipt + 3*npts] = x109*z;
      basis_xz_eval[ipt + 4*npts] = y*(x112 + x30);
      basis_xz_eval[ipt + 5*npts] = radial_eval_alpha_squared*x17*x2 + x105*x17 + x126*x2 + 4.0*x15;
      basis_xz_eval[ipt + 6*npts] = z*(x114 + x56);
      basis_xz_eval[ipt + 7*npts] = x117 + x38;
      basis_xz_eval[ipt + 8*npts] = y*(x128 + x68);
      basis_xz_eval[ipt + 9*npts] = radial_eval_alpha_squared*x20*x5 + x125 + x69;
      basis_xz_eval[ipt + 10*npts] = radial_eval_alpha_squared*x18*x55;
      basis_xz_eval[ipt + 11*npts] = x*(x122 + x56);
      basis_xz_eval[ipt + 12*npts] = x*x130;
      basis_xz_eval[ipt + 13*npts] = x131;
      basis_xz_eval[ipt + 14*npts] = x*x132;

      // Evaluate second derivative of bfn wrt yy
      basis_yy_eval[ipt + 0*npts] = x134;
      basis_yy_eval[ipt + 1*npts] = y*(x135 + x136);
      basis_yy_eval[ipt + 2*npts] = x136*z;
      basis_yy_eval[ipt + 3*npts] = x138 + x80;
      basis_yy_eval[ipt + 4*npts] = x28*(x139 + x140);
      basis_yy_eval[ipt + 5*npts] = x141;
      basis_yy_eval[ipt + 6*npts] = x*x144;
      basis_yy_eval[ipt + 7*npts] = x55*(x146 + x7*x79);
      basis_yy_eval[ipt + 8*npts] = x66*(x147 + x94);
      basis_yy_eval[ipt + 9*npts] = x*x148;
      basis_yy_eval[ipt + 10*npts] = x149;
      basis_yy_eval[ipt + 11*npts] = x144*z;
      basis_yy_eval[ipt + 12*npts] = x150 + x151;
      basis_yy_eval[ipt + 13*npts] = y*(x148 + x96);
      basis_yy_eval[ipt + 14*npts] = x152;

      // Evaluate second derivative of bfn wrt yz
      basis_yz_eval[ipt + 0*npts] = radial_eval_alpha_squared*x0*x28;
      basis_yz_eval[ipt + 1*npts] = z*(x108 + x29);
      basis_yz_eval[ipt + 2*npts] = y*(x111 + x29);
      basis_yz_eval[ipt + 3*npts] = x115*z;
      basis_yz_eval[ipt + 4*npts] = x116 + x40 + x53;
      basis_yz_eval[ipt + 5*npts] = x128*y;
      basis_yz_eval[ipt + 6*npts] = x120;
      basis_yz_eval[ipt + 7*npts] = x*(x123 + x57);
      basis_yz_eval[ipt + 8*npts] = x*(x130 + x68);
      basis_yz_eval[ipt + 9*npts] = x131;
      basis_yz_eval[ipt + 10*npts] = x119*z;
      basis_yz_eval[ipt + 11*npts] = radial_eval_alpha_squared*x11*x18 + x153 + x54;
      basis_yz_eval[ipt + 12*npts] = radial_eval_alpha_squared*x14*x17 + x106*x17 + x126*x14 + 4.0*x9;
      basis_yz_eval[ipt + 13*npts] = radial_eval_alpha_squared*x20*x7 + x153 + x69;
      basis_yz_eval[ipt + 14*npts] = x132*y;

      // Evaluate second derivative of bfn wrt zz
      basis_zz_eval[ipt + 0*npts] = x155;
      basis_zz_eval[ipt + 1*npts] = x156*y;
      basis_zz_eval[ipt + 2*npts] = z*(x135 + x156);
      basis_zz_eval[ipt + 3*npts] = x157;
      basis_zz_eval[ipt + 4*npts] = x28*(x139 + x158);
      basis_zz_eval[ipt + 5*npts] = x159 + x86;
      basis_zz_eval[ipt + 6*npts] = x*x160;
      basis_zz_eval[ipt + 7*npts] = x55*(x161 + x92);
      basis_zz_eval[ipt + 8*npts] = x66*(x11*x79 + x163);
      basis_zz_eval[ipt + 9*npts] = x*x166;
      basis_zz_eval[ipt + 10*npts] = x167;
      basis_zz_eval[ipt + 11*npts] = z*(x160 + x90);
      basis_zz_eval[ipt + 12*npts] = x150 + x168;
      basis_zz_eval[ipt + 13*npts] = x166*y;
      basis_zz_eval[ipt + 14*npts] = x169;

      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x134 + x155 + x74;
      basis_lapl_eval[ipt + 1*npts] = x170*y;
      basis_lapl_eval[ipt + 2*npts] = x170*z;
      basis_lapl_eval[ipt + 3*npts] = x138 + x157 + x72*x8 + x83;
      basis_lapl_eval[ipt + 4*npts] = x28*(x140 + x171 + x5*x72);
      basis_lapl_eval[ipt + 5*npts] = x12*x72 + x141 + x159 + x88;
      basis_lapl_eval[ipt + 6*npts] = x*x172;
      basis_lapl_eval[ipt + 7*npts] = x55*(x173 + x7*x72 + x93);
      basis_lapl_eval[ipt + 8*npts] = x66*(x11*x72 + x174 + x95);
      basis_lapl_eval[ipt + 9*npts] = x*x175;
      basis_lapl_eval[ipt + 10*npts] = x149 + x167 + x98;
      basis_lapl_eval[ipt + 11*npts] = x172*z;
      basis_lapl_eval[ipt + 12*npts] = x151 + x168 + x19*x72 + x99;
      basis_lapl_eval[ipt + 13*npts] = x175*y;
      basis_lapl_eval[ipt + 14*npts] = x100 + x152 + x169;

      // Evaluate Laplacian gradient of bfn (dx)
      basis_lapl_x_eval[ipt + 0*npts] = x*x178 + x*x180 + x0*x182 + 24.0*x13 + 4.0*x136 + 4.0*x156 + x176*x2 + 12.0*x76;
      basis_lapl_x_eval[ipt + 1*npts] = x190*y;
      basis_lapl_x_eval[ipt + 2*npts] = x190*z;
      basis_lapl_x_eval[ipt + 3*npts] = x*x195 + x*x196 + x145*x191 + x161*x191 + x182*x5*x7 + x192*x26 + x193*x93 + x194*x7 + x197;
      basis_lapl_x_eval[ipt + 4*npts] = x28*(x*x198 + x*x199 + x133*x191 + x154*x191 + x182*x5 + x192*x2 + x193*x73 + x194);
      basis_lapl_x_eval[ipt + 5*npts] = x*x200 + x*x201 + x11*x182*x5 + x11*x194 + x147*x191 + x162*x191 + x192*x31 + x193*x95 + x197;
      basis_lapl_x_eval[ipt + 6*npts] = x*x202 + x144 + x160 + x203*x4 + x204*x4 + x205*x5 + x206*x34 + 3.0*x91;
      basis_lapl_x_eval[ipt + 7*npts] = z*(x*x182*x7 + x139 + x173 + x207 + x208 + x209*x4 + x210*x4 + x211);
      basis_lapl_x_eval[ipt + 8*npts] = y*(x*x11*x182 + x139 + x174 + x212 + x213 + x214*x4 + x215*x4 + x216);
      basis_lapl_x_eval[ipt + 9*npts] = x*x217 + x148 + x166 + x206*x43 + x218*x4 + x219*x4 + x220*x5 + 3.0*x97;
      basis_lapl_x_eval[ipt + 10*npts] = x*x223 + x*x224 + x18*x182 + x18*x222 + x221*x7;
      basis_lapl_x_eval[ipt + 11*npts] = z*(x*x203 + x*x204 + x14*x222 + x202 + x226);
      basis_lapl_x_eval[ipt + 12*npts] = x*x227 + x*x228 + x107 + x11*x182*x7 + x110 + x19*x222;
      basis_lapl_x_eval[ipt + 13*npts] = y*(x*x218 + x*x219 + x17*x222 + x217 + x230);
      basis_lapl_x_eval[ipt + 14*npts] = x*x231 + x*x232 + x11*x221 + x182*x20 + x20*x222;
      // Evaluate Laplacian gradient of bfn (dy)
      basis_lapl_y_eval[ipt + 0*npts] = x0*x235 + x0*x238 + x180*y + x233*x5 + x237*y;
      basis_lapl_y_eval[ipt + 1*npts] = 3.0*x136 + x156 + x189*x6 + x206*x26 + x239*y + x240*x6 + x241*x7 + x78;
      basis_lapl_y_eval[ipt + 2*npts] = z*(x189*y + x2*x235 + x226 + x239 + x240*y);
      basis_lapl_y_eval[ipt + 3*npts] = x140*x225 + x158*x242 + x192*x34 + x196*y + x238*x5*x7 + x242*x84 + x243*x5 + x244*y + x245;
      basis_lapl_y_eval[ipt + 4*npts] = z*(x171 + x186 + x199*x6 + x211 + x238*x5*y + x246 + x247*x6 + x92);
      basis_lapl_y_eval[ipt + 5*npts] = x11*x238*x5 + x113 + x12*x235 + x121 + x201*y + x248*y;
      basis_lapl_y_eval[ipt + 6*npts] = x*x251;
      basis_lapl_y_eval[ipt + 7*npts] = x55*(x133*x225 + x14*x192 + x154*x242 + x210*y + x238*x7 + x242*x73 + x243 + x252*y);
      basis_lapl_y_eval[ipt + 8*npts] = x*(x11*x238*y + x163 + x213 + x215*x6 + x253 + x254*x6 + x255 + x92 + x95);
      basis_lapl_y_eval[ipt + 9*npts] = x*(x17*x235 + x219*y + x256 + x257*y + x258);
      basis_lapl_y_eval[ipt + 10*npts] = 24.0*x1 + x14*x176 + 12.0*x142 + 4.0*x160 + x18*x238 + x224*y + x259*y + 4.0*x91;
      basis_lapl_y_eval[ipt + 11*npts] = x251*z;
      basis_lapl_y_eval[ipt + 12*npts] = x11*x238*x7 + x11*x243 + x147*x225 + x162*x242 + x192*x60 + x228*y + x242*x95 + x245 + x260*y;
      basis_lapl_y_eval[ipt + 13*npts] = 3.0*x148 + x166 + x206*x62 + x219*x6 + x220*x7 + x257*x6 + x258*y + x97;
      basis_lapl_y_eval[ipt + 14*npts] = x11*x233 + x20*x235 + x20*x238 + x232*y + x261*y;
      // Evaluate Laplacian gradient of bfn (dz)
      basis_lapl_z_eval[ipt + 0*npts] = x0*x264 + x0*x265 + x178*z + x237*z + x262*x5;
      basis_lapl_z_eval[ipt + 1*npts] = y*(x188*z + x2*x264 + x230 + x240*z + x266);
      basis_lapl_z_eval[ipt + 2*npts] = x10*x188 + x10*x240 + x11*x241 + x136 + 3.0*x156 + x206*x31 + x266*z + x78;
      basis_lapl_z_eval[ipt + 3*npts] = x127 + x129 + x195*z + x244*z + x264*x8 + x265*x5*x7;
      basis_lapl_z_eval[ipt + 4*npts] = y*(x10*x198 + x10*x247 + x140 + x187 + x216 + x246 + x265*x5*z + x85 + x94);
      basis_lapl_z_eval[ipt + 5*npts] = x11*x265*x5 + x140*x267 + x158*x229 + x192*x43 + x200*z + x248*z + x267*x84 + x268*x5 + x269;
      basis_lapl_z_eval[ipt + 6*npts] = x*(x14*x264 + x203*z + x250*z + x256 + x270);
      basis_lapl_z_eval[ipt + 7*npts] = x*(x10*x209 + x10*x252 + x146 + x208 + x249 + x255 + x265*x7*z + x93 + x94);
      basis_lapl_z_eval[ipt + 8*npts] = x66*(x11*x265 + x133*x267 + x154*x229 + x17*x192 + x214*z + x254*z + x267*x73 + x268);
      basis_lapl_z_eval[ipt + 9*npts] = x*x271;
      basis_lapl_z_eval[ipt + 10*npts] = x18*x264 + x18*x265 + x223*z + x259*z + x262*x7;
      basis_lapl_z_eval[ipt + 11*npts] = x10*x203 + x10*x250 + x11*x205 + x144 + 3.0*x160 + x206*x60 + x270*z + x91;
      basis_lapl_z_eval[ipt + 12*npts] = x11*x265*x7 + x145*x267 + x161*x229 + x192*x62 + x227*z + x260*z + x267*x93 + x268*x7 + x269;
      basis_lapl_z_eval[ipt + 13*npts] = x271*y;
      basis_lapl_z_eval[ipt + 14*npts] = 4.0*x148 + 12.0*x164 + x17*x176 + x20*x265 + x231*z + x261*z + 24.0*x3 + 4.0*x97;




#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x0;
      ang_eval_1 = x1*x2;
      ang_eval_2 = x2*x3;
      ang_eval_3 = radial_eval*x8;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = x5*x9;
      ang_eval_1 = radial_eval*x12;
      ang_eval_2 = x13*x14;
      ang_eval_3 = x15*x7;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = x11*x16;
      ang_eval_1 = x13*x17;
      ang_eval_2 = radial_eval*x18;
      ang_eval_3 = x14*x3;
      basis_eval[ipt + 8*npts] = ang_eval_0;
      basis_eval[ipt + 9*npts] = ang_eval_1;
      basis_eval[ipt + 10*npts] = ang_eval_2;
      basis_eval[ipt + 11*npts] = ang_eval_3;

      ang_eval_0 = radial_eval*x19;
      ang_eval_1 = x1*x17;
      ang_eval_2 = radial_eval*x20;
      basis_eval[ipt + 12*npts] = ang_eval_0;
      basis_eval[ipt + 13*npts] = ang_eval_1;
      basis_eval[ipt + 14*npts] = ang_eval_2;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = radial_eval_alpha*x21 + x2*x22;
      dang_eval_y_0 = x0*x48;
      dang_eval_z_0 = x0*x64;
      dang_eval_x_1 = x24*y;
      dang_eval_y_1 = x27 + x49;
      dang_eval_z_1 = x50;
      dang_eval_x_2 = x24*z;
      dang_eval_y_2 = x50;
      dang_eval_z_2 = x32 + x49;
      dang_eval_x_3 = x25*x7 + x27;
      dang_eval_y_3 = x35 + x5*x51;
      dang_eval_z_3 = x64*x8;
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

      dang_eval_x_0 = x28*x30;
      dang_eval_y_0 = x53*z;
      dang_eval_z_0 = y*(x40 + x52);
      dang_eval_x_1 = x11*x25 + x32;
      dang_eval_y_1 = x12*x48;
      dang_eval_z_1 = x44 + x5*x65;
      dang_eval_x_2 = x33 + x35;
      dang_eval_y_2 = x*x54;
      dang_eval_z_2 = x46;
      dang_eval_x_3 = x38*z;
      dang_eval_y_3 = x55*x57;
      dang_eval_z_3 = x*(x36 + x58);
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

      dang_eval_x_0 = x41*y;
      dang_eval_y_0 = x*(x39 + x58);
      dang_eval_z_0 = x66*x68;
      dang_eval_x_1 = x42 + x44;
      dang_eval_y_1 = x47;
      dang_eval_z_1 = x*x69;
      dang_eval_x_2 = x18*x45;
      dang_eval_y_2 = radial_eval_alpha*x59 + x14*x22;
      dang_eval_z_2 = x18*x64;
      dang_eval_x_3 = x46;
      dang_eval_y_3 = x54*z;
      dang_eval_z_3 = x33 + x61;
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

      dang_eval_x_0 = x19*x45;
      dang_eval_y_0 = x11*x51 + x61;
      dang_eval_z_0 = x63 + x65*x7;
      dang_eval_x_1 = x47;
      dang_eval_y_1 = x42 + x63;
      dang_eval_z_1 = x69*y;
      dang_eval_x_2 = x20*x45;
      dang_eval_y_2 = x20*x48;
      dang_eval_z_2 = radial_eval_alpha*x70 + x17*x22;
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
