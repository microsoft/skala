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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_spherical_lapgrad_4(
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
      const auto x0 = 0.5*sqrt_35; 
      const auto x1 = x0*y; 
      const auto x2 = x*x1; 
      const auto x3 = x*x; 
      const auto x4 = x3; 
      const auto x5 = y*y; 
      const auto x6 = x5; 
      const auto x7 = -x6; 
      const auto x8 = x4 + x7; 
      const auto x9 = 0.25*sqrt_70; 
      const auto x10 = x9*z; 
      const auto x11 = x10*y; 
      const auto x12 = 3.0*x4; 
      const auto x13 = x12 + x7; 
      const auto x14 = 0.5*sqrt_5; 
      const auto x15 = x14*y; 
      const auto x16 = x*x15; 
      const auto x17 = z*z; 
      const auto x18 = x17; 
      const auto x19 = -6.0*x18; 
      const auto x20 = x19 + x6; 
      const auto x21 = -x20 - x4; 
      const auto x22 = 0.25*sqrt_10; 
      const auto x23 = x22*z; 
      const auto x24 = x23*y; 
      const auto x25 = -4.0*x18; 
      const auto x26 = 3.0*x6; 
      const auto x27 = x25 + x26; 
      const auto x28 = -x12 - x27; 
      const auto x29 = 0.125*radial_eval; 
      const auto x30 = x*x*x*x; 
      const auto x31 = y*y*y*y; 
      const auto x32 = x4*x6; 
      const auto x33 = 6.0*x32; 
      const auto x34 = x18*x4; 
      const auto x35 = x18*x6; 
      const auto x36 = 3.0*x30 + 3.0*x31 + x33 - 24.0*x34 - 24.0*x35 + 8.0*(z*z*z*z); 
      const auto x37 = x*x23; 
      const auto x38 = 0.25*sqrt_5; 
      const auto x39 = -x30 + x31 + 6.0*x34 - 6.0*x35; 
      const auto x40 = x*x10; 
      const auto x41 = -x26; 
      const auto x42 = x4 + x41; 
      const auto x43 = x30 + x31 - x33; 
      const auto x44 = radial_eval*x13; 
      const auto x45 = x4*x8; 
      const auto x46 = x*x11; 
      const auto x47 = 6.0*radial_eval; 
      const auto x48 = radial_eval_alpha*x13; 
      const auto x49 = x47 + x48; 
      const auto x50 = -x12 - x20; 
      const auto x51 = x21*x4; 
      const auto x52 = -x47; 
      const auto x53 = x*x24*(radial_eval_alpha*x28 + x52); 
      const auto x54 = 12.0*radial_eval; 
      const auto x55 = x*x*x; 
      const auto x56 = 4.0*x; 
      const auto x57 = x*x6 - x18*x56 + x55; 
      const auto x58 = radial_eval_alpha*x; 
      const auto x59 = 9.0*x4; 
      const auto x60 = -x27 - x59; 
      const auto x61 = x28*x4; 
      const auto x62 = 4.0*radial_eval; 
      const auto x63 = 3.0*x; 
      const auto x64 = x18*x63 - x55; 
      const auto x65 = x12 + x41; 
      const auto x66 = radial_eval*x65; 
      const auto x67 = x4*x42; 
      const auto x68 = radial_eval_alpha*x67 + x66; 
      const auto x69 = 0.125*sqrt_35; 
      const auto x70 = x55 - x6*x63; 
      const auto x71 = x*x0; 
      const auto x72 = radial_eval*x42; 
      const auto x73 = x6*x8; 
      const auto x74 = x13*x6; 
      const auto x75 = radial_eval_alpha*x74 + x66; 
      const auto x76 = x*x14; 
      const auto x77 = x19 + x26; 
      const auto x78 = -x4 - x77; 
      const auto x79 = x21*x6; 
      const auto x80 = 9.0*x6; 
      const auto x81 = x12 + x25; 
      const auto x82 = -x80 - x81; 
      const auto x83 = x28*x6; 
      const auto x84 = y*y*y; 
      const auto x85 = 4.0*y; 
      const auto x86 = -x18*x85 + x4*y + x84; 
      const auto x87 = radial_eval_alpha*y; 
      const auto x88 = 3.0*y; 
      const auto x89 = -x18*x88 + x84; 
      const auto x90 = radial_eval_alpha*x42; 
      const auto x91 = x52 + x90; 
      const auto x92 = -x4*x88 + x84; 
      const auto x93 = x1*z; 
      const auto x94 = x9*y; 
      const auto x95 = x13*x18; 
      const auto x96 = x22*y; 
      const auto x97 = -12.0*x18; 
      const auto x98 = x26 + x97; 
      const auto x99 = -x12 - x98; 
      const auto x100 = x18*x28; 
      const auto x101 = radial_eval*x99 + radial_eval_alpha*x100; 
      const auto x102 = z*z*z; 
      const auto x103 = 3.0*z; 
      const auto x104 = 2.0*x102 - x103*x4 - x103*x6; 
      const auto x105 = radial_eval_alpha*z; 
      const auto x106 = x*x22; 
      const auto x107 = x38*z; 
      const auto x108 = x54*x8; 
      const auto x109 = x*x9; 
      const auto x110 = x18*x42; 
      const auto x111 = 2.0*radial_eval_alpha; 
      const auto x112 = x111*x13; 
      const auto x113 = radial_eval_alpha + radial_eval_alpha_squared*x4; 
      const auto x114 = x113*x8; 
      const auto x115 = x112 + x114; 
      const auto x116 = x113*x13; 
      const auto x117 = 12.0*radial_eval_alpha; 
      const auto x118 = x117*x4; 
      const auto x119 = x118 + x47; 
      const auto x120 = x111*x50 + x113*x21; 
      const auto x121 = x6 + x81; 
      const auto x122 = x113*x36 + x121*x54 + 24.0*x57*x58; 
      const auto x123 = -18.0*radial_eval; 
      const auto x124 = x113*x28; 
      const auto x125 = x111*x60 + x124; 
      const auto x126 = -x4; 
      const auto x127 = x126 + x18; 
      const auto x128 = 8.0*x58; 
      const auto x129 = x113*x39 + x127*x54 + x128*x64; 
      const auto x130 = x111*x65; 
      const auto x131 = x113*x42 + x130; 
      const auto x132 = x108 + x113*x43 + x128*x70; 
      const auto x133 = radial_eval_alpha*x3; 
      const auto x134 = radial_eval_alpha*x5; 
      const auto x135 = 6.0*radial_eval_alpha; 
      const auto x136 = x135*x6; 
      const auto x137 = radial_eval_alpha*x65; 
      const auto x138 = -x12 - x77; 
      const auto x139 = 24.0*radial_eval; 
      const auto x140 = x*x139; 
      const auto x141 = x140*y; 
      const auto x142 = 12.0*x58; 
      const auto x143 = 12.0*x87; 
      const auto x144 = radial_eval_alpha_squared*x; 
      const auto x145 = x144*y; 
      const auto x146 = -x135*x4 + x52; 
      const auto x147 = radial_eval_alpha*x56; 
      const auto x148 = radial_eval_alpha*x85; 
      const auto x149 = x*x94; 
      const auto x150 = x135*x18; 
      const auto x151 = -x150; 
      const auto x152 = x*x96*(radial_eval_alpha*x99 + radial_eval_alpha_squared*x100 + x151 + x52); 
      const auto x153 = 96.0*radial_eval*z; 
      const auto x154 = 12.0*x105; 
      const auto x155 = x144*z; 
      const auto x156 = -x59 - x98; 
      const auto x157 = radial_eval_alpha*x17; 
      const auto x158 = x142*x8; 
      const auto x159 = 4.0*radial_eval_alpha; 
      const auto x160 = x157*x65; 
      const auto x161 = x69*z; 
      const auto x162 = x111*x42; 
      const auto x163 = radial_eval_alpha + radial_eval_alpha_squared*x6; 
      const auto x164 = x163*x8; 
      const auto x165 = x162 + x164; 
      const auto x166 = x13*x163 + x130; 
      const auto x167 = x111*x78 + x163*x21; 
      const auto x168 = x163*x28; 
      const auto x169 = x111*x82 + x168; 
      const auto x170 = x27 + x4; 
      const auto x171 = x163*x36 + x170*x54 + 24.0*x86*x87; 
      const auto x172 = x117*x6; 
      const auto x173 = x172 + x47; 
      const auto x174 = -x18 + x6; 
      const auto x175 = 8.0*x87; 
      const auto x176 = x163*x39 + x174*x54 + x175*x89; 
      const auto x177 = x126 + x6; 
      const auto x178 = x163*x43 + x175*x92 + x177*x54; 
      const auto x179 = -x12 - x80 - x97; 
      const auto x180 = radial_eval_alpha_squared*y; 
      const auto x181 = x180*z; 
      const auto x182 = x143*x8; 
      const auto x183 = radial_eval_alpha + radial_eval_alpha_squared*x18; 
      const auto x184 = x183*x8; 
      const auto x185 = x13*x183; 
      const auto x186 = x112 + x185; 
      const auto x187 = 24.0*radial_eval_alpha*x18; 
      const auto x188 = x183*x21 + x187; 
      const auto x189 = x111*x99 + x183*x28; 
      const auto x190 = x139 + x189; 
      const auto x191 = 2.0*x18 - x4 - x6; 
      const auto x192 = 48.0*radial_eval*x191 + 32.0*x104*x105 + x183*x36; 
      const auto x193 = x108 + 24.0*x157*x8 + x183*x39; 
      const auto x194 = x183*x42; 
      const auto x195 = x162 + x194; 
      const auto x196 = x183*x43; 
      const auto x197 = x118 + x166; 
      const auto x198 = x116 + x197; 
      const auto x199 = -x118; 
      const auto x200 = -x172; 
      const auto x201 = x163*x42; 
      const auto x202 = x131 + x200; 
      const auto x203 = x201 + x202; 
      const auto x204 = radial_eval_alpha_cubed*x55 + radial_eval_alpha_squared*x63; 
      const auto x205 = radial_eval_alpha_cubed*x6 + radial_eval_alpha_squared; 
      const auto x206 = x205*x8; 
      const auto x207 = radial_eval_alpha_cubed*x18 + radial_eval_alpha_squared; 
      const auto x208 = x207*x8; 
      const auto x209 = 2.0*radial_eval_alpha_squared; 
      const auto x210 = x209*x3; 
      const auto x211 = 36.0*x58; 
      const auto x212 = 18.0*x*x113; 
      const auto x213 = 6.0*x; 
      const auto x214 = x163*x213; 
      const auto x215 = x183*x213; 
      const auto x216 = 2.0*x144; 
      const auto x217 = x13*x205; 
      const auto x218 = x13*x207; 
      const auto x219 = x205*x21; 
      const auto x220 = x207*x21; 
      const auto x221 = 24.0*radial_eval_alpha_squared; 
      const auto x222 = x111*x138 + x187; 
      const auto x223 = x205*x28; 
      const auto x224 = x207*x28; 
      const auto x225 = x204*x28; 
      const auto x226 = 48.0*x58; 
      const auto x227 = x226*x6; 
      const auto x228 = 24.0*x145; 
      const auto x229 = x205*x36; 
      const auto x230 = x207*x36; 
      const auto x231 = 36.0*radial_eval_alpha; 
      const auto x232 = x111*x156; 
      const auto x233 = 12.0*radial_eval_alpha_squared; 
      const auto x234 = x233*x32; 
      const auto x235 = -x234; 
      const auto x236 = x200 + x235; 
      const auto x237 = 8.0*x145; 
      const auto x238 = 24.0*x17; 
      const auto x239 = x205*x39; 
      const auto x240 = x207*x39; 
      const auto x241 = x163*x65; 
      const auto x242 = x113*x65; 
      const auto x243 = x205*x42; 
      const auto x244 = x207*x42; 
      const auto x245 = x118 + x130 + x183*x65; 
      const auto x246 = x205*x43; 
      const auto x247 = x207*x43; 
      const auto x248 = radial_eval_alpha_cubed*x84 + radial_eval_alpha_squared*x88; 
      const auto x249 = radial_eval_alpha_cubed*x4 + radial_eval_alpha_squared; 
      const auto x250 = x249*x8; 
      const auto x251 = x209*x5; 
      const auto x252 = x13*x249; 
      const auto x253 = x21*x249; 
      const auto x254 = x248*x28; 
      const auto x255 = x249*x28; 
      const auto x256 = x111*x179 + x199; 
      const auto x257 = 48.0*x87; 
      const auto x258 = x257*x4; 
      const auto x259 = 36.0*x87; 
      const auto x260 = x249*x36; 
      const auto x261 = 2.0*x180; 
      const auto x262 = 6.0*y; 
      const auto x263 = -x113*x262 - 18.0*x163*y - x183*x262 - x259; 
      const auto x264 = x249*x39; 
      const auto x265 = x249*x42; 
      const auto x266 = x249*x43; 
      const auto x267 = x209*z; 
      const auto x268 = radial_eval_alpha_cubed*x102 + radial_eval_alpha_squared*x103; 
      const auto x269 = x17*x209; 
      const auto x270 = x269*x65; 
      const auto x271 = x233*x34; 
      const auto x272 = 12.0*z; 
      const auto x273 = 36.0*z; 
      const auto x274 = 48.0*radial_eval_alpha*x18 + x113*x99 + x163*x99 + x17*x223 + x17*x255 + 3.0*x183*x99 + x268*x28*z; 
      const auto x275 = 192.0*x105; 
      const auto x276 = -x233*x35; 
      const auto x277 = 48.0*x105; 
      const auto x278 = 8.0*x181; 
      const auto x279 = 8.0*x155; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x2*x8;
      basis_eval[ipt + 1*npts] = radial_eval*x11*x13;
      basis_eval[ipt + 2*npts] = radial_eval*x16*x21;
      basis_eval[ipt + 3*npts] = radial_eval*x24*x28;
      basis_eval[ipt + 4*npts] = x29*x36;
      basis_eval[ipt + 5*npts] = radial_eval*x28*x37;
      basis_eval[ipt + 6*npts] = radial_eval*x38*x39;
      basis_eval[ipt + 7*npts] = radial_eval*x40*x42;
      basis_eval[ipt + 8*npts] = sqrt_35*x29*x43;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = x1*(radial_eval_alpha*x45 + x44);
      basis_x_eval[ipt + 1*npts] = x46*x49;
      basis_x_eval[ipt + 2*npts] = x15*(radial_eval*x50 + radial_eval_alpha*x51);
      basis_x_eval[ipt + 3*npts] = x53;
      basis_x_eval[ipt + 4*npts] = 0.125*x36*x58 + 0.125*x54*x57;
      basis_x_eval[ipt + 5*npts] = x23*(radial_eval*x60 + radial_eval_alpha*x61);
      basis_x_eval[ipt + 6*npts] = x38*(x39*x58 + x62*x64);
      basis_x_eval[ipt + 7*npts] = x10*x68;
      basis_x_eval[ipt + 8*npts] = x69*(x43*x58 + x62*x70);

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x71*(radial_eval_alpha*x73 + x72);
      basis_y_eval[ipt + 1*npts] = x10*x75;
      basis_y_eval[ipt + 2*npts] = x76*(radial_eval*x78 + radial_eval_alpha*x79);
      basis_y_eval[ipt + 3*npts] = x23*(radial_eval*x82 + radial_eval_alpha*x83);
      basis_y_eval[ipt + 4*npts] = 0.125*x36*x87 + 0.125*x54*x86;
      basis_y_eval[ipt + 5*npts] = x53;
      basis_y_eval[ipt + 6*npts] = x38*(x39*x87 + x62*x89);
      basis_y_eval[ipt + 7*npts] = x46*x91;
      basis_y_eval[ipt + 8*npts] = x69*(x43*x87 + x62*x92);

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x58*x8*x93;
      basis_z_eval[ipt + 1*npts] = x94*(radial_eval_alpha*x95 + x44);
      basis_z_eval[ipt + 2*npts] = x16*z*(radial_eval_alpha*x21 + x54);
      basis_z_eval[ipt + 3*npts] = x101*x96;
      basis_z_eval[ipt + 4*npts] = 2.0*radial_eval*x104 + 0.125*x105*x36;
      basis_z_eval[ipt + 5*npts] = x101*x106;
      basis_z_eval[ipt + 6*npts] = x107*(radial_eval_alpha*x39 + x108);
      basis_z_eval[ipt + 7*npts] = x109*(radial_eval_alpha*x110 + x72);
      basis_z_eval[ipt + 8*npts] = x105*x43*x69;

      // Evaluate second derivative of bfn wrt xx
      basis_xx_eval[ipt + 0*npts] = x2*(x115 + x47);
      basis_xx_eval[ipt + 1*npts] = x11*(x116 + x119);
      basis_xx_eval[ipt + 2*npts] = x16*(x120 + x52);
      basis_xx_eval[ipt + 3*npts] = x24*(x113*x28 - x119);
      basis_xx_eval[ipt + 4*npts] = 0.125*x122;
      basis_xx_eval[ipt + 5*npts] = x37*(x123 + x125);
      basis_xx_eval[ipt + 6*npts] = x129*x38;
      basis_xx_eval[ipt + 7*npts] = x40*(x131 + x47);
      basis_xx_eval[ipt + 8*npts] = x132*x69;

      // Evaluate second derivative of bfn wrt xy
      basis_xy_eval[ipt + 0*npts] = x0*(radial_eval_alpha_squared*x4*x6*x8 + x13*x134 + x133*x42 + x66);
      basis_xy_eval[ipt + 1*npts] = x40*(radial_eval_alpha_squared*x74 + x136 + x137 + x47);
      basis_xy_eval[ipt + 2*npts] = x14*(radial_eval*x138 + radial_eval_alpha_squared*x21*x4*x6 + x133*x78 + x134*x50);
      basis_xy_eval[ipt + 3*npts] = x37*(radial_eval_alpha*x82 + radial_eval_alpha_squared*x83 - x136 + x52);
      basis_xy_eval[ipt + 4*npts] = 0.125*x141 + 0.125*x142*x86 + 0.125*x143*x57 + 0.125*x145*x36;
      basis_xy_eval[ipt + 5*npts] = x24*(radial_eval_alpha*x60 + radial_eval_alpha_squared*x61 + x146);
      basis_xy_eval[ipt + 6*npts] = x38*(x145*x39 + x147*x89 + x148*x64);
      basis_xy_eval[ipt + 7*npts] = x11*(radial_eval_alpha_squared*x67 + x137 + x146);
      basis_xy_eval[ipt + 8*npts] = x69*(-x141 + x145*x43 + x147*x92 + x148*x70);

      // Evaluate second derivative of bfn wrt xz
      basis_xz_eval[ipt + 0*npts] = x93*(radial_eval_alpha_squared*x45 + x48);
      basis_xz_eval[ipt + 1*npts] = x149*(radial_eval_alpha_squared*x95 + x150 + x49);
      basis_xz_eval[ipt + 2*npts] = x15*z*(radial_eval_alpha*x50 + radial_eval_alpha_squared*x51 + x118 + x54);
      basis_xz_eval[ipt + 3*npts] = x152;
      basis_xz_eval[ipt + 4*npts] = -0.125*x*x153 + 2.0*x104*x58 + 0.125*x154*x57 + 0.125*x155*x36;
      basis_xz_eval[ipt + 5*npts] = x22*(radial_eval*x156 + radial_eval_alpha_squared*x18*x28*x4 + x133*x99 + x157*x60);
      basis_xz_eval[ipt + 6*npts] = x107*(x140 + x144*x39 + x158 + x159*x64);
      basis_xz_eval[ipt + 7*npts] = x9*(radial_eval_alpha_squared*x18*x4*x42 + x160 + x68);
      basis_xz_eval[ipt + 8*npts] = x161*(x144*x43 + x159*x70);

      // Evaluate second derivative of bfn wrt yy
      basis_yy_eval[ipt + 0*npts] = x2*(x165 + x52);
      basis_yy_eval[ipt + 1*npts] = x11*(x166 + x52);
      basis_yy_eval[ipt + 2*npts] = x16*(x167 + x52);
      basis_yy_eval[ipt + 3*npts] = x24*(x123 + x169);
      basis_yy_eval[ipt + 4*npts] = 0.125*x171;
      basis_yy_eval[ipt + 5*npts] = x37*(x163*x28 - x173);
      basis_yy_eval[ipt + 6*npts] = x176*x38;
      basis_yy_eval[ipt + 7*npts] = x40*(x163*x42 - x173);
      basis_yy_eval[ipt + 8*npts] = x178*x69;

      // Evaluate second derivative of bfn wrt yz
      basis_yz_eval[ipt + 0*npts] = x71*z*(radial_eval_alpha_squared*x73 + x90);
      basis_yz_eval[ipt + 1*npts] = x9*(radial_eval_alpha_squared*x13*x18*x6 + x160 + x75);
      basis_yz_eval[ipt + 2*npts] = x76*z*(radial_eval_alpha*x78 + radial_eval_alpha_squared*x79 + x172 + x54);
      basis_yz_eval[ipt + 3*npts] = x22*(radial_eval*x179 + radial_eval_alpha_squared*x18*x28*x6 + x134*x99 + x157*x82);
      basis_yz_eval[ipt + 4*npts] = 2.0*x104*x87 - 0.125*x153*y + 0.125*x154*x86 + 0.125*x181*x36;
      basis_yz_eval[ipt + 5*npts] = x152;
      basis_yz_eval[ipt + 6*npts] = x107*(-x139*y + x159*x89 + x180*x39 + x182);
      basis_yz_eval[ipt + 7*npts] = x149*(radial_eval_alpha_squared*x110 + x151 + x91);
      basis_yz_eval[ipt + 8*npts] = x161*(x159*x92 + x180*x43);

      // Evaluate second derivative of bfn wrt zz
      basis_zz_eval[ipt + 0*npts] = x184*x2;
      basis_zz_eval[ipt + 1*npts] = x11*x186;
      basis_zz_eval[ipt + 2*npts] = x16*(x188 + x54);
      basis_zz_eval[ipt + 3*npts] = x190*x24;
      basis_zz_eval[ipt + 4*npts] = 0.125*x192;
      basis_zz_eval[ipt + 5*npts] = x190*x37;
      basis_zz_eval[ipt + 6*npts] = x193*x38;
      basis_zz_eval[ipt + 7*npts] = x195*x40;
      basis_zz_eval[ipt + 8*npts] = x196*x69;

      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x2*(x115 + x165 + x184);
      basis_lapl_eval[ipt + 1*npts] = x11*(x186 + x198);
      basis_lapl_eval[ipt + 2*npts] = x16*(x120 + x167 + x188);
      basis_lapl_eval[ipt + 3*npts] = x24*(x124 + x169 + x189 + x199);
      basis_lapl_eval[ipt + 4*npts] = 0.125*x122 + 0.125*x171 + 0.125*x192;
      basis_lapl_eval[ipt + 5*npts] = x37*(x125 + x168 + x189 + x200);
      basis_lapl_eval[ipt + 6*npts] = x38*(x129 + x176 + x193);
      basis_lapl_eval[ipt + 7*npts] = x40*(x195 + x203);
      basis_lapl_eval[ipt + 8*npts] = x69*(x132 + x178 + x196);

      // Evaluate Laplacian gradient of bfn (dx)
      basis_lapl_x_eval[ipt + 0*npts] = x1*(x*x204*x8 + 3.0*x116 + x185 + x197 + x206*x3 + x208*x3 + x210*x42);
      basis_lapl_x_eval[ipt + 1*npts] = x11*(x*x217 + x*x218 + x13*x204 + x13*x216 + x211 + x212 + x214 + x215 + x216*x65);
      basis_lapl_x_eval[ipt + 2*npts] = x15*(x*x204*x21 + 3.0*x113*x50 + x163*x50 + x183*x50 + x199 + x210*x78 + x219*x3 + x220*x3 + x221*x34 + x222);
      basis_lapl_x_eval[ipt + 3*npts] = x24*(x*x223 + x*x224 - x211 - x212 - x214 - x215 + x216*x82 + x216*x99 + x225);
      basis_lapl_x_eval[ipt + 4*npts] = 0.125*x*x229 + 0.125*x*x230 + 4.0*x104*x155 + 4.5*x113*x57 + 0.125*x121*x211 + 0.125*x142*x170 + 1.5*x163*x57 - 24.0*x18*x58 + 1.5*x183*x57 + 0.125*x191*x226 + 0.125*x204*x36 + 0.125*x227 + 0.125*x228*x86;
      basis_lapl_x_eval[ipt + 5*npts] = x23*(x*x225 + 3.0*x113*x60 + x163*x60 + x183*x60 + x210*x99 + x223*x3 + x224*x3 - x231*x4 + x232 + x236);
      basis_lapl_x_eval[ipt + 6*npts] = x38*(x*x239 + x*x240 + 12.0*x113*x64 + x127*x211 + x142*x174 + x144*x238*x8 + x158 + 4.0*x163*x64 + x18*x226 + 4.0*x183*x64 + x204*x39 + x237*x89);
      basis_lapl_x_eval[ipt + 7*npts] = x10*(x*x204*x42 + x209*x67 + x236 + x241 + 3.0*x242 + x243*x3 + x244*x3 + x245);
      basis_lapl_x_eval[ipt + 8*npts] = x69*(x*x246 + x*x247 + 12.0*x113*x70 + x142*x177 + 4.0*x163*x70 + 4.0*x183*x70 + x204*x43 + x211*x8 - x227 + x237*x92);
      // Evaluate Laplacian gradient of bfn (dy)
      basis_lapl_y_eval[ipt + 0*npts] = x71*(x13*x251 + x194 + 3.0*x201 + x202 + x208*x5 + x248*x8*y + x250*x5);
      basis_lapl_y_eval[ipt + 1*npts] = x10*(x13*x248*y + x200 + x209*x74 + x218*x5 + x234 + 3.0*x241 + x242 + x245 + x252*x5);
      basis_lapl_y_eval[ipt + 2*npts] = x76*(x113*x78 + 3.0*x163*x78 + x183*x78 + x200 + x21*x248*y + x220*x5 + x221*x35 + x222 + x251*x50 + x253*x5);
      basis_lapl_y_eval[ipt + 3*npts] = x23*(x113*x82 + 3.0*x163*x82 + x183*x82 + x224*x5 - x231*x6 + x235 + x251*x99 + x254*y + x255*x5 + x256);
      basis_lapl_y_eval[ipt + 4*npts] = 4.0*x104*x181 + 1.5*x113*x86 + 0.125*x121*x143 + 4.5*x163*x86 + 0.125*x170*x259 - 24.0*x18*x87 + 1.5*x183*x86 + 0.125*x191*x257 + 0.125*x228*x57 + 0.125*x230*y + 0.125*x248*x36 + 0.125*x258 + 0.125*x260*y;
      basis_lapl_y_eval[ipt + 5*npts] = x37*(x224*y + x254 + x255*y + x261*x60 + x261*x99 + x263);
      basis_lapl_y_eval[ipt + 6*npts] = x38*(4.0*x113*x89 + x127*x143 + 12.0*x163*x89 + x174*x259 - x18*x257 + x180*x238*x8 + x182 + 4.0*x183*x89 + x237*x64 + x240*y + x248*x39 + x264*y);
      basis_lapl_y_eval[ipt + 7*npts] = x40*(x244*y + x248*x42 + x261*x42 + x261*x65 + x263 + x265*y);
      basis_lapl_y_eval[ipt + 8*npts] = x69*(4.0*x113*x92 + 12.0*x163*x92 + x177*x259 + x182 + 4.0*x183*x92 + x237*x70 + x247*y + x248*x43 - x258 + x266*y);
      // Evaluate Laplacian gradient of bfn (dz)
      basis_lapl_z_eval[ipt + 0*npts] = x2*(x13*x267 + x206*z + x250*z + x267*x42 + x268*x8);
      basis_lapl_z_eval[ipt + 1*npts] = x94*(x13*x268*z + x17*x217 + x17*x252 + 3.0*x185 + x198 + x270 + x271);
      basis_lapl_z_eval[ipt + 2*npts] = x16*(72.0*x105 + x113*x272 + x163*x272 + x183*x273 + x21*x268 + x219*z + x253*z + x267*x50 + x267*x78);
      basis_lapl_z_eval[ipt + 3*npts] = x96*(x256 + x269*x82 - x271 + x274);
      basis_lapl_z_eval[ipt + 4*npts] = 2.0*x104*x113 + 2.0*x104*x163 + 6.0*x104*x183 + 18.0*x105*x191 + 0.125*x121*x154 + 0.125*x154*x170 + 3.0*x155*x57 + 3.0*x181*x86 + 0.125*x229*z + 0.125*x260*z + 0.125*x268*x36 - 0.125*x275*x4 - 0.125*x275*x6;
      basis_lapl_z_eval[ipt + 5*npts] = x106*(x200 + x232 + x269*x60 + x274 + x276);
      basis_lapl_z_eval[ipt + 6*npts] = x38*(36.0*x105*x8 + x114*x272 + x127*x154 + x154*x174 + x164*x272 + x184*x273 + x239*z + x264*z + x268*x39 + x277*x4 - x277*x6 + x278*x89 + x279*x64);
      basis_lapl_z_eval[ipt + 7*npts] = x109*(x17*x243 + x17*x265 + 3.0*x194 + x203 + x268*x42*z + x270 + x276);
      basis_lapl_z_eval[ipt + 8*npts] = x69*(x154*x177 + x154*x8 + x246*z + x266*z + x268*x43 + x278*x92 + x279*x70);




#if 0
      // Evaluate the angular part of bfn



      double ang_eval_0;
      double ang_eval_1;
      double ang_eval_2;
      double ang_eval_3;


      ang_eval_0 = radial_eval*x2*x8;
      ang_eval_1 = radial_eval*x11*x13;
      ang_eval_2 = radial_eval*x16*x21;
      ang_eval_3 = radial_eval*x24*x28;
      basis_eval[ipt + 0*npts] = ang_eval_0;
      basis_eval[ipt + 1*npts] = ang_eval_1;
      basis_eval[ipt + 2*npts] = ang_eval_2;
      basis_eval[ipt + 3*npts] = ang_eval_3;

      ang_eval_0 = x29*x36;
      ang_eval_1 = radial_eval*x28*x37;
      ang_eval_2 = radial_eval*x38*x39;
      ang_eval_3 = radial_eval*x40*x42;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = sqrt_35*x29*x43;
      basis_eval[ipt + 8*npts] = ang_eval_0;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = x1*(radial_eval_alpha*x45 + x44);
      dang_eval_y_0 = x71*(radial_eval_alpha*x73 + x72);
      dang_eval_z_0 = x58*x8*x93;
      dang_eval_x_1 = x46*x49;
      dang_eval_y_1 = x10*x75;
      dang_eval_z_1 = x94*(radial_eval_alpha*x95 + x44);
      dang_eval_x_2 = x15*(radial_eval*x50 + radial_eval_alpha*x51);
      dang_eval_y_2 = x76*(radial_eval*x78 + radial_eval_alpha*x79);
      dang_eval_z_2 = x16*z*(radial_eval_alpha*x21 + x54);
      dang_eval_x_3 = x53;
      dang_eval_y_3 = x23*(radial_eval*x82 + radial_eval_alpha*x83);
      dang_eval_z_3 = x101*x96;
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

      dang_eval_x_0 = 0.125*x36*x58 + 0.125*x54*x57;
      dang_eval_y_0 = 0.125*x36*x87 + 0.125*x54*x86;
      dang_eval_z_0 = 2.0*radial_eval*x104 + 0.125*x105*x36;
      dang_eval_x_1 = x23*(radial_eval*x60 + radial_eval_alpha*x61);
      dang_eval_y_1 = x53;
      dang_eval_z_1 = x101*x106;
      dang_eval_x_2 = x38*(x39*x58 + x62*x64);
      dang_eval_y_2 = x38*(x39*x87 + x62*x89);
      dang_eval_z_2 = x107*(radial_eval_alpha*x39 + x108);
      dang_eval_x_3 = x10*x68;
      dang_eval_y_3 = x46*x91;
      dang_eval_z_3 = x109*(radial_eval_alpha*x110 + x72);
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

      dang_eval_x_0 = x69*(x43*x58 + x62*x70);
      dang_eval_y_0 = x69*(x43*x87 + x62*x92);
      dang_eval_z_0 = x105*x43*x69;
      basis_x_eval[ipt + 8*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 8*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 8*npts] = dang_eval_z_0;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
