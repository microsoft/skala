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


__global__ __launch_bounds__(128,2) void collocation_device_shell_to_task_kernel_spherical_laplacian_4(
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
      const auto x32 = 6.0*x4*x6; 
      const auto x33 = x18*x4; 
      const auto x34 = x18*x6; 
      const auto x35 = 3.0*x30 + 3.0*x31 + x32 - 24.0*x33 - 24.0*x34 + 8.0*(z*z*z*z); 
      const auto x36 = x*x23; 
      const auto x37 = 0.25*sqrt_5; 
      const auto x38 = -x30 + x31 + 6.0*x33 - 6.0*x34; 
      const auto x39 = x*x10; 
      const auto x40 = -x26; 
      const auto x41 = x4 + x40; 
      const auto x42 = x30 + x31 - x32; 
      const auto x43 = radial_eval*x13; 
      const auto x44 = x4*x8; 
      const auto x45 = x*x11; 
      const auto x46 = 6.0*radial_eval; 
      const auto x47 = radial_eval_alpha*x13; 
      const auto x48 = x46 + x47; 
      const auto x49 = -x12 - x20; 
      const auto x50 = x21*x4; 
      const auto x51 = -x46; 
      const auto x52 = x*x24*(radial_eval_alpha*x28 + x51); 
      const auto x53 = 12.0*radial_eval; 
      const auto x54 = x*x*x; 
      const auto x55 = 4.0*x; 
      const auto x56 = x*x6 - x18*x55 + x54; 
      const auto x57 = radial_eval_alpha*x; 
      const auto x58 = 9.0*x4; 
      const auto x59 = -x27 - x58; 
      const auto x60 = x28*x4; 
      const auto x61 = 4.0*radial_eval; 
      const auto x62 = 3.0*x; 
      const auto x63 = x18*x62 - x54; 
      const auto x64 = x12 + x40; 
      const auto x65 = radial_eval*x64; 
      const auto x66 = x4*x41; 
      const auto x67 = radial_eval_alpha*x66 + x65; 
      const auto x68 = 0.125*sqrt_35; 
      const auto x69 = x54 - x6*x62; 
      const auto x70 = x*x0; 
      const auto x71 = radial_eval*x41; 
      const auto x72 = x6*x8; 
      const auto x73 = x13*x6; 
      const auto x74 = radial_eval_alpha*x73 + x65; 
      const auto x75 = x*x14; 
      const auto x76 = x19 + x26; 
      const auto x77 = -x4 - x76; 
      const auto x78 = x21*x6; 
      const auto x79 = 9.0*x6; 
      const auto x80 = x12 + x25; 
      const auto x81 = -x79 - x80; 
      const auto x82 = x28*x6; 
      const auto x83 = y*y*y; 
      const auto x84 = 4.0*y; 
      const auto x85 = -x18*x84 + x4*y + x83; 
      const auto x86 = radial_eval_alpha*y; 
      const auto x87 = 3.0*y; 
      const auto x88 = -x18*x87 + x83; 
      const auto x89 = radial_eval_alpha*x41; 
      const auto x90 = x51 + x89; 
      const auto x91 = -x4*x87 + x83; 
      const auto x92 = x1*z; 
      const auto x93 = x9*y; 
      const auto x94 = x13*x18; 
      const auto x95 = x22*y; 
      const auto x96 = -12.0*x18; 
      const auto x97 = x26 + x96; 
      const auto x98 = -x12 - x97; 
      const auto x99 = x18*x28; 
      const auto x100 = radial_eval*x98 + radial_eval_alpha*x99; 
      const auto x101 = 3.0*z; 
      const auto x102 = -x101*x4 - x101*x6 + 2.0*(z*z*z); 
      const auto x103 = radial_eval_alpha*z; 
      const auto x104 = x37*z; 
      const auto x105 = x53*x8; 
      const auto x106 = x18*x41; 
      const auto x107 = 2.0*radial_eval_alpha; 
      const auto x108 = x107*x13; 
      const auto x109 = radial_eval_alpha + radial_eval_alpha_squared*x4; 
      const auto x110 = x108 + x109*x8; 
      const auto x111 = x109*x13; 
      const auto x112 = 12.0*radial_eval_alpha; 
      const auto x113 = x112*x4; 
      const auto x114 = x113 + x46; 
      const auto x115 = x107*x49 + x109*x21; 
      const auto x116 = x109*x35 + x53*(x6 + x80) + 24.0*x56*x57; 
      const auto x117 = -18.0*radial_eval; 
      const auto x118 = x109*x28; 
      const auto x119 = x107*x59 + x118; 
      const auto x120 = -x4; 
      const auto x121 = 8.0*x57; 
      const auto x122 = x109*x38 + x121*x63 + x53*(x120 + x18); 
      const auto x123 = x107*x64; 
      const auto x124 = x109*x41 + x123; 
      const auto x125 = x105 + x109*x42 + x121*x69; 
      const auto x126 = radial_eval_alpha*x3; 
      const auto x127 = radial_eval_alpha*x5; 
      const auto x128 = 6.0*radial_eval_alpha; 
      const auto x129 = x128*x6; 
      const auto x130 = radial_eval_alpha*x64; 
      const auto x131 = 24.0*radial_eval; 
      const auto x132 = x*x131; 
      const auto x133 = x132*y; 
      const auto x134 = 12.0*x57; 
      const auto x135 = 12.0*x86; 
      const auto x136 = radial_eval_alpha_squared*x; 
      const auto x137 = x136*y; 
      const auto x138 = -x128*x4 + x51; 
      const auto x139 = radial_eval_alpha*x55; 
      const auto x140 = radial_eval_alpha*x84; 
      const auto x141 = x*x93; 
      const auto x142 = x128*x18; 
      const auto x143 = -x142; 
      const auto x144 = x*x95*(radial_eval_alpha*x98 + radial_eval_alpha_squared*x99 + x143 + x51); 
      const auto x145 = 96.0*radial_eval*z; 
      const auto x146 = 12.0*x103; 
      const auto x147 = radial_eval_alpha*x17; 
      const auto x148 = 4.0*radial_eval_alpha; 
      const auto x149 = x147*x64; 
      const auto x150 = x68*z; 
      const auto x151 = x107*x41; 
      const auto x152 = radial_eval_alpha + radial_eval_alpha_squared*x6; 
      const auto x153 = x151 + x152*x8; 
      const auto x154 = x123 + x13*x152; 
      const auto x155 = x107*x77 + x152*x21; 
      const auto x156 = x152*x28; 
      const auto x157 = x107*x81 + x156; 
      const auto x158 = x152*x35 + x53*(x27 + x4) + 24.0*x85*x86; 
      const auto x159 = x112*x6; 
      const auto x160 = x159 + x46; 
      const auto x161 = 8.0*x86; 
      const auto x162 = x152*x38 + x161*x88 - x53*(x18 - x6); 
      const auto x163 = x152*x42 + x161*x91 + x53*(x120 + x6); 
      const auto x164 = radial_eval_alpha_squared*y; 
      const auto x165 = radial_eval_alpha + radial_eval_alpha_squared*x18; 
      const auto x166 = x165*x8; 
      const auto x167 = x108 + x13*x165; 
      const auto x168 = 24.0*radial_eval_alpha*x18 + x165*x21; 
      const auto x169 = x107*x98 + x165*x28; 
      const auto x170 = x131 + x169; 
      const auto x171 = -48.0*radial_eval*(-2.0*x18 + x4 + x6) + 32.0*x102*x103 + x165*x35; 
      const auto x172 = x105 + 24.0*x147*x8 + x165*x38; 
      const auto x173 = x151 + x165*x41; 
      const auto x174 = x165*x42; 
      const auto x175 = -x159; 


      // Evaluate basis function
      basis_eval[ipt + 0*npts] = radial_eval*x2*x8;
      basis_eval[ipt + 1*npts] = radial_eval*x11*x13;
      basis_eval[ipt + 2*npts] = radial_eval*x16*x21;
      basis_eval[ipt + 3*npts] = radial_eval*x24*x28;
      basis_eval[ipt + 4*npts] = x29*x35;
      basis_eval[ipt + 5*npts] = radial_eval*x28*x36;
      basis_eval[ipt + 6*npts] = radial_eval*x37*x38;
      basis_eval[ipt + 7*npts] = radial_eval*x39*x41;
      basis_eval[ipt + 8*npts] = sqrt_35*x29*x42;


    
      // Evaluate first derivative of bfn wrt x
      basis_x_eval[ipt + 0*npts] = x1*(radial_eval_alpha*x44 + x43);
      basis_x_eval[ipt + 1*npts] = x45*x48;
      basis_x_eval[ipt + 2*npts] = x15*(radial_eval*x49 + radial_eval_alpha*x50);
      basis_x_eval[ipt + 3*npts] = x52;
      basis_x_eval[ipt + 4*npts] = 0.125*x35*x57 + 0.125*x53*x56;
      basis_x_eval[ipt + 5*npts] = x23*(radial_eval*x59 + radial_eval_alpha*x60);
      basis_x_eval[ipt + 6*npts] = x37*(x38*x57 + x61*x63);
      basis_x_eval[ipt + 7*npts] = x10*x67;
      basis_x_eval[ipt + 8*npts] = x68*(x42*x57 + x61*x69);

      // Evaluate first derivative of bfn wrt y
      basis_y_eval[ipt + 0*npts] = x70*(radial_eval_alpha*x72 + x71);
      basis_y_eval[ipt + 1*npts] = x10*x74;
      basis_y_eval[ipt + 2*npts] = x75*(radial_eval*x77 + radial_eval_alpha*x78);
      basis_y_eval[ipt + 3*npts] = x23*(radial_eval*x81 + radial_eval_alpha*x82);
      basis_y_eval[ipt + 4*npts] = 0.125*x35*x86 + 0.125*x53*x85;
      basis_y_eval[ipt + 5*npts] = x52;
      basis_y_eval[ipt + 6*npts] = x37*(x38*x86 + x61*x88);
      basis_y_eval[ipt + 7*npts] = x45*x90;
      basis_y_eval[ipt + 8*npts] = x68*(x42*x86 + x61*x91);

      // Evaluate first derivative of bfn wrt z
      basis_z_eval[ipt + 0*npts] = x57*x8*x92;
      basis_z_eval[ipt + 1*npts] = x93*(radial_eval_alpha*x94 + x43);
      basis_z_eval[ipt + 2*npts] = x16*z*(radial_eval_alpha*x21 + x53);
      basis_z_eval[ipt + 3*npts] = x100*x95;
      basis_z_eval[ipt + 4*npts] = 2.0*radial_eval*x102 + 0.125*x103*x35;
      basis_z_eval[ipt + 5*npts] = x*x100*x22;
      basis_z_eval[ipt + 6*npts] = x104*(radial_eval_alpha*x38 + x105);
      basis_z_eval[ipt + 7*npts] = x*x9*(radial_eval_alpha*x106 + x71);
      basis_z_eval[ipt + 8*npts] = x103*x42*x68;


      // Evaluate Laplacian of bfn 
      basis_lapl_eval[ipt + 0*npts] = x2*(x110 + x153 + x166);
      basis_lapl_eval[ipt + 1*npts] = x11*(x111 + x113 + x154 + x167);
      basis_lapl_eval[ipt + 2*npts] = x16*(x115 + x155 + x168);
      basis_lapl_eval[ipt + 3*npts] = x24*(-x113 + x118 + x157 + x169);
      basis_lapl_eval[ipt + 4*npts] = 0.125*x116 + 0.125*x158 + 0.125*x171;
      basis_lapl_eval[ipt + 5*npts] = x36*(x119 + x156 + x169 + x175);
      basis_lapl_eval[ipt + 6*npts] = x37*(x122 + x162 + x172);
      basis_lapl_eval[ipt + 7*npts] = x39*(x124 + x152*x41 + x173 + x175);
      basis_lapl_eval[ipt + 8*npts] = x68*(x125 + x163 + x174);





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

      ang_eval_0 = x29*x35;
      ang_eval_1 = radial_eval*x28*x36;
      ang_eval_2 = radial_eval*x37*x38;
      ang_eval_3 = radial_eval*x39*x41;
      basis_eval[ipt + 4*npts] = ang_eval_0;
      basis_eval[ipt + 5*npts] = ang_eval_1;
      basis_eval[ipt + 6*npts] = ang_eval_2;
      basis_eval[ipt + 7*npts] = ang_eval_3;

      ang_eval_0 = sqrt_35*x29*x42;
      basis_eval[ipt + 8*npts] = ang_eval_0;


      double dang_eval_x_0, dang_eval_y_0, dang_eval_z_0;
      double dang_eval_x_1, dang_eval_y_1, dang_eval_z_1;
      double dang_eval_x_2, dang_eval_y_2, dang_eval_z_2;
      double dang_eval_x_3, dang_eval_y_3, dang_eval_z_3;

      dang_eval_x_0 = x1*(radial_eval_alpha*x44 + x43);
      dang_eval_y_0 = x70*(radial_eval_alpha*x72 + x71);
      dang_eval_z_0 = x57*x8*x92;
      dang_eval_x_1 = x45*x48;
      dang_eval_y_1 = x10*x74;
      dang_eval_z_1 = x93*(radial_eval_alpha*x94 + x43);
      dang_eval_x_2 = x15*(radial_eval*x49 + radial_eval_alpha*x50);
      dang_eval_y_2 = x75*(radial_eval*x77 + radial_eval_alpha*x78);
      dang_eval_z_2 = x16*z*(radial_eval_alpha*x21 + x53);
      dang_eval_x_3 = x52;
      dang_eval_y_3 = x23*(radial_eval*x81 + radial_eval_alpha*x82);
      dang_eval_z_3 = x100*x95;
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

      dang_eval_x_0 = 0.125*x35*x57 + 0.125*x53*x56;
      dang_eval_y_0 = 0.125*x35*x86 + 0.125*x53*x85;
      dang_eval_z_0 = 2.0*radial_eval*x102 + 0.125*x103*x35;
      dang_eval_x_1 = x23*(radial_eval*x59 + radial_eval_alpha*x60);
      dang_eval_y_1 = x52;
      dang_eval_z_1 = x*x100*x22;
      dang_eval_x_2 = x37*(x38*x57 + x61*x63);
      dang_eval_y_2 = x37*(x38*x86 + x61*x88);
      dang_eval_z_2 = x104*(radial_eval_alpha*x38 + x105);
      dang_eval_x_3 = x10*x67;
      dang_eval_y_3 = x45*x90;
      dang_eval_z_3 = x*x9*(radial_eval_alpha*x106 + x71);
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

      dang_eval_x_0 = x68*(x42*x57 + x61*x69);
      dang_eval_y_0 = x68*(x42*x86 + x61*x91);
      dang_eval_z_0 = x103*x42*x68;
      basis_x_eval[ipt + 8*npts] = dang_eval_x_0;
      basis_y_eval[ipt + 8*npts] = dang_eval_y_0;
      basis_z_eval[ipt + 8*npts] = dang_eval_z_0;

#endif
    } // Loop over points within task
  } // Loop over tasks
        
  } // Loop over shells
} // end kernel

} // namespace GauXC
