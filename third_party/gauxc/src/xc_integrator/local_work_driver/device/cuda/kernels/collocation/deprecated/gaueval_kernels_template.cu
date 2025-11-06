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
//#include <GauXC/device_util.hpp>
#include <iostream>
#include <cassert>

#include "gaueval_kernels.hpp"
#include "gaueval_angular_cartesian.hpp"
#include "gaueval_angular_spherical.hpp"
#include "gaueval_angular_spherical_unnorm.hpp"

namespace GauXC {

__global__
void gaueval_device_$(ang_name)_kernel(
  size_t             nshells,
  size_t             nbf,
  size_t             npts,
  const StaticShell* shells_device,
  const size_t*      offs_device,
  const double*      pts_device,
  double*            eval_device
) {

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nshells ) {

    const size_t ipt = tid_x;
    const size_t ish = tid_y;

    const size_t ibf = offs_device[ish];

    const auto& shell = shells_device[ish];
    const auto* pt    = pts_device + 3*ipt;
  

    const auto* O     = device::array_data( shell.O     );
    const auto* alpha = device::array_data( shell.alpha );
    const auto* coeff = device::array_data( shell.coeff );

    const double xc = pt[0] - O[0];
    const double yc = pt[1] - O[1];
    const double zc = pt[2] - O[2];
  
    const double rsq = xc*xc + yc*yc + zc*zc;
  
    const size_t nprim = shell.nprim; 
    double tmp = 0.;
    for( size_t i = 0; i < nprim; ++i )
      tmp += coeff[i] * std::exp( - alpha[i] * rsq );

    double * bf_eval = eval_device + ibf + ipt*nbf;
    gaueval_$(ang_name)_angular( shell.l, tmp, xc, yc, zc, bf_eval );

  }

}



__global__
void gaueval_device_$(ang_name)_kernel_deriv1(
  size_t             nshells,
  size_t             nbf,
  size_t             npts,
  const StaticShell* shells_device,
  const size_t*      offs_device,
  const double*      pts_device,
  double*            eval_device,
  double*            deval_device_x,
  double*            deval_device_y,
  double*            deval_device_z
) {

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nshells ) {

    const size_t ipt = tid_x;
    const size_t ish = tid_y;

    const size_t ibf = offs_device[ish];

    const auto& shell = shells_device[ish];
    const auto* pt    = pts_device + 3*ipt;
  

    const auto* O     = device::array_data( shell.O     );
    const auto* alpha = device::array_data( shell.alpha );
    const auto* coeff = device::array_data( shell.coeff );

    const double xc = pt[0] - O[0];
    const double yc = pt[1] - O[1];
    const double zc = pt[2] - O[2];
  
    const double rsq = xc*xc + yc*yc + zc*zc;
  
    const size_t nprim = shell.nprim; 
    double tmp = 0.;
    double tmp_x = 0., tmp_y = 0., tmp_z = 0.;
    for( size_t i = 0; i < nprim; ++i ) {

      const double a = alpha[i];
      const double e = coeff[i] * std::exp( - a * rsq );

      const double ae = 2. * a * e;

      tmp   += e;
      tmp_x -= ae * xc;
      tmp_y -= ae * yc;
      tmp_z -= ae * zc;

    }

    double * bf_eval = eval_device    + ibf + ipt*nbf;
    double * dx_eval = deval_device_x + ibf + ipt*nbf;
    double * dy_eval = deval_device_y + ibf + ipt*nbf;
    double * dz_eval = deval_device_z + ibf + ipt*nbf;

    gaueval_$(ang_name)_angular_deriv1( shell.l, tmp, tmp_x, tmp_y, tmp_z, xc, yc, zc, bf_eval, dx_eval, dy_eval, dz_eval );

  }


}


}
