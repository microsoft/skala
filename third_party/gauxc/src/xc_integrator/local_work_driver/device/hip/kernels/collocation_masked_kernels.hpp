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
#include "hip/hip_runtime.h"
#include <iostream>
#include <cassert>

#include <gauxc/shell.hpp>

#include "device/hip/kernels/collocation/collocation_angular_cartesian.hpp"
#include "device/hip/kernels/collocation/collocation_angular_spherical_unnorm.hpp"

namespace GauXC      {


template <typename T>
__global__
void collocation_device_masked_kernel(
  size_t                       nshells,
  size_t                       nbf,
  size_t                       npts,
  const Shell<T>* __restrict__ shells_device,
  const size_t*   __restrict__ mask_device,
  const size_t*   __restrict__ offs_device,
  const T*        __restrict__ pts_device,
  T*              __restrict__ eval_device
) {

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nshells ) {

    const size_t ipt = tid_x;
    const size_t ish = tid_y;

    const size_t ibf = offs_device[ish];

    const auto& shell = shells_device[mask_device[ish]];
    const auto* pt    = pts_device + 3*ipt;
  

    const auto* __restrict__ O     = shell.O_data();
    const auto* __restrict__ alpha = shell.alpha_data();
    const auto* __restrict__ coeff = shell.coeff_data();

    const auto xc = pt[0] - O[0];
    const auto yc = pt[1] - O[1];
    const auto zc = pt[2] - O[2];
  
    const auto rsq = xc*xc + yc*yc + zc*zc;
  
    const size_t nprim = shell.nprim(); 
    auto tmp = 0.;
    for( size_t i = 0; i < nprim; ++i )
      tmp += coeff[i] * std::exp( - alpha[i] * rsq );

    auto * bf_eval = eval_device + ibf*npts + ipt;

    const bool do_sph = shell.pure();
    if( do_sph )
      collocation_spherical_unnorm_angular( npts, shell.l(), tmp, xc, yc, zc, bf_eval );
    else
      collocation_cartesian_angular( npts, shell.l(), tmp, xc, yc, zc, bf_eval );

  }

}








template <typename T>
__global__
void collocation_device_masked_kernel_deriv1(
  size_t                       nshells,
  size_t                       nbf,
  size_t                       npts,
  const Shell<T>* __restrict__ shells_device,
  const size_t*   __restrict__ mask_device,
  const size_t*   __restrict__ offs_device,
  const T*        __restrict__ pts_device,
  T*              __restrict__ eval_device,
  T*              __restrict__ deval_device_x,
  T*              __restrict__ deval_device_y,
  T*              __restrict__ deval_device_z
) {

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nshells ) {

    const size_t ipt = tid_x;
    const size_t ish = tid_y;

    const size_t ibf = offs_device[ish];

    const auto& shell = shells_device[mask_device[ish]];
    const auto* pt    = pts_device + 3*ipt;
  

    const auto* __restrict__ O     = shell.O_data();
    const auto* __restrict__ alpha = shell.alpha_data();
    const auto* __restrict__ coeff = shell.coeff_data();

    const auto xc = pt[0] - O[0];
    const auto yc = pt[1] - O[1];
    const auto zc = pt[2] - O[2];
  
    const auto rsq = xc*xc + yc*yc + zc*zc;
  
    const size_t nprim = shell.nprim(); 
    auto tmp = 0.;
    auto tmp_x = 0., tmp_y = 0., tmp_z = 0.;
    for( size_t i = 0; i < nprim; ++i ) {

      const auto a = alpha[i];
      const auto e = coeff[i] * std::exp( - a * rsq );

      const auto ae = 2. * a * e;

      tmp   += e;
      tmp_x -= ae * xc;
      tmp_y -= ae * yc;
      tmp_z -= ae * zc;

    }

    auto * bf_eval = eval_device    + ibf*npts + ipt;
    auto * dx_eval = deval_device_x + ibf*npts + ipt;
    auto * dy_eval = deval_device_y + ibf*npts + ipt;
    auto * dz_eval = deval_device_z + ibf*npts + ipt;

    const bool do_sph = shell.pure();
    if( do_sph ) 
      collocation_spherical_unnorm_angular_deriv1( npts, shell.l(), tmp, tmp_x, tmp_y, tmp_z, 
                                               xc, yc, zc, bf_eval, dx_eval, 
                                               dy_eval, dz_eval );
    else
      collocation_cartesian_angular_deriv1( npts, shell.l(), tmp, tmp_x, tmp_y, tmp_z, 
                                        xc, yc, zc, bf_eval, dx_eval, 
                                        dy_eval, dz_eval );

  }


}

} // namespace GauXC

