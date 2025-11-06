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
#include <iostream>
#include <cassert>

#include <gauxc/shell.hpp>


namespace GauXC {

__inline__ __device__ void collocation_device_radial_eval(
  const Shell<double>&   shell,
  const double*  pt,
  double*        x,
  double*        y,
  double*        z,
  double*        eval_device
) {

  const auto* O     = shell.O_data();
  const auto* alpha = shell.alpha_data();
  const auto* coeff = shell.coeff_data();

  const double xc = pt[0] - O[0];
  const double yc = pt[1] - O[1];
  const double zc = pt[2] - O[2];
  *x = xc;
  *y = yc;
  *z = zc;
  
  const double rsq = xc*xc + yc*yc + zc*zc;
  
  const uint32_t nprim = shell.nprim(); 
  double tmp = 0.;
  for( uint32_t i = 0; i < nprim; ++i )
    tmp += coeff[i] * std::exp( - alpha[i] * rsq );

  *eval_device = tmp;

}



__inline__ __device__ void collocation_device_radial_eval_deriv1(
  const Shell<double>&   shell,
  const double*  pt,
  double*        x,
  double*        y,
  double*        z,
  double*        eval_device,
  double*        deval_device_x,
  double*        deval_device_y,
  double*        deval_device_z
) {

  const auto* O     = shell.O_data();
  const auto* alpha = shell.alpha_data();
  const auto* coeff = shell.coeff_data();

  const double xc = pt[0] - O[0];
  const double yc = pt[1] - O[1];
  const double zc = pt[2] - O[2];
  *x = xc;
  *y = yc;
  *z = zc;
  
  const double rsq = xc*xc + yc*yc + zc*zc;
  
  const uint32_t nprim = shell.nprim(); 
  double tmp = 0.;
  double tmp_x = 0., tmp_y = 0., tmp_z = 0.;
  for( uint32_t i = 0; i < nprim; ++i ) {

    const double a = alpha[i];
    const double e = coeff[i] * std::exp( - a * rsq );

    const double ae = 2. * a * e;

    tmp   += e;
    tmp_x -= ae * xc;
    tmp_y -= ae * yc;
    tmp_z -= ae * zc;

  }

  *eval_device    = tmp;
  *deval_device_x = tmp_x;
  *deval_device_y = tmp_y;
  *deval_device_z = tmp_z;

}

} // namespace GauXC


