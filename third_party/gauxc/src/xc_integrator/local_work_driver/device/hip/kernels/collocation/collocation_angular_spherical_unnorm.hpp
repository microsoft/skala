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
#include <cassert>

#ifndef GPGAUEVAL_INLINE
#  define GPGAUEVAL_INLINE __noinline__
#endif

namespace GauXC {

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_spherical_unnorm_angular_0(
  int32_t          npts,
  const T          bf,
  const T          x,
  const T          y,
  const T          z,
  T* __restrict__  eval
) {

  eval[npts * 0] = bf;

}

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_spherical_unnorm_angular_0_deriv1(
  const int32_t   npts,
  const T         bf,
  const T         bf_x,
  const T         bf_y,
  const T         bf_z,
  const T         x,
  const T         y,
  const T         z,
  T* __restrict__ eval_x,
  T* __restrict__ eval_y,
  T* __restrict__ eval_z
) {

  eval_x[npts * 0] = bf_x;

  eval_y[npts * 0] = bf_y;

  eval_z[npts * 0] = bf_z;

}

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_spherical_unnorm_angular_1(
  int32_t          npts,
  const T          bf,
  const T          x,
  const T          y,
  const T          z,
  T* __restrict__  eval
) {

  eval[npts * 0] = bf*y;
  eval[npts * 1] = bf*z;
  eval[npts * 2] = bf*x;

}

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_spherical_unnorm_angular_1_deriv1(
  const int32_t   npts,
  const T         bf,
  const T         bf_x,
  const T         bf_y,
  const T         bf_z,
  const T         x,
  const T         y,
  const T         z,
  T* __restrict__ eval_x,
  T* __restrict__ eval_y,
  T* __restrict__ eval_z
) {

  eval_x[npts * 0] = bf_x*y;
  eval_x[npts * 1] = bf_x*z;
  eval_x[npts * 2] = bf + bf_x*x;

  eval_y[npts * 0] = bf + bf_y*y;
  eval_y[npts * 1] = bf_y*z;
  eval_y[npts * 2] = bf_y*x;

  eval_z[npts * 0] = bf_z*y;
  eval_z[npts * 1] = bf + bf_z*z;
  eval_z[npts * 2] = bf_z*x;

}

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_spherical_unnorm_angular_2(
  int32_t          npts,
  const T          bf,
  const T          x,
  const T          y,
  const T          z,
  T* __restrict__  eval
) {

  eval[npts * 0] = sqrt_3*bf*x*y;
  eval[npts * 1] = sqrt_3*bf*y*z;
  eval[npts * 2] = bf*(-x*x - y*y + 2*z*z)/2;
  eval[npts * 3] = sqrt_3*bf*x*z;
  eval[npts * 4] = sqrt_3*bf*(x*x - y*y)/2;

}

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_spherical_unnorm_angular_2_deriv1(
  const int32_t   npts,
  const T         bf,
  const T         bf_x,
  const T         bf_y,
  const T         bf_z,
  const T         x,
  const T         y,
  const T         z,
  T* __restrict__ eval_x,
  T* __restrict__ eval_y,
  T* __restrict__ eval_z
) {

  eval_x[npts * 0] = sqrt_3*y*(bf + bf_x*x);
  eval_x[npts * 1] = sqrt_3*bf_x*y*z;
  eval_x[npts * 2] = -bf*x - bf_x*(x*x + y*y - 2*z*z)/2;
  eval_x[npts * 3] = sqrt_3*z*(bf + bf_x*x);
  eval_x[npts * 4] = sqrt_3*(bf*x + bf_x*(x*x - y*y)/2);

  eval_y[npts * 0] = sqrt_3*x*(bf + bf_y*y);
  eval_y[npts * 1] = sqrt_3*z*(bf + bf_y*y);
  eval_y[npts * 2] = -bf*y - bf_y*(x*x + y*y - 2*z*z)/2;
  eval_y[npts * 3] = sqrt_3*bf_y*x*z;
  eval_y[npts * 4] = sqrt_3*(-bf*y + bf_y*(x*x - y*y)/2);

  eval_z[npts * 0] = sqrt_3*bf_z*x*y;
  eval_z[npts * 1] = sqrt_3*y*(bf + bf_z*z);
  eval_z[npts * 2] = 2*bf*z - bf_z*(x*x + y*y - 2*z*z)/2;
  eval_z[npts * 3] = sqrt_3*x*(bf + bf_z*z);
  eval_z[npts * 4] = sqrt_3*bf_z*(x*x - y*y)/2;

}

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_spherical_unnorm_angular_3(
  int32_t          npts,
  const T          bf,
  const T          x,
  const T          y,
  const T          z,
  T* __restrict__  eval
) {

  eval[npts * 0] = sqrt_10*bf*y*(3*x*x - y*y)/4;
  eval[npts * 1] = sqrt_15*bf*x*y*z;
  eval[npts * 2] = sqrt_6*bf*y*(-x*x - y*y + 4*z*z)/4;
  eval[npts * 3] = bf*z*(-3*x*x - 3*y*y + 2*z*z)/2;
  eval[npts * 4] = sqrt_6*bf*x*(-x*x - y*y + 4*z*z)/4;
  eval[npts * 5] = sqrt_15*bf*z*(x*x - y*y)/2;
  eval[npts * 6] = sqrt_10*bf*x*(x*x - 3*y*y)/4;

}

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_spherical_unnorm_angular_3_deriv1(
  const int32_t   npts,
  const T         bf,
  const T         bf_x,
  const T         bf_y,
  const T         bf_z,
  const T         x,
  const T         y,
  const T         z,
  T* __restrict__ eval_x,
  T* __restrict__ eval_y,
  T* __restrict__ eval_z
) {

  eval_x[npts * 0] = sqrt_10*y*(6*bf*x + bf_x*(3*x*x - y*y))/4;
  eval_x[npts * 1] = sqrt_15*y*z*(bf + bf_x*x);
  eval_x[npts * 2] = -sqrt_6*y*(2*bf*x + bf_x*(x*x + y*y - 4*z*z))/4;
  eval_x[npts * 3] = -z*(6*bf*x + bf_x*(3*x*x + 3*y*y - 2*z*z))/2;
  eval_x[npts * 4] = -sqrt_6*(bf*(3*x*x + y*y - 4*z*z) + bf_x*x*(x*x + y*y - 4*z*z))/4;
  eval_x[npts * 5] = sqrt_15*z*(2*bf*x + bf_x*(x*x - y*y))/2;
  eval_x[npts * 6] = sqrt_10*(3*bf*(x*x - y*y) + bf_x*x*(x*x - 3*y*y))/4;

  eval_y[npts * 0] = sqrt_10*(-3*bf*(-x*x + y*y) + bf_y*y*(3*x*x - y*y))/4;
  eval_y[npts * 1] = sqrt_15*x*z*(bf + bf_y*y);
  eval_y[npts * 2] = -sqrt_6*(bf*(x*x + 3*y*y - 4*z*z) + bf_y*y*(x*x + y*y - 4*z*z))/4;
  eval_y[npts * 3] = -z*(6*bf*y + bf_y*(3*x*x + 3*y*y - 2*z*z))/2;
  eval_y[npts * 4] = -sqrt_6*x*(2*bf*y + bf_y*(x*x + y*y - 4*z*z))/4;
  eval_y[npts * 5] = sqrt_15*z*(-2*bf*y + bf_y*(x*x - y*y))/2;
  eval_y[npts * 6] = sqrt_10*x*(-6*bf*y + bf_y*(x*x - 3*y*y))/4;

  eval_z[npts * 0] = sqrt_10*bf_z*y*(3*x*x - y*y)/4;
  eval_z[npts * 1] = sqrt_15*x*y*(bf + bf_z*z);
  eval_z[npts * 2] = sqrt_6*y*(8*bf*z - bf_z*(x*x + y*y - 4*z*z))/4;
  eval_z[npts * 3] = -3*bf*(x*x + y*y - 2*z*z)/2 - bf_z*z*(3*x*x + 3*y*y - 2*z*z)/2;
  eval_z[npts * 4] = sqrt_6*x*(8*bf*z - bf_z*(x*x + y*y - 4*z*z))/4;
  eval_z[npts * 5] = sqrt_15*(bf + bf_z*z)*(x*x - y*y)/2;
  eval_z[npts * 6] = sqrt_10*bf_z*x*(x*x - 3*y*y)/4;

}


template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_spherical_unnorm_angular(
  const int32_t   npts,
  const int32_t    l,
  const T          bf,
  const T          x,
  const T          y,
  const T          z,
  T* __restrict__  eval
) {

      if( l == 0 ) {
  
        collocation_spherical_unnorm_angular_0( npts, bf, x, y, z, eval );

      } else if( l == 1 ) {
  
        collocation_spherical_unnorm_angular_1( npts, bf, x, y, z, eval );

      } else if( l == 2 ) {
  
        collocation_spherical_unnorm_angular_2( npts, bf, x, y, z, eval );

      } else if( l == 3 ) {
  
        collocation_spherical_unnorm_angular_3( npts, bf, x, y, z, eval );

    } else {
      assert( false && "L < L_MAX" );
    }

} // collocation_spherical_unnorm_angular


template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_spherical_unnorm_angular_deriv1(
  const int32_t    npts,
  const int32_t    l,
  const T          bf,
  const T          bf_x,
  const T          bf_y,
  const T          bf_z,
  const T          x,
  const T          y,
  const T          z,
  T* __restrict__ eval,
  T* __restrict__ eval_x,
  T* __restrict__ eval_y,
  T* __restrict__ eval_z
) {


      if( l == 0 ) {
  
        collocation_spherical_unnorm_angular_0( npts, bf, x, y, z, eval );
        collocation_spherical_unnorm_angular_0_deriv1( npts, bf, bf_x, bf_y, bf_z, x, y, z, eval_x, eval_y, eval_z );

      } else if( l == 1 ) {
  
        collocation_spherical_unnorm_angular_1( npts, bf, x, y, z, eval );
        collocation_spherical_unnorm_angular_1_deriv1( npts, bf, bf_x, bf_y, bf_z, x, y, z, eval_x, eval_y, eval_z );

      } else if( l == 2 ) {
  
        collocation_spherical_unnorm_angular_2( npts, bf, x, y, z, eval );
        collocation_spherical_unnorm_angular_2_deriv1( npts, bf, bf_x, bf_y, bf_z, x, y, z, eval_x, eval_y, eval_z );

      } else if( l == 3 ) {
  
        collocation_spherical_unnorm_angular_3( npts, bf, x, y, z, eval );
        collocation_spherical_unnorm_angular_3_deriv1( npts, bf, bf_x, bf_y, bf_z, x, y, z, eval_x, eval_y, eval_z );

    } else {
      assert( false && "L < L_MAX" );
    }

} // collocation_spherical_unnorm_angular_deriv1


} // namespace GauXC

