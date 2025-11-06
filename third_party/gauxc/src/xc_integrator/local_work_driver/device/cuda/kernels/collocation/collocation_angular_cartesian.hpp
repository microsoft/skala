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

namespace GauXC      {

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_cartesian_angular_0(
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
GPGAUEVAL_INLINE __device__ void collocation_cartesian_angular_0_deriv1(
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
GPGAUEVAL_INLINE __device__ void collocation_cartesian_angular_1(
  int32_t          npts,
  const T          bf,
  const T          x,
  const T          y,
  const T          z,
  T* __restrict__  eval
) {

  eval[npts * 0] = bf*x;
  eval[npts * 1] = bf*y;
  eval[npts * 2] = bf*z;

}

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_cartesian_angular_1_deriv1(
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

  eval_x[npts * 0] = bf + bf_x*x;
  eval_x[npts * 1] = bf_x*y;
  eval_x[npts * 2] = bf_x*z;

  eval_y[npts * 0] = bf_y*x;
  eval_y[npts * 1] = bf + bf_y*y;
  eval_y[npts * 2] = bf_y*z;

  eval_z[npts * 0] = bf_z*x;
  eval_z[npts * 1] = bf_z*y;
  eval_z[npts * 2] = bf + bf_z*z;

}

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_cartesian_angular_2(
  int32_t          npts,
  const T          bf,
  const T          x,
  const T          y,
  const T          z,
  T* __restrict__  eval
) {

  eval[npts * 0] = bf*x*x;
  eval[npts * 1] = bf*x*y;
  eval[npts * 2] = bf*x*z;
  eval[npts * 3] = bf*y*y;
  eval[npts * 4] = bf*y*z;
  eval[npts * 5] = bf*z*z;

}

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_cartesian_angular_2_deriv1(
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

  eval_x[npts * 0] = x*(2*bf + bf_x*x);
  eval_x[npts * 1] = y*(bf + bf_x*x);
  eval_x[npts * 2] = z*(bf + bf_x*x);
  eval_x[npts * 3] = bf_x*y*y;
  eval_x[npts * 4] = bf_x*y*z;
  eval_x[npts * 5] = bf_x*z*z;

  eval_y[npts * 0] = bf_y*x*x;
  eval_y[npts * 1] = x*(bf + bf_y*y);
  eval_y[npts * 2] = bf_y*x*z;
  eval_y[npts * 3] = y*(2*bf + bf_y*y);
  eval_y[npts * 4] = z*(bf + bf_y*y);
  eval_y[npts * 5] = bf_y*z*z;

  eval_z[npts * 0] = bf_z*x*x;
  eval_z[npts * 1] = bf_z*x*y;
  eval_z[npts * 2] = x*(bf + bf_z*z);
  eval_z[npts * 3] = bf_z*y*y;
  eval_z[npts * 4] = y*(bf + bf_z*z);
  eval_z[npts * 5] = z*(2*bf + bf_z*z);

}

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_cartesian_angular_3(
  int32_t          npts,
  const T          bf,
  const T          x,
  const T          y,
  const T          z,
  T* __restrict__  eval
) {

  eval[npts * 0] = bf*x*x*x;
  eval[npts * 1] = bf*x*x*y;
  eval[npts * 2] = bf*x*x*z;
  eval[npts * 3] = bf*x*y*y;
  eval[npts * 4] = bf*x*y*z;
  eval[npts * 5] = bf*x*z*z;
  eval[npts * 6] = bf*y*y*y;
  eval[npts * 7] = bf*y*y*z;
  eval[npts * 8] = bf*y*z*z;
  eval[npts * 9] = bf*z*z*z;

}

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_cartesian_angular_3_deriv1(
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

  eval_x[npts * 0] = x*x*(3*bf + bf_x*x);
  eval_x[npts * 1] = x*y*(2*bf + bf_x*x);
  eval_x[npts * 2] = x*z*(2*bf + bf_x*x);
  eval_x[npts * 3] = y*y*(bf + bf_x*x);
  eval_x[npts * 4] = y*z*(bf + bf_x*x);
  eval_x[npts * 5] = z*z*(bf + bf_x*x);
  eval_x[npts * 6] = bf_x*y*y*y;
  eval_x[npts * 7] = bf_x*y*y*z;
  eval_x[npts * 8] = bf_x*y*z*z;
  eval_x[npts * 9] = bf_x*z*z*z;

  eval_y[npts * 0] = bf_y*x*x*x;
  eval_y[npts * 1] = x*x*(bf + bf_y*y);
  eval_y[npts * 2] = bf_y*x*x*z;
  eval_y[npts * 3] = x*y*(2*bf + bf_y*y);
  eval_y[npts * 4] = x*z*(bf + bf_y*y);
  eval_y[npts * 5] = bf_y*x*z*z;
  eval_y[npts * 6] = y*y*(3*bf + bf_y*y);
  eval_y[npts * 7] = y*z*(2*bf + bf_y*y);
  eval_y[npts * 8] = z*z*(bf + bf_y*y);
  eval_y[npts * 9] = bf_y*z*z*z;

  eval_z[npts * 0] = bf_z*x*x*x;
  eval_z[npts * 1] = bf_z*x*x*y;
  eval_z[npts * 2] = x*x*(bf + bf_z*z);
  eval_z[npts * 3] = bf_z*x*y*y;
  eval_z[npts * 4] = x*y*(bf + bf_z*z);
  eval_z[npts * 5] = x*z*(2*bf + bf_z*z);
  eval_z[npts * 6] = bf_z*y*y*y;
  eval_z[npts * 7] = y*y*(bf + bf_z*z);
  eval_z[npts * 8] = y*z*(2*bf + bf_z*z);
  eval_z[npts * 9] = z*z*(3*bf + bf_z*z);

}

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_cartesian_angular_4(
  int32_t          npts,
  const T          bf,
  const T          x,
  const T          y,
  const T          z,
  T* __restrict__  eval
) {

  eval[npts * 0] = bf*x*x*x*x;
  eval[npts * 1] = bf*x*x*x*y;
  eval[npts * 2] = bf*x*x*x*z;
  eval[npts * 3] = bf*x*x*y*y;
  eval[npts * 4] = bf*x*x*y*z;
  eval[npts * 5] = bf*x*x*z*z;
  eval[npts * 6] = bf*x*y*y*y;
  eval[npts * 7] = bf*x*y*y*z;
  eval[npts * 8] = bf*x*y*z*z;
  eval[npts * 9] = bf*x*z*z*z;
  eval[npts * 10] = bf*y*y*y*y;
  eval[npts * 11] = bf*y*y*y*z;
  eval[npts * 12] = bf*y*y*z*z;
  eval[npts * 13] = bf*y*z*z*z;
  eval[npts * 14] = bf*z*z*z*z;

}

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_cartesian_angular_4_deriv1(
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

  eval_x[npts * 0] = x*x*x*(4*bf + bf_x*x);
  eval_x[npts * 1] = x*x*y*(3*bf + bf_x*x);
  eval_x[npts * 2] = x*x*z*(3*bf + bf_x*x);
  eval_x[npts * 3] = x*y*y*(2*bf + bf_x*x);
  eval_x[npts * 4] = x*y*z*(2*bf + bf_x*x);
  eval_x[npts * 5] = x*z*z*(2*bf + bf_x*x);
  eval_x[npts * 6] = y*y*y*(bf + bf_x*x);
  eval_x[npts * 7] = y*y*z*(bf + bf_x*x);
  eval_x[npts * 8] = y*z*z*(bf + bf_x*x);
  eval_x[npts * 9] = z*z*z*(bf + bf_x*x);
  eval_x[npts * 10] = bf_x*y*y*y*y;
  eval_x[npts * 11] = bf_x*y*y*y*z;
  eval_x[npts * 12] = bf_x*y*y*z*z;
  eval_x[npts * 13] = bf_x*y*z*z*z;
  eval_x[npts * 14] = bf_x*z*z*z*z;

  eval_y[npts * 0] = bf_y*x*x*x*x;
  eval_y[npts * 1] = x*x*x*(bf + bf_y*y);
  eval_y[npts * 2] = bf_y*x*x*x*z;
  eval_y[npts * 3] = x*x*y*(2*bf + bf_y*y);
  eval_y[npts * 4] = x*x*z*(bf + bf_y*y);
  eval_y[npts * 5] = bf_y*x*x*z*z;
  eval_y[npts * 6] = x*y*y*(3*bf + bf_y*y);
  eval_y[npts * 7] = x*y*z*(2*bf + bf_y*y);
  eval_y[npts * 8] = x*z*z*(bf + bf_y*y);
  eval_y[npts * 9] = bf_y*x*z*z*z;
  eval_y[npts * 10] = y*y*y*(4*bf + bf_y*y);
  eval_y[npts * 11] = y*y*z*(3*bf + bf_y*y);
  eval_y[npts * 12] = y*z*z*(2*bf + bf_y*y);
  eval_y[npts * 13] = z*z*z*(bf + bf_y*y);
  eval_y[npts * 14] = bf_y*z*z*z*z;

  eval_z[npts * 0] = bf_z*x*x*x*x;
  eval_z[npts * 1] = bf_z*x*x*x*y;
  eval_z[npts * 2] = x*x*x*(bf + bf_z*z);
  eval_z[npts * 3] = bf_z*x*x*y*y;
  eval_z[npts * 4] = x*x*y*(bf + bf_z*z);
  eval_z[npts * 5] = x*x*z*(2*bf + bf_z*z);
  eval_z[npts * 6] = bf_z*x*y*y*y;
  eval_z[npts * 7] = x*y*y*(bf + bf_z*z);
  eval_z[npts * 8] = x*y*z*(2*bf + bf_z*z);
  eval_z[npts * 9] = x*z*z*(3*bf + bf_z*z);
  eval_z[npts * 10] = bf_z*y*y*y*y;
  eval_z[npts * 11] = y*y*y*(bf + bf_z*z);
  eval_z[npts * 12] = y*y*z*(2*bf + bf_z*z);
  eval_z[npts * 13] = y*z*z*(3*bf + bf_z*z);
  eval_z[npts * 14] = z*z*z*(4*bf + bf_z*z);

}


template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_cartesian_angular(
  const int32_t   npts,
  const int32_t    l,
  const T          bf,
  const T          x,
  const T          y,
  const T          z,
  T* __restrict__  eval
) {

      if( l == 0 ) {
  
        collocation_cartesian_angular_0( npts, bf, x, y, z, eval );

      } else if( l == 1 ) {
  
        collocation_cartesian_angular_1( npts, bf, x, y, z, eval );

      } else if( l == 2 ) {
  
        collocation_cartesian_angular_2( npts, bf, x, y, z, eval );

      } else if( l == 3 ) {
  
        collocation_cartesian_angular_3( npts, bf, x, y, z, eval );

      } else if( l == 4 ) {
  
        collocation_cartesian_angular_4( npts, bf, x, y, z, eval );

    } else {
      assert( false && "L < L_MAX" );
    }

} // collocation_cartesian_angular


template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_cartesian_angular_deriv1(
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
  
        collocation_cartesian_angular_0( npts, bf, x, y, z, eval );
        collocation_cartesian_angular_0_deriv1( npts, bf, bf_x, bf_y, bf_z, x, y, z, eval_x, eval_y, eval_z );

      } else if( l == 1 ) {
  
        collocation_cartesian_angular_1( npts, bf, x, y, z, eval );
        collocation_cartesian_angular_1_deriv1( npts, bf, bf_x, bf_y, bf_z, x, y, z, eval_x, eval_y, eval_z );

      } else if( l == 2 ) {
  
        collocation_cartesian_angular_2( npts, bf, x, y, z, eval );
        collocation_cartesian_angular_2_deriv1( npts, bf, bf_x, bf_y, bf_z, x, y, z, eval_x, eval_y, eval_z );

      } else if( l == 3 ) {
  
        collocation_cartesian_angular_3( npts, bf, x, y, z, eval );
        collocation_cartesian_angular_3_deriv1( npts, bf, bf_x, bf_y, bf_z, x, y, z, eval_x, eval_y, eval_z );

      } else if( l == 4 ) {
  
        collocation_cartesian_angular_4( npts, bf, x, y, z, eval );
        collocation_cartesian_angular_4_deriv1( npts, bf, bf_x, bf_y, bf_z, x, y, z, eval_x, eval_y, eval_z );

    } else {
      assert( false && "L < L_MAX" );
    }

} // collocation_cartesian_angular_deriv1


} // namespace GauXC

