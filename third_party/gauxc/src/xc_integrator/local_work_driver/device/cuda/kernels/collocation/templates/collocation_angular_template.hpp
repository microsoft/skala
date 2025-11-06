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

$for( L in range(L_max + 1) )\
template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_$(name)_angular_$(L)(
  int32_t          npts,
  const T          bf,
  const T          x,
  const T          y,
  const T          z,
  T* __restrict__  eval
) {

  $(body[L])

}

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_$(name)_angular_$(L)_deriv1(
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

  $(body_d1[L])

}

$endfor\

template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_$(name)_angular(
  const int32_t   npts,
  const int32_t    l,
  const T          bf,
  const T          x,
  const T          y,
  const T          z,
  T* __restrict__  eval
) {

$for( L in range(L_max + 1) )\
  $if( L == 0 )\
    if( l == $(L) ) {
  $else\
    } else if( l == $(L) ) {
  $endif
        collocation_$(name)_angular_$(L)( npts, bf, x, y, z, eval );

$endfor\
    } else {
      assert( false && "L < L_MAX" );
    }

} // collocation_$(name)_angular


template <typename T>
GPGAUEVAL_INLINE __device__ void collocation_$(name)_angular_deriv1(
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


$for( L in range(L_max + 1) )\
  $if( L == 0 )\
    if( l == $(L) ) {
  $else\
    } else if( l == $(L) ) {
  $endif
        collocation_$(name)_angular_$(L)( npts, bf, x, y, z, eval );
        collocation_$(name)_angular_$(L)_deriv1( npts, bf, bf_x, bf_y, bf_z, x, y, z, eval_x, eval_y, eval_z );

$endfor\
    } else {
      assert( false && "L < L_MAX" );
    }

} // collocation_$(name)_angular_deriv1


} // namespace GauXC

