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
#include <cuda.h>

namespace GauXC {
namespace cuda  {

template <size_t warp_sz, typename T>
__device__ T warp_reduce_sum(T val) {

  for(int i=(warp_sz/2); i>=1; i/=2)
    val += __shfl_xor_sync(0xffffffff, val, i, warp_sz);

  return val;
}

template <size_t warp_sz, typename T>
__device__ T warp_reduce_prod(T val) {

  for(int i=(warp_sz/2); i>=1; i/=2)
    val *= __shfl_xor_sync(0xffffffff, val, i, warp_sz);

  return val;
}

template <size_t warp_sz, typename T>
__device__ T warp_reduce_max(T val) {

  for(int i=(warp_sz/2); i>=1; i/=2)
    val = fmax( val, __shfl_xor_sync(0xffffffff, val, i, warp_sz) );

  return val;
}

}
}
