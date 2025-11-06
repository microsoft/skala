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
#include <gauxc/util/div_ceil.hpp>
#include "grid_to_center.hpp"
#include "device_specific/hip_device_constants.hpp"
#include "exceptions/hip_exception.hpp"

namespace GauXC {

__global__ void compute_grid_to_center_dist(
        int32_t npts,
        int32_t natoms,
  const double* coords,
  const double* points_x,
  const double* points_y,
  const double* points_z,
        double* dist,
        size_t lddist
) {

  const auto tid_x = threadIdx.x + blockIdx.x*blockDim.x;
  const auto tid_y = threadIdx.y + blockIdx.y*blockDim.y;

  if( tid_x < natoms && tid_y < npts ) {
    const double3* coords_vec = (double3*) coords;
    const auto RA = coords_vec[tid_x];

    const double rix = points_x[tid_y];
    const double riy = points_y[tid_y];
    const double riz = points_z[tid_y];

    const auto rx = RA.x - rix;
    const auto ry = RA.y - riy;
    const auto rz = RA.z - riz;

    dist[ tid_x + tid_y * lddist ] = std::sqrt(rx*rx + ry*ry + rz*rz);
  }

}

void compute_grid_to_center_dist( int32_t npts, int32_t natoms,
  const double* coords, const double* points_x,  const double* points_y, 
  const double* points_z, double* dist, int32_t lddist, hipStream_t stream ) {


  dim3 threads( hip::warp_size, hip::max_warps_per_thread_block );
  dim3 blocks( util::div_ceil( natoms, threads.x ),
               util::div_ceil( npts, threads.y ) );

  hipLaunchKernelGGL( compute_grid_to_center_dist, blocks, threads, 0, stream,
    npts, natoms, coords, points_x, points_y, points_z, dist, lddist );

}

}
