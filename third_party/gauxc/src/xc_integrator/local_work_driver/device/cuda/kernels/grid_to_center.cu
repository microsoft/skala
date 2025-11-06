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
#include <gauxc/util/div_ceil.hpp>
#include "grid_to_center.hpp"
#include "device_specific/cuda_device_constants.hpp"

namespace GauXC {

__global__ void compute_grid_to_center_dist(
        size_t npts,
        size_t natoms,
  const double* coords,
  //const double* points,
  const double* points_x,
  const double* points_y,
  const double* points_z,
        double* dist,
        size_t lddist
) {

  __shared__ double3 point_buffer[cuda::warp_size];
  register double3 coord_reg;

  const int natoms_block = (natoms + cuda::warp_size-1) / cuda::warp_size;
  const int coords_block = (npts + cuda::warp_size-1) / cuda::warp_size;

  const double3* coords_vec = (double3*) coords;
  //const double3* points_vec = (double3*) points;

  for (int j = blockIdx.x; j < natoms_block; j += gridDim.x) {
    const int iAtom = j * cuda::warp_size + threadIdx.x;
    // Load blocks into registers/shared memory
    if (iAtom < natoms) {
      coord_reg = coords_vec[iAtom];
    }
    for (int i = blockIdx.y; i < coords_block; i += gridDim.y) {
      const int iPt_load = i * cuda::warp_size + threadIdx.x;
      if (iPt_load < npts) {
        //point_buffer[threadIdx.x] = points_vec[iPt_load];
        point_buffer[threadIdx.x].x = points_x[iPt_load];
        point_buffer[threadIdx.x].y = points_y[iPt_load];
        point_buffer[threadIdx.x].z = points_z[iPt_load];
      }
      __syncthreads();

      // do the computation
      #pragma unroll 2
      for (int k = threadIdx.y; k < cuda::warp_size; k+=cuda::warp_size/2) {
        const int iPt_sm = k;
        const int iPt = i * cuda::warp_size + iPt_sm;
        const double rx = point_buffer[iPt_sm].x - coord_reg.x;
        const double ry = point_buffer[iPt_sm].y - coord_reg.y;
        const double rz = point_buffer[iPt_sm].z - coord_reg.z;

        if (iAtom < natoms and iPt < npts) {
          dist[ iAtom + iPt * lddist ] = std::sqrt( rx*rx + ry*ry + rz*rz );
        }
      }
      __syncthreads();
    }
  }
}

void compute_grid_to_center_dist( int32_t npts, int32_t natoms,
  const double* coords, const double* points_x, const double* points_y, 
  const double* points_z, double* dist, int32_t lddist, cudaStream_t stream ) {

    const int distance_thread_y = cuda::max_warps_per_thread_block / 2;
    dim3 threads( cuda::warp_size, distance_thread_y );
    dim3 blocks( util::div_ceil( natoms,   threads.x), 
                 util::div_ceil( npts, threads.y * distance_thread_y) );

    compute_grid_to_center_dist<<< blocks, threads, 0, stream>>>(
      npts, natoms, coords, points_x, points_y, points_z, dist, lddist
    );

}

}
