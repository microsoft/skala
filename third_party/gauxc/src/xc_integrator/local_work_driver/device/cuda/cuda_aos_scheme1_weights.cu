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
#include "cuda_aos_scheme1.hpp"
#include "device/cuda/cuda_backend.hpp"
#include "kernels/grid_to_center.hpp"
#include "kernels/cuda_ssf_2d.hu"
#include "kernels/cuda_ssf_1d.hpp"
#include "cuda_aos_scheme1_weights.hpp"
 
namespace GauXC {

void cuda_aos_scheme1_weights_wrapper( int32_t npts, int32_t natoms,
  const double* points_x, const double* points_y, const double* points_z, 
  const double* RAB, int32_t ldRAB, const double* coords, 
  double* dist, int32_t lddist, const int32_t* iparent,
  const double* dist_nearest, double* weights, cudaStream_t stream ) {

  constexpr auto weight_unroll = 
    alg_constants::CudaAoSScheme1::weight_unroll;
  constexpr auto weight_thread_block = 
    alg_constants::CudaAoSScheme1::weight_thread_block;
  constexpr auto weight_thread_block_per_sm = 
    alg_constants::CudaAoSScheme1::weight_thread_block_per_sm;



  // Compute distances from grid to atomic centers
  compute_grid_to_center_dist( npts, natoms, coords, points_x, points_y, points_z, 
   dist, lddist, stream );

#if 0
  // Get the number of SM's on the device
  int num_sm;
  int dev_id = 0;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id);

  // Modify weights
  dim3 threads( cuda::warp_size, weight_thread_block / cuda::warp_size );
  dim3 blocks ( 1, num_sm * weight_thread_block_per_sm );
  modify_weights_ssf_kernel_2d
    <weight_unroll, weight_thread_block, weight_thread_block_per_sm>
    <<< blocks, threads, 0, stream >>> (
      npts, natoms, RAB, ldRAB, coords, dist, lddist, iparent, dist_nearest,
      weights
    );
#else
  partition_weights_ssf_1d( npts, natoms, RAB, ldRAB, coords, dist, lddist,
    iparent, dist_nearest, weights, stream );
#endif

}


void cuda_aos_scheme1_weight_1st_deriv_wrapper(
  int32_t npts, int32_t natoms,
  const double* points_x, const double* points_y, const double* points_z,
  const double* RAB, int32_t ldRAB, const double* coords, 
  double* dist, int32_t lddist, const int32_t* iparent,
  const double* dist_nearest, const double* w_times_f,
  double* exc_grad_w, cudaStream_t stream ){

  // Compute distances from grid to atomic centers
  compute_grid_to_center_dist( npts, natoms, coords, points_x, points_y, points_z, 
   dist, lddist, stream );

  eval_weight_1st_deriv_contracted_ssf_1d( npts, natoms, RAB, ldRAB, coords, points_x, points_y, points_z, dist, lddist,
    iparent, dist_nearest, w_times_f, exc_grad_w, stream );

}





}
