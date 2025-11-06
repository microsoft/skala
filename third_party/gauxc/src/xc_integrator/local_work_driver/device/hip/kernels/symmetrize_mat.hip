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
#include "device/common/symmetrize_mat.hpp"
#include "hip/hip_runtime.h"
#include "device_specific/hip_util.hpp"

namespace GauXC {


__host__ __device__ inline constexpr int div_ceil( int i, int j ){ return (i/j) + !!(i%j); } 

template <int BLOCK_SIZE>
__global__ void symmetrize_matrix_kernel( int N, double* A, int LDA ) {

  __shared__ double shmem[BLOCK_SIZE][BLOCK_SIZE+1];

  const auto n_block = div_ceil(N, BLOCK_SIZE);
  for( int i_block = 0;       i_block < n_block; ++i_block )
  for( int j_block = i_block; j_block < n_block; ++j_block ) { 
  
    const int ij_block = (i_block+1) + (j_block+1)*j_block/2 - 1;
    if( ij_block % gridDim.x != blockIdx.x ) continue;

    const int i_coord = i_block * BLOCK_SIZE;
    const int j_coord = j_block * BLOCK_SIZE;

    const int ix = i_coord + threadIdx.x;
    const int iy = i_coord + threadIdx.y;
    const int jx = j_coord + threadIdx.x;
    const int jy = j_coord + threadIdx.y;

    if( iy < N and jx < N )
      shmem[threadIdx.y][threadIdx.x] = A[iy*LDA + jx];
    __syncthreads();

    if( jy < N and ix < N )
    if( i_coord != j_coord or threadIdx.x < threadIdx.y ) // Diagonal block
      A[jy*LDA + ix] = shmem[threadIdx.x][threadIdx.y];
    __syncthreads();

  }

}

void symmetrize_matrix( int32_t N, double* A, size_t LDA, device_queue queue ) {

  hipStream_t stream = queue.queue_as<util::hip_stream>();
  constexpr int block_size = 32;
  const int n_rc_blocks = div_ceil( N, block_size );
  const size_t n_total_blocks = n_rc_blocks * (n_rc_blocks+1) / 2;
  dim3 threads(block_size,block_size), blocks(std::min(100ul, n_total_blocks));
  
  symmetrize_matrix_kernel<32><<<blocks,threads,0,stream>>>( N, A, LDA );

}

}
