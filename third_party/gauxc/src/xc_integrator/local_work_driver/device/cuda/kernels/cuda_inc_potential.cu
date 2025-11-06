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
#include "device/common/inc_potential.hpp"
#include "device_specific/cuda_device_constants.hpp"
#include <gauxc/util/div_ceil.hpp>
#include "device_specific/cuda_util.hpp"


namespace GauXC {


#define WARP_X 16
#define WARP_Y 1
#define UNROLL_FACTOR 4
#define EFF_UNROLL 4
#define CUT_X 8
#define CUT_Y 8


__global__ __launch_bounds__(1024, 1)
void sym_inc_by_submat_combined_kernel( size_t        ntasks,
                                    XCDeviceTask* device_tasks,
                                    double*       A,
                                    size_t        LDA, 
				    const int block_y,
				    const int block_x ) {

  const int batch_id = blockIdx.z;
  auto& task = device_tasks[ batch_id ];

  const auto* submat_cut_device = task.bfn_screening.submat_cut;
  const auto* submat_block_device = task.bfn_screening.submat_block;
  const auto  LDAS              = task.bfn_screening.nbe;
        auto* ASmall_device     = task.nbe_scr;

  //if( LDAS == LDAB ) return;
  const int tid_xx = threadIdx.x % WARP_X;
  const int tid_xy = threadIdx.x / WARP_X;

  const int tid_yx = threadIdx.y % CUT_X;
  const int tid_yy = threadIdx.y / CUT_X;

  const int start_cut_y = submat_block_device[block_y];
  const int end_cut_y   = submat_block_device[block_y+1];
  const int start_cut_x = submat_block_device[block_x];
  const int end_cut_x   = submat_block_device[block_x+1];

  for( int i_cut = tid_yy + start_cut_y; i_cut < end_cut_y; i_cut += CUT_Y ) {
    const int3 i_data = *((int3*)(submat_cut_device + 3*i_cut));
    const int i_cut_first  = i_data.x;
    const int delta_i      = i_data.y;
    const int i_cut_small  = i_data.z;

  for( int j_cut = tid_yx + start_cut_x; j_cut < end_cut_x; j_cut += CUT_X ) {
    const int3 j_data = *((int3*)(submat_cut_device + 3*j_cut));
    const int j_cut_first  = j_data.x; 
    const int delta_j      = j_data.y;
    const int j_cut_small  = j_data.z;

    auto* ASmall_begin = ASmall_device + i_cut_small + j_cut_small*LDAS;
    auto* ABig_begin   = A   + i_cut_first + j_cut_first*LDA;

    int J;
    for( J = tid_xy; J < (delta_j / EFF_UNROLL) * EFF_UNROLL; J += EFF_UNROLL ) {
      for( int I = tid_xx; I < delta_i; I += WARP_X ) {

        double val[UNROLL_FACTOR];
        double* address[UNROLL_FACTOR];
#pragma unroll
        for (int k = 0; k < UNROLL_FACTOR; k++) {
          val[k] = ASmall_begin[I + (J+k*WARP_Y)*LDAS];
          address[k] = ABig_begin + I + (J+k*WARP_Y)*LDA;
        }
#pragma unroll
        for (int k = 0; k < UNROLL_FACTOR; k++) {
          atomicAdd(address[k], val[k] );
        }
      }
    }

    for ( ; J < delta_j; J += WARP_Y) {
      for( int I = tid_xx; I < delta_i; I += WARP_X ) {
        atomicAdd(ABig_begin + I + J*LDA, ASmall_begin[I + J*LDAS] );
      }
    }

  }
  }
}


void sym_task_inc_potential( size_t        ntasks,
                         XCDeviceTask* device_tasks,
                         double*       V_device,
                         size_t        LDV,
                         size_t        submat_block_size,
                         device_queue  queue ) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>();

  dim3 threads( cuda::warp_size/2, cuda::max_warps_per_thread_block * 2, 1 );
  dim3 blocks( 1,1, ntasks );

  auto n_launch = util::div_ceil( LDV, submat_block_size );
  for (int i = 0; i < n_launch; i++) 
  for (int j = 0; j < n_launch; j++) {
    sym_inc_by_submat_combined_kernel<<< blocks, threads, 0, stream >>>(
      ntasks, device_tasks, V_device, LDV, i, j
    );
  }

}





__global__ __launch_bounds__(1024, 1)
void asym_inc_by_submat_combined_kernel( size_t        ntasks,
                                    XCDeviceTask* device_tasks,
                                    double*       A,
                                    size_t        LDA, 
				    const int block_y,
				    const int block_x ) {

  const int batch_id = blockIdx.z;
  auto& task = device_tasks[ batch_id ];

  const auto* row_submat_cut_device = task.bfn_screening.submat_cut;
  const auto* row_submat_block_device = task.bfn_screening.submat_block;
  const auto* col_submat_cut_device = task.cou_screening.submat_cut;
  const auto* col_submat_block_device = task.cou_screening.submat_block;

  const auto  LDAS              = task.bfn_screening.nbe;
        auto* ASmall_device     = task.nbe_scr;

  //if( LDAS == LDAB ) return;
  const int tid_xx = threadIdx.x % WARP_X;
  const int tid_xy = threadIdx.x / WARP_X;

  const int tid_yx = threadIdx.y % CUT_X;
  const int tid_yy = threadIdx.y / CUT_X;

  const int start_cut_y = row_submat_block_device[block_y];
  const int end_cut_y   = row_submat_block_device[block_y+1];
  const int start_cut_x = col_submat_block_device[block_x];
  const int end_cut_x   = col_submat_block_device[block_x+1];

  for( int i_cut = tid_yy + start_cut_y; i_cut < end_cut_y; i_cut += CUT_Y ) {
    const int3 i_data = *((int3*)(row_submat_cut_device + 3*i_cut));
    const int i_cut_first  = i_data.x;
    const int delta_i      = i_data.y;
    const int i_cut_small  = i_data.z;

  for( int j_cut = tid_yx + start_cut_x; j_cut < end_cut_x; j_cut += CUT_X ) {
    const int3 j_data = *((int3*)(col_submat_cut_device + 3*j_cut));
    const int j_cut_first  = j_data.x; 
    const int delta_j      = j_data.y;
    const int j_cut_small  = j_data.z;

    auto* ASmall_begin = ASmall_device + i_cut_small + j_cut_small*LDAS;
    auto* ABig_begin   = A   + i_cut_first + j_cut_first*LDA;

    int J;
    for( J = tid_xy; J < (delta_j / EFF_UNROLL) * EFF_UNROLL; J += EFF_UNROLL ) {
      for( int I = tid_xx; I < delta_i; I += WARP_X ) {

        double val[UNROLL_FACTOR];
        double* address[UNROLL_FACTOR];
#pragma unroll
        for (int k = 0; k < UNROLL_FACTOR; k++) {
          val[k] = ASmall_begin[I + (J+k*WARP_Y)*LDAS];
          address[k] = ABig_begin + I + (J+k*WARP_Y)*LDA;
        }
#pragma unroll
        for (int k = 0; k < UNROLL_FACTOR; k++) {
          atomicAdd(address[k], val[k] );
        }
      }
    }

    for ( ; J < delta_j; J += WARP_Y) {
      for( int I = tid_xx; I < delta_i; I += WARP_X ) {
        atomicAdd(ABig_begin + I + J*LDA, ASmall_begin[I + J*LDAS] );
      }
    }

  }
  }
}


void asym_task_inc_potential( size_t        ntasks,
                         XCDeviceTask* device_tasks,
                         double*       V_device,
                         size_t        LDV,
                         size_t        submat_block_size,
                         device_queue  queue ) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>();

  dim3 threads( cuda::warp_size/2, cuda::max_warps_per_thread_block * 2, 1 );
  dim3 blocks( 1,1, ntasks );

  auto n_launch = util::div_ceil( LDV, submat_block_size );
  for (int i = 0; i < n_launch; i++) 
  for (int j = 0; j < n_launch; j++) {
    asym_inc_by_submat_combined_kernel<<< blocks, threads, 0, stream >>>(
      ntasks, device_tasks, V_device, LDV, i, j
    );
  }

}


}

