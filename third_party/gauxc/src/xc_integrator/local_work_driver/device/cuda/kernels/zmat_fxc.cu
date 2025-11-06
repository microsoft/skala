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
#include "device/common/zmat_fxc.hpp"
#include <gauxc/util/div_ceil.hpp>
#include "device_specific/cuda_util.hpp"
#include "device_specific/cuda_device_constants.hpp"

namespace GauXC {


template<density_id den_selector>
__global__ void zmat_lda_fxc_kernel( size_t        ntasks,
                                     XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];
  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;
  const auto* FXC_A_device = task.FXC_A_s;
  if constexpr ( den_selector == DEN_Z ) FXC_A_device = task.FXC_A_z;

  const auto* basis_eval_device = task.bf;
  auto* z_matrix_device = task.zmat;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nbf ) {

    const size_t ibfoff = tid_y * npts + tid_x;
    const double fact = 0.5 * FXC_A_device[tid_x];

    z_matrix_device[ ibfoff ] = fact * basis_eval_device[ ibfoff ];
  }

}





template<density_id den_selector>
__global__ void zmat_gga_fxc_kernel( size_t        ntasks,
                                     XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];
  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;

  const auto* basis_eval_device = task.bf;
  const auto* dbasis_x_eval_device = task.dbfx;
  const auto* dbasis_y_eval_device = task.dbfy;
  const auto* dbasis_z_eval_device = task.dbfz;
  const auto* FXC_A_device   = task.FXC_A_s;
  const auto* FXC_Bx_device  = task.FXC_Bx_s;
  const auto* FXC_By_device  = task.FXC_By_s;
  const auto* FXC_Bz_device  = task.FXC_Bz_s;
  if constexpr ( den_selector == DEN_Z ) {
    FXC_A_device   = task.FXC_A_z;
    FXC_Bx_device  = task.FXC_Bx_z;
    FXC_By_device  = task.FXC_By_z;
    FXC_Bz_device  = task.FXC_Bz_z;
  }

  auto* z_matrix_device = task.zmat;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nbf ) {

    const size_t ibfoff = tid_y * npts + tid_x;

    const double dx = FXC_Bx_device[tid_x] * dbasis_x_eval_device[ ibfoff ];
    const double dy = FXC_By_device[tid_x] * dbasis_y_eval_device[ ibfoff ];
    const double dz = FXC_Bz_device[tid_x] * dbasis_z_eval_device[ ibfoff ];

    z_matrix_device[ ibfoff ] = 
      (0.5 * FXC_A_device[tid_x] * basis_eval_device[ ibfoff ] +  dx + dy + dz ); 
  }
}



#define ZMAT_FXC_KERN(xc_approx) \
  cudaStream_t stream = queue.queue_as<util::cuda_stream>(); \
  dim3 threads(cuda::warp_size,cuda::max_warps_per_thread_block,1); \
  dim3 blocks( util::div_ceil( max_npts, threads.x ), \
               util::div_ceil( max_nbf,  threads.y ), \
               ntasks ); \
  if ( sel == DEN_S )       zmat_##xc_approx##_fxc_kernel<DEN_S><<< blocks, threads, 0, stream >>>( ntasks, tasks_device ); \
  else if ( sel == DEN_Z )  zmat_##xc_approx##_fxc_kernel<DEN_Z><<< blocks, threads, 0, stream >>>( ntasks, tasks_device ); \



void zmat_lda_fxc( size_t            ntasks,
                   int32_t           max_nbf,
                   int32_t           max_npts,
                   XCDeviceTask*     tasks_device,
                   density_id sel,
                   device_queue queue ) {
ZMAT_FXC_KERN(lda)
}



void zmat_gga_fxc( size_t            ntasks,
                   int32_t           max_nbf,
                   int32_t           max_npts,
                   XCDeviceTask*     tasks_device,
                   density_id sel,
                   device_queue queue ) {
ZMAT_FXC_KERN(gga)
}



void zmat_mgga_fxc( size_t            ntasks,
                    int32_t           max_nbf,
                    int32_t           max_npts,
                    XCDeviceTask*     tasks_device,
                    bool              do_lapl,
                    density_id sel,
                    device_queue queue ) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>() ;


  dim3 threads(cuda::warp_size,cuda::max_warps_per_thread_block,1);
  dim3 blocks( util::div_ceil( max_npts, threads.x ),
               util::div_ceil( max_nbf,  threads.y ),
               ntasks );

  if(do_lapl)
    GAUXC_GENERIC_EXCEPTION("Fxc contraction + do_lapl NYI");
    
  switch(sel) {
    case DEN_S:
        zmat_gga_fxc_kernel<DEN_S><<< blocks, threads, 0, stream >>>( ntasks, tasks_device );
      break;
    case DEN_Z:
        zmat_gga_fxc_kernel<DEN_Z><<< blocks, threads, 0, stream >>>( ntasks, tasks_device );
      break;
  }

}










template <density_id id>
__global__ void mmat_mgga_fxc_kernel( size_t        ntasks,
                                     XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];
  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;
  auto* FXC_C_s_device   = task.FXC_C_s;
  if constexpr ( id == DEN_Z ) FXC_C_s_device = task.FXC_C_z;

  const auto* dbasis_x_eval_device = task.dbfx;
  const auto* dbasis_y_eval_device = task.dbfy;
  const auto* dbasis_z_eval_device = task.dbfz;

  auto* mmat_x = task.xmat_x;
  auto* mmat_y = task.xmat_y;
  auto* mmat_z = task.xmat_z;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nbf ) {

    const size_t ibfoff = tid_y * npts + tid_x;

    const double fact = 0.25 * FXC_C_s_device[tid_x];

    mmat_x[ ibfoff ] = fact * dbasis_x_eval_device[ ibfoff ]; 
    mmat_y[ ibfoff ] = fact * dbasis_y_eval_device[ ibfoff ]; 
    mmat_z[ ibfoff ] = fact * dbasis_z_eval_device[ ibfoff ]; 
  }
}

void mmat_mgga_fxc( size_t            ntasks,
                    int32_t           max_nbf,
                    int32_t           max_npts,
                    XCDeviceTask*     tasks_device,
                    bool              do_lapl,
                    density_id sel,
                    device_queue queue ) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>() ;


  dim3 threads(cuda::warp_size,cuda::max_warps_per_thread_block,1);
  dim3 blocks( util::div_ceil( max_npts, threads.x ),
               util::div_ceil( max_nbf,  threads.y ),
               ntasks );

  if(do_lapl)
    GAUXC_GENERIC_EXCEPTION("Fxc contraction + do_lapl NYI");
    
  switch(sel) {
    case DEN_S:
        mmat_mgga_fxc_kernel<DEN_S><<< blocks, threads, 0, stream >>>( ntasks, tasks_device );
      break;
    case DEN_Z:
        mmat_mgga_fxc_kernel<DEN_Z><<< blocks, threads, 0, stream >>>( ntasks, tasks_device );
      break;
  }
  
}

}

