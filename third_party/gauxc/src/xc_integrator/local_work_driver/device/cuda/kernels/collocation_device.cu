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
#include "device_specific/cuda_util.hpp"
#include "exceptions/cuda_exception.hpp"
#include <gauxc/xc_task.hpp>

#include "device/common/collocation_device.hpp"
#include "device/cuda/kernels/collocation_masked_kernels.hpp"
#include "device/cuda/kernels/collocation_masked_combined_kernels.hpp"
#include "device/cuda/kernels/collocation_shell_to_task_kernels.hpp"

#include "device_specific/cuda_device_constants.hpp"

#define GAUXC_CUDA_MAX_L 4

namespace GauXC {

 
template <typename T>
void eval_collocation_masked(
  size_t            nshells,
  size_t            nbf,
  size_t            npts,
  const Shell<T>*   shells_device,
  const size_t*     mask_device,
  const size_t*     offs_device,
  const T*          pts_device,
  T*                eval_device,
  device_queue queue
) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>() ;

  auto nmax_threads = util::cuda_kernel_max_threads_per_block( 
    collocation_device_masked_kernel<T>
  );
  auto max_warps_per_thread_block = nmax_threads / cuda::warp_size;

  dim3 threads(cuda::warp_size, max_warps_per_thread_block, 1);
  dim3 blocks( util::div_ceil( npts,    threads.x ),
               util::div_ceil( nshells, threads.y ) );

  collocation_device_masked_kernel<T>
    <<<blocks, threads, 0, stream>>>
    ( nshells, nbf, npts, shells_device, mask_device,
      offs_device, pts_device, eval_device );

}
 
template             
void eval_collocation_masked(
  size_t               nshells,
  size_t               nbf,
  size_t               npts,
  const Shell<double>* shells_device,
  const size_t*        mask_device,
  const size_t*        offs_device,
  const double*        pts_device,
  double*              eval_device,
  device_queue    queue
);




template <typename T>
void eval_collocation_masked_combined(
  size_t            ntasks,
  size_t            npts_max,
  size_t            nshells_max,
  Shell<T>*         shells_device,
  XCDeviceTask*     device_tasks,
  device_queue queue
) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>() ;

  auto nmax_threads = util::cuda_kernel_max_threads_per_block( 
    collocation_device_masked_combined_kernel<T>
  );

  auto max_warps_per_thread_block = nmax_threads / cuda::warp_size;
  dim3 threads(cuda::warp_size, max_warps_per_thread_block, 1);
  dim3 blocks( util::div_ceil( npts_max,    threads.x ),
               util::div_ceil( nshells_max, threads.y ),
               ntasks );

  collocation_device_masked_combined_kernel<T>
    <<<blocks, threads, 0, stream>>>
    ( ntasks, shells_device, device_tasks );
     
}

template
void eval_collocation_masked_combined(
  size_t            ntasks,
  size_t            npts_max,
  size_t            nshells_max,
  Shell<double>*    shells_device,
  XCDeviceTask*     device_tasks,
  device_queue queue
);











template <typename T>
void eval_collocation_masked_deriv1(
  size_t          nshells,
  size_t          nbf,
  size_t          npts,
  const Shell<T>* shells_device,
  const size_t*   mask_device,
  const size_t*   offs_device,
  const T*        pts_device,
  T*              eval_device,
  T*              deval_device_x,
  T*              deval_device_y,
  T*              deval_device_z,
  device_queue queue
) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>() ;

  auto nmax_threads = util::cuda_kernel_max_threads_per_block( 
    collocation_device_masked_combined_kernel<T>
  );

  auto max_warps_per_thread_block = nmax_threads / cuda::warp_size;
  dim3 threads(cuda::warp_size, max_warps_per_thread_block, 1);
  dim3 blocks( util::div_ceil( npts,    threads.x ),
               util::div_ceil( nshells, threads.y ) );

  collocation_device_masked_kernel_deriv1<T>
    <<<blocks, threads, 0, stream>>>
    ( nshells, nbf, npts, shells_device, mask_device, offs_device,
      pts_device, eval_device, deval_device_x, deval_device_y,
      deval_device_z );

}

template
void eval_collocation_masked_deriv1(
  size_t               nshells,
  size_t               nbf,
  size_t               npts,
  const Shell<double>* shells_device,
  const size_t*        mask_device,
  const size_t*        offs_device,
  const double*        pts_device,
  double*              eval_device,
  double*              deval_device_x,
  double*              deval_device_y,
  double*              deval_device_z,
  device_queue    queue
);
















template <typename T>
void eval_collocation_masked_combined_deriv1(
  size_t        ntasks,
  size_t        npts_max,
  size_t        nshells_max,
  Shell<T>*     shells_device,
  XCDeviceTask* device_tasks,
  device_queue queue
) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>() ;

  auto nmax_threads = util::cuda_kernel_max_threads_per_block( 
    collocation_device_masked_combined_kernel_deriv1<T>
  );

  dim3 threads(cuda::warp_size, nmax_threads/cuda::warp_size, 1);
  dim3 blocks( util::div_ceil( npts_max,    threads.x ),
               util::div_ceil( nshells_max, threads.y ),
               ntasks );

  collocation_device_masked_combined_kernel_deriv1<T>
    <<<blocks, threads, 0, stream>>>
    ( ntasks, shells_device, device_tasks );
     
}

template
void eval_collocation_masked_combined_deriv1(
  size_t                ntasks,
  size_t                npts_max,
  size_t                nshells_max,
  Shell<double>*        shells_device,
  XCDeviceTask* device_tasks,
  device_queue queue
);





























uint32_t max_threads_shell_to_task_collocation( int32_t l, bool pure ) {
  if( pure ) {
    switch(l) {
      case 0: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_0 );      
      case 1: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_1 );      
      case 2: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_2 );      
      case 3: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_3 );      
      case 4: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_4 );      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  } else {
    switch(l) {
      
      case 0: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_0 );      
      case 1: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_1 );      
      case 2: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_2 );      
      case 3: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_3 );      
      case 4: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_4 );      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  }
  return 0;
}

template <typename... Args>
void dispatch_shell_to_task_collocation( cudaStream_t stream, int32_t l, 
  bool pure, int32_t ntask_average, int32_t nshells, Args&&... args ) {

  dim3 threads = max_threads_shell_to_task_collocation(l,pure);
  int nwarp_per_block = threads.x / cuda::warp_size;
  int n_task_blocks = util::div_ceil( ntask_average, nwarp_per_block );
  dim3 block(n_task_blocks, 1, nshells);

  if( pure ) {
    switch(l) {
      case 0:
        collocation_device_shell_to_task_kernel_cartesian_0<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 1:
        collocation_device_shell_to_task_kernel_spherical_1<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 2:
        collocation_device_shell_to_task_kernel_spherical_2<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 3:
        collocation_device_shell_to_task_kernel_spherical_3<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 4:
        collocation_device_shell_to_task_kernel_spherical_4<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  } else {
    switch(l) {      
      case 0:
        collocation_device_shell_to_task_kernel_cartesian_0<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 1:
        collocation_device_shell_to_task_kernel_cartesian_1<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 2:
        collocation_device_shell_to_task_kernel_cartesian_2<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 3:
        collocation_device_shell_to_task_kernel_cartesian_3<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 4:
        collocation_device_shell_to_task_kernel_cartesian_4<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  }
}


void eval_collocation_shell_to_task(
  uint32_t                    max_l,
  AngularMomentumShellToTaskBatch* l_batched_shell_to_task,
  XCDeviceTask*               device_tasks,
  device_queue           queue 
) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>() ;

  for( auto l = 0u; l <= max_l; ++l ) {
    auto pure = l_batched_shell_to_task[l].pure;
    auto shell_to_task_device = l_batched_shell_to_task[l].shell_to_task_device;
    auto nshells = l_batched_shell_to_task[l].nshells_in_batch;
    auto ntask_average = std::max(1ul, l_batched_shell_to_task[l].ntask_average);
    dispatch_shell_to_task_collocation( stream, l, pure, ntask_average, nshells, 
      shell_to_task_device, device_tasks );
  }


}


uint32_t max_threads_shell_to_task_collocation_gradient( int32_t l, bool pure ) {
  if( pure ) {
    switch(l) {
      case 0: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_gradient_0 );      
      case 1: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_gradient_1 );      
      case 2: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_gradient_2 );      
      case 3: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_gradient_3 );      
      case 4: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_gradient_4 );      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  } else {
    switch(l) {      
      case 0: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_gradient_0 );      
      case 1: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_gradient_1 );      
      case 2: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_gradient_2 );      
      case 3: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_gradient_3 );      
      case 4: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_gradient_4 );      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  }
  return 0;
}

template <typename... Args>
void dispatch_shell_to_task_collocation_gradient( cudaStream_t stream, int32_t l, 
  bool pure, uint32_t ntask_average, uint32_t nshells, Args&&... args ) {

  dim3 threads = max_threads_shell_to_task_collocation_gradient(l,pure);
  int nwarp_per_block = threads.x / cuda::warp_size;
  int n_task_blocks = util::div_ceil( ntask_average, nwarp_per_block );
  dim3 block(n_task_blocks, 1, nshells);

  if( pure ) {
    switch(l) {
      case 0:
        collocation_device_shell_to_task_kernel_cartesian_gradient_0<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 1:
        collocation_device_shell_to_task_kernel_spherical_gradient_1<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 2:
        collocation_device_shell_to_task_kernel_spherical_gradient_2<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 3:
        collocation_device_shell_to_task_kernel_spherical_gradient_3<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 4:
        collocation_device_shell_to_task_kernel_spherical_gradient_4<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  } else {
    switch(l) {      
      case 0:
        collocation_device_shell_to_task_kernel_cartesian_gradient_0<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 1:
        collocation_device_shell_to_task_kernel_cartesian_gradient_1<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 2:
        collocation_device_shell_to_task_kernel_cartesian_gradient_2<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 3:
        collocation_device_shell_to_task_kernel_cartesian_gradient_3<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 4:
        collocation_device_shell_to_task_kernel_cartesian_gradient_4<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  }

}


void eval_collocation_shell_to_task_gradient(
  uint32_t                    max_l,
  AngularMomentumShellToTaskBatch* l_batched_shell_to_task,
  XCDeviceTask*               device_tasks,
  device_queue           queue 
) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>() ;

  for( auto l = 0u; l <= max_l; ++l ) {
    auto pure = l_batched_shell_to_task[l].pure;
    auto shell_to_task_device = l_batched_shell_to_task[l].shell_to_task_device;
    auto nshells = l_batched_shell_to_task[l].nshells_in_batch;
    auto ntask_average = std::max(1ul, l_batched_shell_to_task[l].ntask_average);
    dispatch_shell_to_task_collocation_gradient( stream, l, pure, 
      ntask_average, nshells, shell_to_task_device, device_tasks );
  }


}


uint32_t max_threads_shell_to_task_collocation_hessian( int32_t l, bool pure ) {
  if( pure ) {
    switch(l) {
      case 0: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_hessian_0 );      
      case 1: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_hessian_1 );      
      case 2: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_hessian_2 );      
      case 3: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_hessian_3 );      
      case 4: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_hessian_4 );      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  } else {
    switch(l) {      
      case 0: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_hessian_0 );      
      case 1: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_hessian_1 );      
      case 2: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_hessian_2 );      
      case 3: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_hessian_3 );      
      case 4: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_hessian_4 );      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  }
  return 0;
}

template <typename... Args>
void dispatch_shell_to_task_collocation_hessian( cudaStream_t stream, int32_t l, 
  bool pure, uint32_t ntask_average, uint32_t nshells, Args&&... args ) {

  dim3 threads = max_threads_shell_to_task_collocation_hessian(l,pure);
  int nwarp_per_block = threads.x / cuda::warp_size;
  int n_task_blocks = util::div_ceil( ntask_average, nwarp_per_block );
  dim3 block(n_task_blocks, 1, nshells);

  if( pure ) {
    switch(l) {
      case 0:
        collocation_device_shell_to_task_kernel_cartesian_hessian_0<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 1:
        collocation_device_shell_to_task_kernel_spherical_hessian_1<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 2:
        collocation_device_shell_to_task_kernel_spherical_hessian_2<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 3:
        collocation_device_shell_to_task_kernel_spherical_hessian_3<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 4:
        collocation_device_shell_to_task_kernel_spherical_hessian_4<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  } else {
    switch(l) {      
      case 0:
        collocation_device_shell_to_task_kernel_cartesian_hessian_0<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 1:
        collocation_device_shell_to_task_kernel_cartesian_hessian_1<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 2:
        collocation_device_shell_to_task_kernel_cartesian_hessian_2<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 3:
        collocation_device_shell_to_task_kernel_cartesian_hessian_3<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 4:
        collocation_device_shell_to_task_kernel_cartesian_hessian_4<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  }

}


void eval_collocation_shell_to_task_hessian(
  uint32_t                    max_l,
  AngularMomentumShellToTaskBatch* l_batched_shell_to_task,
  XCDeviceTask*               device_tasks,
  device_queue           queue 
) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>() ;

  for( auto l = 0u; l <= max_l; ++l ) {
    auto pure = l_batched_shell_to_task[l].pure;
    auto shell_to_task_device = l_batched_shell_to_task[l].shell_to_task_device;
    auto nshells = l_batched_shell_to_task[l].nshells_in_batch;
    auto ntask_average = std::max(1ul, l_batched_shell_to_task[l].ntask_average);
    dispatch_shell_to_task_collocation_hessian( stream, l, pure, 
      ntask_average, nshells, shell_to_task_device, device_tasks );
  }


}


uint32_t max_threads_shell_to_task_collocation_laplacian( int32_t l, bool pure ) {
  if( pure ) {
    switch(l) {
      case 0: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_laplacian_0 );      
      case 1: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_laplacian_1 );
      
      case 2: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_laplacian_2 );
      
      case 3: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_laplacian_3 );
      
      case 4: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_laplacian_4 );
      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  } else {
    switch(l) {      
      case 0: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_laplacian_0 );      
      case 1: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_laplacian_1 );      
      case 2: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_laplacian_2 );      
      case 3: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_laplacian_3 );      
      case 4: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_laplacian_4 );      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  }
  return 0;
}





template <typename... Args>
void dispatch_shell_to_task_collocation_laplacian( cudaStream_t stream, int32_t l,
  bool pure, uint32_t ntask_average, uint32_t nshells, Args&&... args ) {

  dim3 threads = max_threads_shell_to_task_collocation_laplacian(l,pure);
  int nwarp_per_block = threads.x / cuda::warp_size;
  int n_task_blocks = util::div_ceil( ntask_average, nwarp_per_block );
  dim3 block(n_task_blocks, 1, nshells);

  if( pure ) {
    switch(l) {
      case 0:
        collocation_device_shell_to_task_kernel_cartesian_laplacian_0<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;
      
      case 1:
        collocation_device_shell_to_task_kernel_spherical_laplacian_1<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 2:
        collocation_device_shell_to_task_kernel_spherical_laplacian_2<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 3:
        collocation_device_shell_to_task_kernel_spherical_laplacian_3<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 4:
        collocation_device_shell_to_task_kernel_spherical_laplacian_4<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  } else {
    switch(l) {      
      case 0:
        collocation_device_shell_to_task_kernel_cartesian_laplacian_0<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 1:
        collocation_device_shell_to_task_kernel_cartesian_laplacian_1<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 2:
        collocation_device_shell_to_task_kernel_cartesian_laplacian_2<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 3:
        collocation_device_shell_to_task_kernel_cartesian_laplacian_3<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 4:
        collocation_device_shell_to_task_kernel_cartesian_laplacian_4<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  }

}



void eval_collocation_shell_to_task_laplacian(
  uint32_t                    max_l,
  AngularMomentumShellToTaskBatch* l_batched_shell_to_task,
  XCDeviceTask*               device_tasks,
  device_queue           queue
) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>() ;

  for( auto l = 0u; l <= max_l; ++l ) {
    auto pure = l_batched_shell_to_task[l].pure;
    auto shell_to_task_device = l_batched_shell_to_task[l].shell_to_task_device;
    auto nshells = l_batched_shell_to_task[l].nshells_in_batch;
    auto ntask_average = std::max(1ul, l_batched_shell_to_task[l].ntask_average);
    dispatch_shell_to_task_collocation_laplacian( stream, l, pure,
      ntask_average, nshells, shell_to_task_device, device_tasks );
    auto stat = cudaGetLastError();
    GAUXC_CUDA_ERROR("LAP", stat);
  }


}

uint32_t max_threads_shell_to_task_collocation_lapgrad( int32_t l, bool pure ) {
  if( pure ) {
    switch(l) {
      case 0: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_lapgrad_0 );      
      case 1: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_lapgrad_1 );
      
      case 2: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_lapgrad_2 );
      
      case 3: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_lapgrad_3 );
      
      case 4: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_spherical_lapgrad_4 );
      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  } else {
    switch(l) {      
      case 0: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_lapgrad_0 );      
      case 1: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_lapgrad_1 );      
      case 2: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_lapgrad_2 );      
      case 3: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_lapgrad_3 );      
      case 4: return util::cuda_kernel_max_threads_per_block( collocation_device_shell_to_task_kernel_cartesian_lapgrad_4 );      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  }
  return 0;
}





template <typename... Args>
void dispatch_shell_to_task_collocation_lapgrad( cudaStream_t stream, int32_t l,
  bool pure, uint32_t ntask_average, uint32_t nshells, Args&&... args ) {

  dim3 threads = max_threads_shell_to_task_collocation_lapgrad(l,pure);
  int nwarp_per_block = threads.x / cuda::warp_size;
  int n_task_blocks = util::div_ceil( ntask_average, nwarp_per_block );
  dim3 block(n_task_blocks, 1, nshells);

  if( pure ) {
    switch(l) {
      case 0:
        collocation_device_shell_to_task_kernel_cartesian_lapgrad_0<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;
      
      case 1:
        collocation_device_shell_to_task_kernel_spherical_lapgrad_1<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 2:
        collocation_device_shell_to_task_kernel_spherical_lapgrad_2<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 3:
        collocation_device_shell_to_task_kernel_spherical_lapgrad_3<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 4:
        collocation_device_shell_to_task_kernel_spherical_lapgrad_4<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  } else {
    switch(l) {      
      case 0:
        collocation_device_shell_to_task_kernel_cartesian_lapgrad_0<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 1:
        collocation_device_shell_to_task_kernel_cartesian_lapgrad_1<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 2:
        collocation_device_shell_to_task_kernel_cartesian_lapgrad_2<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 3:
        collocation_device_shell_to_task_kernel_cartesian_lapgrad_3<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      case 4:
        collocation_device_shell_to_task_kernel_cartesian_lapgrad_4<<<block,threads,0,stream>>>( nshells, std::forward<Args>(args)... );
        break;      
      default: GAUXC_GENERIC_EXCEPTION("CUDA L_MAX = 4");
    }
  }

}



void eval_collocation_shell_to_task_lapgrad(
  uint32_t                    max_l,
  AngularMomentumShellToTaskBatch* l_batched_shell_to_task,
  XCDeviceTask*               device_tasks,
  device_queue           queue
) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>() ;

  for( auto l = 0u; l <= max_l; ++l ) {
    auto pure = l_batched_shell_to_task[l].pure;
    auto shell_to_task_device = l_batched_shell_to_task[l].shell_to_task_device;
    auto nshells = l_batched_shell_to_task[l].nshells_in_batch;
    auto ntask_average = std::max(1ul, l_batched_shell_to_task[l].ntask_average);
    dispatch_shell_to_task_collocation_lapgrad( stream, l, pure,
      ntask_average, nshells, shell_to_task_device, device_tasks );
    auto stat = cudaGetLastError();
    GAUXC_CUDA_ERROR("LAPGRAD", stat);
  }


}



} // namespace GauXC
