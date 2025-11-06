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
#include "device/common/exx_ek_screening.hpp"
#include <gauxc/util/div_ceil.hpp>
#include <gauxc/shell.hpp>
#include "device_specific/cuda_util.hpp"
#include "cuda_extensions.hpp"
#include "device_specific/cuda_device_constants.hpp"
#include <cub/device/device_scan.cuh>
#include "buffer_adaptor.hpp"
#include "device/common/device_blas.hpp"
//#include <mpi.h>
#include <chrono>
//#include <fstream>
#include "exceptions/cuda_exception.hpp"

namespace GauXC {

__global__ void exx_ek_screening_bfn_stats_kernel( size_t ntasks, 
                                                   double      * max_bfn_sum_device,
                                                   double      * bfn_max_device,
                                                   size_t        LDBFM,
                                                   XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.x;
  if( batch_idx >= ntasks ) return;
  
  auto& task = tasks_device[ batch_idx ];
  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;
  auto* basis_eval_device = task.bf;
  const auto* weights_device    = task.weights;
  //double* bfn_max_device  = task.bfn_max;

  const int warp_lane = threadIdx.x % cuda::warp_size;
  const int warp_id   = threadIdx.x / cuda::warp_size;
  const int nwarp  = blockDim.x  / cuda::warp_size;



  // First scale the basis functions by the weights
  for(int ipt = warp_lane; ipt < npts; ipt += cuda::warp_size) {
    const auto w = std::sqrt(weights_device[ipt]);
    for(int ibf = warp_id; ibf < nbf; ibf += nwarp) {
      const auto val = basis_eval_device[ ipt + ibf*npts ];
      basis_eval_device[ ipt + ibf*npts ] = w * std::abs(val);
    }
  }
  __syncthreads();




  __shared__ double bf_shared[32][32 + 1];
  __shared__ double bfn_sum_shared[32];
  bfn_sum_shared[warp_lane] = 0.0;
  __syncthreads();

  const int npts_chunks = GauXC::util::div_ceil(npts, cuda::warp_size);
  const int  nbf_chunks = GauXC::util::div_ceil(nbf,  nwarp);
  for(int ipts_chunk = 0; ipts_chunk < npts_chunks; ++ipts_chunk) {
    double tmp_bfn_sum = 0.0;
    const int ipt = ipts_chunk * cuda::warp_size + warp_lane;
  for(int  ibf_chunk = 0;  ibf_chunk <  nbf_chunks; ++ ibf_chunk) {
    const int ibf =  ibf_chunk * nwarp + warp_id;

    bf_shared[warp_id][warp_lane] = 0.0;

    // Load in a block of basis functions
    // Warp lane is the point index and warp ID is the bfn idx
    if(ipt < npts and ibf < nbf) 
      bf_shared[warp_id][warp_lane]  = basis_eval_device[ipt + ibf*npts];
    __syncthreads();

    // Do transpose
    // Warp lane is the bfn index and warp ID is the point idx
    auto tmp = bf_shared[warp_lane][warp_id];
    __syncthreads();
 
    // Do a sum reduce over basis functions for the chunk
    // Now every warp has the local bfn chunk sum in lane 0
    // corresponding to the point represented by the warp Id
    tmp_bfn_sum += cuda::warp_reduce_sum<cuda::warp_size>(tmp);
    
  }
    // At this point, every warp contains the total bfn sum
    // for the point corresponding to its warp id. Update the 
    // local value accordingly
    if(warp_lane == 0) {
      double val = bfn_sum_shared[warp_id];
      bfn_sum_shared[warp_id] = fmax( val, tmp_bfn_sum );
    }
    __syncthreads();

  }

  // Get global maximum
  double max_bfn_sum;
  if(warp_id == 0) {
    auto tmp = bfn_sum_shared[warp_lane];
    max_bfn_sum = cuda::warp_reduce_max<cuda::warp_size>(tmp);
  }

  if(threadIdx.x == 0) {
    //task.max_bfn_sum = max_bfn_sum;
    max_bfn_sum_device[batch_idx] =  max_bfn_sum;
    //printf("[GPU] ITASK = %d MAX_SUM = %.6e PTR = %x\n", batch_idx, max_bfn_sum, task.bfn_shell_indirection);
    //printf("[GPU] ITASK = %d NBE = %lu NPTS = %lu \n", batch_idx, nbf, npts);
  }


  __syncthreads();
  for(int ibf = warp_id; ibf < nbf; ibf += nwarp) {
    double max_bf = 0;
    for(int ipt = warp_lane; ipt < npts; ipt += cuda::warp_size) {
      const auto val = basis_eval_device[ipt + ibf*npts];
      max_bf = fmax( max_bf, val );
    }

    // Warp reduce bf max
    max_bf = cuda::warp_reduce_max<cuda::warp_size>(max_bf);
    if(warp_lane == 0) {
      //printf("[GPU] ITASK = %d MAX_BFN(0) = %.6e\n", batch_idx, max_bf);
      bfn_max_device[batch_idx + task.bfn_shell_indirection[ibf]*LDBFM] =
        max_bf; 
    }
  }

}


__global__ void exx_ek_collapse_fmax_to_shells_kernel(
  int                  ntask,
  int                  nshells,
  const Shell<double>* shells_device,
  const int32_t*       shell_to_bf,
  const double*        fmax_bfn_device,
  size_t               LDF_bfn,
  double*              fmax_shell_device,
  size_t               LDF_shell
) {

  const int total_nwarp_x   = (blockDim.x * gridDim.x) / cuda::warp_size;
  const int tid_x           = threadIdx.x + blockIdx.x*blockDim.x;
  const int warp_lane       = tid_x % cuda::warp_size;
  const int warp_id_x       = tid_x / cuda::warp_size;


  //double sh_buffer[10];
  double sh_buffer;

  // Each warp gets a shell
  for(int ish = warp_id_x; ish < nshells; ish += total_nwarp_x) {

    const int sh_sz = shells_device[ish].size();
    const int sh_st = shell_to_bf[ish];

    // Read in tasks in warp-sized chunks
    for(int i_task = warp_lane; i_task < ntask; i_task += cuda::warp_size) {

      // Get shell max
      double sh_max = 0.0;
      for(int ii = 0; ii < sh_sz; ++ii) {
        sh_max = fmax(sh_max, fabs(fmax_bfn_device[i_task + (ii + sh_st)*LDF_bfn]));
      }
      
      // Write to main memory
      fmax_shell_device[i_task + ish*LDF_shell] = sh_max;
      
    }

  }
}


__global__ void exx_ek_shellpair_collision_shared_kernel(
  int32_t       ntasks,
  int32_t       nshell_pairs,
  int32_t       nshells,
  int32_t       shell_buffer_length,
  const double* V_max_sparse_device,
  const size_t* sp_row_ind_device,
  const size_t* sp_col_ind_device,
  const double* F_max_shl_device,
  size_t        LDF,
  const double* max_bf_sum_device,
  double        eps_E,
  double        eps_K,
  uint32_t*     collisions,
  int           LD_coll,
  uint32_t*     rc_collisions,
  int           LD_rc,
  uint32_t*     counts,
  uint32_t*     rc_counts
) {

  extern __shared__ uint32_t s_rc_collisions[];

  const int tid_x = threadIdx.y + blockIdx.x * blockDim.y;
  const int nt_x  = blockDim.y * gridDim.x;

  for(int i_task = tid_x; i_task < ntasks; i_task += nt_x) {

    const auto max_bf_sum = max_bf_sum_device[i_task];

    for (int i = threadIdx.x; i < shell_buffer_length; i+= blockDim.x) {
      s_rc_collisions[i] = 0;
    }
    __syncthreads();

    for(int ij_shell = threadIdx.x; ij_shell < nshell_pairs;  ij_shell+=blockDim.x) {

      const auto i_shell = sp_row_ind_device[ij_shell]; 
      const auto j_shell = sp_col_ind_device[ij_shell]; 

      const auto V_ij = V_max_sparse_device[ij_shell];
      const auto F_i  = F_max_shl_device[i_task + i_shell * LDF];
      const auto F_j  = F_max_shl_device[i_task + j_shell * LDF];

      const double eps_E_compare = F_i * F_j * V_ij;
      const double eps_K_compare = fmax(F_i, F_j) * V_ij * max_bf_sum;
      const bool comp = (eps_K_compare > eps_K or eps_E_compare > eps_E);

      const int ij = ij_shell;
      const int ij_block = ij / 32;
      const int ij_local = ij % 32;
      atomicOr(&(collisions[i_task * LD_coll + ij_block]), (comp ? (1u << ij_local) : 0));

      const int i_block = i_shell / 32;
      const int i_local = i_shell % 32;
      atomicOr(&(s_rc_collisions[i_block]), (comp ? (1u << i_local) : 0));

      const int j_block = j_shell / 32;
      const int j_local = j_shell % 32;
      atomicOr(&(s_rc_collisions[j_block]), (comp ? (1u << j_local) : 0));
    }
    __syncthreads();

    // Write from shared to global memory
    for (int i = threadIdx.x; i < shell_buffer_length; i+= blockDim.x) {
      rc_collisions[i_task * LD_rc + i] = s_rc_collisions[i];
    }
    __syncthreads();


    // TODO use thread block level reduction before writing to global memory
    uint32_t count = 0;
    for(int ij = threadIdx.x; ij < LD_coll; ij+=blockDim.x)  count += __popc(collisions[i_task * LD_coll + ij]);
    atomicAdd(&(counts[i_task]), count);

    count = 0;
    for(int ij = threadIdx.x; ij < LD_rc; ij+=blockDim.x)  count += __popc(rc_collisions[i_task * LD_rc + ij]);
    atomicAdd(&(rc_counts[i_task]), count);
    __syncthreads();
  }

}


__global__ void print_coll(size_t ntasks, size_t nshells, uint32_t* collisions,
  size_t LD_coll) {


  for(auto i_task = 0 ; i_task < ntasks; ++i_task) {

    printf("[GPU] ITASK %d: ", i_task);
    int count = 0;
    for(int i_shell = 0, ij = 0; i_shell < nshells;  ++i_shell      )
    for(int j_shell = 0;         j_shell <= i_shell; ++j_shell, ij++) {

      const int ij_block = ij / 32;
      const int ij_local = ij % 32;
      if( collisions[i_task * LD_coll + ij_block] & (1u << ij_local) ) {
        //printf("(%d, %d) ", i_shell, j_shell);
        count++;
      }
    }
    printf("%d\n", count);

  }
}

__global__ void print_counts(size_t ntasks, uint32_t* counts) {


  for(auto i_task = 0 ; i_task < ntasks; ++i_task) {

    printf("[GPU] ITASK %d: %d\n", i_task,counts[i_task]);

  }
}




template <int32_t buffer_size, typename buffer_type = uint32_t>
__global__ void bitvector_to_position_list_shellpair(
  size_t ntasks,
  size_t nsp,
  size_t LD_bit,
  const uint32_t* collisions,
  const uint32_t* counts,
  uint32_t*       position_list
) {

  constexpr auto warp_size = cuda::warp_size;
  constexpr auto element_size = CHAR_BIT * sizeof(buffer_type);
  constexpr auto buffer_size_bits = element_size * buffer_size;
  __shared__ buffer_type collisions_buffer[warp_size][warp_size][buffer_size];

  // We are converting a large number of small bitvectors into position lists. For this reason, I am assigning a single thread to each bitvector
  // This avoids having to do popcounts and warp wide reductions, but hurts the memory access pattern

  // All threads in a warp must be active to do shared memory loads, so we seperate out the threadId.x
  for (int i_base = threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y; i_base < ntasks; i_base += blockDim.x * blockDim.y * gridDim.x) {
    const int i = i_base + threadIdx.x;
    auto* out = position_list;
    if (i != 0 && i < ntasks) {
      out += counts[i-1];
    } 

    int current = 0;
    size_t nsp_blocks = (nsp + buffer_size_bits - 1) / buffer_size_bits;
    for (int j_block = 0; j_block < nsp_blocks; j_block++) {
      // Each thread has a buffer of length BUFFER_SIZE. All the threads in the warp work to 
      // load this data in a coalesced way (at least as much as possible)
      for (int buffer_loop = 0; buffer_loop < warp_size; buffer_loop += warp_size/buffer_size) {
        const int t_id_x        = threadIdx.x % buffer_size;
        const int buffer_thread = threadIdx.x / buffer_size;
        const int buffer_idx    = buffer_thread + buffer_loop;
        if (j_block * buffer_size_bits + t_id_x * element_size < nsp && i_base + buffer_idx < ntasks) {
          collisions_buffer[threadIdx.y][buffer_idx][t_id_x] = collisions[(i_base + buffer_idx) * LD_bit + j_block * buffer_size + t_id_x];
        }
      }

      __syncwarp();
      if (i < ntasks) {  // Once the data has been loaded, we exclude the threads not corresponding to a bitvector
        // We have loaded in BUFFER_SIZE_BITS elements to be processed by each warp
        for (int j_inner = 0; j_inner < buffer_size_bits && j_block * buffer_size_bits + j_inner < nsp; j_inner++) {
          const int j = buffer_size_bits * j_block + j_inner;
          const int j_int = j_inner / element_size;
          const int j_bit = j_inner % element_size;
          if( collisions_buffer[threadIdx.y][threadIdx.x][j_int] & (1 << (j_bit)) ) {
            out[current++] = j;
          }
        }
      }
      __syncwarp();
    }
  }

}





template <int32_t buffer_size, typename buffer_type = uint32_t>
__global__ void bitvector_to_position_list_shells( 
           size_t  ntasks, 
           size_t  nshells, 
           size_t  LD_bit,
    const uint32_t* collisions, 
    const uint32_t* counts, 
    const int32_t* shell_size,
          uint32_t* position_list, 
           size_t* nbe_list
) {
  constexpr auto warp_size = cuda::warp_size;
  constexpr auto element_size = CHAR_BIT * sizeof(buffer_type);
  constexpr auto buffer_size_bits = element_size * buffer_size;
  __shared__ buffer_type collisions_buffer[warp_size][warp_size][buffer_size];

  // We are converting a large number of small bitvectors into position lists. For this reason, I am assigning a single thread to each bitvector
  // This avoids having to do popcounts and warp wide reductions, but hurts the memory access pattern

  // All threads in a warp must be active to do shared memory loads, so we seperate out the threadId.x
  for (int i_base = threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y; i_base < ntasks; i_base += blockDim.x * blockDim.y * gridDim.x) {
    const int i = i_base + threadIdx.x;
    auto* out = position_list;
    if (i != 0 && i < ntasks) {
      out += counts[i-1];
    } 

    int current = 0;
    size_t nbe = 0;
    size_t nsphere_blocks = (nshells + buffer_size_bits - 1) / buffer_size_bits;
    for (int j_block = 0; j_block < nsphere_blocks; j_block++) {
      // Each thread has a buffer of length BUFFER_SIZE. All the threads in the warp work to 
      // load this data in a coalesced way (at least as much as possible)
      for (int buffer_loop = 0; buffer_loop < warp_size; buffer_loop += warp_size/buffer_size) {
        const int t_id_x        = threadIdx.x % buffer_size;
        const int buffer_thread = threadIdx.x / buffer_size;
        const int buffer_idx    = buffer_thread + buffer_loop;
        if (j_block * buffer_size_bits + t_id_x * element_size < nshells && i_base + buffer_idx < ntasks) {
          collisions_buffer[threadIdx.y][buffer_idx][t_id_x] = collisions[(i_base + buffer_idx) * LD_bit + j_block * buffer_size + t_id_x];
        }
      }

      __syncwarp();
      if (i < ntasks) {  // Once the data has been loaded, we exclude the threads not corresponding to a bitvector
        // We have loaded in BUFFER_SIZE_BITS elements to be processed by each warp
        for (int j_inner = 0; j_inner < buffer_size_bits && j_block * buffer_size_bits + j_inner < nshells; j_inner++) {
          const int j = buffer_size_bits * j_block + j_inner;
          const int j_int = j_inner / element_size;
          const int j_bit = j_inner % element_size;
          if( collisions_buffer[threadIdx.y][threadIdx.x][j_int] & (1 << (j_bit)) ) {
            out[current++] = j;
            nbe += shell_size[j];
          }
        }
      }
      __syncwarp();
    }
    if (i < ntasks) {
      nbe_list[i] = nbe;
    }
  }
}






void exx_ek_screening_bfn_stats( size_t        ntasks,
                                 XCDeviceTask* tasks_device,
                                 double      * max_bfn_sum_device,
                                 double      * bfn_max_device,
                                 size_t        LDBFM,
                                 device_queue queue ) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>() ;
  dim3 threads = 1024;//cuda::max_threads_per_thread_block;
  dim3 blocks  = ntasks;
  exx_ek_screening_bfn_stats_kernel<<<blocks, threads, 0, stream >>>(
    ntasks, max_bfn_sum_device, bfn_max_device, LDBFM, tasks_device );

}


void exx_ek_collapse_fmax_to_shells(
  int                  ntask,
  int                  nshells,
  const Shell<double>* shells_device,
  const int32_t*       shell_to_bf,
  const double*        fmax_bfn_device,
  size_t               LDF_bfn,
  double*              fmax_shell_device,
  size_t               LDF_shell,
  device_queue         queue
) {


  cudaStream_t stream = queue.queue_as<util::cuda_stream>() ;
  dim3 threads = 1024;//cuda::max_threads_per_thread_block;
  dim3 blocks  = std::max(ntask / cuda::warp_size,1u);
  exx_ek_collapse_fmax_to_shells_kernel<<<blocks, threads, 0, stream >>>(
    ntask, nshells, shells_device, shell_to_bf, fmax_bfn_device, LDF_bfn,
    fmax_shell_device, LDF_shell );

}

void exx_ek_shellpair_collision(
  int32_t       ntasks,
  int32_t       nshells,
  int32_t       nbf,
  const double* abs_dmat_device,
  size_t        LDP,
  const double* V_max_sparse_device,
  const size_t* sp_row_ind_device,
  const size_t* sp_col_ind_device,
  const double* max_bf_sum_device,
  const double* bfn_max_device,
  size_t        LDBM,
  const Shell<double>* shells_device,
  const int32_t* shell_to_bf_device,
  const int32_t* shell_sizes_device,
  double        eps_E,
  double        eps_K,
  void*         dyn_stack,
  size_t        dyn_size,
  host_task_iterator tb,
  host_task_iterator te,
  const ShellPairCollection<double>& shpairs,
  device_queue  queue,
  device_blas_handle handle
) {

  using hrt_t = std::chrono::high_resolution_clock;
  using dur_t = std::chrono::duration<double,std::milli>;

  cudaStream_t stream = queue.queue_as<util::cuda_stream>();
  std::vector<uint32_t> counts_host    (ntasks);
  std::vector<uint32_t> rc_counts_host (ntasks);

  const size_t nshell_pairs = shpairs.npairs();
  const size_t LD_coll   = util::div_ceil(nshell_pairs, 32);
  const size_t LD_rc     = util::div_ceil(nshells     , 32);

  // We need 1 bit per shell 
  // This is the number of shells divided by 8
  const int requiredSharedMemoryInBytes = LD_rc * sizeof(uint32_t);

  // By default the maximum amount of shared memory per block is 48KiB, but
  // newer archs can go higher with an opt-in setting
  int dev_id = 0;
  int maxSharedMemoryPerBlock, maxSharedMemoryPerBlockOptin;
  cudaDeviceGetAttribute(&maxSharedMemoryPerBlock,
    cudaDevAttrMaxSharedMemoryPerBlock, dev_id);

  cudaDeviceGetAttribute(&maxSharedMemoryPerBlockOptin,
    cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id);

  if (requiredSharedMemoryInBytes > maxSharedMemoryPerBlock) {
    cudaError_t res = cudaFuncSetAttribute(&exx_ek_shellpair_collision_shared_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemoryPerBlockOptin);

    if (requiredSharedMemoryInBytes > maxSharedMemoryPerBlockOptin) {
      throw cuda_exception(__FILE__, __LINE__, "Number of shell pairs exceeds device shared memory", res);
    }
  }

  buffer_adaptor full_stack(dyn_stack, dyn_size);

  auto collisions    = full_stack.aligned_alloc<uint32_t>(ntasks * LD_coll);
  auto counts        = full_stack.aligned_alloc<uint32_t>(ntasks);
  auto rc_collisions = full_stack.aligned_alloc<uint32_t>(ntasks * LD_rc);
  auto rc_counts     = full_stack.aligned_alloc<uint32_t>(ntasks);

  auto sp_check_st = hrt_t::now();
  util::cuda_set_zero_async( ntasks * LD_coll,collisions.ptr,    stream, "Zero Coll");
  util::cuda_set_zero_async( ntasks * LD_rc,  rc_collisions.ptr, stream, "Zero RC");
  util::cuda_set_zero_async( ntasks,  counts.ptr, stream, "Zero counts");
  util::cuda_set_zero_async( ntasks,  rc_counts.ptr, stream, "Zero rc counts");

  // Compute approximate FMAX and screen
  {
    buffer_adaptor sub_stack( full_stack.stack(), full_stack.nleft() );
    double* fmax_shl_device = nullptr;
    double* fmax_bfn_device = nullptr;
    fmax_bfn_device = sub_stack.aligned_alloc<double>(ntasks * nbf);
    fmax_shl_device = sub_stack.aligned_alloc<double>(ntasks * nshells);
    
    gemm(handle, DeviceBlasOp::NoTrans, DeviceBlasOp::NoTrans,
      ntasks, nbf, nbf,
      1.0, bfn_max_device,  LDBM, abs_dmat_device, nbf,
      0.0, fmax_bfn_device, ntasks
    );

    exx_ek_collapse_fmax_to_shells( ntasks, nshells, shells_device,
      shell_to_bf_device, fmax_bfn_device, ntasks, fmax_shl_device,
      ntasks, queue );

    //#if 1
    //{
    //std::vector<double> fmax_host(ntasks * nshells);
    //util::cuda_copy(ntasks * nshells,fmax_host.data(), fmax_shl_device);
    //std::ofstream ofile("gpu_fmax." + std::to_string(world_rank) + ".txt");
    //for(auto i = 0; i < ntasks; ++i) 
    //for(auto j = 0; j < nshells; ++j) {
    //  ofile << i << " " << fmax_host[i + j*ntasks] << std::endl;
    //}
    //}
    //#else
    //{
    //std::vector<double> fmax_host(ntasks * nbf);
    //util::cuda_copy(ntasks * nbf,fmax_host.data(), fmax_bfn_device);
    //std::ofstream ofile("gpu_fmax." + std::to_string(world_rank) + ".txt");
    //for(auto i = 0; i < ntasks; ++i) 
    //for(auto j = 0; j < nbf; ++j) {
    //  ofile << i << " " << fmax_host[i + j*ntasks] << std::endl;
    //}
    //}
    //#endif
    
    dim3 threads = dim3(512, 1);//cuda::max_threads_per_thread_block;
    dim3 blocks  = GauXC::util::div_ceil(ntasks,threads.y);
    exx_ek_shellpair_collision_shared_kernel<<<blocks, threads,
      requiredSharedMemoryInBytes, stream>>>(
      ntasks, nshell_pairs, nshells, LD_rc, V_max_sparse_device, sp_row_ind_device,
      sp_col_ind_device, fmax_shl_device, ntasks, 
      max_bf_sum_device, eps_E, eps_K, collisions, LD_coll, 
      rc_collisions, LD_rc, counts, rc_counts);
  }
  auto sp_check_en = hrt_t::now();
  //util::cuda_copy(ntasks, counts_host.data(), counts.ptr);
  //util::cuda_copy(ntasks, rc_counts_host.data(), rc_counts.ptr);
  //{
  //std::vector<double> max_bfn_host(ntasks);
  //util::cuda_copy(ntasks,max_bfn_host.data(), max_bf_sum_device);
  //std::ofstream ofile("gpu_max_bfn." + std::to_string(world_rank) + ".txt");
  //for(auto i = 0; i < ntasks; ++i) {
  //  ofile << i << " " << max_bfn_host[i] << std::endl;
  //}
  //}
  //{
  //std::ofstream ofile("gpu_counts." + std::to_string(world_rank) + ".txt");
  //for(auto i = 0; i < ntasks; ++i) {
  //  ofile << i << " " << counts_host[i] << std::endl;
  //}
  //}
  //{
  //std::ofstream ofile("gpu_rc_counts." + std::to_string(world_rank) + ".txt");
  //for(auto i = 0; i < ntasks; ++i) {
  //  ofile << i << " " << rc_counts_host[i] << std::endl;
  //}
  //}

  dur_t sp_check_dur = sp_check_en - sp_check_st;

  cudaError_t stat;

  size_t prefix_sum_bytes = 0;
  stat = cub::DeviceScan::InclusiveSum( NULL, prefix_sum_bytes,
    counts.ptr, counts.ptr, ntasks, stream );


  void* prefix_sum_storage = full_stack.aligned_alloc<char>(prefix_sum_bytes, 16);
  
  auto scan_st = hrt_t::now();

  // Get inclusive sums
  stat = cub::DeviceScan::InclusiveSum( prefix_sum_storage, prefix_sum_bytes,
    counts.ptr, counts.ptr, ntasks, stream );
  stat = cub::DeviceScan::InclusiveSum( prefix_sum_storage, prefix_sum_bytes,
    rc_counts.ptr, rc_counts.ptr, ntasks, stream );

  // Get counts after prefix sum
  util::cuda_copy(ntasks, counts_host.data(), counts.ptr);
  util::cuda_copy(ntasks, rc_counts_host.data(), rc_counts.ptr);
  auto scan_en = hrt_t::now();
  dur_t scan_dur = scan_en - scan_st;

  uint32_t total_sp_count = counts_host[ntasks-1];
  uint32_t total_s_count = rc_counts_host[ntasks-1];

  //size_t global_sp_count = total_sp_count;
  //MPI_Allreduce(MPI_IN_PLACE, &global_sp_count, 1, MPI_UINT64_T, MPI_SUM,
  //  MPI_COMM_WORLD);
  //if(!world_rank) {
  //  printf("*****TOTAL_SP %lu\n", global_sp_count);
  //}

  auto bv_st = hrt_t::now();

  auto position_sp_list_device = full_stack.aligned_alloc<uint32_t>(total_sp_count);
  auto position_s_list_device  = full_stack.aligned_alloc<uint32_t>(total_s_count);
  auto nbe_list                = full_stack.aligned_alloc<size_t>(ntasks);
  {
  dim3 threads(32,32);
  dim3 blocks( util::div_ceil(ntasks, 1024) );
  bitvector_to_position_list_shellpair<8><<<blocks, threads, 0, stream>>>(
    ntasks, nshell_pairs, LD_coll, collisions, counts, position_sp_list_device
  );
  bitvector_to_position_list_shells<8><<<blocks, threads, 0, stream>>>(
    ntasks, nshells, LD_rc, rc_collisions.ptr, rc_counts.ptr, shell_sizes_device,
    position_s_list_device.ptr, nbe_list.ptr
  );
  }

  std::vector<uint32_t> position_sp_list(total_sp_count);
  util::cuda_copy(total_sp_count, position_sp_list.data(), position_sp_list_device.ptr, "Position List ShellPair");

  auto bv_en = hrt_t::now();
  dur_t bv_dur = bv_en - bv_st;


  auto d2h_st = hrt_t::now();
  std::vector<uint32_t> position_s_list(total_s_count);
  std::vector<size_t> nbe_list_host(ntasks);
  util::cuda_copy(total_s_count, position_s_list.data(), position_s_list_device.ptr, "Position List Shell");
  util::cuda_copy(ntasks, nbe_list_host.data(), nbe_list.ptr, "NBE List");
  auto d2h_en = hrt_t::now();
  dur_t d2h_dur = d2h_en - d2h_st;


  auto gen_trip_st = hrt_t::now();
  const auto& shpair_row_ptr = shpairs.row_ptr();
  const auto& shpair_col_ind = shpairs.col_ind();
  std::vector<size_t> shpair_row_ind(nshell_pairs);
  for( auto i = 0; i < nshells; ++i ) {
    const auto j_st = shpair_row_ptr[i];
    const auto j_en = shpair_row_ptr[i+1];
    for( auto _j = j_st; _j < j_en; ++_j ) {
      shpair_row_ind[_j] = i;
    }
  }
  auto gen_trip_en = hrt_t::now();
  dur_t gen_trip_dur = gen_trip_en - gen_trip_st;

  auto finalize_st = hrt_t::now();
  for( auto it = tb; it != te; ++it ) {
    {
    size_t begin = (it == tb) ? 0 : counts_host[std::distance(tb,it)-1];
    size_t end   = counts_host[std::distance(tb,it)];

    it->cou_screening.shell_pair_list.resize(end - begin);
    it->cou_screening.shell_pair_idx_list.resize(end - begin);
    for( auto ij = begin, idx = 0ul; ij < end; ++ij, ++idx) {
      const auto global_ij =  position_sp_list[ij];
      it->cou_screening.shell_pair_idx_list[idx] = global_ij;
      it->cou_screening.shell_pair_list[idx] = std::make_pair(
        shpair_row_ind[global_ij], shpair_col_ind[global_ij]
      );
    }
    }

    {
    size_t begin = (it == tb) ? 0 : rc_counts_host[std::distance(tb,it)-1];
    size_t end   = rc_counts_host[std::distance(tb,it)];

    it->cou_screening.shell_list.resize(end - begin);
    it->cou_screening.nbe = nbe_list_host[std::distance(tb,it)];
    for( auto ij = begin, idx = 0ul; ij < end; ++ij, ++idx) {
      it->cou_screening.shell_list[idx] = position_s_list[ij]; 
    }
    }
    
  }

  auto finalize_en = hrt_t::now();
  dur_t finalize_dur = finalize_en - finalize_st;
  

  //printf("SPC = %.3f SCAN = %.3f BV = %.3f D2H = %.3f GT = %.3f FIN = %.3f\n", 
  //  sp_check_dur.count(), scan_dur.count(), bv_dur.count(),
  //  d2h_dur.count(), gen_trip_dur.count(), finalize_dur.count());
  
}

}
