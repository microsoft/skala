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
#include <math.h>
#include "../include/gpu/chebyshev_boys_computation.hpp"
#include "config_obara_saika.hpp"
#include "integral_0_0.hu"

#include "task_map_base.hu"

#include "device_specific/cuda_device_constants.hpp"
#include "../../cuda_aos_scheme1.hpp"

namespace XGPU {

using namespace GauXC;

  __inline__ __device__ void dev_integral_0_0_driver(size_t npts, 
				   const double *points_x,
				   const double *points_y,
				   const double *points_z,
                   const int nprim_pairs,
                   const GauXC::PrimitivePair<double>* prim_pairs,
				   const double *Xi,
				   const double *Xj,
				   int ldX,
				   double *Gi,
				   double *Gj,
				   int ldG, 
				   const double *weights, 
				   const double *boys_table) {

    double temp;

    //// Load PrimPairs into shared mem
    //const int nprim_pairs = sp->nprim_pairs();
    //#if 1
    //__shared__ GauXC::PrimitivePair<double> prim_pairs[GauXC::detail::nprim_pair_max];
    //__syncthreads();
    //if( threadIdx.x < 32 ) {
    //  const auto pp = sp->prim_pairs();
    //  for(int ij = threadIdx.x; ij < nprim_pairs; ij += 32) {
    //    prim_pairs[ij] = pp[ij];
    //  }
    //}
    //__syncthreads();
    //#else
    //const auto& prim_pairs = sp->prim_pairs();
    //#endif

    const int npts_int = (int) npts;

    #pragma unroll(1)
    for(int p_outer = blockIdx.x * 128; p_outer < npts_int; p_outer += gridDim.x * 128) {

      const double * __restrict__ _point_outer_x = (points_x + p_outer);
      const double * __restrict__ _point_outer_y = (points_y + p_outer);
      const double * __restrict__ _point_outer_z = (points_z + p_outer);

      int p_inner = threadIdx.x;
      if (threadIdx.x < npts_int - p_outer) {

      temp = SCALAR_ZERO();
	    const SCALAR_TYPE xC = SCALAR_LOAD((_point_outer_x + p_inner));
	    const SCALAR_TYPE yC = SCALAR_LOAD((_point_outer_y + p_inner));
	    const SCALAR_TYPE zC = SCALAR_LOAD((_point_outer_z + p_inner));

      for(int ij = 0; ij < nprim_pairs; ++ij) {
        double RHO = prim_pairs[ij].gamma;
      
        double xP = prim_pairs[ij].P.x;
        double yP = prim_pairs[ij].P.y;
        double zP = prim_pairs[ij].P.z;
      
        double eval = prim_pairs[ij].K_coeff_prod;
      
        // Evaluate T Values
        const SCALAR_TYPE X_PC = SCALAR_SUB(xP, xC);
        const SCALAR_TYPE Y_PC = SCALAR_SUB(yP, yC);
        const SCALAR_TYPE Z_PC = SCALAR_SUB(zP, zC);
      
        SCALAR_TYPE TVAL = SCALAR_MUL(X_PC, X_PC);
        TVAL = SCALAR_FMA(Y_PC, Y_PC, TVAL);
        TVAL = SCALAR_FMA(Z_PC, Z_PC, TVAL);
        TVAL = SCALAR_MUL(RHO, TVAL);
      
        // Evaluate VRR Buffer
        const SCALAR_TYPE t00 = boys_element_0(TVAL);
        temp = SCALAR_FMA( eval, t00, temp );
      }
      if (abs(temp) > 1e-12) {
        const double * __restrict__ Xik = (Xi + p_outer + p_inner);
        const double * __restrict__ Xjk = (Xj + p_outer + p_inner);
        double * __restrict__ Gik = (Gi + p_outer + p_inner);
        double * __restrict__ Gjk = (Gj + p_outer + p_inner);
      
        SCALAR_TYPE const_value_v = SCALAR_LOAD((weights + p_outer + p_inner));
      
        double const_value, X_ABp, Y_ABp, Z_ABp, comb_m_i, comb_n_j, comb_p_k;
        SCALAR_TYPE const_value_w;
        SCALAR_TYPE tx, ty, tz, tw, t0;
      
        X_ABp = 1.0; comb_m_i = 1.0;
        Y_ABp = 1.0; comb_n_j = 1.0;
        Z_ABp = 1.0; comb_p_k = 1.0;
        const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
        const_value_w = SCALAR_MUL(const_value_v, const_value);
        tx = SCALAR_LOAD(Xik);
        ty = SCALAR_LOAD(Xjk);
        t0 = SCALAR_MUL(temp, const_value_w);
        tz = SCALAR_MUL(ty, t0);
        tw = SCALAR_MUL(tx, t0);
        atomicAdd(Gik, tz);
        atomicAdd(Gjk, tw);
      }
      }
    }
  }





  __global__ void dev_integral_0_0(size_t npts,
				   double *points_x,
				   double *points_y,
				   double *points_z,
                   const int nprim_pairs,
                   const GauXC::PrimitivePair<double>* prim_pairs,
				   double *Xi,
				   double *Xj,
				   int ldX,
				   double *Gi,
				   double *Gj,
				   int ldG, 
				   double *weights, 
				   double *boys_table) {
    dev_integral_0_0_driver( npts, points_x, points_y, points_z, nprim_pairs, prim_pairs, Xi, Xj, ldX,
      Gi, Gj, ldG, weights, boys_table );
  }



  void integral_0_0(size_t npts,
		    double *points_x,
		    double *points_y,
		    double *points_z,
            const int nprim_pairs,
            const GauXC::PrimitivePair<double>* prim_pairs,
		    double *Xi,
		    double *Xj,
		    int ldX,
		    double *Gi,
		    double *Gj,
		    int ldG, 
		    double *weights, 
		    double *boys_table,
        cudaStream_t stream) {
    int nthreads = 128;
    int nblocks = std::min(uintmax_t(320), GauXC::util::div_ceil(npts,nthreads));
    dev_integral_0_0<<<nblocks, nthreads,0,stream>>>(npts,
				   points_x,
				   points_y,
				   points_z,
        nprim_pairs,prim_pairs,
				   Xi,
				   Xj,
				   ldX,
				   Gi,
				   Gj,
				   ldG, 
				   weights,
				   boys_table);
  }





  __inline__ __device__ void dev_integral_0_0_batched_driver(
           const GauXC::ShellPairToTaskDevice* sp2task,
           GauXC::XCDeviceTask*                device_tasks,
				   double *boys_table) {

    //if (sp2task->shell_pair_device->nprim_pairs() == 0) return;
    const int ntask = sp2task->ntask;

    for( int i_task = blockIdx.y; i_task < ntask; i_task += gridDim.y ) {
    
      const auto iT = sp2task->task_idx_device[i_task];
      const auto* task  = device_tasks + iT;
      const auto  npts  = task->npts;

      const auto  i_off = sp2task->task_shell_off_row_device[i_task]*npts;
      const auto  j_off = sp2task->task_shell_off_col_device[i_task]*npts;

      dev_integral_0_0_driver( 
        npts,
        task->points_x,
        task->points_y,
        task->points_z,
        sp2task->nprim_pairs,
        sp2task->prim_pairs_device,
        task->fmat + i_off,
        task->fmat + j_off,
        npts,
        task->gmat + i_off,
        task->gmat + j_off,
        npts,
        task->weights, boys_table );
    }

  }

  __global__ void dev_integral_0_0_batched(
           const GauXC::ShellPairToTaskDevice* sp2task,
           GauXC::XCDeviceTask*                device_tasks,
				   double *boys_table) {
    dev_integral_0_0_batched_driver( sp2task, device_tasks, boys_table );
  }

  void integral_0_0_batched(size_t ntask_sp,
        const GauXC::ShellPairToTaskDevice* sp2task,
        GauXC::XCDeviceTask*                device_tasks,
		    double *boys_table,
        cudaStream_t stream) {

    int nthreads = 128;
    int nblocks_x = 160;
    int nblocks_y = ntask_sp;
    dim3 nblocks(nblocks_x, nblocks_y);
    dev_integral_0_0_batched<<<nblocks,nthreads,0,stream>>>(
      sp2task, device_tasks, boys_table );

  }





  __inline__ __device__ void dev_integral_0_0_soa_batched_driver(
           int32_t                         ntask,
           const int32_t*                  sp2task_idx_device,
           const int32_t*                  sp2task_shell_off_row_device,
           const int32_t*                  sp2task_shell_off_col_device,
           const int32_t                   nprim_pairs,
           const GauXC::PrimitivePair<double>* prim_pairs_device,
           const int32_t*                  task_npts,
           const double**                  task_points_x,
           const double**                  task_points_y,
           const double**                  task_points_z,
           const double**                  task_weights,
           const double**                  task_fmat,
           double**                        task_gmat,
				   double *                        boys_table) {

    for( int i_task = blockIdx.y; i_task < ntask; i_task += gridDim.y ) {
    
      const auto iT   = sp2task_idx_device[i_task];
      const auto npts = task_npts[iT];

      const auto  i_off = sp2task_shell_off_row_device[i_task] * npts;
      const auto  j_off = sp2task_shell_off_col_device[i_task] * npts;

      dev_integral_0_0_driver( 
        npts,
        task_points_x[iT],
        task_points_y[iT],
        task_points_z[iT],
        nprim_pairs, prim_pairs_device,
        task_fmat[iT] + i_off,
        task_fmat[iT] + j_off,
        npts,
        task_gmat[iT] + i_off,
        task_gmat[iT] + j_off,
        npts,
        task_weights[iT], boys_table );
    }

  }

  __global__ void dev_integral_0_0_soa_batched(
           int32_t                         ntask,
           const int32_t*                  sp2task_idx_device,
           const int32_t*                  sp2task_shell_off_row_device,
           const int32_t*                  sp2task_shell_off_col_device,
           const int32_t                   nprim_pairs,
           const GauXC::PrimitivePair<double>* prim_pairs_device,
           const int32_t*                  task_npts,
           const double**                   task_points_x,
           const double**                   task_points_y,
           const double**                   task_points_z,
           const double**                   task_weights,
           const double**                   task_fmat,
           double**                         task_gmat,
				   double *boys_table) {
    dev_integral_0_0_soa_batched_driver( ntask, sp2task_idx_device, 
      sp2task_shell_off_row_device, sp2task_shell_off_col_device, nprim_pairs,
      prim_pairs_device,
      task_npts, task_points_x, task_points_y, task_points_z, task_weights,
      task_fmat, task_gmat, boys_table );
  }


  __global__ void 
  __launch_bounds__(128, 16)
  dev_integral_0_0_shell_batched(
           int nsp,
           const GauXC::ShellPairToTaskDevice* sp2task,
           GauXC::XCDeviceTask*                device_tasks,
				   double *boys_table) {

    for( int i = blockIdx.z; i < nsp; i += gridDim.z ) {
      dev_integral_0_0_batched_driver( sp2task + i, device_tasks, boys_table );
    }

  }

  void integral_0_0_shell_batched(
        size_t nsp,
        size_t max_ntask,
        const GauXC::ShellPairToTaskDevice* sp2task,
        GauXC::XCDeviceTask*                device_tasks,
		    double *boys_table,
        cudaStream_t stream) {

    size_t xy_max = (1ul << 16) - 1;
    int nthreads = 128;
    int nblocks_x = 1;
    int nblocks_y = std::min(max_ntask, xy_max);
    int nblocks_z = std::min(nsp,  xy_max);
    dim3 nblocks(nblocks_x, nblocks_y, nblocks_z);

    dev_integral_0_0_shell_batched<<<nblocks,nthreads,0,stream>>>(
      nsp, sp2task, device_tasks, boys_table );

  }


template<ObaraSaikaType type_, int points_per_subtask_, int primpair_shared_limit_>
struct DeviceTask00 {
  static constexpr int max_primpair_shared_limit = 32;

  static constexpr int primpair_shared_limit = primpair_shared_limit_;
  static constexpr int points_per_subtask = points_per_subtask_;
  static constexpr int num_threads = points_per_subtask_;
  static constexpr ObaraSaikaType type = type_;

  static_assert(ObaraSaikaType::swap != type, "DeviceTask00 does not support swap");
  static constexpr bool diag = (ObaraSaikaType::diag == type);

  static constexpr bool use_shared = (primpair_shared_limit > 0) && 
                                     (primpair_shared_limit <= max_primpair_shared_limit);
  static constexpr int num_warps = points_per_subtask / GauXC::cuda::warp_size;
  // Cannot declare shared memory array with length 0
  static constexpr int prim_buffer_size = (use_shared) ? num_warps * primpair_shared_limit : 1;

  using Params = ObaraSaikaBaseParams<type>;

  __inline__ __device__ static void compute( 
    const int i,
    const int npts,
    const int nprim_pairs,
    // Point data
    double4 (&s_task_data)[points_per_subtask],
    // Shell Pair Data
    const GauXC::PrimitivePair<double>* prim_pairs,
    // Output Data
    const Params param,
    int ldX,
    int ldG, 
    // Other
    double *boys_table) {

    // Unpack Params;
    const double *Xi = param.Xi;
    const double *Xj = param.Xj;
    double *Gi = param.Gi;
    double *Gj = param.Gj;

    const int laneId = threadIdx.x % GauXC::cuda::warp_size;
    const int warpId __attribute__((unused)) = threadIdx.x / GauXC::cuda::warp_size;

    __shared__ GauXC::PrimitivePair<double> s_prim_pairs[prim_buffer_size] __attribute__((unused));

    if constexpr (use_shared) {
      load_primpair_shared(laneId, warpId, nprim_pairs,
        &(prim_pairs[0]), &(s_prim_pairs[warpId * primpair_shared_limit]));
      __syncwarp();
    }

    // Loop over points in shared in batches of 32
    for (int i = 0; i <  num_warps; i++) {
      double temp = SCALAR_ZERO();

      const int pointIndex = i * GauXC::cuda::warp_size + laneId;

      if (pointIndex < npts) {

        const double point_x = s_task_data[pointIndex].x;
        const double point_y = s_task_data[pointIndex].y;
        const double point_z = s_task_data[pointIndex].z;
        const double weight = s_task_data[pointIndex].w;

        for (int ij = 0; ij < nprim_pairs; ij++) {
          const GauXC::PrimitivePair<double>* prim_pairs_use = nullptr; 
          if constexpr (use_shared) prim_pairs_use = &(s_prim_pairs[warpId * primpair_shared_limit]);
          else                      prim_pairs_use = &(prim_pairs[0]);

          double RHO = prim_pairs_use[ij].gamma;
          double xP = prim_pairs_use[ij].P.x;
          double yP = prim_pairs_use[ij].P.y;
          double zP = prim_pairs_use[ij].P.z;
          double eval = prim_pairs_use[ij].K_coeff_prod;
       
          // Evaluate T Values
          const SCALAR_TYPE X_PC = SCALAR_SUB(xP, point_x);
          const SCALAR_TYPE Y_PC = SCALAR_SUB(yP, point_y);
          const SCALAR_TYPE Z_PC = SCALAR_SUB(zP, point_z);
        
          SCALAR_TYPE TVAL = SCALAR_MUL(X_PC, X_PC);
          TVAL = SCALAR_FMA(Y_PC, Y_PC, TVAL);
          TVAL = SCALAR_FMA(Z_PC, Z_PC, TVAL);
          TVAL = SCALAR_MUL(RHO, TVAL);
        
          // Evaluate VRR Buffer
          const SCALAR_TYPE t00 = boys_element_0(TVAL);
          temp = SCALAR_FMA( eval, t00, temp );
        }

        // Output
        if (diag || abs(temp) > 1e-12) {
          const double * __restrict__ Xik = (Xi + pointIndex);
          const double * __restrict__ Xjk = (Xj + pointIndex);
          double * __restrict__ Gik = (Gi + pointIndex);
          double * __restrict__ Gjk = (Gj + pointIndex);

          SCALAR_TYPE const_value_v = weight;
        
          double const_value, X_ABp, Y_ABp, Z_ABp, comb_m_i, comb_n_j, comb_p_k;
          SCALAR_TYPE const_value_w;
          SCALAR_TYPE tx, ty, tz, tw, t0;
        
          X_ABp = 1.0; comb_m_i = 1.0;
          Y_ABp = 1.0; comb_n_j = 1.0;
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = SCALAR_LOAD(Xik);
          ty = SCALAR_LOAD(Xjk);
          t0 = SCALAR_MUL(temp, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_MUL(tx, t0);
          atomicAdd(Gik, tz);
          if constexpr (!diag) atomicAdd(Gjk, tw);
        }
      }
    }
    __syncwarp();
  }
};

template <int primpair_limit>
using AM00 = DeviceTask00<ObaraSaikaType::base,
  alg_constants::CudaAoSScheme1::ObaraSaika::points_per_subtask, primpair_limit>;

template <int primpair_limit>
using AM0 = DeviceTask00<ObaraSaikaType::diag,
  alg_constants::CudaAoSScheme1::ObaraSaika::points_per_subtask, primpair_limit>;

  void integral_0_0_task_batched(
    size_t ntasks, size_t nsubtask,
    int max_primpair, size_t max_nsp,
    GauXC::XCDeviceTask*                device_tasks,
    const GauXC::TaskToShellPairDevice* task2sp,
    const std::array<int32_t, 4>*  subtasks,
    const int32_t* nprim_pairs_device,
    GauXC::PrimitivePair<double>** prim_pair_ptr_device,
    double* sp_X_AB_device,
    double* sp_Y_AB_device,
    double* sp_Z_AB_device,
    double *boys_table,
    cudaStream_t stream) {

    int nblocks_x = nsubtask;
    int nblocks_y = 8; 
    int nblocks_z = 1;
    dim3 nblocks(nblocks_x, nblocks_y, nblocks_z);
    dim3 nthreads(alg_constants::CudaAoSScheme1::ObaraSaika::points_per_subtask);
    
    dev_integral_task_map_dispatcher<AM00>(
      nblocks, nthreads, max_primpair, stream, 
      ntasks, nsubtask,
      device_tasks, task2sp, 
      (int4*) subtasks, nprim_pairs_device, prim_pair_ptr_device,
      sp_X_AB_device, sp_Y_AB_device, sp_Z_AB_device,
      boys_table );
  }

  void integral_0_task_batched(
    size_t ntasks, size_t nsubtask,
    int max_primpair, size_t max_nsp,
    GauXC::XCDeviceTask*                device_tasks,
    const GauXC::TaskToShellPairDevice* task2sp,
    const std::array<int32_t, 4>*  subtasks,
    const int32_t* nprim_pairs_device,
    GauXC::PrimitivePair<double>** prim_pair_ptr_device,
    double* sp_X_AB_device,
    double* sp_Y_AB_device,
    double* sp_Z_AB_device,
    double *boys_table,
    cudaStream_t stream) {

    int nblocks_x = nsubtask;
    int nblocks_y = 8; 
    int nblocks_z = 1;
    dim3 nblocks(nblocks_x, nblocks_y, nblocks_z);
    dim3 nthreads(alg_constants::CudaAoSScheme1::ObaraSaika::points_per_subtask);
    
    dev_integral_task_map_dispatcher<AM0>(
      nblocks, nthreads, max_primpair, stream, 
      ntasks, nsubtask,
      device_tasks, task2sp, 
      (int4*) subtasks, nprim_pairs_device, prim_pair_ptr_device,
      sp_X_AB_device, sp_Y_AB_device, sp_Z_AB_device,
      boys_table );
  }

}
