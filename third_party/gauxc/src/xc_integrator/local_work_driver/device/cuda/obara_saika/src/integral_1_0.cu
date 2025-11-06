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
#include "integral_1_0.hu"

#include "task_map_base.hu"

#include "device_specific/cuda_device_constants.hpp"
#include "../../cuda_aos_scheme1.hpp"

namespace XGPU {

using namespace GauXC;

  __inline__ __device__ void dev_integral_1_0_driver(size_t npts,
				   double *_points_x,
				   double *_points_y,
				   double *_points_z,
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
    //__shared__ double temp[128 * 3];
    double temp_0, temp_1, temp_2;


    const int npts_int = (int) npts;
    
    for(int p_outer = blockIdx.x * blockDim.x; p_outer < npts_int; p_outer += gridDim.x * blockDim.x) {
      double *_point_outer_x = (_points_x + p_outer);
      double *_point_outer_y = (_points_y + p_outer);
      double *_point_outer_z = (_points_z + p_outer);

      int p_inner = threadIdx.x;
      if (threadIdx.x < npts_int - p_outer) {

      //for(int i = 0; i < 3; ++i) SCALAR_STORE((temp + i * blockDim.x + threadIdx.x), SCALAR_ZERO());
      temp_0 = SCALAR_ZERO();
      temp_1 = SCALAR_ZERO();
      temp_2 = SCALAR_ZERO();

      for(int ij = 0; ij < nprim_pairs; ++ij) {
	double RHO = prim_pairs[ij].gamma;
	double X_PA = prim_pairs[ij].PA.x;
	double Y_PA = prim_pairs[ij].PA.y;
	double Z_PA = prim_pairs[ij].PA.z;

	double xP = prim_pairs[ij].P.x;
	double yP = prim_pairs[ij].P.y;
	double zP = prim_pairs[ij].P.z;

	double eval = prim_pairs[ij].K_coeff_prod;

	// Evaluate T Values
	SCALAR_TYPE xC = SCALAR_LOAD((_point_outer_x + p_inner));
	SCALAR_TYPE yC = SCALAR_LOAD((_point_outer_y + p_inner));
	SCALAR_TYPE zC = SCALAR_LOAD((_point_outer_z + p_inner));

	SCALAR_TYPE X_PC = SCALAR_SUB(xP, xC);
	SCALAR_TYPE Y_PC = SCALAR_SUB(yP, yC);
	SCALAR_TYPE Z_PC = SCALAR_SUB(zP, zC);

	SCALAR_TYPE TVAL = SCALAR_MUL(X_PC, X_PC);
	TVAL = SCALAR_FMA(Y_PC, Y_PC, TVAL);
	TVAL = SCALAR_FMA(Z_PC, Z_PC, TVAL);
	TVAL = SCALAR_MUL(RHO, TVAL);
	
	SCALAR_TYPE t00, t01, TVAL_inv_e;

	// Evaluate Boys function
	boys_element<1>(&TVAL, &TVAL_inv_e, &t01, boys_table);

	// Evaluate VRR Buffer
	SCALAR_TYPE t10, tx;

	t00 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(TVAL, t01), TVAL_inv_e), SCALAR_SET1(2.00000000000000000000));

	t00 = SCALAR_MUL(eval, t00);
	t01 = SCALAR_MUL(eval, t01);
	t10 = SCALAR_MUL(X_PA, t00);
	t10 = SCALAR_FNMA(X_PC, t01, t10);
	//tx = SCALAR_LOAD((temp + 0 * blockDim.x + threadIdx.x));
	tx = temp_0;
	tx = SCALAR_ADD(tx, t10);
	//SCALAR_STORE((temp + 0 * blockDim.x + threadIdx.x), tx);
  temp_0 = tx;
	t10 = SCALAR_MUL(Y_PA, t00);
	t10 = SCALAR_FNMA(Y_PC, t01, t10);
	//tx = SCALAR_LOAD((temp + 1 * blockDim.x + threadIdx.x));
	tx = temp_1;
	tx = SCALAR_ADD(tx, t10);
	//SCALAR_STORE((temp + 1 * blockDim.x + threadIdx.x), tx);
  temp_1 = tx;
	t10 = SCALAR_MUL(Z_PA, t00);
	t10 = SCALAR_FNMA(Z_PC, t01, t10);
	//tx = SCALAR_LOAD((temp + 2 * blockDim.x + threadIdx.x));
	tx = temp_2;
	tx = SCALAR_ADD(tx, t10);
	//SCALAR_STORE((temp + 2 * blockDim.x + threadIdx.x), tx);
  temp_2 = tx;
      }

  
      if (abs(temp_0) > 1e-12 || abs(temp_1) > 1e-12 || abs(temp_2) > 1e-12) {
	double *Xik = (Xi + p_outer + p_inner);
	double *Xjk = (Xj + p_outer + p_inner);
	double *Gik = (Gi + p_outer + p_inner);
	double *Gjk = (Gj + p_outer + p_inner);

	SCALAR_TYPE const_value_v = SCALAR_LOAD((weights + p_outer + p_inner));

	double const_value, X_ABp, Y_ABp, Z_ABp, comb_m_i, comb_n_j, comb_p_k;
	SCALAR_TYPE const_value_w;
	SCALAR_TYPE tx, ty, tz, tw, t0, t1, t2;

	X_ABp = 1.0; comb_m_i = 1.0;
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);

  #if 0
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	//t0 = SCALAR_LOAD((temp + 0 * blockDim.x + threadIdx.x));
  t0 = temp_0;
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	SCALAR_STORE((Gik + 0 * ldG), tz);
	SCALAR_STORE((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	//t1 = SCALAR_LOAD((temp + 1 * blockDim.x + threadIdx.x));
  t1 = temp_1;
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	SCALAR_STORE((Gik + 1 * ldG), tz);
	SCALAR_STORE((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	//t2 = SCALAR_LOAD((temp + 2 * blockDim.x + threadIdx.x));
  t2 = temp_2;
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	SCALAR_STORE((Gik + 2 * ldG), tz);
	SCALAR_STORE((Gjk + 0 * ldG), tw);
  #else
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));

	t0 = SCALAR_MUL(temp_0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_MUL(tx, t0);
	atomicAdd((Gik + 0 * ldG), tz);
                                   
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_MUL(temp_1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);

	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_MUL(temp_2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);

	atomicAdd((Gjk + 0 * ldG), tw);
  #endif
      }
      }
    }
  }

  __global__ void dev_integral_1_0(size_t npts,
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
    dev_integral_1_0_driver( npts, points_x, points_y, points_z, nprim_pairs, prim_pairs, Xi, Xj, ldX,
      Gi, Gj, ldG, weights, boys_table );
  }

    void integral_1_0(size_t npts,
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
      dev_integral_1_0<<<320, 128, 0, stream>>>(npts,
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




  template <bool swap>
  __inline__ __device__ void dev_integral_1_0_batched_driver(
           const GauXC::ShellPairToTaskDevice* sp2task,
           GauXC::XCDeviceTask*                device_tasks,
				   double *boys_table) {

    //if (sp2task->shell_pair_device->nprim_pairs() == 0) return;
    const int ntask = sp2task->ntask;
    #pragma unroll 1
    for( int i_task = blockIdx.y; i_task < ntask; i_task += gridDim.y ) {
    
      const auto iT = sp2task->task_idx_device[i_task];
      const auto* task  = device_tasks + iT;
      const auto  npts  = task->npts;

      int i_off, j_off;
      if constexpr ( swap ) {
        j_off = sp2task->task_shell_off_row_device[i_task]*npts;
        i_off = sp2task->task_shell_off_col_device[i_task]*npts;
      } else {
        i_off = sp2task->task_shell_off_row_device[i_task]*npts;
        j_off = sp2task->task_shell_off_col_device[i_task]*npts;
      }


      dev_integral_1_0_driver( 
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

  template <bool swap>
  __global__ void dev_integral_1_0_batched(
           const GauXC::ShellPairToTaskDevice* sp2task,
           GauXC::XCDeviceTask*                device_tasks,
				   double *boys_table) {
    dev_integral_1_0_batched_driver<swap>(sp2task,device_tasks,boys_table);
  }


  void integral_1_0_batched(bool swap, size_t ntask_sp,
        const GauXC::ShellPairToTaskDevice* sp2task,
        GauXC::XCDeviceTask*                device_tasks,
		    double *boys_table,
        cudaStream_t stream) {

    int nthreads = 128;
    int nblocks_x = 160;
    int nblocks_y = ntask_sp;
    dim3 nblocks(nblocks_x, nblocks_y);

    if(swap)
      dev_integral_1_0_batched<true><<<nblocks,nthreads,0,stream>>>(
        sp2task, device_tasks, boys_table );
    else
      dev_integral_1_0_batched<false><<<nblocks,nthreads,0,stream>>>(
        sp2task, device_tasks, boys_table );

  }

  template <bool swap>
  __global__ void dev_integral_1_0_shell_batched(
           int nsp,
           const GauXC::ShellPairToTaskDevice* sp2task,
           GauXC::XCDeviceTask*                device_tasks,
				   double *boys_table) {
    for(int i = blockIdx.z; i < nsp; i+= gridDim.z ) {
      dev_integral_1_0_batched_driver<swap>(sp2task+i,device_tasks,boys_table);
    }
  }

  void integral_1_0_shell_batched(
        bool swap,
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
    if(swap)
      dev_integral_1_0_shell_batched<true><<<nblocks,nthreads,0,stream>>>(
        nsp, sp2task, device_tasks, boys_table );
    else
      dev_integral_1_0_shell_batched<false><<<nblocks,nthreads,0,stream>>>(
        nsp, sp2task, device_tasks, boys_table );

  }

template<ObaraSaikaType type_, int points_per_subtask_, int primpair_shared_limit_,
         bool pure_bra>
struct DeviceTask10 {
  static constexpr int max_primpair_shared_limit = 32;

  static constexpr int primpair_shared_limit = primpair_shared_limit_;
  static constexpr int points_per_subtask = points_per_subtask_;
  static constexpr int num_threads = points_per_subtask_;
  static constexpr ObaraSaikaType type = type_;

  static_assert(ObaraSaikaType::diag != type, "DeviceTask10 does not support diag");

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

    static constexpr bool use_shared = (primpair_shared_limit > 0);
    static constexpr int num_warps = points_per_subtask / GauXC::cuda::warp_size;
    // Cannot declare shared memory array with length 0
    static constexpr int prim_buffer_size = (use_shared) ? num_warps * primpair_shared_limit : 1;

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
      double temp_0 = SCALAR_ZERO();
      double temp_1 = SCALAR_ZERO();
      double temp_2 = SCALAR_ZERO();

      const int pointIndex = i * GauXC::cuda::warp_size + laneId;

      if (pointIndex < npts) {
        const double point_x = s_task_data[pointIndex].x;
        const double point_y = s_task_data[pointIndex].y;
        const double point_z = s_task_data[pointIndex].z;
        const double weight = s_task_data[pointIndex].w;

        for(int ij = 0; ij < nprim_pairs; ++ij) {
          const GauXC::PrimitivePair<double>* prim_pairs_use = nullptr; 
          if constexpr (use_shared) prim_pairs_use = &(s_prim_pairs[warpId * primpair_shared_limit]);
          else                      prim_pairs_use = &(prim_pairs[0]);

          double RHO = prim_pairs_use[ij].gamma;
          double X_PA = prim_pairs_use[ij].PA.x;
          double Y_PA = prim_pairs_use[ij].PA.y;
          double Z_PA = prim_pairs_use[ij].PA.z;

          double xP = prim_pairs_use[ij].P.x;
          double yP = prim_pairs_use[ij].P.y;
          double zP = prim_pairs_use[ij].P.z;

          double eval = prim_pairs_use[ij].K_coeff_prod;

          // Evaluate T Values
          SCALAR_TYPE X_PC = SCALAR_SUB(xP, point_x);
          SCALAR_TYPE Y_PC = SCALAR_SUB(yP, point_y);
          SCALAR_TYPE Z_PC = SCALAR_SUB(zP, point_z);

          SCALAR_TYPE TVAL = SCALAR_MUL(X_PC, X_PC);
          TVAL = SCALAR_FMA(Y_PC, Y_PC, TVAL);
          TVAL = SCALAR_FMA(Z_PC, Z_PC, TVAL);
          TVAL = SCALAR_MUL(RHO, TVAL);
          
          SCALAR_TYPE t00, t01, TVAL_inv_e;

          // Evaluate Boys function
          boys_element<1>(&TVAL, &TVAL_inv_e, &t01, boys_table);

          // Evaluate VRR Buffer
          SCALAR_TYPE t10, tx;

          t00 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(TVAL, t01), TVAL_inv_e), SCALAR_SET1(2.00000000000000000000));

          t00 = SCALAR_MUL(eval, t00);
          t01 = SCALAR_MUL(eval, t01);
          t10 = SCALAR_MUL(X_PA, t00);
          t10 = SCALAR_FNMA(X_PC, t01, t10);
          tx = temp_0;
          tx = SCALAR_ADD(tx, t10);
          temp_0 = tx;
          t10 = SCALAR_MUL(Y_PA, t00);
          t10 = SCALAR_FNMA(Y_PC, t01, t10);
          tx = temp_1;
          tx = SCALAR_ADD(tx, t10);
          temp_1 = tx;
          t10 = SCALAR_MUL(Z_PA, t00);
          t10 = SCALAR_FNMA(Z_PC, t01, t10);
          tx = temp_2;
          tx = SCALAR_ADD(tx, t10);
          temp_2 = tx;
        }

    
        if (abs(temp_0) > 1e-12 || abs(temp_1) > 1e-12 || abs(temp_2) > 1e-12) {
          const double * __restrict__ Xik = (Xi + pointIndex);
          const double * __restrict__ Xjk = (Xj + pointIndex);
          double * __restrict__ Gik = (Gi + pointIndex);
          double * __restrict__ Gjk = (Gj + pointIndex);

          SCALAR_TYPE const_value_v = weight;

          double const_value, X_ABp, Y_ABp, Z_ABp, comb_m_i, comb_n_j, comb_p_k;
          SCALAR_TYPE const_value_w;
          SCALAR_TYPE tx, ty, tz, tw, t0, t1, t2;

          SCALAR_TYPE Xik_0, Xik_1, Xik_2;

          if constexpr (pure_bra) {
            Xik_0 = SCALAR_LOAD((Xik + 2*ldX));
            Xik_1 = SCALAR_LOAD((Xik + 0*ldX));
            Xik_2 = SCALAR_LOAD((Xik + 1*ldX));
          } else {
            Xik_0 = SCALAR_LOAD((Xik + 0*ldX));
            Xik_1 = SCALAR_LOAD((Xik + 1*ldX));
            Xik_2 = SCALAR_LOAD((Xik + 2*ldX));
          }

          X_ABp = 1.0; comb_m_i = 1.0;
          Y_ABp = 1.0; comb_n_j = 1.0;
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);

          tx = Xik_0;
          ty = SCALAR_LOAD((Xjk + 0 * ldX));

          t0 = SCALAR_MUL(temp_0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_MUL(tx, t0);
          if constexpr (pure_bra) atomicAdd((Gik + 2 * ldG), tz);
          else                    atomicAdd((Gik + 0 * ldG), tz);
                                         
          tx = Xik_1;
          t1 = SCALAR_MUL(temp_1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          if constexpr (pure_bra) atomicAdd((Gik + 0 * ldG), tz);
          else                    atomicAdd((Gik + 1 * ldG), tz);

          tx = Xik_2;
          t2 = SCALAR_MUL(temp_2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          if constexpr (pure_bra) atomicAdd((Gik + 1 * ldG), tz);
          else                    atomicAdd((Gik + 2 * ldG), tz);

          atomicAdd((Gjk + 0 * ldG), tw);
        }
      }
    }
    __syncwarp();
  }
};

template <int primpair_limit>
using AM10_swap_cart = DeviceTask10<ObaraSaikaType::swap,
  alg_constants::CudaAoSScheme1::ObaraSaika::points_per_subtask, 
  primpair_limit, false>;

template <int primpair_limit>
using AM10_cart = DeviceTask10<ObaraSaikaType::base,
  alg_constants::CudaAoSScheme1::ObaraSaika::points_per_subtask, 
  primpair_limit, false>;

template <int primpair_limit>
using AM10_swap_sph = DeviceTask10<ObaraSaikaType::swap,
  alg_constants::CudaAoSScheme1::ObaraSaika::points_per_subtask, 
  primpair_limit, true>;

template <int primpair_limit>
using AM10_sph = DeviceTask10<ObaraSaikaType::base,
  alg_constants::CudaAoSScheme1::ObaraSaika::points_per_subtask, 
  primpair_limit, true>;

  void integral_1_0_task_batched(
    bool swap,
    bool sph,
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

    if (swap) {
      if(sph)
        dev_integral_task_map_dispatcher<AM10_swap_sph>(
          nblocks, nthreads, max_primpair, stream, 
          ntasks, nsubtask,
          device_tasks, task2sp, 
          (int4*) subtasks, nprim_pairs_device, prim_pair_ptr_device,
          sp_X_AB_device, sp_Y_AB_device, sp_Z_AB_device,
          boys_table );
      else
        dev_integral_task_map_dispatcher<AM10_swap_cart>(
          nblocks, nthreads, max_primpair, stream, 
          ntasks, nsubtask,
          device_tasks, task2sp, 
          (int4*) subtasks, nprim_pairs_device, prim_pair_ptr_device,
          sp_X_AB_device, sp_Y_AB_device, sp_Z_AB_device,
          boys_table );
    } else {
      if(sph)
        dev_integral_task_map_dispatcher<AM10_sph>(
          nblocks, nthreads, max_primpair, stream, 
          ntasks, nsubtask,
          device_tasks, task2sp, 
          (int4*) subtasks, nprim_pairs_device, prim_pair_ptr_device,
          sp_X_AB_device, sp_Y_AB_device, sp_Z_AB_device,
          boys_table );
      else
        dev_integral_task_map_dispatcher<AM10_cart>(
          nblocks, nthreads, max_primpair, stream, 
          ntasks, nsubtask,
          device_tasks, task2sp, 
          (int4*) subtasks, nprim_pairs_device, prim_pair_ptr_device,
          sp_X_AB_device, sp_Y_AB_device, sp_Z_AB_device,
          boys_table );
    }
  }
}
