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
#include "integral_1.hu"

namespace XGPU {
  __inline__ __device__ void dev_integral_1_driver(size_t npts,
				 double *points_x,
				 double *points_y,
				 double *points_z,
                 const int nprim_pairs,
                 const GauXC::PrimitivePair<double>* prim_pairs,
				 double *Xi,
				 int ldX,
				 double *Gi,
				 int ldG,
				 double *weights,
				 double *boys_table) {
    __shared__ double temp[128 * 9];
    
    for(size_t p_outer = blockIdx.x * blockDim.x; p_outer < npts; p_outer += gridDim.x * blockDim.x) {
      double *_point_outer_x = (points_x + p_outer);
      double *_point_outer_y = (points_y + p_outer);
      double *_point_outer_z = (points_z + p_outer);

      size_t p_inner = (threadIdx.x < (npts - p_outer)) ? threadIdx.x : (npts - p_outer);

      for(int i = 0; i < 9; ++i) SCALAR_STORE((temp + i * blockDim.x + threadIdx.x), SCALAR_ZERO());

      for(int ij = 0; ij < nprim_pairs; ++ij) {
	double RHO = prim_pairs[ij].gamma;
	double RHO_INV = prim_pairs[ij].gamma_inv;

	double xA = prim_pairs[ij].P.x;
	double yA = prim_pairs[ij].P.y;
	double zA = prim_pairs[ij].P.z;
	
	constexpr double X_PA = 0.0;
	constexpr double Y_PA = 0.0;
	constexpr double Z_PA = 0.0;

	double eval = prim_pairs[ij].K_coeff_prod;

	// Evaluate T Values
	SCALAR_TYPE xC = SCALAR_LOAD((_point_outer_x + p_inner));
	SCALAR_TYPE yC = SCALAR_LOAD((_point_outer_y + p_inner));
	SCALAR_TYPE zC = SCALAR_LOAD((_point_outer_z + p_inner));

	SCALAR_TYPE X_PC = SCALAR_SUB(xA, xC);
	SCALAR_TYPE Y_PC = SCALAR_SUB(yA, yC);
	SCALAR_TYPE Z_PC = SCALAR_SUB(zA, zC);

	SCALAR_TYPE TVAL = SCALAR_MUL(X_PC, X_PC);
	TVAL = SCALAR_FMA(Y_PC, Y_PC, TVAL);
        TVAL = SCALAR_FMA(Z_PC, Z_PC, TVAL);
	TVAL = SCALAR_MUL(RHO, TVAL);

	SCALAR_TYPE t00, t01, t02, TVAL_inv_e;

	// Evaluate Boys function
	boys_element<2>(&TVAL, &TVAL_inv_e, &t02, boys_table);

	// Evaluate VRR Buffer
	SCALAR_TYPE t10, t11, t20, tx, ty;

	t01 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(TVAL, t02), TVAL_inv_e), SCALAR_SET1(0.66666666666666662966));
	t00 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(TVAL, t01), TVAL_inv_e), SCALAR_SET1(2.00000000000000000000));
	
	t00 = SCALAR_MUL(eval, t00);
	t01 = SCALAR_MUL(eval, t01);
	t02 = SCALAR_MUL(eval, t02);
	t10 = SCALAR_MUL(X_PA, t00);
	t10 = SCALAR_FNMA(X_PC, t01, t10);
	t11 = SCALAR_MUL(X_PA, t01);
	t11 = SCALAR_FNMA(X_PC, t02, t11);
	tx = SCALAR_LOAD((temp + 0 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t10);
	SCALAR_STORE((temp + 0 * blockDim.x + threadIdx.x), tx);
	t20 = SCALAR_MUL(X_PA, t10);
	t20 = SCALAR_FNMA(X_PC, t11, t20);
	tx = SCALAR_SUB(t00, t01);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t20 = SCALAR_FMA(tx, ty, t20);
	tx = SCALAR_LOAD((temp + 3 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t20);
	SCALAR_STORE((temp + 3 * blockDim.x + threadIdx.x), tx);
	t20 = SCALAR_MUL(Y_PA, t10);
	t20 = SCALAR_FNMA(Y_PC, t11, t20);
	tx = SCALAR_LOAD((temp + 4 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t20);
	SCALAR_STORE((temp + 4 * blockDim.x + threadIdx.x), tx);
	t20 = SCALAR_MUL(Z_PA, t10);
	t20 = SCALAR_FNMA(Z_PC, t11, t20);
	tx = SCALAR_LOAD((temp + 5 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t20);
	SCALAR_STORE((temp + 5 * blockDim.x + threadIdx.x), tx);
	t10 = SCALAR_MUL(Y_PA, t00);
	t10 = SCALAR_FNMA(Y_PC, t01, t10);
	t11 = SCALAR_MUL(Y_PA, t01);
	t11 = SCALAR_FNMA(Y_PC, t02, t11);
	tx = SCALAR_LOAD((temp + 1 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t10);
	SCALAR_STORE((temp + 1 * blockDim.x + threadIdx.x), tx);
	t20 = SCALAR_MUL(Y_PA, t10);
	t20 = SCALAR_FNMA(Y_PC, t11, t20);
	tx = SCALAR_SUB(t00, t01);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t20 = SCALAR_FMA(tx, ty, t20);
	tx = SCALAR_LOAD((temp + 6 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t20);
	SCALAR_STORE((temp + 6 * blockDim.x + threadIdx.x), tx);
	t20 = SCALAR_MUL(Z_PA, t10);
	t20 = SCALAR_FNMA(Z_PC, t11, t20);
	tx = SCALAR_LOAD((temp + 7 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t20);
	SCALAR_STORE((temp + 7 * blockDim.x + threadIdx.x), tx);
	t10 = SCALAR_MUL(Z_PA, t00);
	t10 = SCALAR_FNMA(Z_PC, t01, t10);
	t11 = SCALAR_MUL(Z_PA, t01);
	t11 = SCALAR_FNMA(Z_PC, t02, t11);
	tx = SCALAR_LOAD((temp + 2 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t10);
	SCALAR_STORE((temp + 2 * blockDim.x + threadIdx.x), tx);
	t20 = SCALAR_MUL(Z_PA, t10);
	t20 = SCALAR_FNMA(Z_PC, t11, t20);
	tx = SCALAR_SUB(t00, t01);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t20 = SCALAR_FMA(tx, ty, t20);
	tx = SCALAR_LOAD((temp + 8 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t20);
	SCALAR_STORE((temp + 8 * blockDim.x + threadIdx.x), tx);
      }

      if(threadIdx.x < npts - p_outer) {
	double *Xik = (Xi + p_outer + p_inner);
	double *Gik = (Gi + p_outer + p_inner);

	SCALAR_TYPE tx, wg, xik, gik;
	tx  = SCALAR_LOAD((temp + 3 * blockDim.x + threadIdx.x));
	wg  = SCALAR_LOAD((weights + p_outer + p_inner));

	xik = SCALAR_LOAD((Xik + 0 * ldX));
	gik = SCALAR_LOAD((Gik + 0 * ldG));

	tx = SCALAR_MUL(tx, wg);
	gik = SCALAR_FMA(tx, xik, gik);
	SCALAR_STORE((Gik + 0 * ldG), gik);
	tx  = SCALAR_LOAD((temp + 4 * blockDim.x + threadIdx.x));
	wg  = SCALAR_LOAD((weights + p_outer + p_inner));

	xik = SCALAR_LOAD((Xik + 0 * ldX));
	gik = SCALAR_LOAD((Gik + 1 * ldG));

	tx = SCALAR_MUL(tx, wg);
	gik = SCALAR_FMA(tx, xik, gik);
	SCALAR_STORE((Gik + 1 * ldG), gik);
	tx  = SCALAR_LOAD((temp + 5 * blockDim.x + threadIdx.x));
	wg  = SCALAR_LOAD((weights + p_outer + p_inner));

	xik = SCALAR_LOAD((Xik + 0 * ldX));
	gik = SCALAR_LOAD((Gik + 2 * ldG));

	tx = SCALAR_MUL(tx, wg);
	gik = SCALAR_FMA(tx, xik, gik);
	SCALAR_STORE((Gik + 2 * ldG), gik);
	tx  = SCALAR_LOAD((temp + 4 * blockDim.x + threadIdx.x));
	wg  = SCALAR_LOAD((weights + p_outer + p_inner));

	xik = SCALAR_LOAD((Xik + 1 * ldX));
	gik = SCALAR_LOAD((Gik + 0 * ldG));

	tx = SCALAR_MUL(tx, wg);
	gik = SCALAR_FMA(tx, xik, gik);
	SCALAR_STORE((Gik + 0 * ldG), gik);
	tx  = SCALAR_LOAD((temp + 6 * blockDim.x + threadIdx.x));
	wg  = SCALAR_LOAD((weights + p_outer + p_inner));

	xik = SCALAR_LOAD((Xik + 1 * ldX));
	gik = SCALAR_LOAD((Gik + 1 * ldG));

	tx = SCALAR_MUL(tx, wg);
	gik = SCALAR_FMA(tx, xik, gik);
	SCALAR_STORE((Gik + 1 * ldG), gik);
	tx  = SCALAR_LOAD((temp + 7 * blockDim.x + threadIdx.x));
	wg  = SCALAR_LOAD((weights + p_outer + p_inner));

	xik = SCALAR_LOAD((Xik + 1 * ldX));
	gik = SCALAR_LOAD((Gik + 2 * ldG));

	tx = SCALAR_MUL(tx, wg);
	gik = SCALAR_FMA(tx, xik, gik);
	SCALAR_STORE((Gik + 2 * ldG), gik);
	tx  = SCALAR_LOAD((temp + 5 * blockDim.x + threadIdx.x));
	wg  = SCALAR_LOAD((weights + p_outer + p_inner));

	xik = SCALAR_LOAD((Xik + 2 * ldX));
	gik = SCALAR_LOAD((Gik + 0 * ldG));

	tx = SCALAR_MUL(tx, wg);
	gik = SCALAR_FMA(tx, xik, gik);
	SCALAR_STORE((Gik + 0 * ldG), gik);
	tx  = SCALAR_LOAD((temp + 7 * blockDim.x + threadIdx.x));
	wg  = SCALAR_LOAD((weights + p_outer + p_inner));

	xik = SCALAR_LOAD((Xik + 2 * ldX));
	gik = SCALAR_LOAD((Gik + 1 * ldG));

	tx = SCALAR_MUL(tx, wg);
	gik = SCALAR_FMA(tx, xik, gik);
	SCALAR_STORE((Gik + 1 * ldG), gik);
	tx  = SCALAR_LOAD((temp + 8 * blockDim.x + threadIdx.x));
	wg  = SCALAR_LOAD((weights + p_outer + p_inner));

	xik = SCALAR_LOAD((Xik + 2 * ldX));
	gik = SCALAR_LOAD((Gik + 2 * ldG));

	tx = SCALAR_MUL(tx, wg);
	gik = SCALAR_FMA(tx, xik, gik);
	SCALAR_STORE((Gik + 2 * ldG), gik);
      }
    }
  }

  __global__ void dev_integral_1(size_t npts,
				   double *points_x,
				   double *points_y,
				   double *points_z,
                   const int nprim_pairs,
                   const GauXC::PrimitivePair<double>* prim_pairs,
				   double *Xi,
				   int ldX,
				   double *Gi,
				   int ldG, 
				   double *weights, 
				   double *boys_table) {
    dev_integral_1_driver( npts, points_x, points_y, points_z, nprim_pairs, prim_pairs, Xi, ldX,
      Gi, ldG, weights, boys_table );
  }

  void integral_1(size_t npts,
		  double *_points_x,	
		  double *_points_y,	
		  double *_points_z,	
          const int nprim_pairs,
          const GauXC::PrimitivePair<double>* prim_pairs,
		  double *Xi,
		  int ldX,
		  double *Gi,
		  int ldG, 
		  double *weights,
		  double *boys_table,
      cudaStream_t stream) {
    dev_integral_1<<<320, 128, 0, stream>>>(npts,
				 _points_x,
				 _points_y,
				 _points_z,
         nprim_pairs, prim_pairs,
				 Xi,
				 ldX,
				 Gi,
				 ldG, 
				 weights, 
				 boys_table);
  }

  __global__ void dev_integral_1_batched(
           const GauXC::ShellPairToTaskDevice* sp2task,
           GauXC::XCDeviceTask*                device_tasks,
				   double *boys_table) {

    const int ntask = sp2task->ntask;
    for( int i_task = blockIdx.y; i_task < ntask; i_task += gridDim.y ) {
    
      const auto iT = sp2task->task_idx_device[i_task];
      const auto* task  = device_tasks + iT;
      const auto  npts  = task->npts;

      const auto  i_off = sp2task->task_shell_off_row_device[i_task]*npts;

      dev_integral_1_driver( 
        npts,
        task->points_x,
        task->points_y,
        task->points_z,
        sp2task->nprim_pairs,
        sp2task->prim_pairs_device,
        task->fmat + i_off,
        npts,
        task->gmat + i_off,
        npts,
        task->weights, boys_table );
    }

  }


  void integral_1_batched(size_t ntask_sp,
        const GauXC::ShellPairToTaskDevice* sp2task,
        GauXC::XCDeviceTask*                device_tasks,
		    double *boys_table,
        cudaStream_t stream) {

    int nthreads = 128;
    int nblocks_x = 160;
    int nblocks_y = ntask_sp;
    dim3 nblocks(nblocks_x, nblocks_y);
    dev_integral_1_batched<<<nblocks,nthreads,0,stream>>>(
      sp2task, device_tasks, boys_table );

  }
}
