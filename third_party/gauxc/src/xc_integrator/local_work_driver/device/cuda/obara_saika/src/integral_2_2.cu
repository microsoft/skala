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
#include "integral_2_2.hu"

#include "task_map_base.hu"

#include "device_specific/cuda_device_constants.hpp"
#include "../../cuda_aos_scheme1.hpp"

namespace XGPU {

using namespace GauXC;

  __inline__ __device__ void dev_integral_2_2_driver(double X_AB,
				   double Y_AB,
				   double Z_AB,
				   size_t npts,
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
    __shared__ double temp[128 * 31];

    __shared__ double outBuffer[128][6];

    for(size_t p_outer = blockIdx.x * blockDim.x; p_outer < npts; p_outer += gridDim.x * blockDim.x) {
      for (int i = 0; i < 6; i++) {
        outBuffer[threadIdx.x][i] = 0.0;
      }

      double *_point_outer_x = (_points_x + p_outer);
      double *_point_outer_y = (_points_y + p_outer);
      double *_point_outer_z = (_points_z + p_outer);

      size_t p_inner = (threadIdx.x < (npts - p_outer)) ? threadIdx.x : (npts - p_outer);

      for(int i = 0; i < 31; ++i) SCALAR_STORE((temp + i * blockDim.x + threadIdx.x), SCALAR_ZERO());

      for(int ij = 0; ij < nprim_pairs; ++ij) {
	double RHO = prim_pairs[ij].gamma;
	double RHO_INV = prim_pairs[ij].gamma_inv;
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

	SCALAR_TYPE t00, t01, t02, t03, t04, TVAL_inv_e;

	// Evaluate Boys function
	boys_element<4>(&TVAL, &TVAL_inv_e, &t04, boys_table);

	// Evaluate VRR Buffer
	SCALAR_TYPE t10, t11, t12, t13, t20, t21, t22, t30, t31, t40, tx, ty;

	t03 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(TVAL, t04), TVAL_inv_e), SCALAR_SET1(0.28571428571428569843));
	t02 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(TVAL, t03), TVAL_inv_e), SCALAR_SET1(0.40000000000000002220));
	t01 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(TVAL, t02), TVAL_inv_e), SCALAR_SET1(0.66666666666666662966));
	t00 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(TVAL, t01), TVAL_inv_e), SCALAR_SET1(2.00000000000000000000));

	t00 = SCALAR_MUL(eval, t00);
	t01 = SCALAR_MUL(eval, t01);
	t02 = SCALAR_MUL(eval, t02);
	t03 = SCALAR_MUL(eval, t03);
	t04 = SCALAR_MUL(eval, t04);
	t10 = SCALAR_MUL(X_PA, t00);
	t10 = SCALAR_FNMA(X_PC, t01, t10);
	t11 = SCALAR_MUL(X_PA, t01);
	t11 = SCALAR_FNMA(X_PC, t02, t11);
	t12 = SCALAR_MUL(X_PA, t02);
	t12 = SCALAR_FNMA(X_PC, t03, t12);
	t13 = SCALAR_MUL(X_PA, t03);
	t13 = SCALAR_FNMA(X_PC, t04, t13);
	t20 = SCALAR_MUL(X_PA, t10);
	t20 = SCALAR_FNMA(X_PC, t11, t20);
	tx = SCALAR_SUB(t00, t01);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t20 = SCALAR_FMA(tx, ty, t20);
	t21 = SCALAR_MUL(X_PA, t11);
	t21 = SCALAR_FNMA(X_PC, t12, t21);
	tx = SCALAR_SUB(t01, t02);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t21 = SCALAR_FMA(tx, ty, t21);
	t22 = SCALAR_MUL(X_PA, t12);
	t22 = SCALAR_FNMA(X_PC, t13, t22);
	tx = SCALAR_SUB(t02, t03);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t22 = SCALAR_FMA(tx, ty, t22);
	tx = SCALAR_LOAD((temp + 0 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t20);
	SCALAR_STORE((temp + 0 * blockDim.x + threadIdx.x), tx);
	t30 = SCALAR_MUL(X_PA, t20);
	t30 = SCALAR_FNMA(X_PC, t21, t30);
	tx = SCALAR_SUB(t10, t11);
	ty = SCALAR_SET1(0.5 * 2);
	ty = SCALAR_MUL(ty, RHO_INV);
	t30 = SCALAR_FMA(tx, ty, t30);
	t31 = SCALAR_MUL(X_PA, t21);
	t31 = SCALAR_FNMA(X_PC, t22, t31);
	tx = SCALAR_SUB(t11, t12);
	ty = SCALAR_SET1(0.5 * 2);
	ty = SCALAR_MUL(ty, RHO_INV);
	t31 = SCALAR_FMA(tx, ty, t31);
	tx = SCALAR_LOAD((temp + 6 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t30);
	SCALAR_STORE((temp + 6 * blockDim.x + threadIdx.x), tx);
	t40 = SCALAR_MUL(X_PA, t30);
	t40 = SCALAR_FNMA(X_PC, t31, t40);
	tx = SCALAR_SUB(t20, t21);
	ty = SCALAR_SET1(0.5 * 3);
	ty = SCALAR_MUL(ty, RHO_INV);
	t40 = SCALAR_FMA(tx, ty, t40);
	tx = SCALAR_LOAD((temp + 16 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t40);
	SCALAR_STORE((temp + 16 * blockDim.x + threadIdx.x), tx);
	t40 = SCALAR_MUL(Y_PA, t30);
	t40 = SCALAR_FNMA(Y_PC, t31, t40);
	tx = SCALAR_LOAD((temp + 17 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t40);
	SCALAR_STORE((temp + 17 * blockDim.x + threadIdx.x), tx);
	t40 = SCALAR_MUL(Z_PA, t30);
	t40 = SCALAR_FNMA(Z_PC, t31, t40);
	tx = SCALAR_LOAD((temp + 18 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t40);
	SCALAR_STORE((temp + 18 * blockDim.x + threadIdx.x), tx);
	t30 = SCALAR_MUL(Y_PA, t20);
	t30 = SCALAR_FNMA(Y_PC, t21, t30);
	t31 = SCALAR_MUL(Y_PA, t21);
	t31 = SCALAR_FNMA(Y_PC, t22, t31);
	tx = SCALAR_LOAD((temp + 7 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t30);
	SCALAR_STORE((temp + 7 * blockDim.x + threadIdx.x), tx);
	t40 = SCALAR_MUL(Y_PA, t30);
	t40 = SCALAR_FNMA(Y_PC, t31, t40);
	tx = SCALAR_SUB(t20, t21);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t40 = SCALAR_FMA(tx, ty, t40);
	tx = SCALAR_LOAD((temp + 19 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t40);
	SCALAR_STORE((temp + 19 * blockDim.x + threadIdx.x), tx);
	t40 = SCALAR_MUL(Z_PA, t30);
	t40 = SCALAR_FNMA(Z_PC, t31, t40);
	tx = SCALAR_LOAD((temp + 20 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t40);
	SCALAR_STORE((temp + 20 * blockDim.x + threadIdx.x), tx);
	t30 = SCALAR_MUL(Z_PA, t20);
	t30 = SCALAR_FNMA(Z_PC, t21, t30);
	t31 = SCALAR_MUL(Z_PA, t21);
	t31 = SCALAR_FNMA(Z_PC, t22, t31);
	tx = SCALAR_LOAD((temp + 8 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t30);
	SCALAR_STORE((temp + 8 * blockDim.x + threadIdx.x), tx);
	t40 = SCALAR_MUL(Z_PA, t30);
	t40 = SCALAR_FNMA(Z_PC, t31, t40);
	tx = SCALAR_SUB(t20, t21);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t40 = SCALAR_FMA(tx, ty, t40);
	tx = SCALAR_LOAD((temp + 21 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t40);
	SCALAR_STORE((temp + 21 * blockDim.x + threadIdx.x), tx);
	t20 = SCALAR_MUL(Y_PA, t10);
	t20 = SCALAR_FNMA(Y_PC, t11, t20);
	t21 = SCALAR_MUL(Y_PA, t11);
	t21 = SCALAR_FNMA(Y_PC, t12, t21);
	t22 = SCALAR_MUL(Y_PA, t12);
	t22 = SCALAR_FNMA(Y_PC, t13, t22);
	tx = SCALAR_LOAD((temp + 1 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t20);
	SCALAR_STORE((temp + 1 * blockDim.x + threadIdx.x), tx);
	t30 = SCALAR_MUL(Y_PA, t20);
	t30 = SCALAR_FNMA(Y_PC, t21, t30);
	tx = SCALAR_SUB(t10, t11);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t30 = SCALAR_FMA(tx, ty, t30);
	t31 = SCALAR_MUL(Y_PA, t21);
	t31 = SCALAR_FNMA(Y_PC, t22, t31);
	tx = SCALAR_SUB(t11, t12);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t31 = SCALAR_FMA(tx, ty, t31);
	tx = SCALAR_LOAD((temp + 9 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t30);
	SCALAR_STORE((temp + 9 * blockDim.x + threadIdx.x), tx);
	t40 = SCALAR_MUL(Y_PA, t30);
	t40 = SCALAR_FNMA(Y_PC, t31, t40);
	tx = SCALAR_SUB(t20, t21);
	ty = SCALAR_SET1(0.5 * 2);
	ty = SCALAR_MUL(ty, RHO_INV);
	t40 = SCALAR_FMA(tx, ty, t40);
	tx = SCALAR_LOAD((temp + 22 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t40);
	SCALAR_STORE((temp + 22 * blockDim.x + threadIdx.x), tx);
	t40 = SCALAR_MUL(Z_PA, t30);
	t40 = SCALAR_FNMA(Z_PC, t31, t40);
	tx = SCALAR_LOAD((temp + 23 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t40);
	SCALAR_STORE((temp + 23 * blockDim.x + threadIdx.x), tx);
	t30 = SCALAR_MUL(Z_PA, t20);
	t30 = SCALAR_FNMA(Z_PC, t21, t30);
	t31 = SCALAR_MUL(Z_PA, t21);
	t31 = SCALAR_FNMA(Z_PC, t22, t31);
	tx = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t30);
	SCALAR_STORE((temp + 10 * blockDim.x + threadIdx.x), tx);
	t40 = SCALAR_MUL(Z_PA, t30);
	t40 = SCALAR_FNMA(Z_PC, t31, t40);
	tx = SCALAR_SUB(t20, t21);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t40 = SCALAR_FMA(tx, ty, t40);
	tx = SCALAR_LOAD((temp + 24 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t40);
	SCALAR_STORE((temp + 24 * blockDim.x + threadIdx.x), tx);
	t20 = SCALAR_MUL(Z_PA, t10);
	t20 = SCALAR_FNMA(Z_PC, t11, t20);
	t21 = SCALAR_MUL(Z_PA, t11);
	t21 = SCALAR_FNMA(Z_PC, t12, t21);
	t22 = SCALAR_MUL(Z_PA, t12);
	t22 = SCALAR_FNMA(Z_PC, t13, t22);
	tx = SCALAR_LOAD((temp + 2 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t20);
	SCALAR_STORE((temp + 2 * blockDim.x + threadIdx.x), tx);
	t30 = SCALAR_MUL(Z_PA, t20);
	t30 = SCALAR_FNMA(Z_PC, t21, t30);
	tx = SCALAR_SUB(t10, t11);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t30 = SCALAR_FMA(tx, ty, t30);
	t31 = SCALAR_MUL(Z_PA, t21);
	t31 = SCALAR_FNMA(Z_PC, t22, t31);
	tx = SCALAR_SUB(t11, t12);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t31 = SCALAR_FMA(tx, ty, t31);
	tx = SCALAR_LOAD((temp + 11 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t30);
	SCALAR_STORE((temp + 11 * blockDim.x + threadIdx.x), tx);
	t40 = SCALAR_MUL(Z_PA, t30);
	t40 = SCALAR_FNMA(Z_PC, t31, t40);
	tx = SCALAR_SUB(t20, t21);
	ty = SCALAR_SET1(0.5 * 2);
	ty = SCALAR_MUL(ty, RHO_INV);
	t40 = SCALAR_FMA(tx, ty, t40);
	tx = SCALAR_LOAD((temp + 25 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t40);
	SCALAR_STORE((temp + 25 * blockDim.x + threadIdx.x), tx);
	t10 = SCALAR_MUL(Y_PA, t00);
	t10 = SCALAR_FNMA(Y_PC, t01, t10);
	t11 = SCALAR_MUL(Y_PA, t01);
	t11 = SCALAR_FNMA(Y_PC, t02, t11);
	t12 = SCALAR_MUL(Y_PA, t02);
	t12 = SCALAR_FNMA(Y_PC, t03, t12);
	t13 = SCALAR_MUL(Y_PA, t03);
	t13 = SCALAR_FNMA(Y_PC, t04, t13);
	t20 = SCALAR_MUL(Y_PA, t10);
	t20 = SCALAR_FNMA(Y_PC, t11, t20);
	tx = SCALAR_SUB(t00, t01);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t20 = SCALAR_FMA(tx, ty, t20);
	t21 = SCALAR_MUL(Y_PA, t11);
	t21 = SCALAR_FNMA(Y_PC, t12, t21);
	tx = SCALAR_SUB(t01, t02);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t21 = SCALAR_FMA(tx, ty, t21);
	t22 = SCALAR_MUL(Y_PA, t12);
	t22 = SCALAR_FNMA(Y_PC, t13, t22);
	tx = SCALAR_SUB(t02, t03);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t22 = SCALAR_FMA(tx, ty, t22);
	tx = SCALAR_LOAD((temp + 3 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t20);
	SCALAR_STORE((temp + 3 * blockDim.x + threadIdx.x), tx);
	t30 = SCALAR_MUL(Y_PA, t20);
	t30 = SCALAR_FNMA(Y_PC, t21, t30);
	tx = SCALAR_SUB(t10, t11);
	ty = SCALAR_SET1(0.5 * 2);
	ty = SCALAR_MUL(ty, RHO_INV);
	t30 = SCALAR_FMA(tx, ty, t30);
	t31 = SCALAR_MUL(Y_PA, t21);
	t31 = SCALAR_FNMA(Y_PC, t22, t31);
	tx = SCALAR_SUB(t11, t12);
	ty = SCALAR_SET1(0.5 * 2);
	ty = SCALAR_MUL(ty, RHO_INV);
	t31 = SCALAR_FMA(tx, ty, t31);
	tx = SCALAR_LOAD((temp + 12 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t30);
	SCALAR_STORE((temp + 12 * blockDim.x + threadIdx.x), tx);
	t40 = SCALAR_MUL(Y_PA, t30);
	t40 = SCALAR_FNMA(Y_PC, t31, t40);
	tx = SCALAR_SUB(t20, t21);
	ty = SCALAR_SET1(0.5 * 3);
	ty = SCALAR_MUL(ty, RHO_INV);
	t40 = SCALAR_FMA(tx, ty, t40);
	tx = SCALAR_LOAD((temp + 26 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t40);
	SCALAR_STORE((temp + 26 * blockDim.x + threadIdx.x), tx);
	t40 = SCALAR_MUL(Z_PA, t30);
	t40 = SCALAR_FNMA(Z_PC, t31, t40);
	tx = SCALAR_LOAD((temp + 27 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t40);
	SCALAR_STORE((temp + 27 * blockDim.x + threadIdx.x), tx);
	t30 = SCALAR_MUL(Z_PA, t20);
	t30 = SCALAR_FNMA(Z_PC, t21, t30);
	t31 = SCALAR_MUL(Z_PA, t21);
	t31 = SCALAR_FNMA(Z_PC, t22, t31);
	tx = SCALAR_LOAD((temp + 13 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t30);
	SCALAR_STORE((temp + 13 * blockDim.x + threadIdx.x), tx);
	t40 = SCALAR_MUL(Z_PA, t30);
	t40 = SCALAR_FNMA(Z_PC, t31, t40);
	tx = SCALAR_SUB(t20, t21);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t40 = SCALAR_FMA(tx, ty, t40);
	tx = SCALAR_LOAD((temp + 28 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t40);
	SCALAR_STORE((temp + 28 * blockDim.x + threadIdx.x), tx);
	t20 = SCALAR_MUL(Z_PA, t10);
	t20 = SCALAR_FNMA(Z_PC, t11, t20);
	t21 = SCALAR_MUL(Z_PA, t11);
	t21 = SCALAR_FNMA(Z_PC, t12, t21);
	t22 = SCALAR_MUL(Z_PA, t12);
	t22 = SCALAR_FNMA(Z_PC, t13, t22);
	tx = SCALAR_LOAD((temp + 4 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t20);
	SCALAR_STORE((temp + 4 * blockDim.x + threadIdx.x), tx);
	t30 = SCALAR_MUL(Z_PA, t20);
	t30 = SCALAR_FNMA(Z_PC, t21, t30);
	tx = SCALAR_SUB(t10, t11);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t30 = SCALAR_FMA(tx, ty, t30);
	t31 = SCALAR_MUL(Z_PA, t21);
	t31 = SCALAR_FNMA(Z_PC, t22, t31);
	tx = SCALAR_SUB(t11, t12);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t31 = SCALAR_FMA(tx, ty, t31);
	tx = SCALAR_LOAD((temp + 14 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t30);
	SCALAR_STORE((temp + 14 * blockDim.x + threadIdx.x), tx);
	t40 = SCALAR_MUL(Z_PA, t30);
	t40 = SCALAR_FNMA(Z_PC, t31, t40);
	tx = SCALAR_SUB(t20, t21);
	ty = SCALAR_SET1(0.5 * 2);
	ty = SCALAR_MUL(ty, RHO_INV);
	t40 = SCALAR_FMA(tx, ty, t40);
	tx = SCALAR_LOAD((temp + 29 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t40);
	SCALAR_STORE((temp + 29 * blockDim.x + threadIdx.x), tx);
	t10 = SCALAR_MUL(Z_PA, t00);
	t10 = SCALAR_FNMA(Z_PC, t01, t10);
	t11 = SCALAR_MUL(Z_PA, t01);
	t11 = SCALAR_FNMA(Z_PC, t02, t11);
	t12 = SCALAR_MUL(Z_PA, t02);
	t12 = SCALAR_FNMA(Z_PC, t03, t12);
	t13 = SCALAR_MUL(Z_PA, t03);
	t13 = SCALAR_FNMA(Z_PC, t04, t13);
	t20 = SCALAR_MUL(Z_PA, t10);
	t20 = SCALAR_FNMA(Z_PC, t11, t20);
	tx = SCALAR_SUB(t00, t01);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t20 = SCALAR_FMA(tx, ty, t20);
	t21 = SCALAR_MUL(Z_PA, t11);
	t21 = SCALAR_FNMA(Z_PC, t12, t21);
	tx = SCALAR_SUB(t01, t02);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t21 = SCALAR_FMA(tx, ty, t21);
	t22 = SCALAR_MUL(Z_PA, t12);
	t22 = SCALAR_FNMA(Z_PC, t13, t22);
	tx = SCALAR_SUB(t02, t03);
	ty = SCALAR_SET1(0.5 * 1);
	ty = SCALAR_MUL(ty, RHO_INV);
	t22 = SCALAR_FMA(tx, ty, t22);
	tx = SCALAR_LOAD((temp + 5 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t20);
	SCALAR_STORE((temp + 5 * blockDim.x + threadIdx.x), tx);
	t30 = SCALAR_MUL(Z_PA, t20);
	t30 = SCALAR_FNMA(Z_PC, t21, t30);
	tx = SCALAR_SUB(t10, t11);
	ty = SCALAR_SET1(0.5 * 2);
	ty = SCALAR_MUL(ty, RHO_INV);
	t30 = SCALAR_FMA(tx, ty, t30);
	t31 = SCALAR_MUL(Z_PA, t21);
	t31 = SCALAR_FNMA(Z_PC, t22, t31);
	tx = SCALAR_SUB(t11, t12);
	ty = SCALAR_SET1(0.5 * 2);
	ty = SCALAR_MUL(ty, RHO_INV);
	t31 = SCALAR_FMA(tx, ty, t31);
	tx = SCALAR_LOAD((temp + 15 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t30);
	SCALAR_STORE((temp + 15 * blockDim.x + threadIdx.x), tx);
	t40 = SCALAR_MUL(Z_PA, t30);
	t40 = SCALAR_FNMA(Z_PC, t31, t40);
	tx = SCALAR_SUB(t20, t21);
	ty = SCALAR_SET1(0.5 * 3);
	ty = SCALAR_MUL(ty, RHO_INV);
	t40 = SCALAR_FMA(tx, ty, t40);
	tx = SCALAR_LOAD((temp + 30 * blockDim.x + threadIdx.x));
	tx = SCALAR_ADD(tx, t40);
	SCALAR_STORE((temp + 30 * blockDim.x + threadIdx.x), tx);
      }

      if(threadIdx.x < npts - p_outer) {
	double *Xik = (Xi + p_outer + p_inner);
	double *Xjk = (Xj + p_outer + p_inner);
	double *Gik = (Gi + p_outer + p_inner);
	double *Gjk = (Gj + p_outer + p_inner);

	SCALAR_TYPE const_value_v = SCALAR_LOAD((weights + p_outer + p_inner));

	double const_value, X_ABp, Y_ABp, Z_ABp, comb_m_i, comb_n_j, comb_p_k;
	SCALAR_TYPE const_value_w;
	SCALAR_TYPE tx, ty, tz, tw, t0, t1, t2, t3, t4, t5;

  #if 0
	X_ABp = 1.0; comb_m_i = 1.0;
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t0 = SCALAR_LOAD((temp + 16 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t1 = SCALAR_LOAD((temp + 17 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t2 = SCALAR_LOAD((temp + 18 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t3 = SCALAR_LOAD((temp + 19 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t4 = SCALAR_LOAD((temp + 20 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t5 = SCALAR_LOAD((temp + 21 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	X_ABp = SCALAR_MUL(X_ABp, X_AB); comb_m_i = SCALAR_MUL(comb_m_i * 2, SCALAR_RECIPROCAL(1));
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t0 = SCALAR_LOAD((temp + 6 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t1 = SCALAR_LOAD((temp + 7 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t2 = SCALAR_LOAD((temp + 8 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t3 = SCALAR_LOAD((temp + 9 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t4 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t5 = SCALAR_LOAD((temp + 11 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	X_ABp = SCALAR_MUL(X_ABp, X_AB); comb_m_i = SCALAR_MUL(comb_m_i * 1, SCALAR_RECIPROCAL(2));
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t0 = SCALAR_LOAD((temp + 0 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t1 = SCALAR_LOAD((temp + 1 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t2 = SCALAR_LOAD((temp + 2 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t3 = SCALAR_LOAD((temp + 3 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t4 = SCALAR_LOAD((temp + 4 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 0 * ldG));
	t5 = SCALAR_LOAD((temp + 5 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 0 * ldG), tw);
	X_ABp = 1.0; comb_m_i = 1.0;
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t0 = SCALAR_LOAD((temp + 17 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t1 = SCALAR_LOAD((temp + 19 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t2 = SCALAR_LOAD((temp + 20 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t3 = SCALAR_LOAD((temp + 22 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t4 = SCALAR_LOAD((temp + 23 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t5 = SCALAR_LOAD((temp + 24 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	Y_ABp = SCALAR_MUL(Y_ABp, Y_AB); comb_n_j = SCALAR_MUL(comb_n_j * 1, SCALAR_RECIPROCAL(1));
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t0 = SCALAR_LOAD((temp + 6 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t1 = SCALAR_LOAD((temp + 7 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t2 = SCALAR_LOAD((temp + 8 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t3 = SCALAR_LOAD((temp + 9 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t4 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t5 = SCALAR_LOAD((temp + 11 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	X_ABp = SCALAR_MUL(X_ABp, X_AB); comb_m_i = SCALAR_MUL(comb_m_i * 1, SCALAR_RECIPROCAL(1));
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t0 = SCALAR_LOAD((temp + 7 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t1 = SCALAR_LOAD((temp + 9 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t2 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t3 = SCALAR_LOAD((temp + 12 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t4 = SCALAR_LOAD((temp + 13 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t5 = SCALAR_LOAD((temp + 14 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	Y_ABp = SCALAR_MUL(Y_ABp, Y_AB); comb_n_j = SCALAR_MUL(comb_n_j * 1, SCALAR_RECIPROCAL(1));
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t0 = SCALAR_LOAD((temp + 0 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t1 = SCALAR_LOAD((temp + 1 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t2 = SCALAR_LOAD((temp + 2 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t3 = SCALAR_LOAD((temp + 3 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t4 = SCALAR_LOAD((temp + 4 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 1 * ldG));
	t5 = SCALAR_LOAD((temp + 5 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 1 * ldG), tw);
	X_ABp = 1.0; comb_m_i = 1.0;
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t0 = SCALAR_LOAD((temp + 18 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t1 = SCALAR_LOAD((temp + 20 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t2 = SCALAR_LOAD((temp + 21 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t3 = SCALAR_LOAD((temp + 23 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t4 = SCALAR_LOAD((temp + 24 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t5 = SCALAR_LOAD((temp + 25 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 1, SCALAR_RECIPROCAL(1));
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t0 = SCALAR_LOAD((temp + 6 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t1 = SCALAR_LOAD((temp + 7 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t2 = SCALAR_LOAD((temp + 8 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t3 = SCALAR_LOAD((temp + 9 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t4 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t5 = SCALAR_LOAD((temp + 11 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	X_ABp = SCALAR_MUL(X_ABp, X_AB); comb_m_i = SCALAR_MUL(comb_m_i * 1, SCALAR_RECIPROCAL(1));
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t0 = SCALAR_LOAD((temp + 8 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t1 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t2 = SCALAR_LOAD((temp + 11 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t3 = SCALAR_LOAD((temp + 13 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t4 = SCALAR_LOAD((temp + 14 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t5 = SCALAR_LOAD((temp + 15 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 1, SCALAR_RECIPROCAL(1));
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t0 = SCALAR_LOAD((temp + 0 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t1 = SCALAR_LOAD((temp + 1 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t2 = SCALAR_LOAD((temp + 2 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t3 = SCALAR_LOAD((temp + 3 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t4 = SCALAR_LOAD((temp + 4 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 2 * ldG));
	t5 = SCALAR_LOAD((temp + 5 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 2 * ldG), tw);
	X_ABp = 1.0; comb_m_i = 1.0;
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t0 = SCALAR_LOAD((temp + 19 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t1 = SCALAR_LOAD((temp + 22 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t2 = SCALAR_LOAD((temp + 23 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t3 = SCALAR_LOAD((temp + 26 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t4 = SCALAR_LOAD((temp + 27 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t5 = SCALAR_LOAD((temp + 28 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	Y_ABp = SCALAR_MUL(Y_ABp, Y_AB); comb_n_j = SCALAR_MUL(comb_n_j * 2, SCALAR_RECIPROCAL(1));
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t0 = SCALAR_LOAD((temp + 7 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t1 = SCALAR_LOAD((temp + 9 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t2 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t3 = SCALAR_LOAD((temp + 12 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t4 = SCALAR_LOAD((temp + 13 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t5 = SCALAR_LOAD((temp + 14 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	Y_ABp = SCALAR_MUL(Y_ABp, Y_AB); comb_n_j = SCALAR_MUL(comb_n_j * 1, SCALAR_RECIPROCAL(2));
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t0 = SCALAR_LOAD((temp + 0 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t1 = SCALAR_LOAD((temp + 1 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t2 = SCALAR_LOAD((temp + 2 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t3 = SCALAR_LOAD((temp + 3 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t4 = SCALAR_LOAD((temp + 4 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 3 * ldG));
	t5 = SCALAR_LOAD((temp + 5 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 3 * ldG), tw);
	X_ABp = 1.0; comb_m_i = 1.0;
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t0 = SCALAR_LOAD((temp + 20 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t1 = SCALAR_LOAD((temp + 23 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t2 = SCALAR_LOAD((temp + 24 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t3 = SCALAR_LOAD((temp + 27 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t4 = SCALAR_LOAD((temp + 28 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t5 = SCALAR_LOAD((temp + 29 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 1, SCALAR_RECIPROCAL(1));
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t0 = SCALAR_LOAD((temp + 7 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t1 = SCALAR_LOAD((temp + 9 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t2 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t3 = SCALAR_LOAD((temp + 12 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t4 = SCALAR_LOAD((temp + 13 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t5 = SCALAR_LOAD((temp + 14 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	Y_ABp = SCALAR_MUL(Y_ABp, Y_AB); comb_n_j = SCALAR_MUL(comb_n_j * 1, SCALAR_RECIPROCAL(1));
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t0 = SCALAR_LOAD((temp + 8 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t1 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t2 = SCALAR_LOAD((temp + 11 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t3 = SCALAR_LOAD((temp + 13 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t4 = SCALAR_LOAD((temp + 14 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t5 = SCALAR_LOAD((temp + 15 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 1, SCALAR_RECIPROCAL(1));
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t0 = SCALAR_LOAD((temp + 0 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t1 = SCALAR_LOAD((temp + 1 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t2 = SCALAR_LOAD((temp + 2 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t3 = SCALAR_LOAD((temp + 3 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t4 = SCALAR_LOAD((temp + 4 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 4 * ldG));
	t5 = SCALAR_LOAD((temp + 5 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 4 * ldG), tw);
	X_ABp = 1.0; comb_m_i = 1.0;
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t0 = SCALAR_LOAD((temp + 21 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t1 = SCALAR_LOAD((temp + 24 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t2 = SCALAR_LOAD((temp + 25 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t3 = SCALAR_LOAD((temp + 28 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t4 = SCALAR_LOAD((temp + 29 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t5 = SCALAR_LOAD((temp + 30 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 2, SCALAR_RECIPROCAL(1));
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t0 = SCALAR_LOAD((temp + 8 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t1 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t2 = SCALAR_LOAD((temp + 11 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t3 = SCALAR_LOAD((temp + 13 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t4 = SCALAR_LOAD((temp + 14 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t5 = SCALAR_LOAD((temp + 15 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 1, SCALAR_RECIPROCAL(2));
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 0 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t0 = SCALAR_LOAD((temp + 0 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_FMA(ty, t0, tz);
	tw = SCALAR_FMA(tx, t0, tw);
	atomicAdd((Gik + 0 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 1 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t1 = SCALAR_LOAD((temp + 1 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_FMA(ty, t1, tz);
	tw = SCALAR_FMA(tx, t1, tw);
	atomicAdd((Gik + 1 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 2 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t2 = SCALAR_LOAD((temp + 2 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_FMA(ty, t2, tz);
	tw = SCALAR_FMA(tx, t2, tw);
	atomicAdd((Gik + 2 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 3 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t3 = SCALAR_LOAD((temp + 3 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_FMA(ty, t3, tz);
	tw = SCALAR_FMA(tx, t3, tw);
	atomicAdd((Gik + 3 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 4 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t4 = SCALAR_LOAD((temp + 4 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_FMA(ty, t4, tz);
	tw = SCALAR_FMA(tx, t4, tw);
	atomicAdd((Gik + 4 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	tz = SCALAR_LOAD((Gik + 5 * ldG));
	tw = SCALAR_LOAD((Gjk + 5 * ldG));
	t5 = SCALAR_LOAD((temp + 5 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_FMA(ty, t5, tz);
	tw = SCALAR_FMA(tx, t5, tw);
	atomicAdd((Gik + 5 * ldG), tz);
	atomicAdd((Gjk + 5 * ldG), tw);
  #else
	X_ABp = 1.0; comb_m_i = 1.0;
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 0 * ldX));
	t0 = SCALAR_LOAD((temp + 16 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_MUL(tx, t0);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 17 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 18 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 19 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 20 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 21 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
                                
	X_ABp = SCALAR_MUL(X_ABp, X_AB); comb_m_i = SCALAR_MUL(comb_m_i * 2, SCALAR_RECIPROCAL(1));
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	t0 = SCALAR_LOAD((temp + 6 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_FMA(tx, t0, tw);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 7 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 8 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 9 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 11 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
                                
	X_ABp = SCALAR_MUL(X_ABp, X_AB); comb_m_i = SCALAR_MUL(comb_m_i * 1, SCALAR_RECIPROCAL(2));
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	t0 = SCALAR_LOAD((temp + 0 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_FMA(tx, t0, tw);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 1 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 2 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 3 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 4 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 5 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
	atomicAdd((Gjk + 0 * ldG), tw);
  


	X_ABp = 1.0; comb_m_i = 1.0;
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 1 * ldX));
	t0 = SCALAR_LOAD((temp + 17 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_MUL(tx, t0);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 19 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 20 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 22 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 23 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 24 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
                                
	Y_ABp = SCALAR_MUL(Y_ABp, Y_AB); comb_n_j = SCALAR_MUL(comb_n_j * 1, SCALAR_RECIPROCAL(1));
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	t0 = SCALAR_LOAD((temp + 6 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_FMA(tx, t0, tw);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 7 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 8 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 9 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 11 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
                                
	X_ABp = SCALAR_MUL(X_ABp, X_AB); comb_m_i = SCALAR_MUL(comb_m_i * 1, SCALAR_RECIPROCAL(1));
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	t0 = SCALAR_LOAD((temp + 7 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_FMA(tx, t0, tw);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 9 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 12 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 13 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 14 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
                                
	Y_ABp = SCALAR_MUL(Y_ABp, Y_AB); comb_n_j = SCALAR_MUL(comb_n_j * 1, SCALAR_RECIPROCAL(1));
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	t0 = SCALAR_LOAD((temp + 0 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_FMA(tx, t0, tw);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 1 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 2 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 3 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 4 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 5 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
	atomicAdd((Gjk + 1 * ldG), tw);




	X_ABp = 1.0; comb_m_i = 1.0;
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 2 * ldX));
	t0 = SCALAR_LOAD((temp + 18 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_MUL(tx, t0);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 20 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 21 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 23 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 24 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 25 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
                                
	Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 1, SCALAR_RECIPROCAL(1));
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	t0 = SCALAR_LOAD((temp + 6 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_FMA(tx, t0, tw);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 7 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 8 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 9 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 11 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
                                
	X_ABp = SCALAR_MUL(X_ABp, X_AB); comb_m_i = SCALAR_MUL(comb_m_i * 1, SCALAR_RECIPROCAL(1));
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	t0 = SCALAR_LOAD((temp + 8 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_FMA(tx, t0, tw);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 11 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 13 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 14 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 15 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
                                
	Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 1, SCALAR_RECIPROCAL(1));
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	t0 = SCALAR_LOAD((temp + 0 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_FMA(tx, t0, tw);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 1 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 2 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 3 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 4 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 5 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
	atomicAdd((Gjk + 2 * ldG), tw);





	X_ABp = 1.0; comb_m_i = 1.0;
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 3 * ldX));
	t0 = SCALAR_LOAD((temp + 19 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_MUL(tx, t0);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 22 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 23 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 26 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 27 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 28 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
                                
	Y_ABp = SCALAR_MUL(Y_ABp, Y_AB); comb_n_j = SCALAR_MUL(comb_n_j * 2, SCALAR_RECIPROCAL(1));
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	t0 = SCALAR_LOAD((temp + 7 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_FMA(tx, t0, tw);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 9 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 12 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 13 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 14 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
                                
	Y_ABp = SCALAR_MUL(Y_ABp, Y_AB); comb_n_j = SCALAR_MUL(comb_n_j * 1, SCALAR_RECIPROCAL(2));
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	t0 = SCALAR_LOAD((temp + 0 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_FMA(tx, t0, tw);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 1 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 2 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 3 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 4 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 5 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
	atomicAdd((Gjk + 3 * ldG), tw);



	X_ABp = 1.0; comb_m_i = 1.0;
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 4 * ldX));
	t0 = SCALAR_LOAD((temp + 20 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_MUL(tx, t0);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 23 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 24 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 27 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 28 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 29 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
                                
	Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 1, SCALAR_RECIPROCAL(1));
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	t0 = SCALAR_LOAD((temp + 7 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_FMA(tx, t0, tw);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 9 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 12 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 13 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 14 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
                                
	Y_ABp = SCALAR_MUL(Y_ABp, Y_AB); comb_n_j = SCALAR_MUL(comb_n_j * 1, SCALAR_RECIPROCAL(1));
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	t0 = SCALAR_LOAD((temp + 8 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_FMA(tx, t0, tw);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 11 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 13 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 14 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 15 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
                                
	Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 1, SCALAR_RECIPROCAL(1));
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	t0 = SCALAR_LOAD((temp + 0 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_FMA(tx, t0, tw);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 1 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 2 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 3 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 4 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 5 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
	atomicAdd((Gjk + 4 * ldG), tw);




	X_ABp = 1.0; comb_m_i = 1.0;
	Y_ABp = 1.0; comb_n_j = 1.0;
	Z_ABp = 1.0; comb_p_k = 1.0;
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	ty = SCALAR_LOAD((Xjk + 5 * ldX));
	t0 = SCALAR_LOAD((temp + 21 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_MUL(tx, t0);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 24 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 25 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 28 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 29 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 30 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
                                
	Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 2, SCALAR_RECIPROCAL(1));
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	t0 = SCALAR_LOAD((temp + 8 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_FMA(tx, t0, tw);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 10 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 11 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 13 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 14 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 15 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
                                
	Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 1, SCALAR_RECIPROCAL(2));
	const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
	const_value_w = SCALAR_MUL(const_value_v, const_value);
	tx = SCALAR_LOAD((Xik + 0 * ldX));
	t0 = SCALAR_LOAD((temp + 0 * blockDim.x + threadIdx.x));
	t0 = SCALAR_MUL(t0, const_value_w);
	tz = SCALAR_MUL(ty, t0);
	tw = SCALAR_FMA(tx, t0, tw);
	//atomicAdd((Gik + 0 * ldG), tz);
	outBuffer[threadIdx.x][0] += tz;
                                
	tx = SCALAR_LOAD((Xik + 1 * ldX));
	t1 = SCALAR_LOAD((temp + 1 * blockDim.x + threadIdx.x));
	t1 = SCALAR_MUL(t1, const_value_w);
	tz = SCALAR_MUL(ty, t1);
	tw = SCALAR_FMA(tx, t1, tw);
	//atomicAdd((Gik + 1 * ldG), tz);
	outBuffer[threadIdx.x][1] += tz;
                                
	tx = SCALAR_LOAD((Xik + 2 * ldX));
	t2 = SCALAR_LOAD((temp + 2 * blockDim.x + threadIdx.x));
	t2 = SCALAR_MUL(t2, const_value_w);
	tz = SCALAR_MUL(ty, t2);
	tw = SCALAR_FMA(tx, t2, tw);
	//atomicAdd((Gik + 2 * ldG), tz);
	outBuffer[threadIdx.x][2] += tz;
                                
	tx = SCALAR_LOAD((Xik + 3 * ldX));
	t3 = SCALAR_LOAD((temp + 3 * blockDim.x + threadIdx.x));
	t3 = SCALAR_MUL(t3, const_value_w);
	tz = SCALAR_MUL(ty, t3);
	tw = SCALAR_FMA(tx, t3, tw);
	//atomicAdd((Gik + 3 * ldG), tz);
	outBuffer[threadIdx.x][3] += tz;
                                
	tx = SCALAR_LOAD((Xik + 4 * ldX));
	t4 = SCALAR_LOAD((temp + 4 * blockDim.x + threadIdx.x));
	t4 = SCALAR_MUL(t4, const_value_w);
	tz = SCALAR_MUL(ty, t4);
	tw = SCALAR_FMA(tx, t4, tw);
	//atomicAdd((Gik + 4 * ldG), tz);
	outBuffer[threadIdx.x][4] += tz;
                                
	tx = SCALAR_LOAD((Xik + 5 * ldX));
	t5 = SCALAR_LOAD((temp + 5 * blockDim.x + threadIdx.x));
	t5 = SCALAR_MUL(t5, const_value_w);
	tz = SCALAR_MUL(ty, t5);
	tw = SCALAR_FMA(tx, t5, tw);
	//atomicAdd((Gik + 5 * ldG), tz);
	outBuffer[threadIdx.x][5] += tz;
	atomicAdd((Gjk + 5 * ldG), tw);

	atomicAdd((Gik + 0 * ldG), outBuffer[threadIdx.x][0]);
	atomicAdd((Gik + 1 * ldG), outBuffer[threadIdx.x][1]);
	atomicAdd((Gik + 2 * ldG), outBuffer[threadIdx.x][2]);
	atomicAdd((Gik + 3 * ldG), outBuffer[threadIdx.x][3]);
	atomicAdd((Gik + 4 * ldG), outBuffer[threadIdx.x][4]);
	atomicAdd((Gik + 5 * ldG), outBuffer[threadIdx.x][5]);

  #endif
      }
    }
  }

  __global__ void dev_integral_2_2(
           double X_AB,
				   double Y_AB,
				   double Z_AB,
           size_t npts,
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
    dev_integral_2_2_driver( X_AB, Y_AB, Z_AB, npts, points_x, points_y, 
      points_z, nprim_pairs, prim_pairs, Xi, Xj, ldX, Gi, Gj, ldG, weights, boys_table );
  }

  void integral_2_2(double X_AB,
		    double Y_AB,
		    double Z_AB,
		    size_t npts,
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
    dev_integral_2_2<<<320, 128, 0, stream>>>(X_AB,
				   Y_AB,
				   Z_AB,
				   npts,
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

  __inline__ __device__ void dev_integral_2_2_batched_driver (
           double X_AB,
				   double Y_AB,
				   double Z_AB,
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


      dev_integral_2_2_driver( 
        X_AB, Y_AB, Z_AB,
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

  __global__ void dev_integral_2_2_batched(
           double X_AB,
				   double Y_AB,
				   double Z_AB,
           const GauXC::ShellPairToTaskDevice* sp2task,
           GauXC::XCDeviceTask*                device_tasks,
				   double *boys_table) {
   dev_integral_2_2_batched_driver(X_AB,Y_AB,Z_AB,sp2task,device_tasks,boys_table);
 }



  void integral_2_2_batched(size_t ntask_sp,
        double X_AB,
				double Y_AB,
				double Z_AB,
        const GauXC::ShellPairToTaskDevice* sp2task,
        GauXC::XCDeviceTask*                device_tasks,
		    double *boys_table,
        cudaStream_t stream) {

    int nthreads = 128;
    int nblocks_x = 160;
    int nblocks_y = ntask_sp;
    dim3 nblocks(nblocks_x, nblocks_y);

    dev_integral_2_2_batched<<<nblocks,nthreads,0,stream>>>(
      X_AB, Y_AB, Z_AB, sp2task, device_tasks, boys_table );

  }



  __global__ void dev_integral_2_2_shell_batched(
           int nsp,
           const GauXC::ShellPairToTaskDevice* sp2task,
           GauXC::XCDeviceTask*                device_tasks,
				   double *boys_table) {

   for( int i = blockIdx.z; i < nsp; i += gridDim.z ) {
     auto sp = sp2task + i;
     const auto X_AB = sp->X_AB;
     const auto Y_AB = sp->Y_AB;
     const auto Z_AB = sp->Z_AB;
     dev_integral_2_2_batched_driver(X_AB,Y_AB,Z_AB,sp,device_tasks,boys_table);
   }
 }

  void integral_2_2_shell_batched(
        size_t nsp,
        size_t max_ntask,
        const GauXC::ShellPairToTaskDevice* sp2task,
        GauXC::XCDeviceTask*                device_tasks,
		    double *boys_table,
        cudaStream_t stream) {

    int nthreads = 128;
    int nblocks_x = 1;
    int nblocks_y = max_ntask;
    int nblocks_z = nsp;
    dim3 nblocks(nblocks_x, nblocks_y, nblocks_z);
    dev_integral_2_2_shell_batched<<<nblocks,nthreads,0,stream>>>(
      nsp, sp2task, device_tasks, boys_table );

  }

template<ObaraSaikaType type_, int points_per_subtask_, int primpair_shared_limit_,
         bool pure_bra, bool pure_ket>
struct DeviceTask22 {
  static constexpr int max_primpair_shared_limit = 8;

  static constexpr int primpair_shared_limit = primpair_shared_limit_;
  static constexpr int points_per_subtask = points_per_subtask_;
  static constexpr int num_threads = points_per_subtask_;
  static constexpr ObaraSaikaType type = type_;

  static_assert(ObaraSaikaType::swap != type, "DeviceTask22 does not support swap");
  static constexpr bool diag = (ObaraSaikaType::diag == type);

  static constexpr bool use_shared = (primpair_shared_limit > 0) && 
                                     (primpair_shared_limit <= max_primpair_shared_limit);
  static constexpr int num_warps = points_per_subtask / GauXC::cuda::warp_size;
  // Cannot declare shared memory array with length 0
  static constexpr int prim_buffer_size = (use_shared) ? num_warps * primpair_shared_limit : 1;

  using Params = ObaraSaikaParamsWithAB<type>;

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
    const double X_AB = param.X_AB;
    const double Y_AB = param.Y_AB;
    const double Z_AB = param.Z_AB;

    const int laneId = threadIdx.x % GauXC::cuda::warp_size;
    const int warpId __attribute__((unused)) = threadIdx.x / GauXC::cuda::warp_size;

    __shared__ GauXC::PrimitivePair<double> s_prim_pairs[prim_buffer_size] __attribute__((unused));

    if constexpr (use_shared) {
      load_primpair_shared(laneId, warpId, nprim_pairs,
        &(prim_pairs[0]), &(s_prim_pairs[warpId * primpair_shared_limit]));
      __syncwarp();
    }

    double outBuffer[6];
    double temp[num_threads * 31];

    // Loop over points in shared in batches of 32
    for (int i = 0; i <  num_warps; i++) {

      for (int j = 0; j < 6; j++) {
        outBuffer[j] = 0.0;
      }

      for(int j = 0; j < 31; ++j) SCALAR_STORE((temp + j), SCALAR_ZERO());

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
          double RHO_INV = prim_pairs_use[ij].gamma_inv;
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

          SCALAR_TYPE t00, t01, t02, t03, t04, TVAL_inv_e;

          // Evaluate Boys function
          boys_element<4>(&TVAL, &TVAL_inv_e, &t04, boys_table);

          // Evaluate VRR Buffer
          SCALAR_TYPE t10, t11, t12, t13, t20, t21, t22, t30, t31, t40, tx, ty;

          t03 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(TVAL, t04), TVAL_inv_e), SCALAR_SET1(0.28571428571428569843));
          t02 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(TVAL, t03), TVAL_inv_e), SCALAR_SET1(0.40000000000000002220));
          t01 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(TVAL, t02), TVAL_inv_e), SCALAR_SET1(0.66666666666666662966));
          t00 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(TVAL, t01), TVAL_inv_e), SCALAR_SET1(2.00000000000000000000));

          t00 = SCALAR_MUL(eval, t00);
          t01 = SCALAR_MUL(eval, t01);
          t02 = SCALAR_MUL(eval, t02);
          t03 = SCALAR_MUL(eval, t03);
          t04 = SCALAR_MUL(eval, t04);
          t10 = SCALAR_MUL(X_PA, t00);
          t10 = SCALAR_FNMA(X_PC, t01, t10);
          t11 = SCALAR_MUL(X_PA, t01);
          t11 = SCALAR_FNMA(X_PC, t02, t11);
          t12 = SCALAR_MUL(X_PA, t02);
          t12 = SCALAR_FNMA(X_PC, t03, t12);
          t13 = SCALAR_MUL(X_PA, t03);
          t13 = SCALAR_FNMA(X_PC, t04, t13);
          t20 = SCALAR_MUL(X_PA, t10);
          t20 = SCALAR_FNMA(X_PC, t11, t20);
          tx = SCALAR_SUB(t00, t01);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t20 = SCALAR_FMA(tx, ty, t20);
          t21 = SCALAR_MUL(X_PA, t11);
          t21 = SCALAR_FNMA(X_PC, t12, t21);
          tx = SCALAR_SUB(t01, t02);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t21 = SCALAR_FMA(tx, ty, t21);
          t22 = SCALAR_MUL(X_PA, t12);
          t22 = SCALAR_FNMA(X_PC, t13, t22);
          tx = SCALAR_SUB(t02, t03);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t22 = SCALAR_FMA(tx, ty, t22);
          tx = SCALAR_LOAD((temp + 0 ));
          tx = SCALAR_ADD(tx, t20);
          SCALAR_STORE((temp + 0 ), tx);
          t30 = SCALAR_MUL(X_PA, t20);
          t30 = SCALAR_FNMA(X_PC, t21, t30);
          tx = SCALAR_SUB(t10, t11);
          ty = SCALAR_SET1(0.5 * 2);
          ty = SCALAR_MUL(ty, RHO_INV);
          t30 = SCALAR_FMA(tx, ty, t30);
          t31 = SCALAR_MUL(X_PA, t21);
          t31 = SCALAR_FNMA(X_PC, t22, t31);
          tx = SCALAR_SUB(t11, t12);
          ty = SCALAR_SET1(0.5 * 2);
          ty = SCALAR_MUL(ty, RHO_INV);
          t31 = SCALAR_FMA(tx, ty, t31);
          tx = SCALAR_LOAD((temp + 6 ));
          tx = SCALAR_ADD(tx, t30);
          SCALAR_STORE((temp + 6 ), tx);
          t40 = SCALAR_MUL(X_PA, t30);
          t40 = SCALAR_FNMA(X_PC, t31, t40);
          tx = SCALAR_SUB(t20, t21);
          ty = SCALAR_SET1(0.5 * 3);
          ty = SCALAR_MUL(ty, RHO_INV);
          t40 = SCALAR_FMA(tx, ty, t40);
          tx = SCALAR_LOAD((temp + 16 ));
          tx = SCALAR_ADD(tx, t40);
          SCALAR_STORE((temp + 16 ), tx);
          t40 = SCALAR_MUL(Y_PA, t30);
          t40 = SCALAR_FNMA(Y_PC, t31, t40);
          tx = SCALAR_LOAD((temp + 17 ));
          tx = SCALAR_ADD(tx, t40);
          SCALAR_STORE((temp + 17 ), tx);
          t40 = SCALAR_MUL(Z_PA, t30);
          t40 = SCALAR_FNMA(Z_PC, t31, t40);
          tx = SCALAR_LOAD((temp + 18 ));
          tx = SCALAR_ADD(tx, t40);
          SCALAR_STORE((temp + 18 ), tx);
          t30 = SCALAR_MUL(Y_PA, t20);
          t30 = SCALAR_FNMA(Y_PC, t21, t30);
          t31 = SCALAR_MUL(Y_PA, t21);
          t31 = SCALAR_FNMA(Y_PC, t22, t31);
          tx = SCALAR_LOAD((temp + 7 ));
          tx = SCALAR_ADD(tx, t30);
          SCALAR_STORE((temp + 7 ), tx);
          t40 = SCALAR_MUL(Y_PA, t30);
          t40 = SCALAR_FNMA(Y_PC, t31, t40);
          tx = SCALAR_SUB(t20, t21);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t40 = SCALAR_FMA(tx, ty, t40);
          tx = SCALAR_LOAD((temp + 19 ));
          tx = SCALAR_ADD(tx, t40);
          SCALAR_STORE((temp + 19 ), tx);
          t40 = SCALAR_MUL(Z_PA, t30);
          t40 = SCALAR_FNMA(Z_PC, t31, t40);
          tx = SCALAR_LOAD((temp + 20 ));
          tx = SCALAR_ADD(tx, t40);
          SCALAR_STORE((temp + 20 ), tx);
          t30 = SCALAR_MUL(Z_PA, t20);
          t30 = SCALAR_FNMA(Z_PC, t21, t30);
          t31 = SCALAR_MUL(Z_PA, t21);
          t31 = SCALAR_FNMA(Z_PC, t22, t31);
          tx = SCALAR_LOAD((temp + 8 ));
          tx = SCALAR_ADD(tx, t30);
          SCALAR_STORE((temp + 8 ), tx);
          t40 = SCALAR_MUL(Z_PA, t30);
          t40 = SCALAR_FNMA(Z_PC, t31, t40);
          tx = SCALAR_SUB(t20, t21);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t40 = SCALAR_FMA(tx, ty, t40);
          tx = SCALAR_LOAD((temp + 21 ));
          tx = SCALAR_ADD(tx, t40);
          SCALAR_STORE((temp + 21 ), tx);
          t20 = SCALAR_MUL(Y_PA, t10);
          t20 = SCALAR_FNMA(Y_PC, t11, t20);
          t21 = SCALAR_MUL(Y_PA, t11);
          t21 = SCALAR_FNMA(Y_PC, t12, t21);
          t22 = SCALAR_MUL(Y_PA, t12);
          t22 = SCALAR_FNMA(Y_PC, t13, t22);
          tx = SCALAR_LOAD((temp + 1 ));
          tx = SCALAR_ADD(tx, t20);
          SCALAR_STORE((temp + 1 ), tx);
          t30 = SCALAR_MUL(Y_PA, t20);
          t30 = SCALAR_FNMA(Y_PC, t21, t30);
          tx = SCALAR_SUB(t10, t11);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t30 = SCALAR_FMA(tx, ty, t30);
          t31 = SCALAR_MUL(Y_PA, t21);
          t31 = SCALAR_FNMA(Y_PC, t22, t31);
          tx = SCALAR_SUB(t11, t12);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t31 = SCALAR_FMA(tx, ty, t31);
          tx = SCALAR_LOAD((temp + 9 ));
          tx = SCALAR_ADD(tx, t30);
          SCALAR_STORE((temp + 9 ), tx);
          t40 = SCALAR_MUL(Y_PA, t30);
          t40 = SCALAR_FNMA(Y_PC, t31, t40);
          tx = SCALAR_SUB(t20, t21);
          ty = SCALAR_SET1(0.5 * 2);
          ty = SCALAR_MUL(ty, RHO_INV);
          t40 = SCALAR_FMA(tx, ty, t40);
          tx = SCALAR_LOAD((temp + 22 ));
          tx = SCALAR_ADD(tx, t40);
          SCALAR_STORE((temp + 22 ), tx);
          t40 = SCALAR_MUL(Z_PA, t30);
          t40 = SCALAR_FNMA(Z_PC, t31, t40);
          tx = SCALAR_LOAD((temp + 23 ));
          tx = SCALAR_ADD(tx, t40);
          SCALAR_STORE((temp + 23 ), tx);
          t30 = SCALAR_MUL(Z_PA, t20);
          t30 = SCALAR_FNMA(Z_PC, t21, t30);
          t31 = SCALAR_MUL(Z_PA, t21);
          t31 = SCALAR_FNMA(Z_PC, t22, t31);
          tx = SCALAR_LOAD((temp + 10 ));
          tx = SCALAR_ADD(tx, t30);
          SCALAR_STORE((temp + 10 ), tx);
          t40 = SCALAR_MUL(Z_PA, t30);
          t40 = SCALAR_FNMA(Z_PC, t31, t40);
          tx = SCALAR_SUB(t20, t21);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t40 = SCALAR_FMA(tx, ty, t40);
          tx = SCALAR_LOAD((temp + 24 ));
          tx = SCALAR_ADD(tx, t40);
          SCALAR_STORE((temp + 24 ), tx);
          t20 = SCALAR_MUL(Z_PA, t10);
          t20 = SCALAR_FNMA(Z_PC, t11, t20);
          t21 = SCALAR_MUL(Z_PA, t11);
          t21 = SCALAR_FNMA(Z_PC, t12, t21);
          t22 = SCALAR_MUL(Z_PA, t12);
          t22 = SCALAR_FNMA(Z_PC, t13, t22);
          tx = SCALAR_LOAD((temp + 2 ));
          tx = SCALAR_ADD(tx, t20);
          SCALAR_STORE((temp + 2 ), tx);
          t30 = SCALAR_MUL(Z_PA, t20);
          t30 = SCALAR_FNMA(Z_PC, t21, t30);
          tx = SCALAR_SUB(t10, t11);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t30 = SCALAR_FMA(tx, ty, t30);
          t31 = SCALAR_MUL(Z_PA, t21);
          t31 = SCALAR_FNMA(Z_PC, t22, t31);
          tx = SCALAR_SUB(t11, t12);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t31 = SCALAR_FMA(tx, ty, t31);
          tx = SCALAR_LOAD((temp + 11 ));
          tx = SCALAR_ADD(tx, t30);
          SCALAR_STORE((temp + 11 ), tx);
          t40 = SCALAR_MUL(Z_PA, t30);
          t40 = SCALAR_FNMA(Z_PC, t31, t40);
          tx = SCALAR_SUB(t20, t21);
          ty = SCALAR_SET1(0.5 * 2);
          ty = SCALAR_MUL(ty, RHO_INV);
          t40 = SCALAR_FMA(tx, ty, t40);
          tx = SCALAR_LOAD((temp + 25 ));
          tx = SCALAR_ADD(tx, t40);
          SCALAR_STORE((temp + 25 ), tx);
          t10 = SCALAR_MUL(Y_PA, t00);
          t10 = SCALAR_FNMA(Y_PC, t01, t10);
          t11 = SCALAR_MUL(Y_PA, t01);
          t11 = SCALAR_FNMA(Y_PC, t02, t11);
          t12 = SCALAR_MUL(Y_PA, t02);
          t12 = SCALAR_FNMA(Y_PC, t03, t12);
          t13 = SCALAR_MUL(Y_PA, t03);
          t13 = SCALAR_FNMA(Y_PC, t04, t13);
          t20 = SCALAR_MUL(Y_PA, t10);
          t20 = SCALAR_FNMA(Y_PC, t11, t20);
          tx = SCALAR_SUB(t00, t01);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t20 = SCALAR_FMA(tx, ty, t20);
          t21 = SCALAR_MUL(Y_PA, t11);
          t21 = SCALAR_FNMA(Y_PC, t12, t21);
          tx = SCALAR_SUB(t01, t02);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t21 = SCALAR_FMA(tx, ty, t21);
          t22 = SCALAR_MUL(Y_PA, t12);
          t22 = SCALAR_FNMA(Y_PC, t13, t22);
          tx = SCALAR_SUB(t02, t03);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t22 = SCALAR_FMA(tx, ty, t22);
          tx = SCALAR_LOAD((temp + 3 ));
          tx = SCALAR_ADD(tx, t20);
          SCALAR_STORE((temp + 3 ), tx);
          t30 = SCALAR_MUL(Y_PA, t20);
          t30 = SCALAR_FNMA(Y_PC, t21, t30);
          tx = SCALAR_SUB(t10, t11);
          ty = SCALAR_SET1(0.5 * 2);
          ty = SCALAR_MUL(ty, RHO_INV);
          t30 = SCALAR_FMA(tx, ty, t30);
          t31 = SCALAR_MUL(Y_PA, t21);
          t31 = SCALAR_FNMA(Y_PC, t22, t31);
          tx = SCALAR_SUB(t11, t12);
          ty = SCALAR_SET1(0.5 * 2);
          ty = SCALAR_MUL(ty, RHO_INV);
          t31 = SCALAR_FMA(tx, ty, t31);
          tx = SCALAR_LOAD((temp + 12 ));
          tx = SCALAR_ADD(tx, t30);
          SCALAR_STORE((temp + 12 ), tx);
          t40 = SCALAR_MUL(Y_PA, t30);
          t40 = SCALAR_FNMA(Y_PC, t31, t40);
          tx = SCALAR_SUB(t20, t21);
          ty = SCALAR_SET1(0.5 * 3);
          ty = SCALAR_MUL(ty, RHO_INV);
          t40 = SCALAR_FMA(tx, ty, t40);
          tx = SCALAR_LOAD((temp + 26 ));
          tx = SCALAR_ADD(tx, t40);
          SCALAR_STORE((temp + 26 ), tx);
          t40 = SCALAR_MUL(Z_PA, t30);
          t40 = SCALAR_FNMA(Z_PC, t31, t40);
          tx = SCALAR_LOAD((temp + 27 ));
          tx = SCALAR_ADD(tx, t40);
          SCALAR_STORE((temp + 27 ), tx);
          t30 = SCALAR_MUL(Z_PA, t20);
          t30 = SCALAR_FNMA(Z_PC, t21, t30);
          t31 = SCALAR_MUL(Z_PA, t21);
          t31 = SCALAR_FNMA(Z_PC, t22, t31);
          tx = SCALAR_LOAD((temp + 13 ));
          tx = SCALAR_ADD(tx, t30);
          SCALAR_STORE((temp + 13 ), tx);
          t40 = SCALAR_MUL(Z_PA, t30);
          t40 = SCALAR_FNMA(Z_PC, t31, t40);
          tx = SCALAR_SUB(t20, t21);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t40 = SCALAR_FMA(tx, ty, t40);
          tx = SCALAR_LOAD((temp + 28 ));
          tx = SCALAR_ADD(tx, t40);
          SCALAR_STORE((temp + 28 ), tx);
          t20 = SCALAR_MUL(Z_PA, t10);
          t20 = SCALAR_FNMA(Z_PC, t11, t20);
          t21 = SCALAR_MUL(Z_PA, t11);
          t21 = SCALAR_FNMA(Z_PC, t12, t21);
          t22 = SCALAR_MUL(Z_PA, t12);
          t22 = SCALAR_FNMA(Z_PC, t13, t22);
          tx = SCALAR_LOAD((temp + 4 ));
          tx = SCALAR_ADD(tx, t20);
          SCALAR_STORE((temp + 4 ), tx);
          t30 = SCALAR_MUL(Z_PA, t20);
          t30 = SCALAR_FNMA(Z_PC, t21, t30);
          tx = SCALAR_SUB(t10, t11);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t30 = SCALAR_FMA(tx, ty, t30);
          t31 = SCALAR_MUL(Z_PA, t21);
          t31 = SCALAR_FNMA(Z_PC, t22, t31);
          tx = SCALAR_SUB(t11, t12);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t31 = SCALAR_FMA(tx, ty, t31);
          tx = SCALAR_LOAD((temp + 14 ));
          tx = SCALAR_ADD(tx, t30);
          SCALAR_STORE((temp + 14 ), tx);
          t40 = SCALAR_MUL(Z_PA, t30);
          t40 = SCALAR_FNMA(Z_PC, t31, t40);
          tx = SCALAR_SUB(t20, t21);
          ty = SCALAR_SET1(0.5 * 2);
          ty = SCALAR_MUL(ty, RHO_INV);
          t40 = SCALAR_FMA(tx, ty, t40);
          tx = SCALAR_LOAD((temp + 29 ));
          tx = SCALAR_ADD(tx, t40);
          SCALAR_STORE((temp + 29 ), tx);
          t10 = SCALAR_MUL(Z_PA, t00);
          t10 = SCALAR_FNMA(Z_PC, t01, t10);
          t11 = SCALAR_MUL(Z_PA, t01);
          t11 = SCALAR_FNMA(Z_PC, t02, t11);
          t12 = SCALAR_MUL(Z_PA, t02);
          t12 = SCALAR_FNMA(Z_PC, t03, t12);
          t13 = SCALAR_MUL(Z_PA, t03);
          t13 = SCALAR_FNMA(Z_PC, t04, t13);
          t20 = SCALAR_MUL(Z_PA, t10);
          t20 = SCALAR_FNMA(Z_PC, t11, t20);
          tx = SCALAR_SUB(t00, t01);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t20 = SCALAR_FMA(tx, ty, t20);
          t21 = SCALAR_MUL(Z_PA, t11);
          t21 = SCALAR_FNMA(Z_PC, t12, t21);
          tx = SCALAR_SUB(t01, t02);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t21 = SCALAR_FMA(tx, ty, t21);
          t22 = SCALAR_MUL(Z_PA, t12);
          t22 = SCALAR_FNMA(Z_PC, t13, t22);
          tx = SCALAR_SUB(t02, t03);
          ty = SCALAR_SET1(0.5 * 1);
          ty = SCALAR_MUL(ty, RHO_INV);
          t22 = SCALAR_FMA(tx, ty, t22);
          tx = SCALAR_LOAD((temp + 5 ));
          tx = SCALAR_ADD(tx, t20);
          SCALAR_STORE((temp + 5 ), tx);
          t30 = SCALAR_MUL(Z_PA, t20);
          t30 = SCALAR_FNMA(Z_PC, t21, t30);
          tx = SCALAR_SUB(t10, t11);
          ty = SCALAR_SET1(0.5 * 2);
          ty = SCALAR_MUL(ty, RHO_INV);
          t30 = SCALAR_FMA(tx, ty, t30);
          t31 = SCALAR_MUL(Z_PA, t21);
          t31 = SCALAR_FNMA(Z_PC, t22, t31);
          tx = SCALAR_SUB(t11, t12);
          ty = SCALAR_SET1(0.5 * 2);
          ty = SCALAR_MUL(ty, RHO_INV);
          t31 = SCALAR_FMA(tx, ty, t31);
          tx = SCALAR_LOAD((temp + 15 ));
          tx = SCALAR_ADD(tx, t30);
          SCALAR_STORE((temp + 15 ), tx);
          t40 = SCALAR_MUL(Z_PA, t30);
          t40 = SCALAR_FNMA(Z_PC, t31, t40);
          tx = SCALAR_SUB(t20, t21);
          ty = SCALAR_SET1(0.5 * 3);
          ty = SCALAR_MUL(ty, RHO_INV);
          t40 = SCALAR_FMA(tx, ty, t40);
          tx = SCALAR_LOAD((temp + 30 ));
          tx = SCALAR_ADD(tx, t40);
          SCALAR_STORE((temp + 30 ), tx);
        }

        bool nonzero = false;
        for(int i = 0; i < 31; ++i) {
          nonzero = nonzero || abs(temp[i ]) > 1e-12;
        }

        if (diag || nonzero) {
          const double * __restrict__ Xik = (Xi + pointIndex);
          const double * __restrict__ Xjk = (Xj + pointIndex);
          double * __restrict__ Gik = (Gi + pointIndex);
          double * __restrict__ Gjk = (Gj + pointIndex);

          SCALAR_TYPE const_value_v = weight;

          double const_value, X_ABp, Y_ABp, Z_ABp, comb_m_i, comb_n_j, comb_p_k;
          SCALAR_TYPE const_value_w;
          SCALAR_TYPE tx, ty, tz, tw, t0, t1, t2, t3, t4, t5;

          SCALAR_TYPE Xik_0, Xik_1, Xik_2, Xik_3, Xik_4, Xik_5;
          SCALAR_TYPE Xjk_0, Xjk_1, Xjk_2, Xjk_3, Xjk_4, Xjk_5;
          SCALAR_TYPE Gjk_0, Gjk_1, Gjk_2, Gjk_3, Gjk_4, Gjk_5;

          if constexpr (pure_bra) {
            SCALAR_TYPE Xik_m2 = SCALAR_LOAD((Xik + 0*ldX));
            SCALAR_TYPE Xik_m1 = SCALAR_LOAD((Xik + 1*ldX));
            SCALAR_TYPE Xik_z0 = SCALAR_LOAD((Xik + 2*ldX));
            SCALAR_TYPE Xik_p1 = SCALAR_LOAD((Xik + 3*ldX));
            SCALAR_TYPE Xik_p2 = SCALAR_LOAD((Xik + 4*ldX));

            ::cuda::std::tie(Xik_0, Xik_1, Xik_2, Xik_3, Xik_4, Xik_5) =
              sph::itform_l2(Xik_m2, Xik_m1, Xik_z0, Xik_p1, Xik_p2);
          } else {
            Xik_0 = SCALAR_LOAD((Xik + 0*ldX));
            Xik_1 = SCALAR_LOAD((Xik + 1*ldX));
            Xik_2 = SCALAR_LOAD((Xik + 2*ldX));
            Xik_3 = SCALAR_LOAD((Xik + 3*ldX));
            Xik_4 = SCALAR_LOAD((Xik + 4*ldX));
            Xik_5 = SCALAR_LOAD((Xik + 5*ldX));
          }

          if constexpr (pure_ket) {
            SCALAR_TYPE Xjk_m2 = SCALAR_LOAD((Xjk + 0*ldX));
            SCALAR_TYPE Xjk_m1 = SCALAR_LOAD((Xjk + 1*ldX));
            SCALAR_TYPE Xjk_z0 = SCALAR_LOAD((Xjk + 2*ldX));
            SCALAR_TYPE Xjk_p1 = SCALAR_LOAD((Xjk + 3*ldX));
            SCALAR_TYPE Xjk_p2 = SCALAR_LOAD((Xjk + 4*ldX));

            ::cuda::std::tie(Xjk_0, Xjk_1, Xjk_2, Xjk_3, Xjk_4, Xjk_5) =
              sph::itform_l2(Xjk_m2, Xjk_m1, Xjk_z0, Xjk_p1, Xjk_p2);
          } else {
            Xjk_0 = SCALAR_LOAD((Xjk + 0*ldX));
            Xjk_1 = SCALAR_LOAD((Xjk + 1*ldX));
            Xjk_2 = SCALAR_LOAD((Xjk + 2*ldX));
            Xjk_3 = SCALAR_LOAD((Xjk + 3*ldX));
            Xjk_4 = SCALAR_LOAD((Xjk + 4*ldX));
            Xjk_5 = SCALAR_LOAD((Xjk + 5*ldX));
          }

          Gjk_0 = 0;
          Gjk_1 = 0;
          Gjk_2 = 0;
          Gjk_3 = 0;
          Gjk_4 = 0;
          Gjk_5 = 0;

          X_ABp = 1.0; comb_m_i = 1.0;
          Y_ABp = 1.0; comb_n_j = 1.0;
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          ty = Xjk_0;
          t0 = SCALAR_LOAD((temp + 16 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_MUL(tx, t0);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 17 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 18 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 19 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 20 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 21 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
                                  
          X_ABp = SCALAR_MUL(X_ABp, X_AB); comb_m_i = SCALAR_MUL(comb_m_i * 2, SCALAR_RECIPROCAL(1));
          Y_ABp = 1.0; comb_n_j = 1.0;
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          t0 = SCALAR_LOAD((temp + 6 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_FMA(tx, t0, tw);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 7 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 8 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 9 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 10 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 11 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
                                  
          X_ABp = SCALAR_MUL(X_ABp, X_AB); comb_m_i = SCALAR_MUL(comb_m_i * 1, SCALAR_RECIPROCAL(2));
          Y_ABp = 1.0; comb_n_j = 1.0;
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          t0 = SCALAR_LOAD((temp + 0 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_FMA(tx, t0, tw);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 1 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 2 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 3 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 4 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 5 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
          //if constexpr (!diag) atomicAdd((Gjk + 0 * ldG), tw);
          if constexpr (!diag) Gjk_0 += tw;
    


          X_ABp = 1.0; comb_m_i = 1.0;
          Y_ABp = 1.0; comb_n_j = 1.0;
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          ty = Xjk_1;
          t0 = SCALAR_LOAD((temp + 17 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_MUL(tx, t0);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 19 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 20 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 22 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 23 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 24 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
                                  
          Y_ABp = SCALAR_MUL(Y_ABp, Y_AB); comb_n_j = SCALAR_MUL(comb_n_j * 1, SCALAR_RECIPROCAL(1));
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          t0 = SCALAR_LOAD((temp + 6 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_FMA(tx, t0, tw);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 7 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 8 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 9 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 10 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 11 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
                                  
          X_ABp = SCALAR_MUL(X_ABp, X_AB); comb_m_i = SCALAR_MUL(comb_m_i * 1, SCALAR_RECIPROCAL(1));
          Y_ABp = 1.0; comb_n_j = 1.0;
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          t0 = SCALAR_LOAD((temp + 7 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_FMA(tx, t0, tw);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 9 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 10 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 12 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 13 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 14 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
                                  
          Y_ABp = SCALAR_MUL(Y_ABp, Y_AB); comb_n_j = SCALAR_MUL(comb_n_j * 1, SCALAR_RECIPROCAL(1));
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          t0 = SCALAR_LOAD((temp + 0 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_FMA(tx, t0, tw);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 1 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 2 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 3 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 4 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 5 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
          //if constexpr (!diag) atomicAdd((Gjk + 1 * ldG), tw);
          if constexpr (!diag) Gjk_1 += tw;




          X_ABp = 1.0; comb_m_i = 1.0;
          Y_ABp = 1.0; comb_n_j = 1.0;
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          ty = Xjk_2;
          t0 = SCALAR_LOAD((temp + 18 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_MUL(tx, t0);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 20 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 21 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 23 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 24 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 25 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
                                  
          Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 1, SCALAR_RECIPROCAL(1));
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          t0 = SCALAR_LOAD((temp + 6 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_FMA(tx, t0, tw);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 7 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 8 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 9 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 10 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 11 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
                                  
          X_ABp = SCALAR_MUL(X_ABp, X_AB); comb_m_i = SCALAR_MUL(comb_m_i * 1, SCALAR_RECIPROCAL(1));
          Y_ABp = 1.0; comb_n_j = 1.0;
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          t0 = SCALAR_LOAD((temp + 8 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_FMA(tx, t0, tw);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 10 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 11 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 13 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 14 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 15 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
                                  
          Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 1, SCALAR_RECIPROCAL(1));
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          t0 = SCALAR_LOAD((temp + 0 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_FMA(tx, t0, tw);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 1 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 2 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 3 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 4 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 5 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
          //if constexpr (!diag) atomicAdd((Gjk + 2 * ldG), tw);
          if constexpr (!diag) Gjk_2 += tw;





          X_ABp = 1.0; comb_m_i = 1.0;
          Y_ABp = 1.0; comb_n_j = 1.0;
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          ty = Xjk_3;
          t0 = SCALAR_LOAD((temp + 19 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_MUL(tx, t0);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 22 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 23 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 26 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 27 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 28 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
                                  
          Y_ABp = SCALAR_MUL(Y_ABp, Y_AB); comb_n_j = SCALAR_MUL(comb_n_j * 2, SCALAR_RECIPROCAL(1));
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          t0 = SCALAR_LOAD((temp + 7 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_FMA(tx, t0, tw);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 9 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 10 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 12 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 13 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 14 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
                                  
          Y_ABp = SCALAR_MUL(Y_ABp, Y_AB); comb_n_j = SCALAR_MUL(comb_n_j * 1, SCALAR_RECIPROCAL(2));
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          t0 = SCALAR_LOAD((temp + 0 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_FMA(tx, t0, tw);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 1 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 2 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 3 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 4 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 5 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
          //if constexpr (!diag) atomicAdd((Gjk + 3 * ldG), tw);
          if constexpr (!diag) Gjk_3 += tw;



          X_ABp = 1.0; comb_m_i = 1.0;
          Y_ABp = 1.0; comb_n_j = 1.0;
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          ty = Xjk_4;
          t0 = SCALAR_LOAD((temp + 20 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_MUL(tx, t0);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 23 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 24 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 27 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 28 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 29 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
                                  
          Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 1, SCALAR_RECIPROCAL(1));
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          t0 = SCALAR_LOAD((temp + 7 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_FMA(tx, t0, tw);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 9 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 10 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 12 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 13 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 14 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
                                  
          Y_ABp = SCALAR_MUL(Y_ABp, Y_AB); comb_n_j = SCALAR_MUL(comb_n_j * 1, SCALAR_RECIPROCAL(1));
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          t0 = SCALAR_LOAD((temp + 8 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_FMA(tx, t0, tw);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 10 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 11 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 13 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 14 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 15 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
                                  
          Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 1, SCALAR_RECIPROCAL(1));
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          t0 = SCALAR_LOAD((temp + 0 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_FMA(tx, t0, tw);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 1 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 2 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 3 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 4 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 5 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
          //if constexpr (!diag) atomicAdd((Gjk + 4 * ldG), tw);
          if constexpr (!diag) Gjk_4 += tw;




          X_ABp = 1.0; comb_m_i = 1.0;
          Y_ABp = 1.0; comb_n_j = 1.0;
          Z_ABp = 1.0; comb_p_k = 1.0;
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          ty = Xjk_5;
          t0 = SCALAR_LOAD((temp + 21 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_MUL(tx, t0);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 24 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 25 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 28 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 29 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 30 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
                                  
          Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 2, SCALAR_RECIPROCAL(1));
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          t0 = SCALAR_LOAD((temp + 8 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_FMA(tx, t0, tw);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 10 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 11 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 13 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 14 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 15 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
                                  
          Z_ABp = SCALAR_MUL(Z_ABp, Z_AB); comb_p_k = SCALAR_MUL(comb_p_k * 1, SCALAR_RECIPROCAL(2));
          const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
          const_value_w = SCALAR_MUL(const_value_v, const_value);
          tx = Xik_0;
          t0 = SCALAR_LOAD((temp + 0 ));
          t0 = SCALAR_MUL(t0, const_value_w);
          tz = SCALAR_MUL(ty, t0);
          tw = SCALAR_FMA(tx, t0, tw);
          outBuffer[0] += tz;
                                  
          tx = Xik_1;
          t1 = SCALAR_LOAD((temp + 1 ));
          t1 = SCALAR_MUL(t1, const_value_w);
          tz = SCALAR_MUL(ty, t1);
          tw = SCALAR_FMA(tx, t1, tw);
          outBuffer[1] += tz;
                                  
          tx = Xik_2;
          t2 = SCALAR_LOAD((temp + 2 ));
          t2 = SCALAR_MUL(t2, const_value_w);
          tz = SCALAR_MUL(ty, t2);
          tw = SCALAR_FMA(tx, t2, tw);
          outBuffer[2] += tz;
                                  
          tx = Xik_3;
          t3 = SCALAR_LOAD((temp + 3 ));
          t3 = SCALAR_MUL(t3, const_value_w);
          tz = SCALAR_MUL(ty, t3);
          tw = SCALAR_FMA(tx, t3, tw);
          outBuffer[3] += tz;
                                  
          tx = Xik_4;
          t4 = SCALAR_LOAD((temp + 4 ));
          t4 = SCALAR_MUL(t4, const_value_w);
          tz = SCALAR_MUL(ty, t4);
          tw = SCALAR_FMA(tx, t4, tw);
          outBuffer[4] += tz;
                                  
          tx = Xik_5;
          t5 = SCALAR_LOAD((temp + 5 ));
          t5 = SCALAR_MUL(t5, const_value_w);
          tz = SCALAR_MUL(ty, t5);
          tw = SCALAR_FMA(tx, t5, tw);
          outBuffer[5] += tz;
          //if constexpr (!diag) atomicAdd((Gjk + 5 * ldG), tw);
          if constexpr (!diag) Gjk_5 += tw;

          if constexpr (!diag) {
            if constexpr (pure_ket) {
              SCALAR_TYPE Gjk_m2, Gjk_m1, Gjk_z0, Gjk_p1, Gjk_p2;
              
              ::cuda::std::tie(Gjk_m2, Gjk_m1, Gjk_z0, Gjk_p1, Gjk_p2) =
                sph::tform_l2(Gjk_0, Gjk_1, Gjk_2, Gjk_3, Gjk_4, Gjk_5);
              atomicAdd((Gjk + 0 * ldG), Gjk_m2);
              atomicAdd((Gjk + 1 * ldG), Gjk_m1);
              atomicAdd((Gjk + 2 * ldG), Gjk_z0);
              atomicAdd((Gjk + 3 * ldG), Gjk_p1);
              atomicAdd((Gjk + 4 * ldG), Gjk_p2);
            } else {
              atomicAdd((Gjk + 0 * ldG), Gjk_0);
              atomicAdd((Gjk + 1 * ldG), Gjk_1);
              atomicAdd((Gjk + 2 * ldG), Gjk_2);
              atomicAdd((Gjk + 3 * ldG), Gjk_3);
              atomicAdd((Gjk + 4 * ldG), Gjk_4);
              atomicAdd((Gjk + 5 * ldG), Gjk_5);
            }
          }

          if constexpr (pure_bra) {
            SCALAR_TYPE Gik_m2, Gik_m1, Gik_z0, Gik_p1, Gik_p2;
              
            ::cuda::std::tie(Gik_m2, Gik_m1, Gik_z0, Gik_p1, Gik_p2) =
              sph::tform_l2(outBuffer[0], outBuffer[1], outBuffer[2], 
                            outBuffer[3], outBuffer[4], outBuffer[5]);
            atomicAdd((Gik + 0 * ldG), Gik_m2);
            atomicAdd((Gik + 1 * ldG), Gik_m1);
            atomicAdd((Gik + 2 * ldG), Gik_z0);
            atomicAdd((Gik + 3 * ldG), Gik_p1);
            atomicAdd((Gik + 4 * ldG), Gik_p2);
          } else {
            atomicAdd((Gik + 0 * ldG), outBuffer[0]);
            atomicAdd((Gik + 1 * ldG), outBuffer[1]);
            atomicAdd((Gik + 2 * ldG), outBuffer[2]);
            atomicAdd((Gik + 3 * ldG), outBuffer[3]);
            atomicAdd((Gik + 4 * ldG), outBuffer[4]);
            atomicAdd((Gik + 5 * ldG), outBuffer[5]);
          }
        }
      }
    }
    __syncwarp();
  }
};

template <int primpair_limit>
using AM22_cart = DeviceTask22<ObaraSaikaType::base,
  alg_constants::CudaAoSScheme1::ObaraSaika::points_per_subtask, 
  primpair_limit, false, false>;

template <int primpair_limit>
using AM2_cart = DeviceTask22<ObaraSaikaType::diag,
  alg_constants::CudaAoSScheme1::ObaraSaika::points_per_subtask, 
  primpair_limit, false, false>;

template <int primpair_limit>
using AM22_sph = DeviceTask22<ObaraSaikaType::base,
  alg_constants::CudaAoSScheme1::ObaraSaika::points_per_subtask, 
  primpair_limit, true, true>;

template <int primpair_limit>
using AM2_sph = DeviceTask22<ObaraSaikaType::diag,
  alg_constants::CudaAoSScheme1::ObaraSaika::points_per_subtask, 
  primpair_limit, true, true>;

  void integral_2_2_task_batched(
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
    
    if(sph)    
      dev_integral_task_map_dispatcher<AM22_sph>(
        nblocks, nthreads, max_primpair, stream, 
        ntasks, nsubtask,
        device_tasks, task2sp, 
        (int4*) subtasks, nprim_pairs_device, prim_pair_ptr_device,
        sp_X_AB_device, sp_Y_AB_device, sp_Z_AB_device,
        boys_table );
    else
      dev_integral_task_map_dispatcher<AM22_cart>(
        nblocks, nthreads, max_primpair, stream, 
        ntasks, nsubtask,
        device_tasks, task2sp, 
        (int4*) subtasks, nprim_pairs_device, prim_pair_ptr_device,
        sp_X_AB_device, sp_Y_AB_device, sp_Z_AB_device,
        boys_table );
  }

  void integral_2_task_batched(
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
    
    if(sph)
      dev_integral_task_map_dispatcher<AM2_sph>(
        nblocks, nthreads, max_primpair, stream, 
        ntasks, nsubtask,
        device_tasks, task2sp, 
        (int4*) subtasks, nprim_pairs_device, prim_pair_ptr_device,
        sp_X_AB_device, sp_Y_AB_device, sp_Z_AB_device,
        boys_table );
    else
      dev_integral_task_map_dispatcher<AM2_cart>(
        nblocks, nthreads, max_primpair, stream, 
        ntasks, nsubtask,
        device_tasks, task2sp, 
        (int4*) subtasks, nprim_pairs_device, prim_pair_ptr_device,
        sp_X_AB_device, sp_Y_AB_device, sp_Z_AB_device,
        boys_table );
  }

}
