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
#include "integral_2.hu"

namespace XGPU {
  __inline__ __device__ void dev_integral_2_driver(size_t npts,
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
    __shared__ double temp[128 * 31];
    
    for(size_t p_outer = blockIdx.x * blockDim.x; p_outer < npts; p_outer += gridDim.x * blockDim.x) {
      double *_point_outer_x = (points_x + p_outer);
      double *_point_outer_y = (points_y + p_outer);
      double *_point_outer_z = (points_z + p_outer);

      size_t p_inner = (threadIdx.x < (npts - p_outer)) ? threadIdx.x : (npts - p_outer);

      for(int i = 0; i < 31; ++i) SCALAR_STORE((temp + i * blockDim.x + threadIdx.x), SCALAR_ZERO());

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
	double *Gik = (Gi + p_outer + p_inner);

	for(int c0 = 0; c0 <= 2; ++c0) {
	  for(int c1 = 0; c1 <= c0; ++c1) {
	    int m = 2 - c0;
	    int p = c1;

	    int idxB = (((2 - m) * (2 - m + 1)) >> 1) + p;

	    int mv, pv;

	    SCALAR_TYPE tx, wg, xik, gik;
	    mv = 2 + m; pv = 0 + p;
	    tx  = SCALAR_LOAD((temp + (16 + (((4 - mv) * (4 - mv + 1)) >> 1) + pv) * blockDim.x + threadIdx.x));
	    wg  = SCALAR_LOAD((weights + p_outer + p_inner));

	    xik = SCALAR_LOAD((Xik + idxB * ldX));
	    gik = SCALAR_LOAD((Gik + 0 * ldG));

	    tx = SCALAR_MUL(tx, wg);
	    gik = SCALAR_FMA(tx, xik, gik);
	    SCALAR_STORE((Gik + 0 * ldG), gik);
	    mv = 1 + m; pv = 0 + p;
	    tx  = SCALAR_LOAD((temp + (16 + (((4 - mv) * (4 - mv + 1)) >> 1) + pv) * blockDim.x + threadIdx.x));
	    wg  = SCALAR_LOAD((weights + p_outer + p_inner));

	    xik = SCALAR_LOAD((Xik + idxB * ldX));
	    gik = SCALAR_LOAD((Gik + 1 * ldG));

	    tx = SCALAR_MUL(tx, wg);
	    gik = SCALAR_FMA(tx, xik, gik);
	    SCALAR_STORE((Gik + 1 * ldG), gik);
	    mv = 1 + m; pv = 1 + p;
	    tx  = SCALAR_LOAD((temp + (16 + (((4 - mv) * (4 - mv + 1)) >> 1) + pv) * blockDim.x + threadIdx.x));
	    wg  = SCALAR_LOAD((weights + p_outer + p_inner));

	    xik = SCALAR_LOAD((Xik + idxB * ldX));
	    gik = SCALAR_LOAD((Gik + 2 * ldG));

	    tx = SCALAR_MUL(tx, wg);
	    gik = SCALAR_FMA(tx, xik, gik);
	    SCALAR_STORE((Gik + 2 * ldG), gik);
	    mv = 0 + m; pv = 0 + p;
	    tx  = SCALAR_LOAD((temp + (16 + (((4 - mv) * (4 - mv + 1)) >> 1) + pv) * blockDim.x + threadIdx.x));
	    wg  = SCALAR_LOAD((weights + p_outer + p_inner));

	    xik = SCALAR_LOAD((Xik + idxB * ldX));
	    gik = SCALAR_LOAD((Gik + 3 * ldG));

	    tx = SCALAR_MUL(tx, wg);
	    gik = SCALAR_FMA(tx, xik, gik);
	    SCALAR_STORE((Gik + 3 * ldG), gik);
	    mv = 0 + m; pv = 1 + p;
	    tx  = SCALAR_LOAD((temp + (16 + (((4 - mv) * (4 - mv + 1)) >> 1) + pv) * blockDim.x + threadIdx.x));
	    wg  = SCALAR_LOAD((weights + p_outer + p_inner));

	    xik = SCALAR_LOAD((Xik + idxB * ldX));
	    gik = SCALAR_LOAD((Gik + 4 * ldG));

	    tx = SCALAR_MUL(tx, wg);
	    gik = SCALAR_FMA(tx, xik, gik);
	    SCALAR_STORE((Gik + 4 * ldG), gik);
	    mv = 0 + m; pv = 2 + p;
	    tx  = SCALAR_LOAD((temp + (16 + (((4 - mv) * (4 - mv + 1)) >> 1) + pv) * blockDim.x + threadIdx.x));
	    wg  = SCALAR_LOAD((weights + p_outer + p_inner));

	    xik = SCALAR_LOAD((Xik + idxB * ldX));
	    gik = SCALAR_LOAD((Gik + 5 * ldG));

	    tx = SCALAR_MUL(tx, wg);
	    gik = SCALAR_FMA(tx, xik, gik);
	    SCALAR_STORE((Gik + 5 * ldG), gik);
	  }
	}
      }
    }
  }

  __global__ void dev_integral_2(size_t npts,
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
    dev_integral_2_driver( npts, points_x, points_y, points_z, nprim_pairs, prim_pairs, Xi, ldX,
      Gi, ldG, weights, boys_table );
  }

  void integral_2(size_t npts,
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
    dev_integral_2<<<320, 128, 0, stream>>>(npts,
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

  __global__ void dev_integral_2_batched(
           const GauXC::ShellPairToTaskDevice* sp2task,
           GauXC::XCDeviceTask*                device_tasks,
				   double *boys_table) {

    const int ntask = sp2task->ntask;
    for( int i_task = blockIdx.y; i_task < ntask; i_task += gridDim.y ) {
    
      const auto iT = sp2task->task_idx_device[i_task];
      const auto* task  = device_tasks + iT;
      const auto  npts  = task->npts;

      const auto  i_off = sp2task->task_shell_off_row_device[i_task]*npts;

      dev_integral_2_driver( 
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


  void integral_2_batched(size_t ntask_sp,
        const GauXC::ShellPairToTaskDevice* sp2task,
        GauXC::XCDeviceTask*                device_tasks,
		    double *boys_table,
        cudaStream_t stream) {

    int nthreads = 128;
    int nblocks_x = 160;
    int nblocks_y = ntask_sp;
    dim3 nblocks(nblocks_x, nblocks_y);
    dev_integral_2_batched<<<nblocks,nthreads,0,stream>>>(
      sp2task, device_tasks, boys_table );

  }
}
