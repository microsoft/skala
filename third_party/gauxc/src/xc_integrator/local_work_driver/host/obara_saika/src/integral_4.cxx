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
#include "../include/cpu/chebyshev_boys_computation.hpp"
#include "../include/cpu/integral_data_types.hpp"
#include "config_obara_saika.hpp"
#include "integral_4.hpp"

#define PI 3.14159265358979323846

namespace XCPU {
void integral_4(size_t npts,
               double *_points,
               point rA,
               point /*rB*/,
               int nprim_pairs,
               prim_pair *prim_pairs,
               double *Xi,
               int ldX,
               double *Gi,
               int ldG, 
               double *weights,
               double *boys_table) {
   __attribute__((__aligned__(64))) double buffer[145 * NPTS_LOCAL + 3 * NPTS_LOCAL];

   double *temp       = (buffer + 0);
   double *Tval       = (buffer + 145 * NPTS_LOCAL + 0 * NPTS_LOCAL);
   double *Tval_inv_e = (buffer + 145 * NPTS_LOCAL + 1 * NPTS_LOCAL);
   double *FmT        = (buffer + 145 * NPTS_LOCAL + 2 * NPTS_LOCAL);

   size_t npts_upper = NPTS_LOCAL * (npts / NPTS_LOCAL);
   size_t p_outer = 0;
   for(p_outer = 0; p_outer < npts_upper; p_outer += NPTS_LOCAL) {
      double *_point_outer = (_points + p_outer);

      double xA = rA.x;
      double yA = rA.y;
      double zA = rA.z;

      for(int i = 0; i < 145 * NPTS_LOCAL; i += SIMD_LENGTH) SIMD_ALIGNED_STORE((temp + i), SIMD_ZERO());

      for(int ij = 0; ij < nprim_pairs; ++ij) {
         double RHO = prim_pairs[ij].gamma;
         double RHO_INV = 1.0 / RHO;

         constexpr double X_PA = 0.0;
         constexpr double Y_PA = 0.0;
         constexpr double Z_PA = 0.0;

         double eval = prim_pairs[ij].K_coeff_prod;

         // Evaluate T Values
         for(size_t p_inner = 0; p_inner < NPTS_LOCAL; p_inner += SIMD_LENGTH) {
            SIMD_TYPE xC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 0 * npts));
            SIMD_TYPE yC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 1 * npts));
            SIMD_TYPE zC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 2 * npts));

            SIMD_TYPE X_PC = SIMD_SUB(SIMD_DUPLICATE(&(xA)), xC);
            SIMD_TYPE Y_PC = SIMD_SUB(SIMD_DUPLICATE(&(yA)), yC);
            SIMD_TYPE Z_PC = SIMD_SUB(SIMD_DUPLICATE(&(zA)), zC);

            X_PC = SIMD_MUL(X_PC, X_PC);
            X_PC = SIMD_FMA(Y_PC, Y_PC, X_PC);
            X_PC = SIMD_FMA(Z_PC, Z_PC, X_PC);
            X_PC = SIMD_MUL(SIMD_DUPLICATE(&(RHO)), X_PC);
            SIMD_ALIGNED_STORE((Tval + p_inner), X_PC);
         }

         // Evaluate Boys function
         boys_elements<8>(NPTS_LOCAL, Tval, Tval_inv_e, FmT, boys_table);

         // Evaluate VRR Buffer
         for(size_t p_inner = 0; p_inner < NPTS_LOCAL; p_inner += SIMD_LENGTH) {
            SIMD_TYPE xC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 0 * npts));
            SIMD_TYPE yC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 1 * npts));
            SIMD_TYPE zC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 2 * npts));

            SIMD_TYPE X_PC = SIMD_SUB(SIMD_DUPLICATE(&(xA)), xC);
            SIMD_TYPE Y_PC = SIMD_SUB(SIMD_DUPLICATE(&(yA)), yC);
            SIMD_TYPE Z_PC = SIMD_SUB(SIMD_DUPLICATE(&(zA)), zC);

            SIMD_TYPE tval, tval_inv_e, tx, ty, t00, t01, t02, t03, t04, t05, t06, t07, t08, t10, t11, t12, t13, t14, t15, t16, t17, t20, t21, t22, t23, t24, t25, t26, t30, t31, t32, t33, t34, t35, t40, t41, t42, t43, t44, t50, t51, t52, t53, t60, t61, t62, t70, t71, t80;

            tval = SIMD_ALIGNED_LOAD((Tval + p_inner));
            tval_inv_e = SIMD_ALIGNED_LOAD((Tval_inv_e + p_inner));

            t08 = SIMD_ALIGNED_LOAD((FmT + p_inner));
            t07 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t08), tval_inv_e), SIMD_SET1(0.13333333333333333148));
            t06 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t07), tval_inv_e), SIMD_SET1(0.15384615384615385469));
            t05 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t06), tval_inv_e), SIMD_SET1(0.18181818181818182323));
            t04 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t05), tval_inv_e), SIMD_SET1(0.22222222222222220989));
            t03 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t04), tval_inv_e), SIMD_SET1(0.28571428571428569843));
            t02 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t03), tval_inv_e), SIMD_SET1(0.40000000000000002220));
            t01 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t02), tval_inv_e), SIMD_SET1(0.66666666666666662966));
            t00 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t01), tval_inv_e), SIMD_SET1(2.00000000000000000000));

            t00 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t00);
            t01 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t01);
            t02 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t02);
            t03 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t03);
            t04 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t04);
            t05 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t05);
            t06 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t06);
            t07 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t07);
            t08 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t08);
            t10 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t00);
            t10 = SIMD_FNMA(X_PC, t01, t10);
            t11 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t01);
            t11 = SIMD_FNMA(X_PC, t02, t11);
            t12 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t02);
            t12 = SIMD_FNMA(X_PC, t03, t12);
            t13 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t03);
            t13 = SIMD_FNMA(X_PC, t04, t13);
            t14 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t04);
            t14 = SIMD_FNMA(X_PC, t05, t14);
            t15 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t05);
            t15 = SIMD_FNMA(X_PC, t06, t15);
            t16 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t06);
            t16 = SIMD_FNMA(X_PC, t07, t16);
            t17 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t07);
            t17 = SIMD_FNMA(X_PC, t08, t17);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t10);
            t20 = SIMD_FNMA(X_PC, t11, t20);
            tx = SIMD_SUB(t00, t01);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t20 = SIMD_FMA(tx, ty, t20);
            t21 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t11);
            t21 = SIMD_FNMA(X_PC, t12, t21);
            tx = SIMD_SUB(t01, t02);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t21 = SIMD_FMA(tx, ty, t21);
            t22 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t12);
            t22 = SIMD_FNMA(X_PC, t13, t22);
            tx = SIMD_SUB(t02, t03);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t22 = SIMD_FMA(tx, ty, t22);
            t23 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t13);
            t23 = SIMD_FNMA(X_PC, t14, t23);
            tx = SIMD_SUB(t03, t04);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t23 = SIMD_FMA(tx, ty, t23);
            t24 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t14);
            t24 = SIMD_FNMA(X_PC, t15, t24);
            tx = SIMD_SUB(t04, t05);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t24 = SIMD_FMA(tx, ty, t24);
            t25 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t15);
            t25 = SIMD_FNMA(X_PC, t16, t25);
            tx = SIMD_SUB(t05, t06);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t25 = SIMD_FMA(tx, ty, t25);
            t26 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t16);
            t26 = SIMD_FNMA(X_PC, t17, t26);
            tx = SIMD_SUB(t06, t07);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t26 = SIMD_FMA(tx, ty, t26);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t20);
            t30 = SIMD_FNMA(X_PC, t21, t30);
            tx = SIMD_SUB(t10, t11);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t30 = SIMD_FMA(tx, ty, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t21);
            t31 = SIMD_FNMA(X_PC, t22, t31);
            tx = SIMD_SUB(t11, t12);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t31 = SIMD_FMA(tx, ty, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t22);
            t32 = SIMD_FNMA(X_PC, t23, t32);
            tx = SIMD_SUB(t12, t13);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t32 = SIMD_FMA(tx, ty, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t23);
            t33 = SIMD_FNMA(X_PC, t24, t33);
            tx = SIMD_SUB(t13, t14);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t33 = SIMD_FMA(tx, ty, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t24);
            t34 = SIMD_FNMA(X_PC, t25, t34);
            tx = SIMD_SUB(t14, t15);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t34 = SIMD_FMA(tx, ty, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t25);
            t35 = SIMD_FNMA(X_PC, t26, t35);
            tx = SIMD_SUB(t15, t16);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t35 = SIMD_FMA(tx, ty, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t30);
            t40 = SIMD_FNMA(X_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t31);
            t41 = SIMD_FNMA(X_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t32);
            t42 = SIMD_FNMA(X_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t33);
            t43 = SIMD_FNMA(X_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t34);
            t44 = SIMD_FNMA(X_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 0 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t40);
            t50 = SIMD_FNMA(X_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t41);
            t51 = SIMD_FNMA(X_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t42);
            t52 = SIMD_FNMA(X_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t43);
            t53 = SIMD_FNMA(X_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 15 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 15 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t50);
            t60 = SIMD_FNMA(X_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t51);
            t61 = SIMD_FNMA(X_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t52);
            t62 = SIMD_FNMA(X_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 36 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 36 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t60);
            t70 = SIMD_FNMA(X_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t61);
            t71 = SIMD_FNMA(X_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 64 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 64 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t70);
            t80 = SIMD_FNMA(X_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 7);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 100 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 100 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t70);
            t80 = SIMD_FNMA(Y_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 101 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 101 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 102 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 102 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t60);
            t70 = SIMD_FNMA(Y_PC, t61, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t61);
            t71 = SIMD_FNMA(Y_PC, t62, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 65 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 65 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t70);
            t80 = SIMD_FNMA(Y_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 103 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 103 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 104 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 104 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 66 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 66 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 105 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 105 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t50);
            t60 = SIMD_FNMA(Y_PC, t51, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t51);
            t61 = SIMD_FNMA(Y_PC, t52, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t52);
            t62 = SIMD_FNMA(Y_PC, t53, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 37 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 37 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t60);
            t70 = SIMD_FNMA(Y_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t61);
            t71 = SIMD_FNMA(Y_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 67 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 67 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t70);
            t80 = SIMD_FNMA(Y_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 106 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 106 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 107 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 68 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 68 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 108 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 38 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 38 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 69 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 69 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 109 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 109 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t40);
            t50 = SIMD_FNMA(Y_PC, t41, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t41);
            t51 = SIMD_FNMA(Y_PC, t42, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t42);
            t52 = SIMD_FNMA(Y_PC, t43, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t43);
            t53 = SIMD_FNMA(Y_PC, t44, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 16 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 16 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t50);
            t60 = SIMD_FNMA(Y_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t51);
            t61 = SIMD_FNMA(Y_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t52);
            t62 = SIMD_FNMA(Y_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 39 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 39 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t60);
            t70 = SIMD_FNMA(Y_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t61);
            t71 = SIMD_FNMA(Y_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 70 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 70 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t70);
            t80 = SIMD_FNMA(Y_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 110 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 111 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 71 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 71 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 112 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 40 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 40 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 72 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 72 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 113 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 17 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 17 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 41 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 41 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 73 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 73 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 114 * NPTS_LOCAL + p_inner), tx);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t30);
            t40 = SIMD_FNMA(Y_PC, t31, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t31);
            t41 = SIMD_FNMA(Y_PC, t32, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t32);
            t42 = SIMD_FNMA(Y_PC, t33, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t33);
            t43 = SIMD_FNMA(Y_PC, t34, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t34);
            t44 = SIMD_FNMA(Y_PC, t35, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 1 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 1 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t40);
            t50 = SIMD_FNMA(Y_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t41);
            t51 = SIMD_FNMA(Y_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t42);
            t52 = SIMD_FNMA(Y_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t43);
            t53 = SIMD_FNMA(Y_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 18 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 18 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t50);
            t60 = SIMD_FNMA(Y_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t51);
            t61 = SIMD_FNMA(Y_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t52);
            t62 = SIMD_FNMA(Y_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 42 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 42 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t60);
            t70 = SIMD_FNMA(Y_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t61);
            t71 = SIMD_FNMA(Y_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 74 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 74 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t70);
            t80 = SIMD_FNMA(Y_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 115 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 115 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 116 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 75 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 75 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 117 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 43 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 43 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 76 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 76 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 118 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 19 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 19 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 44 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 44 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 77 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 77 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 119 * NPTS_LOCAL + p_inner), tx);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 2 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 2 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 20 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 20 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 45 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 45 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 78 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 78 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 120 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 120 * NPTS_LOCAL + p_inner), tx);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t20);
            t30 = SIMD_FNMA(Y_PC, t21, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t21);
            t31 = SIMD_FNMA(Y_PC, t22, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t22);
            t32 = SIMD_FNMA(Y_PC, t23, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t23);
            t33 = SIMD_FNMA(Y_PC, t24, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t24);
            t34 = SIMD_FNMA(Y_PC, t25, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t25);
            t35 = SIMD_FNMA(Y_PC, t26, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t30);
            t40 = SIMD_FNMA(Y_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t31);
            t41 = SIMD_FNMA(Y_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t32);
            t42 = SIMD_FNMA(Y_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t33);
            t43 = SIMD_FNMA(Y_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t34);
            t44 = SIMD_FNMA(Y_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 3 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 3 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t40);
            t50 = SIMD_FNMA(Y_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t41);
            t51 = SIMD_FNMA(Y_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t42);
            t52 = SIMD_FNMA(Y_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t43);
            t53 = SIMD_FNMA(Y_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 21 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 21 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t50);
            t60 = SIMD_FNMA(Y_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t51);
            t61 = SIMD_FNMA(Y_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t52);
            t62 = SIMD_FNMA(Y_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 46 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 46 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t60);
            t70 = SIMD_FNMA(Y_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t61);
            t71 = SIMD_FNMA(Y_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 79 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 79 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t70);
            t80 = SIMD_FNMA(Y_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 121 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 121 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 122 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 80 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 80 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 123 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 47 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 47 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 81 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 81 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 124 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 22 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 22 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 48 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 48 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 82 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 82 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 125 * NPTS_LOCAL + p_inner), tx);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 4 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 4 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 23 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 23 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 49 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 49 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 83 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 83 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 126 * NPTS_LOCAL + p_inner), tx);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t20);
            t30 = SIMD_FNMA(Z_PC, t21, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t21);
            t31 = SIMD_FNMA(Z_PC, t22, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t22);
            t32 = SIMD_FNMA(Z_PC, t23, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t23);
            t33 = SIMD_FNMA(Z_PC, t24, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t24);
            t34 = SIMD_FNMA(Z_PC, t25, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t25);
            t35 = SIMD_FNMA(Z_PC, t26, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 5 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 5 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 24 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 24 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 50 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 50 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 84 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 84 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 127 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 127 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t10);
            t20 = SIMD_FNMA(Y_PC, t11, t20);
            t21 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t11);
            t21 = SIMD_FNMA(Y_PC, t12, t21);
            t22 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t12);
            t22 = SIMD_FNMA(Y_PC, t13, t22);
            t23 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t13);
            t23 = SIMD_FNMA(Y_PC, t14, t23);
            t24 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t14);
            t24 = SIMD_FNMA(Y_PC, t15, t24);
            t25 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t15);
            t25 = SIMD_FNMA(Y_PC, t16, t25);
            t26 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t16);
            t26 = SIMD_FNMA(Y_PC, t17, t26);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t20);
            t30 = SIMD_FNMA(Y_PC, t21, t30);
            tx = SIMD_SUB(t10, t11);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t30 = SIMD_FMA(tx, ty, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t21);
            t31 = SIMD_FNMA(Y_PC, t22, t31);
            tx = SIMD_SUB(t11, t12);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t31 = SIMD_FMA(tx, ty, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t22);
            t32 = SIMD_FNMA(Y_PC, t23, t32);
            tx = SIMD_SUB(t12, t13);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t32 = SIMD_FMA(tx, ty, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t23);
            t33 = SIMD_FNMA(Y_PC, t24, t33);
            tx = SIMD_SUB(t13, t14);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t33 = SIMD_FMA(tx, ty, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t24);
            t34 = SIMD_FNMA(Y_PC, t25, t34);
            tx = SIMD_SUB(t14, t15);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t34 = SIMD_FMA(tx, ty, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t25);
            t35 = SIMD_FNMA(Y_PC, t26, t35);
            tx = SIMD_SUB(t15, t16);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t35 = SIMD_FMA(tx, ty, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t30);
            t40 = SIMD_FNMA(Y_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t31);
            t41 = SIMD_FNMA(Y_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t32);
            t42 = SIMD_FNMA(Y_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t33);
            t43 = SIMD_FNMA(Y_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t34);
            t44 = SIMD_FNMA(Y_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 6 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 6 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t40);
            t50 = SIMD_FNMA(Y_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t41);
            t51 = SIMD_FNMA(Y_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t42);
            t52 = SIMD_FNMA(Y_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t43);
            t53 = SIMD_FNMA(Y_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 25 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 25 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t50);
            t60 = SIMD_FNMA(Y_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t51);
            t61 = SIMD_FNMA(Y_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t52);
            t62 = SIMD_FNMA(Y_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 51 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 51 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t60);
            t70 = SIMD_FNMA(Y_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t61);
            t71 = SIMD_FNMA(Y_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 85 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 85 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t70);
            t80 = SIMD_FNMA(Y_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 128 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 128 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 129 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 129 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 86 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 86 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 130 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 52 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 52 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 87 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 87 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 131 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 26 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 26 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 53 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 53 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 88 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 88 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 132 * NPTS_LOCAL + p_inner), tx);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 7 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 7 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 27 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 27 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 54 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 54 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 89 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 89 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 133 * NPTS_LOCAL + p_inner), tx);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t20);
            t30 = SIMD_FNMA(Z_PC, t21, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t21);
            t31 = SIMD_FNMA(Z_PC, t22, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t22);
            t32 = SIMD_FNMA(Z_PC, t23, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t23);
            t33 = SIMD_FNMA(Z_PC, t24, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t24);
            t34 = SIMD_FNMA(Z_PC, t25, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t25);
            t35 = SIMD_FNMA(Z_PC, t26, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 8 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 8 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 28 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 28 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 55 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 55 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 90 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 90 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 134 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 134 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t10);
            t20 = SIMD_FNMA(Z_PC, t11, t20);
            t21 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t11);
            t21 = SIMD_FNMA(Z_PC, t12, t21);
            t22 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t12);
            t22 = SIMD_FNMA(Z_PC, t13, t22);
            t23 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t13);
            t23 = SIMD_FNMA(Z_PC, t14, t23);
            t24 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t14);
            t24 = SIMD_FNMA(Z_PC, t15, t24);
            t25 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t15);
            t25 = SIMD_FNMA(Z_PC, t16, t25);
            t26 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t16);
            t26 = SIMD_FNMA(Z_PC, t17, t26);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t20);
            t30 = SIMD_FNMA(Z_PC, t21, t30);
            tx = SIMD_SUB(t10, t11);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t30 = SIMD_FMA(tx, ty, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t21);
            t31 = SIMD_FNMA(Z_PC, t22, t31);
            tx = SIMD_SUB(t11, t12);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t31 = SIMD_FMA(tx, ty, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t22);
            t32 = SIMD_FNMA(Z_PC, t23, t32);
            tx = SIMD_SUB(t12, t13);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t32 = SIMD_FMA(tx, ty, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t23);
            t33 = SIMD_FNMA(Z_PC, t24, t33);
            tx = SIMD_SUB(t13, t14);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t33 = SIMD_FMA(tx, ty, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t24);
            t34 = SIMD_FNMA(Z_PC, t25, t34);
            tx = SIMD_SUB(t14, t15);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t34 = SIMD_FMA(tx, ty, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t25);
            t35 = SIMD_FNMA(Z_PC, t26, t35);
            tx = SIMD_SUB(t15, t16);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t35 = SIMD_FMA(tx, ty, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 9 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 9 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 29 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 29 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 56 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 56 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 91 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 91 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 135 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 135 * NPTS_LOCAL + p_inner), tx);
            t10 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t00);
            t10 = SIMD_FNMA(Y_PC, t01, t10);
            t11 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t01);
            t11 = SIMD_FNMA(Y_PC, t02, t11);
            t12 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t02);
            t12 = SIMD_FNMA(Y_PC, t03, t12);
            t13 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t03);
            t13 = SIMD_FNMA(Y_PC, t04, t13);
            t14 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t04);
            t14 = SIMD_FNMA(Y_PC, t05, t14);
            t15 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t05);
            t15 = SIMD_FNMA(Y_PC, t06, t15);
            t16 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t06);
            t16 = SIMD_FNMA(Y_PC, t07, t16);
            t17 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t07);
            t17 = SIMD_FNMA(Y_PC, t08, t17);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t10);
            t20 = SIMD_FNMA(Y_PC, t11, t20);
            tx = SIMD_SUB(t00, t01);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t20 = SIMD_FMA(tx, ty, t20);
            t21 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t11);
            t21 = SIMD_FNMA(Y_PC, t12, t21);
            tx = SIMD_SUB(t01, t02);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t21 = SIMD_FMA(tx, ty, t21);
            t22 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t12);
            t22 = SIMD_FNMA(Y_PC, t13, t22);
            tx = SIMD_SUB(t02, t03);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t22 = SIMD_FMA(tx, ty, t22);
            t23 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t13);
            t23 = SIMD_FNMA(Y_PC, t14, t23);
            tx = SIMD_SUB(t03, t04);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t23 = SIMD_FMA(tx, ty, t23);
            t24 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t14);
            t24 = SIMD_FNMA(Y_PC, t15, t24);
            tx = SIMD_SUB(t04, t05);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t24 = SIMD_FMA(tx, ty, t24);
            t25 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t15);
            t25 = SIMD_FNMA(Y_PC, t16, t25);
            tx = SIMD_SUB(t05, t06);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t25 = SIMD_FMA(tx, ty, t25);
            t26 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t16);
            t26 = SIMD_FNMA(Y_PC, t17, t26);
            tx = SIMD_SUB(t06, t07);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t26 = SIMD_FMA(tx, ty, t26);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t20);
            t30 = SIMD_FNMA(Y_PC, t21, t30);
            tx = SIMD_SUB(t10, t11);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t30 = SIMD_FMA(tx, ty, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t21);
            t31 = SIMD_FNMA(Y_PC, t22, t31);
            tx = SIMD_SUB(t11, t12);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t31 = SIMD_FMA(tx, ty, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t22);
            t32 = SIMD_FNMA(Y_PC, t23, t32);
            tx = SIMD_SUB(t12, t13);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t32 = SIMD_FMA(tx, ty, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t23);
            t33 = SIMD_FNMA(Y_PC, t24, t33);
            tx = SIMD_SUB(t13, t14);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t33 = SIMD_FMA(tx, ty, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t24);
            t34 = SIMD_FNMA(Y_PC, t25, t34);
            tx = SIMD_SUB(t14, t15);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t34 = SIMD_FMA(tx, ty, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t25);
            t35 = SIMD_FNMA(Y_PC, t26, t35);
            tx = SIMD_SUB(t15, t16);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t35 = SIMD_FMA(tx, ty, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t30);
            t40 = SIMD_FNMA(Y_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t31);
            t41 = SIMD_FNMA(Y_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t32);
            t42 = SIMD_FNMA(Y_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t33);
            t43 = SIMD_FNMA(Y_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t34);
            t44 = SIMD_FNMA(Y_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 10 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 10 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t40);
            t50 = SIMD_FNMA(Y_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t41);
            t51 = SIMD_FNMA(Y_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t42);
            t52 = SIMD_FNMA(Y_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t43);
            t53 = SIMD_FNMA(Y_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 30 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 30 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t50);
            t60 = SIMD_FNMA(Y_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t51);
            t61 = SIMD_FNMA(Y_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t52);
            t62 = SIMD_FNMA(Y_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 57 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 57 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t60);
            t70 = SIMD_FNMA(Y_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t61);
            t71 = SIMD_FNMA(Y_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 92 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 92 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t70);
            t80 = SIMD_FNMA(Y_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 7);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 136 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 136 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 137 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 137 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 93 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 93 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 138 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 138 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 58 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 58 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 94 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 94 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 139 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 139 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 31 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 31 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 59 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 59 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 95 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 95 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 140 * NPTS_LOCAL + p_inner), tx);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 11 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 11 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 32 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 32 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 60 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 60 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 96 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 96 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 141 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 141 * NPTS_LOCAL + p_inner), tx);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t20);
            t30 = SIMD_FNMA(Z_PC, t21, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t21);
            t31 = SIMD_FNMA(Z_PC, t22, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t22);
            t32 = SIMD_FNMA(Z_PC, t23, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t23);
            t33 = SIMD_FNMA(Z_PC, t24, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t24);
            t34 = SIMD_FNMA(Z_PC, t25, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t25);
            t35 = SIMD_FNMA(Z_PC, t26, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 12 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 12 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 33 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 33 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 61 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 61 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 97 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 97 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 142 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 142 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t10);
            t20 = SIMD_FNMA(Z_PC, t11, t20);
            t21 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t11);
            t21 = SIMD_FNMA(Z_PC, t12, t21);
            t22 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t12);
            t22 = SIMD_FNMA(Z_PC, t13, t22);
            t23 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t13);
            t23 = SIMD_FNMA(Z_PC, t14, t23);
            t24 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t14);
            t24 = SIMD_FNMA(Z_PC, t15, t24);
            t25 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t15);
            t25 = SIMD_FNMA(Z_PC, t16, t25);
            t26 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t16);
            t26 = SIMD_FNMA(Z_PC, t17, t26);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t20);
            t30 = SIMD_FNMA(Z_PC, t21, t30);
            tx = SIMD_SUB(t10, t11);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t30 = SIMD_FMA(tx, ty, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t21);
            t31 = SIMD_FNMA(Z_PC, t22, t31);
            tx = SIMD_SUB(t11, t12);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t31 = SIMD_FMA(tx, ty, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t22);
            t32 = SIMD_FNMA(Z_PC, t23, t32);
            tx = SIMD_SUB(t12, t13);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t32 = SIMD_FMA(tx, ty, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t23);
            t33 = SIMD_FNMA(Z_PC, t24, t33);
            tx = SIMD_SUB(t13, t14);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t33 = SIMD_FMA(tx, ty, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t24);
            t34 = SIMD_FNMA(Z_PC, t25, t34);
            tx = SIMD_SUB(t14, t15);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t34 = SIMD_FMA(tx, ty, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t25);
            t35 = SIMD_FNMA(Z_PC, t26, t35);
            tx = SIMD_SUB(t15, t16);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t35 = SIMD_FMA(tx, ty, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 13 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 13 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 34 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 34 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 62 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 62 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 98 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 98 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 143 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 143 * NPTS_LOCAL + p_inner), tx);
            t10 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t00);
            t10 = SIMD_FNMA(Z_PC, t01, t10);
            t11 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t01);
            t11 = SIMD_FNMA(Z_PC, t02, t11);
            t12 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t02);
            t12 = SIMD_FNMA(Z_PC, t03, t12);
            t13 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t03);
            t13 = SIMD_FNMA(Z_PC, t04, t13);
            t14 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t04);
            t14 = SIMD_FNMA(Z_PC, t05, t14);
            t15 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t05);
            t15 = SIMD_FNMA(Z_PC, t06, t15);
            t16 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t06);
            t16 = SIMD_FNMA(Z_PC, t07, t16);
            t17 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t07);
            t17 = SIMD_FNMA(Z_PC, t08, t17);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t10);
            t20 = SIMD_FNMA(Z_PC, t11, t20);
            tx = SIMD_SUB(t00, t01);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t20 = SIMD_FMA(tx, ty, t20);
            t21 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t11);
            t21 = SIMD_FNMA(Z_PC, t12, t21);
            tx = SIMD_SUB(t01, t02);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t21 = SIMD_FMA(tx, ty, t21);
            t22 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t12);
            t22 = SIMD_FNMA(Z_PC, t13, t22);
            tx = SIMD_SUB(t02, t03);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t22 = SIMD_FMA(tx, ty, t22);
            t23 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t13);
            t23 = SIMD_FNMA(Z_PC, t14, t23);
            tx = SIMD_SUB(t03, t04);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t23 = SIMD_FMA(tx, ty, t23);
            t24 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t14);
            t24 = SIMD_FNMA(Z_PC, t15, t24);
            tx = SIMD_SUB(t04, t05);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t24 = SIMD_FMA(tx, ty, t24);
            t25 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t15);
            t25 = SIMD_FNMA(Z_PC, t16, t25);
            tx = SIMD_SUB(t05, t06);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t25 = SIMD_FMA(tx, ty, t25);
            t26 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t16);
            t26 = SIMD_FNMA(Z_PC, t17, t26);
            tx = SIMD_SUB(t06, t07);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t26 = SIMD_FMA(tx, ty, t26);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t20);
            t30 = SIMD_FNMA(Z_PC, t21, t30);
            tx = SIMD_SUB(t10, t11);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t30 = SIMD_FMA(tx, ty, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t21);
            t31 = SIMD_FNMA(Z_PC, t22, t31);
            tx = SIMD_SUB(t11, t12);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t31 = SIMD_FMA(tx, ty, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t22);
            t32 = SIMD_FNMA(Z_PC, t23, t32);
            tx = SIMD_SUB(t12, t13);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t32 = SIMD_FMA(tx, ty, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t23);
            t33 = SIMD_FNMA(Z_PC, t24, t33);
            tx = SIMD_SUB(t13, t14);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t33 = SIMD_FMA(tx, ty, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t24);
            t34 = SIMD_FNMA(Z_PC, t25, t34);
            tx = SIMD_SUB(t14, t15);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t34 = SIMD_FMA(tx, ty, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t25);
            t35 = SIMD_FNMA(Z_PC, t26, t35);
            tx = SIMD_SUB(t15, t16);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t35 = SIMD_FMA(tx, ty, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 14 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 14 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 35 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 35 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 63 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 63 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 99 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 99 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 7);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 144 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 144 * NPTS_LOCAL + p_inner), tx);
         }
      }

      for(size_t p_inner = 0; p_inner < NPTS_LOCAL; p_inner += SIMD_LENGTH) {
         double *Xik = (Xi + p_outer + p_inner);
         double *Gik = (Gi + p_outer + p_inner);

         SIMD_TYPE tx, wg, xik, gik;
         tx  = SIMD_ALIGNED_LOAD((temp + 100 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 101 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 102 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 103 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 104 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 105 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 106 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 109 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 101 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 103 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 104 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 106 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 115 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 102 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 104 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 105 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 109 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 120 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 103 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 106 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 115 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 121 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 104 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 105 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 109 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 120 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 127 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 106 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 115 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 121 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 128 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 129 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 129 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 134 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 109 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 120 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 127 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 134 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 135 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 115 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 121 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 128 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 129 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 136 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 137 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 138 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 139 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 129 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 137 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 138 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 139 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 141 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 138 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 139 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 141 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 142 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 134 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 139 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 141 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 142 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 143 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 120 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 127 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 134 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 135 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 141 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 142 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 143 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 144 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
      }
   }

   // cleanup code
   for(; p_outer < npts; p_outer += NPTS_LOCAL) {
      size_t npts_inner = std::min((size_t) NPTS_LOCAL, npts - p_outer);
      double *_point_outer = (_points + p_outer);

      double xA = rA.x;
      double yA = rA.y;
      double zA = rA.z;

      for(int i = 0; i < 145 * NPTS_LOCAL; i += SIMD_LENGTH) SIMD_ALIGNED_STORE((temp + i), SIMD_ZERO());

      for(int ij = 0; ij < nprim_pairs; ++ij) {
         double RHO = prim_pairs[ij].gamma;
         double RHO_INV = 1.0 / RHO;

         constexpr double X_PA = 0.0;
         constexpr double Y_PA = 0.0;
         constexpr double Z_PA = 0.0;

         double eval = prim_pairs[ij].K_coeff_prod;

         // Evaluate T Values
         size_t npts_inner_upper = SIMD_LENGTH * (npts_inner / SIMD_LENGTH);
         size_t p_inner = 0;
         for(p_inner = 0; p_inner < npts_inner_upper; p_inner += SIMD_LENGTH) {
            SIMD_TYPE xC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 0 * npts));
            SIMD_TYPE yC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 1 * npts));
            SIMD_TYPE zC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 2 * npts));

            SIMD_TYPE X_PC = SIMD_SUB(SIMD_DUPLICATE(&(xA)), xC);
            SIMD_TYPE Y_PC = SIMD_SUB(SIMD_DUPLICATE(&(yA)), yC);
            SIMD_TYPE Z_PC = SIMD_SUB(SIMD_DUPLICATE(&(zA)), zC);

            X_PC = SIMD_MUL(X_PC, X_PC);
            X_PC = SIMD_FMA(Y_PC, Y_PC, X_PC);
            X_PC = SIMD_FMA(Z_PC, Z_PC, X_PC);
            X_PC = SIMD_MUL(SIMD_DUPLICATE(&(RHO)), X_PC);
            SIMD_ALIGNED_STORE((Tval + p_inner), X_PC);
         }

         for(; p_inner < npts_inner; p_inner += SCALAR_LENGTH) {
            SCALAR_TYPE xC = SCALAR_LOAD((_point_outer + p_inner + 0 * npts));
            SCALAR_TYPE yC = SCALAR_LOAD((_point_outer + p_inner + 1 * npts));
            SCALAR_TYPE zC = SCALAR_LOAD((_point_outer + p_inner + 2 * npts));

            SCALAR_TYPE X_PC = SCALAR_SUB(SCALAR_DUPLICATE(&(xA)), xC);
            SCALAR_TYPE Y_PC = SCALAR_SUB(SCALAR_DUPLICATE(&(yA)), yC);
            SCALAR_TYPE Z_PC = SCALAR_SUB(SCALAR_DUPLICATE(&(zA)), zC);

            X_PC = SCALAR_MUL(X_PC, X_PC);
            X_PC = SCALAR_FMA(Y_PC, Y_PC, X_PC);
            X_PC = SCALAR_FMA(Z_PC, Z_PC, X_PC);
            X_PC = SCALAR_MUL(SCALAR_DUPLICATE(&(RHO)), X_PC);
            SCALAR_STORE((Tval + p_inner), X_PC);
         }

         // Evaluate Boys function
         boys_elements<8>(npts_inner, Tval, Tval_inv_e, FmT, boys_table);

         // Evaluate VRR Buffer
         p_inner = 0;
         for(p_inner = 0; p_inner < npts_inner_upper; p_inner += SIMD_LENGTH) {
            SIMD_TYPE xC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 0 * npts));
            SIMD_TYPE yC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 1 * npts));
            SIMD_TYPE zC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 2 * npts));

            SIMD_TYPE X_PC = SIMD_SUB(SIMD_DUPLICATE(&(xA)), xC);
            SIMD_TYPE Y_PC = SIMD_SUB(SIMD_DUPLICATE(&(yA)), yC);
            SIMD_TYPE Z_PC = SIMD_SUB(SIMD_DUPLICATE(&(zA)), zC);

            SIMD_TYPE tval, tval_inv_e, tx, ty, t00, t01, t02, t03, t04, t05, t06, t07, t08, t10, t11, t12, t13, t14, t15, t16, t17, t20, t21, t22, t23, t24, t25, t26, t30, t31, t32, t33, t34, t35, t40, t41, t42, t43, t44, t50, t51, t52, t53, t60, t61, t62, t70, t71, t80;

            tval = SIMD_ALIGNED_LOAD((Tval + p_inner));
            tval_inv_e = SIMD_ALIGNED_LOAD((Tval_inv_e + p_inner));

            t08 = SIMD_ALIGNED_LOAD((FmT + p_inner));
            t07 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t08), tval_inv_e), SIMD_SET1(0.13333333333333333148));
            t06 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t07), tval_inv_e), SIMD_SET1(0.15384615384615385469));
            t05 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t06), tval_inv_e), SIMD_SET1(0.18181818181818182323));
            t04 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t05), tval_inv_e), SIMD_SET1(0.22222222222222220989));
            t03 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t04), tval_inv_e), SIMD_SET1(0.28571428571428569843));
            t02 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t03), tval_inv_e), SIMD_SET1(0.40000000000000002220));
            t01 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t02), tval_inv_e), SIMD_SET1(0.66666666666666662966));
            t00 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t01), tval_inv_e), SIMD_SET1(2.00000000000000000000));

            t00 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t00);
            t01 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t01);
            t02 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t02);
            t03 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t03);
            t04 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t04);
            t05 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t05);
            t06 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t06);
            t07 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t07);
            t08 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t08);
            t10 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t00);
            t10 = SIMD_FNMA(X_PC, t01, t10);
            t11 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t01);
            t11 = SIMD_FNMA(X_PC, t02, t11);
            t12 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t02);
            t12 = SIMD_FNMA(X_PC, t03, t12);
            t13 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t03);
            t13 = SIMD_FNMA(X_PC, t04, t13);
            t14 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t04);
            t14 = SIMD_FNMA(X_PC, t05, t14);
            t15 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t05);
            t15 = SIMD_FNMA(X_PC, t06, t15);
            t16 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t06);
            t16 = SIMD_FNMA(X_PC, t07, t16);
            t17 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t07);
            t17 = SIMD_FNMA(X_PC, t08, t17);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t10);
            t20 = SIMD_FNMA(X_PC, t11, t20);
            tx = SIMD_SUB(t00, t01);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t20 = SIMD_FMA(tx, ty, t20);
            t21 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t11);
            t21 = SIMD_FNMA(X_PC, t12, t21);
            tx = SIMD_SUB(t01, t02);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t21 = SIMD_FMA(tx, ty, t21);
            t22 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t12);
            t22 = SIMD_FNMA(X_PC, t13, t22);
            tx = SIMD_SUB(t02, t03);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t22 = SIMD_FMA(tx, ty, t22);
            t23 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t13);
            t23 = SIMD_FNMA(X_PC, t14, t23);
            tx = SIMD_SUB(t03, t04);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t23 = SIMD_FMA(tx, ty, t23);
            t24 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t14);
            t24 = SIMD_FNMA(X_PC, t15, t24);
            tx = SIMD_SUB(t04, t05);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t24 = SIMD_FMA(tx, ty, t24);
            t25 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t15);
            t25 = SIMD_FNMA(X_PC, t16, t25);
            tx = SIMD_SUB(t05, t06);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t25 = SIMD_FMA(tx, ty, t25);
            t26 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t16);
            t26 = SIMD_FNMA(X_PC, t17, t26);
            tx = SIMD_SUB(t06, t07);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t26 = SIMD_FMA(tx, ty, t26);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t20);
            t30 = SIMD_FNMA(X_PC, t21, t30);
            tx = SIMD_SUB(t10, t11);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t30 = SIMD_FMA(tx, ty, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t21);
            t31 = SIMD_FNMA(X_PC, t22, t31);
            tx = SIMD_SUB(t11, t12);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t31 = SIMD_FMA(tx, ty, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t22);
            t32 = SIMD_FNMA(X_PC, t23, t32);
            tx = SIMD_SUB(t12, t13);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t32 = SIMD_FMA(tx, ty, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t23);
            t33 = SIMD_FNMA(X_PC, t24, t33);
            tx = SIMD_SUB(t13, t14);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t33 = SIMD_FMA(tx, ty, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t24);
            t34 = SIMD_FNMA(X_PC, t25, t34);
            tx = SIMD_SUB(t14, t15);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t34 = SIMD_FMA(tx, ty, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t25);
            t35 = SIMD_FNMA(X_PC, t26, t35);
            tx = SIMD_SUB(t15, t16);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t35 = SIMD_FMA(tx, ty, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t30);
            t40 = SIMD_FNMA(X_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t31);
            t41 = SIMD_FNMA(X_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t32);
            t42 = SIMD_FNMA(X_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t33);
            t43 = SIMD_FNMA(X_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t34);
            t44 = SIMD_FNMA(X_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 0 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t40);
            t50 = SIMD_FNMA(X_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t41);
            t51 = SIMD_FNMA(X_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t42);
            t52 = SIMD_FNMA(X_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t43);
            t53 = SIMD_FNMA(X_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 15 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 15 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t50);
            t60 = SIMD_FNMA(X_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t51);
            t61 = SIMD_FNMA(X_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t52);
            t62 = SIMD_FNMA(X_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 36 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 36 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t60);
            t70 = SIMD_FNMA(X_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t61);
            t71 = SIMD_FNMA(X_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 64 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 64 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t70);
            t80 = SIMD_FNMA(X_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 7);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 100 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 100 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t70);
            t80 = SIMD_FNMA(Y_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 101 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 101 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 102 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 102 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t60);
            t70 = SIMD_FNMA(Y_PC, t61, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t61);
            t71 = SIMD_FNMA(Y_PC, t62, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 65 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 65 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t70);
            t80 = SIMD_FNMA(Y_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 103 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 103 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 104 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 104 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 66 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 66 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 105 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 105 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t50);
            t60 = SIMD_FNMA(Y_PC, t51, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t51);
            t61 = SIMD_FNMA(Y_PC, t52, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t52);
            t62 = SIMD_FNMA(Y_PC, t53, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 37 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 37 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t60);
            t70 = SIMD_FNMA(Y_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t61);
            t71 = SIMD_FNMA(Y_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 67 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 67 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t70);
            t80 = SIMD_FNMA(Y_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 106 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 106 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 107 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 68 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 68 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 108 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 38 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 38 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 69 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 69 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 109 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 109 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t40);
            t50 = SIMD_FNMA(Y_PC, t41, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t41);
            t51 = SIMD_FNMA(Y_PC, t42, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t42);
            t52 = SIMD_FNMA(Y_PC, t43, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t43);
            t53 = SIMD_FNMA(Y_PC, t44, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 16 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 16 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t50);
            t60 = SIMD_FNMA(Y_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t51);
            t61 = SIMD_FNMA(Y_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t52);
            t62 = SIMD_FNMA(Y_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 39 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 39 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t60);
            t70 = SIMD_FNMA(Y_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t61);
            t71 = SIMD_FNMA(Y_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 70 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 70 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t70);
            t80 = SIMD_FNMA(Y_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 110 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 111 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 71 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 71 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 112 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 40 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 40 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 72 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 72 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 113 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 17 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 17 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 41 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 41 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 73 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 73 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 114 * NPTS_LOCAL + p_inner), tx);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t30);
            t40 = SIMD_FNMA(Y_PC, t31, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t31);
            t41 = SIMD_FNMA(Y_PC, t32, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t32);
            t42 = SIMD_FNMA(Y_PC, t33, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t33);
            t43 = SIMD_FNMA(Y_PC, t34, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t34);
            t44 = SIMD_FNMA(Y_PC, t35, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 1 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 1 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t40);
            t50 = SIMD_FNMA(Y_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t41);
            t51 = SIMD_FNMA(Y_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t42);
            t52 = SIMD_FNMA(Y_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t43);
            t53 = SIMD_FNMA(Y_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 18 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 18 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t50);
            t60 = SIMD_FNMA(Y_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t51);
            t61 = SIMD_FNMA(Y_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t52);
            t62 = SIMD_FNMA(Y_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 42 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 42 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t60);
            t70 = SIMD_FNMA(Y_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t61);
            t71 = SIMD_FNMA(Y_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 74 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 74 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t70);
            t80 = SIMD_FNMA(Y_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 115 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 115 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 116 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 75 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 75 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 117 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 43 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 43 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 76 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 76 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 118 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 19 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 19 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 44 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 44 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 77 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 77 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 119 * NPTS_LOCAL + p_inner), tx);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 2 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 2 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 20 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 20 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 45 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 45 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 78 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 78 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 120 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 120 * NPTS_LOCAL + p_inner), tx);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t20);
            t30 = SIMD_FNMA(Y_PC, t21, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t21);
            t31 = SIMD_FNMA(Y_PC, t22, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t22);
            t32 = SIMD_FNMA(Y_PC, t23, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t23);
            t33 = SIMD_FNMA(Y_PC, t24, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t24);
            t34 = SIMD_FNMA(Y_PC, t25, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t25);
            t35 = SIMD_FNMA(Y_PC, t26, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t30);
            t40 = SIMD_FNMA(Y_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t31);
            t41 = SIMD_FNMA(Y_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t32);
            t42 = SIMD_FNMA(Y_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t33);
            t43 = SIMD_FNMA(Y_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t34);
            t44 = SIMD_FNMA(Y_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 3 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 3 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t40);
            t50 = SIMD_FNMA(Y_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t41);
            t51 = SIMD_FNMA(Y_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t42);
            t52 = SIMD_FNMA(Y_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t43);
            t53 = SIMD_FNMA(Y_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 21 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 21 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t50);
            t60 = SIMD_FNMA(Y_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t51);
            t61 = SIMD_FNMA(Y_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t52);
            t62 = SIMD_FNMA(Y_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 46 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 46 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t60);
            t70 = SIMD_FNMA(Y_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t61);
            t71 = SIMD_FNMA(Y_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 79 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 79 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t70);
            t80 = SIMD_FNMA(Y_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 121 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 121 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 122 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 80 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 80 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 123 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 47 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 47 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 81 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 81 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 124 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 22 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 22 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 48 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 48 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 82 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 82 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 125 * NPTS_LOCAL + p_inner), tx);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 4 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 4 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 23 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 23 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 49 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 49 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 83 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 83 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 126 * NPTS_LOCAL + p_inner), tx);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t20);
            t30 = SIMD_FNMA(Z_PC, t21, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t21);
            t31 = SIMD_FNMA(Z_PC, t22, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t22);
            t32 = SIMD_FNMA(Z_PC, t23, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t23);
            t33 = SIMD_FNMA(Z_PC, t24, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t24);
            t34 = SIMD_FNMA(Z_PC, t25, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t25);
            t35 = SIMD_FNMA(Z_PC, t26, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 5 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 5 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 24 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 24 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 50 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 50 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 84 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 84 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 127 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 127 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t10);
            t20 = SIMD_FNMA(Y_PC, t11, t20);
            t21 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t11);
            t21 = SIMD_FNMA(Y_PC, t12, t21);
            t22 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t12);
            t22 = SIMD_FNMA(Y_PC, t13, t22);
            t23 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t13);
            t23 = SIMD_FNMA(Y_PC, t14, t23);
            t24 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t14);
            t24 = SIMD_FNMA(Y_PC, t15, t24);
            t25 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t15);
            t25 = SIMD_FNMA(Y_PC, t16, t25);
            t26 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t16);
            t26 = SIMD_FNMA(Y_PC, t17, t26);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t20);
            t30 = SIMD_FNMA(Y_PC, t21, t30);
            tx = SIMD_SUB(t10, t11);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t30 = SIMD_FMA(tx, ty, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t21);
            t31 = SIMD_FNMA(Y_PC, t22, t31);
            tx = SIMD_SUB(t11, t12);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t31 = SIMD_FMA(tx, ty, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t22);
            t32 = SIMD_FNMA(Y_PC, t23, t32);
            tx = SIMD_SUB(t12, t13);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t32 = SIMD_FMA(tx, ty, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t23);
            t33 = SIMD_FNMA(Y_PC, t24, t33);
            tx = SIMD_SUB(t13, t14);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t33 = SIMD_FMA(tx, ty, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t24);
            t34 = SIMD_FNMA(Y_PC, t25, t34);
            tx = SIMD_SUB(t14, t15);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t34 = SIMD_FMA(tx, ty, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t25);
            t35 = SIMD_FNMA(Y_PC, t26, t35);
            tx = SIMD_SUB(t15, t16);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t35 = SIMD_FMA(tx, ty, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t30);
            t40 = SIMD_FNMA(Y_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t31);
            t41 = SIMD_FNMA(Y_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t32);
            t42 = SIMD_FNMA(Y_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t33);
            t43 = SIMD_FNMA(Y_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t34);
            t44 = SIMD_FNMA(Y_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 6 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 6 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t40);
            t50 = SIMD_FNMA(Y_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t41);
            t51 = SIMD_FNMA(Y_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t42);
            t52 = SIMD_FNMA(Y_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t43);
            t53 = SIMD_FNMA(Y_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 25 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 25 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t50);
            t60 = SIMD_FNMA(Y_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t51);
            t61 = SIMD_FNMA(Y_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t52);
            t62 = SIMD_FNMA(Y_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 51 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 51 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t60);
            t70 = SIMD_FNMA(Y_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t61);
            t71 = SIMD_FNMA(Y_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 85 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 85 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t70);
            t80 = SIMD_FNMA(Y_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 128 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 128 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 129 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 129 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 86 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 86 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 130 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 52 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 52 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 87 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 87 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 131 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 26 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 26 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 53 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 53 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 88 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 88 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 132 * NPTS_LOCAL + p_inner), tx);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 7 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 7 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 27 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 27 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 54 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 54 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 89 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 89 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 133 * NPTS_LOCAL + p_inner), tx);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t20);
            t30 = SIMD_FNMA(Z_PC, t21, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t21);
            t31 = SIMD_FNMA(Z_PC, t22, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t22);
            t32 = SIMD_FNMA(Z_PC, t23, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t23);
            t33 = SIMD_FNMA(Z_PC, t24, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t24);
            t34 = SIMD_FNMA(Z_PC, t25, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t25);
            t35 = SIMD_FNMA(Z_PC, t26, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 8 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 8 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 28 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 28 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 55 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 55 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 90 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 90 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 134 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 134 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t10);
            t20 = SIMD_FNMA(Z_PC, t11, t20);
            t21 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t11);
            t21 = SIMD_FNMA(Z_PC, t12, t21);
            t22 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t12);
            t22 = SIMD_FNMA(Z_PC, t13, t22);
            t23 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t13);
            t23 = SIMD_FNMA(Z_PC, t14, t23);
            t24 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t14);
            t24 = SIMD_FNMA(Z_PC, t15, t24);
            t25 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t15);
            t25 = SIMD_FNMA(Z_PC, t16, t25);
            t26 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t16);
            t26 = SIMD_FNMA(Z_PC, t17, t26);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t20);
            t30 = SIMD_FNMA(Z_PC, t21, t30);
            tx = SIMD_SUB(t10, t11);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t30 = SIMD_FMA(tx, ty, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t21);
            t31 = SIMD_FNMA(Z_PC, t22, t31);
            tx = SIMD_SUB(t11, t12);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t31 = SIMD_FMA(tx, ty, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t22);
            t32 = SIMD_FNMA(Z_PC, t23, t32);
            tx = SIMD_SUB(t12, t13);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t32 = SIMD_FMA(tx, ty, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t23);
            t33 = SIMD_FNMA(Z_PC, t24, t33);
            tx = SIMD_SUB(t13, t14);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t33 = SIMD_FMA(tx, ty, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t24);
            t34 = SIMD_FNMA(Z_PC, t25, t34);
            tx = SIMD_SUB(t14, t15);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t34 = SIMD_FMA(tx, ty, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t25);
            t35 = SIMD_FNMA(Z_PC, t26, t35);
            tx = SIMD_SUB(t15, t16);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t35 = SIMD_FMA(tx, ty, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 9 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 9 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 29 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 29 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 56 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 56 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 91 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 91 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 135 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 135 * NPTS_LOCAL + p_inner), tx);
            t10 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t00);
            t10 = SIMD_FNMA(Y_PC, t01, t10);
            t11 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t01);
            t11 = SIMD_FNMA(Y_PC, t02, t11);
            t12 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t02);
            t12 = SIMD_FNMA(Y_PC, t03, t12);
            t13 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t03);
            t13 = SIMD_FNMA(Y_PC, t04, t13);
            t14 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t04);
            t14 = SIMD_FNMA(Y_PC, t05, t14);
            t15 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t05);
            t15 = SIMD_FNMA(Y_PC, t06, t15);
            t16 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t06);
            t16 = SIMD_FNMA(Y_PC, t07, t16);
            t17 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t07);
            t17 = SIMD_FNMA(Y_PC, t08, t17);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t10);
            t20 = SIMD_FNMA(Y_PC, t11, t20);
            tx = SIMD_SUB(t00, t01);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t20 = SIMD_FMA(tx, ty, t20);
            t21 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t11);
            t21 = SIMD_FNMA(Y_PC, t12, t21);
            tx = SIMD_SUB(t01, t02);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t21 = SIMD_FMA(tx, ty, t21);
            t22 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t12);
            t22 = SIMD_FNMA(Y_PC, t13, t22);
            tx = SIMD_SUB(t02, t03);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t22 = SIMD_FMA(tx, ty, t22);
            t23 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t13);
            t23 = SIMD_FNMA(Y_PC, t14, t23);
            tx = SIMD_SUB(t03, t04);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t23 = SIMD_FMA(tx, ty, t23);
            t24 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t14);
            t24 = SIMD_FNMA(Y_PC, t15, t24);
            tx = SIMD_SUB(t04, t05);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t24 = SIMD_FMA(tx, ty, t24);
            t25 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t15);
            t25 = SIMD_FNMA(Y_PC, t16, t25);
            tx = SIMD_SUB(t05, t06);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t25 = SIMD_FMA(tx, ty, t25);
            t26 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t16);
            t26 = SIMD_FNMA(Y_PC, t17, t26);
            tx = SIMD_SUB(t06, t07);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t26 = SIMD_FMA(tx, ty, t26);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t20);
            t30 = SIMD_FNMA(Y_PC, t21, t30);
            tx = SIMD_SUB(t10, t11);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t30 = SIMD_FMA(tx, ty, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t21);
            t31 = SIMD_FNMA(Y_PC, t22, t31);
            tx = SIMD_SUB(t11, t12);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t31 = SIMD_FMA(tx, ty, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t22);
            t32 = SIMD_FNMA(Y_PC, t23, t32);
            tx = SIMD_SUB(t12, t13);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t32 = SIMD_FMA(tx, ty, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t23);
            t33 = SIMD_FNMA(Y_PC, t24, t33);
            tx = SIMD_SUB(t13, t14);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t33 = SIMD_FMA(tx, ty, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t24);
            t34 = SIMD_FNMA(Y_PC, t25, t34);
            tx = SIMD_SUB(t14, t15);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t34 = SIMD_FMA(tx, ty, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t25);
            t35 = SIMD_FNMA(Y_PC, t26, t35);
            tx = SIMD_SUB(t15, t16);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t35 = SIMD_FMA(tx, ty, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t30);
            t40 = SIMD_FNMA(Y_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t31);
            t41 = SIMD_FNMA(Y_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t32);
            t42 = SIMD_FNMA(Y_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t33);
            t43 = SIMD_FNMA(Y_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t34);
            t44 = SIMD_FNMA(Y_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 10 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 10 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t40);
            t50 = SIMD_FNMA(Y_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t41);
            t51 = SIMD_FNMA(Y_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t42);
            t52 = SIMD_FNMA(Y_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t43);
            t53 = SIMD_FNMA(Y_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 30 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 30 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t50);
            t60 = SIMD_FNMA(Y_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t51);
            t61 = SIMD_FNMA(Y_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t52);
            t62 = SIMD_FNMA(Y_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 57 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 57 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t60);
            t70 = SIMD_FNMA(Y_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t61);
            t71 = SIMD_FNMA(Y_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 92 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 92 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t70);
            t80 = SIMD_FNMA(Y_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 7);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 136 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 136 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 137 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 137 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 93 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 93 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 138 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 138 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 58 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 58 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 94 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 94 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 139 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 139 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 31 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 31 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 59 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 59 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 95 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 95 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 140 * NPTS_LOCAL + p_inner), tx);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 11 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 11 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 32 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 32 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 60 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 60 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 96 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 96 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 141 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 141 * NPTS_LOCAL + p_inner), tx);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t20);
            t30 = SIMD_FNMA(Z_PC, t21, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t21);
            t31 = SIMD_FNMA(Z_PC, t22, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t22);
            t32 = SIMD_FNMA(Z_PC, t23, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t23);
            t33 = SIMD_FNMA(Z_PC, t24, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t24);
            t34 = SIMD_FNMA(Z_PC, t25, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t25);
            t35 = SIMD_FNMA(Z_PC, t26, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 12 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 12 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 33 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 33 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 61 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 61 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 97 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 97 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 142 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 142 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t10);
            t20 = SIMD_FNMA(Z_PC, t11, t20);
            t21 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t11);
            t21 = SIMD_FNMA(Z_PC, t12, t21);
            t22 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t12);
            t22 = SIMD_FNMA(Z_PC, t13, t22);
            t23 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t13);
            t23 = SIMD_FNMA(Z_PC, t14, t23);
            t24 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t14);
            t24 = SIMD_FNMA(Z_PC, t15, t24);
            t25 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t15);
            t25 = SIMD_FNMA(Z_PC, t16, t25);
            t26 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t16);
            t26 = SIMD_FNMA(Z_PC, t17, t26);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t20);
            t30 = SIMD_FNMA(Z_PC, t21, t30);
            tx = SIMD_SUB(t10, t11);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t30 = SIMD_FMA(tx, ty, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t21);
            t31 = SIMD_FNMA(Z_PC, t22, t31);
            tx = SIMD_SUB(t11, t12);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t31 = SIMD_FMA(tx, ty, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t22);
            t32 = SIMD_FNMA(Z_PC, t23, t32);
            tx = SIMD_SUB(t12, t13);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t32 = SIMD_FMA(tx, ty, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t23);
            t33 = SIMD_FNMA(Z_PC, t24, t33);
            tx = SIMD_SUB(t13, t14);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t33 = SIMD_FMA(tx, ty, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t24);
            t34 = SIMD_FNMA(Z_PC, t25, t34);
            tx = SIMD_SUB(t14, t15);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t34 = SIMD_FMA(tx, ty, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t25);
            t35 = SIMD_FNMA(Z_PC, t26, t35);
            tx = SIMD_SUB(t15, t16);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t35 = SIMD_FMA(tx, ty, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 13 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 13 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 34 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 34 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 62 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 62 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 98 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 98 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 143 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 143 * NPTS_LOCAL + p_inner), tx);
            t10 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t00);
            t10 = SIMD_FNMA(Z_PC, t01, t10);
            t11 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t01);
            t11 = SIMD_FNMA(Z_PC, t02, t11);
            t12 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t02);
            t12 = SIMD_FNMA(Z_PC, t03, t12);
            t13 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t03);
            t13 = SIMD_FNMA(Z_PC, t04, t13);
            t14 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t04);
            t14 = SIMD_FNMA(Z_PC, t05, t14);
            t15 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t05);
            t15 = SIMD_FNMA(Z_PC, t06, t15);
            t16 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t06);
            t16 = SIMD_FNMA(Z_PC, t07, t16);
            t17 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t07);
            t17 = SIMD_FNMA(Z_PC, t08, t17);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t10);
            t20 = SIMD_FNMA(Z_PC, t11, t20);
            tx = SIMD_SUB(t00, t01);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t20 = SIMD_FMA(tx, ty, t20);
            t21 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t11);
            t21 = SIMD_FNMA(Z_PC, t12, t21);
            tx = SIMD_SUB(t01, t02);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t21 = SIMD_FMA(tx, ty, t21);
            t22 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t12);
            t22 = SIMD_FNMA(Z_PC, t13, t22);
            tx = SIMD_SUB(t02, t03);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t22 = SIMD_FMA(tx, ty, t22);
            t23 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t13);
            t23 = SIMD_FNMA(Z_PC, t14, t23);
            tx = SIMD_SUB(t03, t04);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t23 = SIMD_FMA(tx, ty, t23);
            t24 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t14);
            t24 = SIMD_FNMA(Z_PC, t15, t24);
            tx = SIMD_SUB(t04, t05);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t24 = SIMD_FMA(tx, ty, t24);
            t25 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t15);
            t25 = SIMD_FNMA(Z_PC, t16, t25);
            tx = SIMD_SUB(t05, t06);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t25 = SIMD_FMA(tx, ty, t25);
            t26 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t16);
            t26 = SIMD_FNMA(Z_PC, t17, t26);
            tx = SIMD_SUB(t06, t07);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t26 = SIMD_FMA(tx, ty, t26);
            t30 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t20);
            t30 = SIMD_FNMA(Z_PC, t21, t30);
            tx = SIMD_SUB(t10, t11);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t30 = SIMD_FMA(tx, ty, t30);
            t31 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t21);
            t31 = SIMD_FNMA(Z_PC, t22, t31);
            tx = SIMD_SUB(t11, t12);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t31 = SIMD_FMA(tx, ty, t31);
            t32 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t22);
            t32 = SIMD_FNMA(Z_PC, t23, t32);
            tx = SIMD_SUB(t12, t13);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t32 = SIMD_FMA(tx, ty, t32);
            t33 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t23);
            t33 = SIMD_FNMA(Z_PC, t24, t33);
            tx = SIMD_SUB(t13, t14);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t33 = SIMD_FMA(tx, ty, t33);
            t34 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t24);
            t34 = SIMD_FNMA(Z_PC, t25, t34);
            tx = SIMD_SUB(t14, t15);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t34 = SIMD_FMA(tx, ty, t34);
            t35 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t25);
            t35 = SIMD_FNMA(Z_PC, t26, t35);
            tx = SIMD_SUB(t15, t16);
            ty = SIMD_SET1(0.5 * 2);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t35 = SIMD_FMA(tx, ty, t35);
            t40 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t30);
            t40 = SIMD_FNMA(Z_PC, t31, t40);
            tx = SIMD_SUB(t20, t21);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t40 = SIMD_FMA(tx, ty, t40);
            t41 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t31);
            t41 = SIMD_FNMA(Z_PC, t32, t41);
            tx = SIMD_SUB(t21, t22);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t41 = SIMD_FMA(tx, ty, t41);
            t42 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t32);
            t42 = SIMD_FNMA(Z_PC, t33, t42);
            tx = SIMD_SUB(t22, t23);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t42 = SIMD_FMA(tx, ty, t42);
            t43 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t33);
            t43 = SIMD_FNMA(Z_PC, t34, t43);
            tx = SIMD_SUB(t23, t24);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t43 = SIMD_FMA(tx, ty, t43);
            t44 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t34);
            t44 = SIMD_FNMA(Z_PC, t35, t44);
            tx = SIMD_SUB(t24, t25);
            ty = SIMD_SET1(0.5 * 3);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t44 = SIMD_FMA(tx, ty, t44);
            tx = SIMD_ALIGNED_LOAD((temp + 14 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t40);
            SIMD_ALIGNED_STORE((temp + 14 * NPTS_LOCAL + p_inner), tx);
            t50 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t40);
            t50 = SIMD_FNMA(Z_PC, t41, t50);
            tx = SIMD_SUB(t30, t31);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t50 = SIMD_FMA(tx, ty, t50);
            t51 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t41);
            t51 = SIMD_FNMA(Z_PC, t42, t51);
            tx = SIMD_SUB(t31, t32);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t51 = SIMD_FMA(tx, ty, t51);
            t52 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t42);
            t52 = SIMD_FNMA(Z_PC, t43, t52);
            tx = SIMD_SUB(t32, t33);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t52 = SIMD_FMA(tx, ty, t52);
            t53 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t43);
            t53 = SIMD_FNMA(Z_PC, t44, t53);
            tx = SIMD_SUB(t33, t34);
            ty = SIMD_SET1(0.5 * 4);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t53 = SIMD_FMA(tx, ty, t53);
            tx = SIMD_ALIGNED_LOAD((temp + 35 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t50);
            SIMD_ALIGNED_STORE((temp + 35 * NPTS_LOCAL + p_inner), tx);
            t60 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t50);
            t60 = SIMD_FNMA(Z_PC, t51, t60);
            tx = SIMD_SUB(t40, t41);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t60 = SIMD_FMA(tx, ty, t60);
            t61 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t51);
            t61 = SIMD_FNMA(Z_PC, t52, t61);
            tx = SIMD_SUB(t41, t42);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t61 = SIMD_FMA(tx, ty, t61);
            t62 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t52);
            t62 = SIMD_FNMA(Z_PC, t53, t62);
            tx = SIMD_SUB(t42, t43);
            ty = SIMD_SET1(0.5 * 5);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t62 = SIMD_FMA(tx, ty, t62);
            tx = SIMD_ALIGNED_LOAD((temp + 63 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t60);
            SIMD_ALIGNED_STORE((temp + 63 * NPTS_LOCAL + p_inner), tx);
            t70 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t60);
            t70 = SIMD_FNMA(Z_PC, t61, t70);
            tx = SIMD_SUB(t50, t51);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t70 = SIMD_FMA(tx, ty, t70);
            t71 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t61);
            t71 = SIMD_FNMA(Z_PC, t62, t71);
            tx = SIMD_SUB(t51, t52);
            ty = SIMD_SET1(0.5 * 6);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t71 = SIMD_FMA(tx, ty, t71);
            tx = SIMD_ALIGNED_LOAD((temp + 99 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t70);
            SIMD_ALIGNED_STORE((temp + 99 * NPTS_LOCAL + p_inner), tx);
            t80 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t70);
            t80 = SIMD_FNMA(Z_PC, t71, t80);
            tx = SIMD_SUB(t60, t61);
            ty = SIMD_SET1(0.5 * 7);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t80 = SIMD_FMA(tx, ty, t80);
            tx = SIMD_ALIGNED_LOAD((temp + 144 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t80);
            SIMD_ALIGNED_STORE((temp + 144 * NPTS_LOCAL + p_inner), tx);
         }

         for(; p_inner < npts_inner; p_inner += SCALAR_LENGTH) {
            SCALAR_TYPE xC = SCALAR_LOAD((_point_outer + p_inner + 0 * npts));
            SCALAR_TYPE yC = SCALAR_LOAD((_point_outer + p_inner + 1 * npts));
            SCALAR_TYPE zC = SCALAR_LOAD((_point_outer + p_inner + 2 * npts));

            SCALAR_TYPE X_PC = SCALAR_SUB(SCALAR_DUPLICATE(&(xA)), xC);
            SCALAR_TYPE Y_PC = SCALAR_SUB(SCALAR_DUPLICATE(&(yA)), yC);
            SCALAR_TYPE Z_PC = SCALAR_SUB(SCALAR_DUPLICATE(&(zA)), zC);

            SCALAR_TYPE tval, tval_inv_e, tx, ty, t00, t01, t02, t03, t04, t05, t06, t07, t08, t10, t11, t12, t13, t14, t15, t16, t17, t20, t21, t22, t23, t24, t25, t26, t30, t31, t32, t33, t34, t35, t40, t41, t42, t43, t44, t50, t51, t52, t53, t60, t61, t62, t70, t71, t80;

            tval = SCALAR_LOAD((Tval + p_inner));
            tval_inv_e = SCALAR_LOAD((Tval_inv_e + p_inner));

            t08 = SCALAR_LOAD((FmT + p_inner));
            t07 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(tval, t08), tval_inv_e), SCALAR_SET1(0.13333333333333333148));
            t06 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(tval, t07), tval_inv_e), SCALAR_SET1(0.15384615384615385469));
            t05 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(tval, t06), tval_inv_e), SCALAR_SET1(0.18181818181818182323));
            t04 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(tval, t05), tval_inv_e), SCALAR_SET1(0.22222222222222220989));
            t03 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(tval, t04), tval_inv_e), SCALAR_SET1(0.28571428571428569843));
            t02 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(tval, t03), tval_inv_e), SCALAR_SET1(0.40000000000000002220));
            t01 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(tval, t02), tval_inv_e), SCALAR_SET1(0.66666666666666662966));
            t00 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(tval, t01), tval_inv_e), SCALAR_SET1(2.00000000000000000000));

            t00 = SCALAR_MUL(SCALAR_DUPLICATE(&(eval)), t00);
            t01 = SCALAR_MUL(SCALAR_DUPLICATE(&(eval)), t01);
            t02 = SCALAR_MUL(SCALAR_DUPLICATE(&(eval)), t02);
            t03 = SCALAR_MUL(SCALAR_DUPLICATE(&(eval)), t03);
            t04 = SCALAR_MUL(SCALAR_DUPLICATE(&(eval)), t04);
            t05 = SCALAR_MUL(SCALAR_DUPLICATE(&(eval)), t05);
            t06 = SCALAR_MUL(SCALAR_DUPLICATE(&(eval)), t06);
            t07 = SCALAR_MUL(SCALAR_DUPLICATE(&(eval)), t07);
            t08 = SCALAR_MUL(SCALAR_DUPLICATE(&(eval)), t08);
            t10 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t00);
            t10 = SCALAR_FNMA(X_PC, t01, t10);
            t11 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t01);
            t11 = SCALAR_FNMA(X_PC, t02, t11);
            t12 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t02);
            t12 = SCALAR_FNMA(X_PC, t03, t12);
            t13 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t03);
            t13 = SCALAR_FNMA(X_PC, t04, t13);
            t14 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t04);
            t14 = SCALAR_FNMA(X_PC, t05, t14);
            t15 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t05);
            t15 = SCALAR_FNMA(X_PC, t06, t15);
            t16 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t06);
            t16 = SCALAR_FNMA(X_PC, t07, t16);
            t17 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t07);
            t17 = SCALAR_FNMA(X_PC, t08, t17);
            t20 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t10);
            t20 = SCALAR_FNMA(X_PC, t11, t20);
            tx = SCALAR_SUB(t00, t01);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t20 = SCALAR_FMA(tx, ty, t20);
            t21 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t11);
            t21 = SCALAR_FNMA(X_PC, t12, t21);
            tx = SCALAR_SUB(t01, t02);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t21 = SCALAR_FMA(tx, ty, t21);
            t22 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t12);
            t22 = SCALAR_FNMA(X_PC, t13, t22);
            tx = SCALAR_SUB(t02, t03);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t22 = SCALAR_FMA(tx, ty, t22);
            t23 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t13);
            t23 = SCALAR_FNMA(X_PC, t14, t23);
            tx = SCALAR_SUB(t03, t04);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t23 = SCALAR_FMA(tx, ty, t23);
            t24 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t14);
            t24 = SCALAR_FNMA(X_PC, t15, t24);
            tx = SCALAR_SUB(t04, t05);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t24 = SCALAR_FMA(tx, ty, t24);
            t25 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t15);
            t25 = SCALAR_FNMA(X_PC, t16, t25);
            tx = SCALAR_SUB(t05, t06);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t25 = SCALAR_FMA(tx, ty, t25);
            t26 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t16);
            t26 = SCALAR_FNMA(X_PC, t17, t26);
            tx = SCALAR_SUB(t06, t07);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t26 = SCALAR_FMA(tx, ty, t26);
            t30 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t20);
            t30 = SCALAR_FNMA(X_PC, t21, t30);
            tx = SCALAR_SUB(t10, t11);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t30 = SCALAR_FMA(tx, ty, t30);
            t31 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t21);
            t31 = SCALAR_FNMA(X_PC, t22, t31);
            tx = SCALAR_SUB(t11, t12);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t31 = SCALAR_FMA(tx, ty, t31);
            t32 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t22);
            t32 = SCALAR_FNMA(X_PC, t23, t32);
            tx = SCALAR_SUB(t12, t13);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t32 = SCALAR_FMA(tx, ty, t32);
            t33 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t23);
            t33 = SCALAR_FNMA(X_PC, t24, t33);
            tx = SCALAR_SUB(t13, t14);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t33 = SCALAR_FMA(tx, ty, t33);
            t34 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t24);
            t34 = SCALAR_FNMA(X_PC, t25, t34);
            tx = SCALAR_SUB(t14, t15);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t34 = SCALAR_FMA(tx, ty, t34);
            t35 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t25);
            t35 = SCALAR_FNMA(X_PC, t26, t35);
            tx = SCALAR_SUB(t15, t16);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t35 = SCALAR_FMA(tx, ty, t35);
            t40 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t30);
            t40 = SCALAR_FNMA(X_PC, t31, t40);
            tx = SCALAR_SUB(t20, t21);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t40 = SCALAR_FMA(tx, ty, t40);
            t41 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t31);
            t41 = SCALAR_FNMA(X_PC, t32, t41);
            tx = SCALAR_SUB(t21, t22);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t41 = SCALAR_FMA(tx, ty, t41);
            t42 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t32);
            t42 = SCALAR_FNMA(X_PC, t33, t42);
            tx = SCALAR_SUB(t22, t23);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t42 = SCALAR_FMA(tx, ty, t42);
            t43 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t33);
            t43 = SCALAR_FNMA(X_PC, t34, t43);
            tx = SCALAR_SUB(t23, t24);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t43 = SCALAR_FMA(tx, ty, t43);
            t44 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t34);
            t44 = SCALAR_FNMA(X_PC, t35, t44);
            tx = SCALAR_SUB(t24, t25);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t44 = SCALAR_FMA(tx, ty, t44);
            tx = SCALAR_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t40);
            SCALAR_STORE((temp + 0 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t40);
            t50 = SCALAR_FNMA(X_PC, t41, t50);
            tx = SCALAR_SUB(t30, t31);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t50 = SCALAR_FMA(tx, ty, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t41);
            t51 = SCALAR_FNMA(X_PC, t42, t51);
            tx = SCALAR_SUB(t31, t32);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t51 = SCALAR_FMA(tx, ty, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t42);
            t52 = SCALAR_FNMA(X_PC, t43, t52);
            tx = SCALAR_SUB(t32, t33);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t52 = SCALAR_FMA(tx, ty, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t43);
            t53 = SCALAR_FNMA(X_PC, t44, t53);
            tx = SCALAR_SUB(t33, t34);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t53 = SCALAR_FMA(tx, ty, t53);
            tx = SCALAR_LOAD((temp + 15 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 15 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t50);
            t60 = SCALAR_FNMA(X_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t51);
            t61 = SCALAR_FNMA(X_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t52);
            t62 = SCALAR_FNMA(X_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 36 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 36 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t60);
            t70 = SCALAR_FNMA(X_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 6);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t61);
            t71 = SCALAR_FNMA(X_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 6);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 64 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 64 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t70);
            t80 = SCALAR_FNMA(X_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 7);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 100 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 100 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t70);
            t80 = SCALAR_FNMA(Y_PC, t71, t80);
            tx = SCALAR_LOAD((temp + 101 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 101 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_LOAD((temp + 102 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 102 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t60);
            t70 = SCALAR_FNMA(Y_PC, t61, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t61);
            t71 = SCALAR_FNMA(Y_PC, t62, t71);
            tx = SCALAR_LOAD((temp + 65 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 65 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t70);
            t80 = SCALAR_FNMA(Y_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 103 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 103 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_LOAD((temp + 104 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 104 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_LOAD((temp + 66 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 66 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 105 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 105 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t50);
            t60 = SCALAR_FNMA(Y_PC, t51, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t51);
            t61 = SCALAR_FNMA(Y_PC, t52, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t52);
            t62 = SCALAR_FNMA(Y_PC, t53, t62);
            tx = SCALAR_LOAD((temp + 37 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 37 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t60);
            t70 = SCALAR_FNMA(Y_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t61);
            t71 = SCALAR_FNMA(Y_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 67 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 67 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t70);
            t80 = SCALAR_FNMA(Y_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 106 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 106 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 107 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_LOAD((temp + 68 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 68 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 108 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_LOAD((temp + 38 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 38 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 69 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 69 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 109 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 109 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t40);
            t50 = SCALAR_FNMA(Y_PC, t41, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t41);
            t51 = SCALAR_FNMA(Y_PC, t42, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t42);
            t52 = SCALAR_FNMA(Y_PC, t43, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t43);
            t53 = SCALAR_FNMA(Y_PC, t44, t53);
            tx = SCALAR_LOAD((temp + 16 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 16 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t50);
            t60 = SCALAR_FNMA(Y_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t51);
            t61 = SCALAR_FNMA(Y_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t52);
            t62 = SCALAR_FNMA(Y_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 39 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 39 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t60);
            t70 = SCALAR_FNMA(Y_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t61);
            t71 = SCALAR_FNMA(Y_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 70 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 70 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t70);
            t80 = SCALAR_FNMA(Y_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 110 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 111 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_LOAD((temp + 71 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 71 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 112 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_LOAD((temp + 40 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 40 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 72 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 72 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 113 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t40);
            t50 = SCALAR_FNMA(Z_PC, t41, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t41);
            t51 = SCALAR_FNMA(Z_PC, t42, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t42);
            t52 = SCALAR_FNMA(Z_PC, t43, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t43);
            t53 = SCALAR_FNMA(Z_PC, t44, t53);
            tx = SCALAR_LOAD((temp + 17 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 17 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 41 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 41 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 73 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 73 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 114 * NPTS_LOCAL + p_inner), tx);
            t40 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t30);
            t40 = SCALAR_FNMA(Y_PC, t31, t40);
            t41 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t31);
            t41 = SCALAR_FNMA(Y_PC, t32, t41);
            t42 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t32);
            t42 = SCALAR_FNMA(Y_PC, t33, t42);
            t43 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t33);
            t43 = SCALAR_FNMA(Y_PC, t34, t43);
            t44 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t34);
            t44 = SCALAR_FNMA(Y_PC, t35, t44);
            tx = SCALAR_LOAD((temp + 1 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t40);
            SCALAR_STORE((temp + 1 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t40);
            t50 = SCALAR_FNMA(Y_PC, t41, t50);
            tx = SCALAR_SUB(t30, t31);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t50 = SCALAR_FMA(tx, ty, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t41);
            t51 = SCALAR_FNMA(Y_PC, t42, t51);
            tx = SCALAR_SUB(t31, t32);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t51 = SCALAR_FMA(tx, ty, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t42);
            t52 = SCALAR_FNMA(Y_PC, t43, t52);
            tx = SCALAR_SUB(t32, t33);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t52 = SCALAR_FMA(tx, ty, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t43);
            t53 = SCALAR_FNMA(Y_PC, t44, t53);
            tx = SCALAR_SUB(t33, t34);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t53 = SCALAR_FMA(tx, ty, t53);
            tx = SCALAR_LOAD((temp + 18 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 18 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t50);
            t60 = SCALAR_FNMA(Y_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t51);
            t61 = SCALAR_FNMA(Y_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t52);
            t62 = SCALAR_FNMA(Y_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 42 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 42 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t60);
            t70 = SCALAR_FNMA(Y_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t61);
            t71 = SCALAR_FNMA(Y_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 74 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 74 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t70);
            t80 = SCALAR_FNMA(Y_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 115 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 115 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 116 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_LOAD((temp + 75 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 75 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 117 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_LOAD((temp + 43 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 43 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 76 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 76 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 118 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t40);
            t50 = SCALAR_FNMA(Z_PC, t41, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t41);
            t51 = SCALAR_FNMA(Z_PC, t42, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t42);
            t52 = SCALAR_FNMA(Z_PC, t43, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t43);
            t53 = SCALAR_FNMA(Z_PC, t44, t53);
            tx = SCALAR_LOAD((temp + 19 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 19 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 44 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 44 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 77 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 77 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 119 * NPTS_LOCAL + p_inner), tx);
            t40 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t30);
            t40 = SCALAR_FNMA(Z_PC, t31, t40);
            t41 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t31);
            t41 = SCALAR_FNMA(Z_PC, t32, t41);
            t42 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t32);
            t42 = SCALAR_FNMA(Z_PC, t33, t42);
            t43 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t33);
            t43 = SCALAR_FNMA(Z_PC, t34, t43);
            t44 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t34);
            t44 = SCALAR_FNMA(Z_PC, t35, t44);
            tx = SCALAR_LOAD((temp + 2 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t40);
            SCALAR_STORE((temp + 2 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t40);
            t50 = SCALAR_FNMA(Z_PC, t41, t50);
            tx = SCALAR_SUB(t30, t31);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t50 = SCALAR_FMA(tx, ty, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t41);
            t51 = SCALAR_FNMA(Z_PC, t42, t51);
            tx = SCALAR_SUB(t31, t32);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t51 = SCALAR_FMA(tx, ty, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t42);
            t52 = SCALAR_FNMA(Z_PC, t43, t52);
            tx = SCALAR_SUB(t32, t33);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t52 = SCALAR_FMA(tx, ty, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t43);
            t53 = SCALAR_FNMA(Z_PC, t44, t53);
            tx = SCALAR_SUB(t33, t34);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t53 = SCALAR_FMA(tx, ty, t53);
            tx = SCALAR_LOAD((temp + 20 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 20 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 45 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 45 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 78 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 78 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 120 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 120 * NPTS_LOCAL + p_inner), tx);
            t30 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t20);
            t30 = SCALAR_FNMA(Y_PC, t21, t30);
            t31 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t21);
            t31 = SCALAR_FNMA(Y_PC, t22, t31);
            t32 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t22);
            t32 = SCALAR_FNMA(Y_PC, t23, t32);
            t33 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t23);
            t33 = SCALAR_FNMA(Y_PC, t24, t33);
            t34 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t24);
            t34 = SCALAR_FNMA(Y_PC, t25, t34);
            t35 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t25);
            t35 = SCALAR_FNMA(Y_PC, t26, t35);
            t40 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t30);
            t40 = SCALAR_FNMA(Y_PC, t31, t40);
            tx = SCALAR_SUB(t20, t21);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t40 = SCALAR_FMA(tx, ty, t40);
            t41 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t31);
            t41 = SCALAR_FNMA(Y_PC, t32, t41);
            tx = SCALAR_SUB(t21, t22);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t41 = SCALAR_FMA(tx, ty, t41);
            t42 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t32);
            t42 = SCALAR_FNMA(Y_PC, t33, t42);
            tx = SCALAR_SUB(t22, t23);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t42 = SCALAR_FMA(tx, ty, t42);
            t43 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t33);
            t43 = SCALAR_FNMA(Y_PC, t34, t43);
            tx = SCALAR_SUB(t23, t24);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t43 = SCALAR_FMA(tx, ty, t43);
            t44 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t34);
            t44 = SCALAR_FNMA(Y_PC, t35, t44);
            tx = SCALAR_SUB(t24, t25);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t44 = SCALAR_FMA(tx, ty, t44);
            tx = SCALAR_LOAD((temp + 3 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t40);
            SCALAR_STORE((temp + 3 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t40);
            t50 = SCALAR_FNMA(Y_PC, t41, t50);
            tx = SCALAR_SUB(t30, t31);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t50 = SCALAR_FMA(tx, ty, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t41);
            t51 = SCALAR_FNMA(Y_PC, t42, t51);
            tx = SCALAR_SUB(t31, t32);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t51 = SCALAR_FMA(tx, ty, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t42);
            t52 = SCALAR_FNMA(Y_PC, t43, t52);
            tx = SCALAR_SUB(t32, t33);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t52 = SCALAR_FMA(tx, ty, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t43);
            t53 = SCALAR_FNMA(Y_PC, t44, t53);
            tx = SCALAR_SUB(t33, t34);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t53 = SCALAR_FMA(tx, ty, t53);
            tx = SCALAR_LOAD((temp + 21 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 21 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t50);
            t60 = SCALAR_FNMA(Y_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t51);
            t61 = SCALAR_FNMA(Y_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t52);
            t62 = SCALAR_FNMA(Y_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 46 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 46 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t60);
            t70 = SCALAR_FNMA(Y_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t61);
            t71 = SCALAR_FNMA(Y_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 79 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 79 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t70);
            t80 = SCALAR_FNMA(Y_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 121 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 121 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 122 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_LOAD((temp + 80 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 80 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 123 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_LOAD((temp + 47 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 47 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 81 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 81 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 124 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t40);
            t50 = SCALAR_FNMA(Z_PC, t41, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t41);
            t51 = SCALAR_FNMA(Z_PC, t42, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t42);
            t52 = SCALAR_FNMA(Z_PC, t43, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t43);
            t53 = SCALAR_FNMA(Z_PC, t44, t53);
            tx = SCALAR_LOAD((temp + 22 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 22 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 48 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 48 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 82 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 82 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 125 * NPTS_LOCAL + p_inner), tx);
            t40 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t30);
            t40 = SCALAR_FNMA(Z_PC, t31, t40);
            t41 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t31);
            t41 = SCALAR_FNMA(Z_PC, t32, t41);
            t42 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t32);
            t42 = SCALAR_FNMA(Z_PC, t33, t42);
            t43 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t33);
            t43 = SCALAR_FNMA(Z_PC, t34, t43);
            t44 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t34);
            t44 = SCALAR_FNMA(Z_PC, t35, t44);
            tx = SCALAR_LOAD((temp + 4 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t40);
            SCALAR_STORE((temp + 4 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t40);
            t50 = SCALAR_FNMA(Z_PC, t41, t50);
            tx = SCALAR_SUB(t30, t31);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t50 = SCALAR_FMA(tx, ty, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t41);
            t51 = SCALAR_FNMA(Z_PC, t42, t51);
            tx = SCALAR_SUB(t31, t32);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t51 = SCALAR_FMA(tx, ty, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t42);
            t52 = SCALAR_FNMA(Z_PC, t43, t52);
            tx = SCALAR_SUB(t32, t33);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t52 = SCALAR_FMA(tx, ty, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t43);
            t53 = SCALAR_FNMA(Z_PC, t44, t53);
            tx = SCALAR_SUB(t33, t34);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t53 = SCALAR_FMA(tx, ty, t53);
            tx = SCALAR_LOAD((temp + 23 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 23 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 49 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 49 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 83 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 83 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 126 * NPTS_LOCAL + p_inner), tx);
            t30 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t20);
            t30 = SCALAR_FNMA(Z_PC, t21, t30);
            t31 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t21);
            t31 = SCALAR_FNMA(Z_PC, t22, t31);
            t32 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t22);
            t32 = SCALAR_FNMA(Z_PC, t23, t32);
            t33 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t23);
            t33 = SCALAR_FNMA(Z_PC, t24, t33);
            t34 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t24);
            t34 = SCALAR_FNMA(Z_PC, t25, t34);
            t35 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t25);
            t35 = SCALAR_FNMA(Z_PC, t26, t35);
            t40 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t30);
            t40 = SCALAR_FNMA(Z_PC, t31, t40);
            tx = SCALAR_SUB(t20, t21);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t40 = SCALAR_FMA(tx, ty, t40);
            t41 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t31);
            t41 = SCALAR_FNMA(Z_PC, t32, t41);
            tx = SCALAR_SUB(t21, t22);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t41 = SCALAR_FMA(tx, ty, t41);
            t42 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t32);
            t42 = SCALAR_FNMA(Z_PC, t33, t42);
            tx = SCALAR_SUB(t22, t23);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t42 = SCALAR_FMA(tx, ty, t42);
            t43 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t33);
            t43 = SCALAR_FNMA(Z_PC, t34, t43);
            tx = SCALAR_SUB(t23, t24);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t43 = SCALAR_FMA(tx, ty, t43);
            t44 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t34);
            t44 = SCALAR_FNMA(Z_PC, t35, t44);
            tx = SCALAR_SUB(t24, t25);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t44 = SCALAR_FMA(tx, ty, t44);
            tx = SCALAR_LOAD((temp + 5 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t40);
            SCALAR_STORE((temp + 5 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t40);
            t50 = SCALAR_FNMA(Z_PC, t41, t50);
            tx = SCALAR_SUB(t30, t31);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t50 = SCALAR_FMA(tx, ty, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t41);
            t51 = SCALAR_FNMA(Z_PC, t42, t51);
            tx = SCALAR_SUB(t31, t32);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t51 = SCALAR_FMA(tx, ty, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t42);
            t52 = SCALAR_FNMA(Z_PC, t43, t52);
            tx = SCALAR_SUB(t32, t33);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t52 = SCALAR_FMA(tx, ty, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t43);
            t53 = SCALAR_FNMA(Z_PC, t44, t53);
            tx = SCALAR_SUB(t33, t34);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t53 = SCALAR_FMA(tx, ty, t53);
            tx = SCALAR_LOAD((temp + 24 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 24 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 50 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 50 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 84 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 84 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 127 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 127 * NPTS_LOCAL + p_inner), tx);
            t20 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t10);
            t20 = SCALAR_FNMA(Y_PC, t11, t20);
            t21 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t11);
            t21 = SCALAR_FNMA(Y_PC, t12, t21);
            t22 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t12);
            t22 = SCALAR_FNMA(Y_PC, t13, t22);
            t23 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t13);
            t23 = SCALAR_FNMA(Y_PC, t14, t23);
            t24 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t14);
            t24 = SCALAR_FNMA(Y_PC, t15, t24);
            t25 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t15);
            t25 = SCALAR_FNMA(Y_PC, t16, t25);
            t26 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t16);
            t26 = SCALAR_FNMA(Y_PC, t17, t26);
            t30 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t20);
            t30 = SCALAR_FNMA(Y_PC, t21, t30);
            tx = SCALAR_SUB(t10, t11);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t30 = SCALAR_FMA(tx, ty, t30);
            t31 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t21);
            t31 = SCALAR_FNMA(Y_PC, t22, t31);
            tx = SCALAR_SUB(t11, t12);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t31 = SCALAR_FMA(tx, ty, t31);
            t32 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t22);
            t32 = SCALAR_FNMA(Y_PC, t23, t32);
            tx = SCALAR_SUB(t12, t13);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t32 = SCALAR_FMA(tx, ty, t32);
            t33 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t23);
            t33 = SCALAR_FNMA(Y_PC, t24, t33);
            tx = SCALAR_SUB(t13, t14);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t33 = SCALAR_FMA(tx, ty, t33);
            t34 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t24);
            t34 = SCALAR_FNMA(Y_PC, t25, t34);
            tx = SCALAR_SUB(t14, t15);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t34 = SCALAR_FMA(tx, ty, t34);
            t35 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t25);
            t35 = SCALAR_FNMA(Y_PC, t26, t35);
            tx = SCALAR_SUB(t15, t16);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t35 = SCALAR_FMA(tx, ty, t35);
            t40 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t30);
            t40 = SCALAR_FNMA(Y_PC, t31, t40);
            tx = SCALAR_SUB(t20, t21);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t40 = SCALAR_FMA(tx, ty, t40);
            t41 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t31);
            t41 = SCALAR_FNMA(Y_PC, t32, t41);
            tx = SCALAR_SUB(t21, t22);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t41 = SCALAR_FMA(tx, ty, t41);
            t42 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t32);
            t42 = SCALAR_FNMA(Y_PC, t33, t42);
            tx = SCALAR_SUB(t22, t23);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t42 = SCALAR_FMA(tx, ty, t42);
            t43 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t33);
            t43 = SCALAR_FNMA(Y_PC, t34, t43);
            tx = SCALAR_SUB(t23, t24);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t43 = SCALAR_FMA(tx, ty, t43);
            t44 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t34);
            t44 = SCALAR_FNMA(Y_PC, t35, t44);
            tx = SCALAR_SUB(t24, t25);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t44 = SCALAR_FMA(tx, ty, t44);
            tx = SCALAR_LOAD((temp + 6 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t40);
            SCALAR_STORE((temp + 6 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t40);
            t50 = SCALAR_FNMA(Y_PC, t41, t50);
            tx = SCALAR_SUB(t30, t31);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t50 = SCALAR_FMA(tx, ty, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t41);
            t51 = SCALAR_FNMA(Y_PC, t42, t51);
            tx = SCALAR_SUB(t31, t32);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t51 = SCALAR_FMA(tx, ty, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t42);
            t52 = SCALAR_FNMA(Y_PC, t43, t52);
            tx = SCALAR_SUB(t32, t33);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t52 = SCALAR_FMA(tx, ty, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t43);
            t53 = SCALAR_FNMA(Y_PC, t44, t53);
            tx = SCALAR_SUB(t33, t34);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t53 = SCALAR_FMA(tx, ty, t53);
            tx = SCALAR_LOAD((temp + 25 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 25 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t50);
            t60 = SCALAR_FNMA(Y_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t51);
            t61 = SCALAR_FNMA(Y_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t52);
            t62 = SCALAR_FNMA(Y_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 51 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 51 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t60);
            t70 = SCALAR_FNMA(Y_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t61);
            t71 = SCALAR_FNMA(Y_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 85 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 85 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t70);
            t80 = SCALAR_FNMA(Y_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 6);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 128 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 128 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_LOAD((temp + 129 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 129 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_LOAD((temp + 86 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 86 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 130 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_LOAD((temp + 52 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 52 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 87 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 87 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 131 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t40);
            t50 = SCALAR_FNMA(Z_PC, t41, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t41);
            t51 = SCALAR_FNMA(Z_PC, t42, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t42);
            t52 = SCALAR_FNMA(Z_PC, t43, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t43);
            t53 = SCALAR_FNMA(Z_PC, t44, t53);
            tx = SCALAR_LOAD((temp + 26 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 26 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 53 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 53 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 88 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 88 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 132 * NPTS_LOCAL + p_inner), tx);
            t40 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t30);
            t40 = SCALAR_FNMA(Z_PC, t31, t40);
            t41 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t31);
            t41 = SCALAR_FNMA(Z_PC, t32, t41);
            t42 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t32);
            t42 = SCALAR_FNMA(Z_PC, t33, t42);
            t43 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t33);
            t43 = SCALAR_FNMA(Z_PC, t34, t43);
            t44 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t34);
            t44 = SCALAR_FNMA(Z_PC, t35, t44);
            tx = SCALAR_LOAD((temp + 7 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t40);
            SCALAR_STORE((temp + 7 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t40);
            t50 = SCALAR_FNMA(Z_PC, t41, t50);
            tx = SCALAR_SUB(t30, t31);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t50 = SCALAR_FMA(tx, ty, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t41);
            t51 = SCALAR_FNMA(Z_PC, t42, t51);
            tx = SCALAR_SUB(t31, t32);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t51 = SCALAR_FMA(tx, ty, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t42);
            t52 = SCALAR_FNMA(Z_PC, t43, t52);
            tx = SCALAR_SUB(t32, t33);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t52 = SCALAR_FMA(tx, ty, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t43);
            t53 = SCALAR_FNMA(Z_PC, t44, t53);
            tx = SCALAR_SUB(t33, t34);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t53 = SCALAR_FMA(tx, ty, t53);
            tx = SCALAR_LOAD((temp + 27 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 27 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 54 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 54 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 89 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 89 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 133 * NPTS_LOCAL + p_inner), tx);
            t30 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t20);
            t30 = SCALAR_FNMA(Z_PC, t21, t30);
            t31 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t21);
            t31 = SCALAR_FNMA(Z_PC, t22, t31);
            t32 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t22);
            t32 = SCALAR_FNMA(Z_PC, t23, t32);
            t33 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t23);
            t33 = SCALAR_FNMA(Z_PC, t24, t33);
            t34 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t24);
            t34 = SCALAR_FNMA(Z_PC, t25, t34);
            t35 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t25);
            t35 = SCALAR_FNMA(Z_PC, t26, t35);
            t40 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t30);
            t40 = SCALAR_FNMA(Z_PC, t31, t40);
            tx = SCALAR_SUB(t20, t21);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t40 = SCALAR_FMA(tx, ty, t40);
            t41 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t31);
            t41 = SCALAR_FNMA(Z_PC, t32, t41);
            tx = SCALAR_SUB(t21, t22);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t41 = SCALAR_FMA(tx, ty, t41);
            t42 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t32);
            t42 = SCALAR_FNMA(Z_PC, t33, t42);
            tx = SCALAR_SUB(t22, t23);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t42 = SCALAR_FMA(tx, ty, t42);
            t43 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t33);
            t43 = SCALAR_FNMA(Z_PC, t34, t43);
            tx = SCALAR_SUB(t23, t24);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t43 = SCALAR_FMA(tx, ty, t43);
            t44 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t34);
            t44 = SCALAR_FNMA(Z_PC, t35, t44);
            tx = SCALAR_SUB(t24, t25);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t44 = SCALAR_FMA(tx, ty, t44);
            tx = SCALAR_LOAD((temp + 8 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t40);
            SCALAR_STORE((temp + 8 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t40);
            t50 = SCALAR_FNMA(Z_PC, t41, t50);
            tx = SCALAR_SUB(t30, t31);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t50 = SCALAR_FMA(tx, ty, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t41);
            t51 = SCALAR_FNMA(Z_PC, t42, t51);
            tx = SCALAR_SUB(t31, t32);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t51 = SCALAR_FMA(tx, ty, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t42);
            t52 = SCALAR_FNMA(Z_PC, t43, t52);
            tx = SCALAR_SUB(t32, t33);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t52 = SCALAR_FMA(tx, ty, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t43);
            t53 = SCALAR_FNMA(Z_PC, t44, t53);
            tx = SCALAR_SUB(t33, t34);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t53 = SCALAR_FMA(tx, ty, t53);
            tx = SCALAR_LOAD((temp + 28 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 28 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 55 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 55 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 90 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 90 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 134 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 134 * NPTS_LOCAL + p_inner), tx);
            t20 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t10);
            t20 = SCALAR_FNMA(Z_PC, t11, t20);
            t21 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t11);
            t21 = SCALAR_FNMA(Z_PC, t12, t21);
            t22 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t12);
            t22 = SCALAR_FNMA(Z_PC, t13, t22);
            t23 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t13);
            t23 = SCALAR_FNMA(Z_PC, t14, t23);
            t24 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t14);
            t24 = SCALAR_FNMA(Z_PC, t15, t24);
            t25 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t15);
            t25 = SCALAR_FNMA(Z_PC, t16, t25);
            t26 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t16);
            t26 = SCALAR_FNMA(Z_PC, t17, t26);
            t30 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t20);
            t30 = SCALAR_FNMA(Z_PC, t21, t30);
            tx = SCALAR_SUB(t10, t11);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t30 = SCALAR_FMA(tx, ty, t30);
            t31 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t21);
            t31 = SCALAR_FNMA(Z_PC, t22, t31);
            tx = SCALAR_SUB(t11, t12);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t31 = SCALAR_FMA(tx, ty, t31);
            t32 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t22);
            t32 = SCALAR_FNMA(Z_PC, t23, t32);
            tx = SCALAR_SUB(t12, t13);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t32 = SCALAR_FMA(tx, ty, t32);
            t33 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t23);
            t33 = SCALAR_FNMA(Z_PC, t24, t33);
            tx = SCALAR_SUB(t13, t14);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t33 = SCALAR_FMA(tx, ty, t33);
            t34 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t24);
            t34 = SCALAR_FNMA(Z_PC, t25, t34);
            tx = SCALAR_SUB(t14, t15);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t34 = SCALAR_FMA(tx, ty, t34);
            t35 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t25);
            t35 = SCALAR_FNMA(Z_PC, t26, t35);
            tx = SCALAR_SUB(t15, t16);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t35 = SCALAR_FMA(tx, ty, t35);
            t40 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t30);
            t40 = SCALAR_FNMA(Z_PC, t31, t40);
            tx = SCALAR_SUB(t20, t21);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t40 = SCALAR_FMA(tx, ty, t40);
            t41 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t31);
            t41 = SCALAR_FNMA(Z_PC, t32, t41);
            tx = SCALAR_SUB(t21, t22);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t41 = SCALAR_FMA(tx, ty, t41);
            t42 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t32);
            t42 = SCALAR_FNMA(Z_PC, t33, t42);
            tx = SCALAR_SUB(t22, t23);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t42 = SCALAR_FMA(tx, ty, t42);
            t43 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t33);
            t43 = SCALAR_FNMA(Z_PC, t34, t43);
            tx = SCALAR_SUB(t23, t24);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t43 = SCALAR_FMA(tx, ty, t43);
            t44 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t34);
            t44 = SCALAR_FNMA(Z_PC, t35, t44);
            tx = SCALAR_SUB(t24, t25);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t44 = SCALAR_FMA(tx, ty, t44);
            tx = SCALAR_LOAD((temp + 9 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t40);
            SCALAR_STORE((temp + 9 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t40);
            t50 = SCALAR_FNMA(Z_PC, t41, t50);
            tx = SCALAR_SUB(t30, t31);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t50 = SCALAR_FMA(tx, ty, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t41);
            t51 = SCALAR_FNMA(Z_PC, t42, t51);
            tx = SCALAR_SUB(t31, t32);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t51 = SCALAR_FMA(tx, ty, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t42);
            t52 = SCALAR_FNMA(Z_PC, t43, t52);
            tx = SCALAR_SUB(t32, t33);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t52 = SCALAR_FMA(tx, ty, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t43);
            t53 = SCALAR_FNMA(Z_PC, t44, t53);
            tx = SCALAR_SUB(t33, t34);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t53 = SCALAR_FMA(tx, ty, t53);
            tx = SCALAR_LOAD((temp + 29 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 29 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 56 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 56 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 91 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 91 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 6);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 135 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 135 * NPTS_LOCAL + p_inner), tx);
            t10 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t00);
            t10 = SCALAR_FNMA(Y_PC, t01, t10);
            t11 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t01);
            t11 = SCALAR_FNMA(Y_PC, t02, t11);
            t12 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t02);
            t12 = SCALAR_FNMA(Y_PC, t03, t12);
            t13 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t03);
            t13 = SCALAR_FNMA(Y_PC, t04, t13);
            t14 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t04);
            t14 = SCALAR_FNMA(Y_PC, t05, t14);
            t15 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t05);
            t15 = SCALAR_FNMA(Y_PC, t06, t15);
            t16 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t06);
            t16 = SCALAR_FNMA(Y_PC, t07, t16);
            t17 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t07);
            t17 = SCALAR_FNMA(Y_PC, t08, t17);
            t20 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t10);
            t20 = SCALAR_FNMA(Y_PC, t11, t20);
            tx = SCALAR_SUB(t00, t01);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t20 = SCALAR_FMA(tx, ty, t20);
            t21 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t11);
            t21 = SCALAR_FNMA(Y_PC, t12, t21);
            tx = SCALAR_SUB(t01, t02);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t21 = SCALAR_FMA(tx, ty, t21);
            t22 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t12);
            t22 = SCALAR_FNMA(Y_PC, t13, t22);
            tx = SCALAR_SUB(t02, t03);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t22 = SCALAR_FMA(tx, ty, t22);
            t23 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t13);
            t23 = SCALAR_FNMA(Y_PC, t14, t23);
            tx = SCALAR_SUB(t03, t04);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t23 = SCALAR_FMA(tx, ty, t23);
            t24 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t14);
            t24 = SCALAR_FNMA(Y_PC, t15, t24);
            tx = SCALAR_SUB(t04, t05);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t24 = SCALAR_FMA(tx, ty, t24);
            t25 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t15);
            t25 = SCALAR_FNMA(Y_PC, t16, t25);
            tx = SCALAR_SUB(t05, t06);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t25 = SCALAR_FMA(tx, ty, t25);
            t26 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t16);
            t26 = SCALAR_FNMA(Y_PC, t17, t26);
            tx = SCALAR_SUB(t06, t07);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t26 = SCALAR_FMA(tx, ty, t26);
            t30 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t20);
            t30 = SCALAR_FNMA(Y_PC, t21, t30);
            tx = SCALAR_SUB(t10, t11);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t30 = SCALAR_FMA(tx, ty, t30);
            t31 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t21);
            t31 = SCALAR_FNMA(Y_PC, t22, t31);
            tx = SCALAR_SUB(t11, t12);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t31 = SCALAR_FMA(tx, ty, t31);
            t32 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t22);
            t32 = SCALAR_FNMA(Y_PC, t23, t32);
            tx = SCALAR_SUB(t12, t13);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t32 = SCALAR_FMA(tx, ty, t32);
            t33 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t23);
            t33 = SCALAR_FNMA(Y_PC, t24, t33);
            tx = SCALAR_SUB(t13, t14);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t33 = SCALAR_FMA(tx, ty, t33);
            t34 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t24);
            t34 = SCALAR_FNMA(Y_PC, t25, t34);
            tx = SCALAR_SUB(t14, t15);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t34 = SCALAR_FMA(tx, ty, t34);
            t35 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t25);
            t35 = SCALAR_FNMA(Y_PC, t26, t35);
            tx = SCALAR_SUB(t15, t16);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t35 = SCALAR_FMA(tx, ty, t35);
            t40 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t30);
            t40 = SCALAR_FNMA(Y_PC, t31, t40);
            tx = SCALAR_SUB(t20, t21);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t40 = SCALAR_FMA(tx, ty, t40);
            t41 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t31);
            t41 = SCALAR_FNMA(Y_PC, t32, t41);
            tx = SCALAR_SUB(t21, t22);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t41 = SCALAR_FMA(tx, ty, t41);
            t42 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t32);
            t42 = SCALAR_FNMA(Y_PC, t33, t42);
            tx = SCALAR_SUB(t22, t23);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t42 = SCALAR_FMA(tx, ty, t42);
            t43 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t33);
            t43 = SCALAR_FNMA(Y_PC, t34, t43);
            tx = SCALAR_SUB(t23, t24);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t43 = SCALAR_FMA(tx, ty, t43);
            t44 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t34);
            t44 = SCALAR_FNMA(Y_PC, t35, t44);
            tx = SCALAR_SUB(t24, t25);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t44 = SCALAR_FMA(tx, ty, t44);
            tx = SCALAR_LOAD((temp + 10 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t40);
            SCALAR_STORE((temp + 10 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t40);
            t50 = SCALAR_FNMA(Y_PC, t41, t50);
            tx = SCALAR_SUB(t30, t31);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t50 = SCALAR_FMA(tx, ty, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t41);
            t51 = SCALAR_FNMA(Y_PC, t42, t51);
            tx = SCALAR_SUB(t31, t32);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t51 = SCALAR_FMA(tx, ty, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t42);
            t52 = SCALAR_FNMA(Y_PC, t43, t52);
            tx = SCALAR_SUB(t32, t33);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t52 = SCALAR_FMA(tx, ty, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t43);
            t53 = SCALAR_FNMA(Y_PC, t44, t53);
            tx = SCALAR_SUB(t33, t34);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t53 = SCALAR_FMA(tx, ty, t53);
            tx = SCALAR_LOAD((temp + 30 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 30 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t50);
            t60 = SCALAR_FNMA(Y_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t51);
            t61 = SCALAR_FNMA(Y_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t52);
            t62 = SCALAR_FNMA(Y_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 57 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 57 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t60);
            t70 = SCALAR_FNMA(Y_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 6);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t61);
            t71 = SCALAR_FNMA(Y_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 6);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 92 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 92 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t70);
            t80 = SCALAR_FNMA(Y_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 7);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 136 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 136 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_LOAD((temp + 137 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 137 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_LOAD((temp + 93 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 93 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 138 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 138 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_LOAD((temp + 58 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 58 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 94 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 94 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 139 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 139 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t40);
            t50 = SCALAR_FNMA(Z_PC, t41, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t41);
            t51 = SCALAR_FNMA(Z_PC, t42, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t42);
            t52 = SCALAR_FNMA(Z_PC, t43, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t43);
            t53 = SCALAR_FNMA(Z_PC, t44, t53);
            tx = SCALAR_LOAD((temp + 31 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 31 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 59 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 59 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 95 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 95 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 140 * NPTS_LOCAL + p_inner), tx);
            t40 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t30);
            t40 = SCALAR_FNMA(Z_PC, t31, t40);
            t41 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t31);
            t41 = SCALAR_FNMA(Z_PC, t32, t41);
            t42 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t32);
            t42 = SCALAR_FNMA(Z_PC, t33, t42);
            t43 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t33);
            t43 = SCALAR_FNMA(Z_PC, t34, t43);
            t44 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t34);
            t44 = SCALAR_FNMA(Z_PC, t35, t44);
            tx = SCALAR_LOAD((temp + 11 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t40);
            SCALAR_STORE((temp + 11 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t40);
            t50 = SCALAR_FNMA(Z_PC, t41, t50);
            tx = SCALAR_SUB(t30, t31);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t50 = SCALAR_FMA(tx, ty, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t41);
            t51 = SCALAR_FNMA(Z_PC, t42, t51);
            tx = SCALAR_SUB(t31, t32);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t51 = SCALAR_FMA(tx, ty, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t42);
            t52 = SCALAR_FNMA(Z_PC, t43, t52);
            tx = SCALAR_SUB(t32, t33);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t52 = SCALAR_FMA(tx, ty, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t43);
            t53 = SCALAR_FNMA(Z_PC, t44, t53);
            tx = SCALAR_SUB(t33, t34);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t53 = SCALAR_FMA(tx, ty, t53);
            tx = SCALAR_LOAD((temp + 32 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 32 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 60 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 60 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 96 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 96 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 141 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 141 * NPTS_LOCAL + p_inner), tx);
            t30 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t20);
            t30 = SCALAR_FNMA(Z_PC, t21, t30);
            t31 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t21);
            t31 = SCALAR_FNMA(Z_PC, t22, t31);
            t32 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t22);
            t32 = SCALAR_FNMA(Z_PC, t23, t32);
            t33 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t23);
            t33 = SCALAR_FNMA(Z_PC, t24, t33);
            t34 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t24);
            t34 = SCALAR_FNMA(Z_PC, t25, t34);
            t35 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t25);
            t35 = SCALAR_FNMA(Z_PC, t26, t35);
            t40 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t30);
            t40 = SCALAR_FNMA(Z_PC, t31, t40);
            tx = SCALAR_SUB(t20, t21);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t40 = SCALAR_FMA(tx, ty, t40);
            t41 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t31);
            t41 = SCALAR_FNMA(Z_PC, t32, t41);
            tx = SCALAR_SUB(t21, t22);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t41 = SCALAR_FMA(tx, ty, t41);
            t42 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t32);
            t42 = SCALAR_FNMA(Z_PC, t33, t42);
            tx = SCALAR_SUB(t22, t23);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t42 = SCALAR_FMA(tx, ty, t42);
            t43 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t33);
            t43 = SCALAR_FNMA(Z_PC, t34, t43);
            tx = SCALAR_SUB(t23, t24);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t43 = SCALAR_FMA(tx, ty, t43);
            t44 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t34);
            t44 = SCALAR_FNMA(Z_PC, t35, t44);
            tx = SCALAR_SUB(t24, t25);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t44 = SCALAR_FMA(tx, ty, t44);
            tx = SCALAR_LOAD((temp + 12 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t40);
            SCALAR_STORE((temp + 12 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t40);
            t50 = SCALAR_FNMA(Z_PC, t41, t50);
            tx = SCALAR_SUB(t30, t31);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t50 = SCALAR_FMA(tx, ty, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t41);
            t51 = SCALAR_FNMA(Z_PC, t42, t51);
            tx = SCALAR_SUB(t31, t32);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t51 = SCALAR_FMA(tx, ty, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t42);
            t52 = SCALAR_FNMA(Z_PC, t43, t52);
            tx = SCALAR_SUB(t32, t33);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t52 = SCALAR_FMA(tx, ty, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t43);
            t53 = SCALAR_FNMA(Z_PC, t44, t53);
            tx = SCALAR_SUB(t33, t34);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t53 = SCALAR_FMA(tx, ty, t53);
            tx = SCALAR_LOAD((temp + 33 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 33 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 61 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 61 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 97 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 97 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 142 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 142 * NPTS_LOCAL + p_inner), tx);
            t20 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t10);
            t20 = SCALAR_FNMA(Z_PC, t11, t20);
            t21 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t11);
            t21 = SCALAR_FNMA(Z_PC, t12, t21);
            t22 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t12);
            t22 = SCALAR_FNMA(Z_PC, t13, t22);
            t23 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t13);
            t23 = SCALAR_FNMA(Z_PC, t14, t23);
            t24 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t14);
            t24 = SCALAR_FNMA(Z_PC, t15, t24);
            t25 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t15);
            t25 = SCALAR_FNMA(Z_PC, t16, t25);
            t26 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t16);
            t26 = SCALAR_FNMA(Z_PC, t17, t26);
            t30 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t20);
            t30 = SCALAR_FNMA(Z_PC, t21, t30);
            tx = SCALAR_SUB(t10, t11);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t30 = SCALAR_FMA(tx, ty, t30);
            t31 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t21);
            t31 = SCALAR_FNMA(Z_PC, t22, t31);
            tx = SCALAR_SUB(t11, t12);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t31 = SCALAR_FMA(tx, ty, t31);
            t32 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t22);
            t32 = SCALAR_FNMA(Z_PC, t23, t32);
            tx = SCALAR_SUB(t12, t13);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t32 = SCALAR_FMA(tx, ty, t32);
            t33 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t23);
            t33 = SCALAR_FNMA(Z_PC, t24, t33);
            tx = SCALAR_SUB(t13, t14);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t33 = SCALAR_FMA(tx, ty, t33);
            t34 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t24);
            t34 = SCALAR_FNMA(Z_PC, t25, t34);
            tx = SCALAR_SUB(t14, t15);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t34 = SCALAR_FMA(tx, ty, t34);
            t35 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t25);
            t35 = SCALAR_FNMA(Z_PC, t26, t35);
            tx = SCALAR_SUB(t15, t16);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t35 = SCALAR_FMA(tx, ty, t35);
            t40 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t30);
            t40 = SCALAR_FNMA(Z_PC, t31, t40);
            tx = SCALAR_SUB(t20, t21);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t40 = SCALAR_FMA(tx, ty, t40);
            t41 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t31);
            t41 = SCALAR_FNMA(Z_PC, t32, t41);
            tx = SCALAR_SUB(t21, t22);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t41 = SCALAR_FMA(tx, ty, t41);
            t42 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t32);
            t42 = SCALAR_FNMA(Z_PC, t33, t42);
            tx = SCALAR_SUB(t22, t23);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t42 = SCALAR_FMA(tx, ty, t42);
            t43 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t33);
            t43 = SCALAR_FNMA(Z_PC, t34, t43);
            tx = SCALAR_SUB(t23, t24);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t43 = SCALAR_FMA(tx, ty, t43);
            t44 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t34);
            t44 = SCALAR_FNMA(Z_PC, t35, t44);
            tx = SCALAR_SUB(t24, t25);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t44 = SCALAR_FMA(tx, ty, t44);
            tx = SCALAR_LOAD((temp + 13 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t40);
            SCALAR_STORE((temp + 13 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t40);
            t50 = SCALAR_FNMA(Z_PC, t41, t50);
            tx = SCALAR_SUB(t30, t31);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t50 = SCALAR_FMA(tx, ty, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t41);
            t51 = SCALAR_FNMA(Z_PC, t42, t51);
            tx = SCALAR_SUB(t31, t32);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t51 = SCALAR_FMA(tx, ty, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t42);
            t52 = SCALAR_FNMA(Z_PC, t43, t52);
            tx = SCALAR_SUB(t32, t33);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t52 = SCALAR_FMA(tx, ty, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t43);
            t53 = SCALAR_FNMA(Z_PC, t44, t53);
            tx = SCALAR_SUB(t33, t34);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t53 = SCALAR_FMA(tx, ty, t53);
            tx = SCALAR_LOAD((temp + 34 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 34 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 62 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 62 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 98 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 98 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 6);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 143 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 143 * NPTS_LOCAL + p_inner), tx);
            t10 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t00);
            t10 = SCALAR_FNMA(Z_PC, t01, t10);
            t11 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t01);
            t11 = SCALAR_FNMA(Z_PC, t02, t11);
            t12 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t02);
            t12 = SCALAR_FNMA(Z_PC, t03, t12);
            t13 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t03);
            t13 = SCALAR_FNMA(Z_PC, t04, t13);
            t14 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t04);
            t14 = SCALAR_FNMA(Z_PC, t05, t14);
            t15 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t05);
            t15 = SCALAR_FNMA(Z_PC, t06, t15);
            t16 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t06);
            t16 = SCALAR_FNMA(Z_PC, t07, t16);
            t17 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t07);
            t17 = SCALAR_FNMA(Z_PC, t08, t17);
            t20 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t10);
            t20 = SCALAR_FNMA(Z_PC, t11, t20);
            tx = SCALAR_SUB(t00, t01);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t20 = SCALAR_FMA(tx, ty, t20);
            t21 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t11);
            t21 = SCALAR_FNMA(Z_PC, t12, t21);
            tx = SCALAR_SUB(t01, t02);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t21 = SCALAR_FMA(tx, ty, t21);
            t22 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t12);
            t22 = SCALAR_FNMA(Z_PC, t13, t22);
            tx = SCALAR_SUB(t02, t03);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t22 = SCALAR_FMA(tx, ty, t22);
            t23 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t13);
            t23 = SCALAR_FNMA(Z_PC, t14, t23);
            tx = SCALAR_SUB(t03, t04);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t23 = SCALAR_FMA(tx, ty, t23);
            t24 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t14);
            t24 = SCALAR_FNMA(Z_PC, t15, t24);
            tx = SCALAR_SUB(t04, t05);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t24 = SCALAR_FMA(tx, ty, t24);
            t25 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t15);
            t25 = SCALAR_FNMA(Z_PC, t16, t25);
            tx = SCALAR_SUB(t05, t06);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t25 = SCALAR_FMA(tx, ty, t25);
            t26 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t16);
            t26 = SCALAR_FNMA(Z_PC, t17, t26);
            tx = SCALAR_SUB(t06, t07);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t26 = SCALAR_FMA(tx, ty, t26);
            t30 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t20);
            t30 = SCALAR_FNMA(Z_PC, t21, t30);
            tx = SCALAR_SUB(t10, t11);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t30 = SCALAR_FMA(tx, ty, t30);
            t31 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t21);
            t31 = SCALAR_FNMA(Z_PC, t22, t31);
            tx = SCALAR_SUB(t11, t12);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t31 = SCALAR_FMA(tx, ty, t31);
            t32 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t22);
            t32 = SCALAR_FNMA(Z_PC, t23, t32);
            tx = SCALAR_SUB(t12, t13);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t32 = SCALAR_FMA(tx, ty, t32);
            t33 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t23);
            t33 = SCALAR_FNMA(Z_PC, t24, t33);
            tx = SCALAR_SUB(t13, t14);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t33 = SCALAR_FMA(tx, ty, t33);
            t34 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t24);
            t34 = SCALAR_FNMA(Z_PC, t25, t34);
            tx = SCALAR_SUB(t14, t15);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t34 = SCALAR_FMA(tx, ty, t34);
            t35 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t25);
            t35 = SCALAR_FNMA(Z_PC, t26, t35);
            tx = SCALAR_SUB(t15, t16);
            ty = SCALAR_SET1(0.5 * 2);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t35 = SCALAR_FMA(tx, ty, t35);
            t40 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t30);
            t40 = SCALAR_FNMA(Z_PC, t31, t40);
            tx = SCALAR_SUB(t20, t21);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t40 = SCALAR_FMA(tx, ty, t40);
            t41 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t31);
            t41 = SCALAR_FNMA(Z_PC, t32, t41);
            tx = SCALAR_SUB(t21, t22);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t41 = SCALAR_FMA(tx, ty, t41);
            t42 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t32);
            t42 = SCALAR_FNMA(Z_PC, t33, t42);
            tx = SCALAR_SUB(t22, t23);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t42 = SCALAR_FMA(tx, ty, t42);
            t43 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t33);
            t43 = SCALAR_FNMA(Z_PC, t34, t43);
            tx = SCALAR_SUB(t23, t24);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t43 = SCALAR_FMA(tx, ty, t43);
            t44 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t34);
            t44 = SCALAR_FNMA(Z_PC, t35, t44);
            tx = SCALAR_SUB(t24, t25);
            ty = SCALAR_SET1(0.5 * 3);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t44 = SCALAR_FMA(tx, ty, t44);
            tx = SCALAR_LOAD((temp + 14 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t40);
            SCALAR_STORE((temp + 14 * NPTS_LOCAL + p_inner), tx);
            t50 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t40);
            t50 = SCALAR_FNMA(Z_PC, t41, t50);
            tx = SCALAR_SUB(t30, t31);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t50 = SCALAR_FMA(tx, ty, t50);
            t51 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t41);
            t51 = SCALAR_FNMA(Z_PC, t42, t51);
            tx = SCALAR_SUB(t31, t32);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t51 = SCALAR_FMA(tx, ty, t51);
            t52 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t42);
            t52 = SCALAR_FNMA(Z_PC, t43, t52);
            tx = SCALAR_SUB(t32, t33);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t52 = SCALAR_FMA(tx, ty, t52);
            t53 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t43);
            t53 = SCALAR_FNMA(Z_PC, t44, t53);
            tx = SCALAR_SUB(t33, t34);
            ty = SCALAR_SET1(0.5 * 4);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t53 = SCALAR_FMA(tx, ty, t53);
            tx = SCALAR_LOAD((temp + 35 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t50);
            SCALAR_STORE((temp + 35 * NPTS_LOCAL + p_inner), tx);
            t60 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t50);
            t60 = SCALAR_FNMA(Z_PC, t51, t60);
            tx = SCALAR_SUB(t40, t41);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t60 = SCALAR_FMA(tx, ty, t60);
            t61 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t51);
            t61 = SCALAR_FNMA(Z_PC, t52, t61);
            tx = SCALAR_SUB(t41, t42);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t61 = SCALAR_FMA(tx, ty, t61);
            t62 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t52);
            t62 = SCALAR_FNMA(Z_PC, t53, t62);
            tx = SCALAR_SUB(t42, t43);
            ty = SCALAR_SET1(0.5 * 5);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t62 = SCALAR_FMA(tx, ty, t62);
            tx = SCALAR_LOAD((temp + 63 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t60);
            SCALAR_STORE((temp + 63 * NPTS_LOCAL + p_inner), tx);
            t70 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t60);
            t70 = SCALAR_FNMA(Z_PC, t61, t70);
            tx = SCALAR_SUB(t50, t51);
            ty = SCALAR_SET1(0.5 * 6);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t70 = SCALAR_FMA(tx, ty, t70);
            t71 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t61);
            t71 = SCALAR_FNMA(Z_PC, t62, t71);
            tx = SCALAR_SUB(t51, t52);
            ty = SCALAR_SET1(0.5 * 6);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t71 = SCALAR_FMA(tx, ty, t71);
            tx = SCALAR_LOAD((temp + 99 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t70);
            SCALAR_STORE((temp + 99 * NPTS_LOCAL + p_inner), tx);
            t80 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t70);
            t80 = SCALAR_FNMA(Z_PC, t71, t80);
            tx = SCALAR_SUB(t60, t61);
            ty = SCALAR_SET1(0.5 * 7);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t80 = SCALAR_FMA(tx, ty, t80);
            tx = SCALAR_LOAD((temp + 144 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t80);
            SCALAR_STORE((temp + 144 * NPTS_LOCAL + p_inner), tx);
         }
      }

      size_t npts_inner_upper = SIMD_LENGTH * (npts_inner / SIMD_LENGTH);
      size_t p_inner = 0;
      for(p_inner = 0; p_inner < npts_inner_upper; p_inner += SIMD_LENGTH) {
         double *Xik = (Xi + p_outer + p_inner);
         double *Gik = (Gi + p_outer + p_inner);

         SIMD_TYPE tx, wg, xik, gik;
         tx  = SIMD_ALIGNED_LOAD((temp + 100 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 101 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 102 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 103 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 104 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 105 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 106 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 109 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 101 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 103 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 104 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 106 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 115 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 102 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 104 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 105 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 109 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 120 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 103 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 106 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 115 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 121 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 3 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 104 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 4 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 105 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 109 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 120 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 127 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 5 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 106 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 115 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 121 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 128 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 129 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 6 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 129 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 7 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 134 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 8 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 109 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 120 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 127 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 134 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 135 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 9 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 115 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 121 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 128 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 129 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 136 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 137 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 138 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 139 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 10 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 129 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 137 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 138 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 139 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 141 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 11 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 138 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 139 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 141 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 142 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 12 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 134 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 139 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 141 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 142 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 143 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 13 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 120 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 3 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 3 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 4 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 4 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 127 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 5 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 5 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 6 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 6 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 7 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 7 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 134 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 8 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 8 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 135 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 9 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 9 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 10 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 10 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 141 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 11 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 11 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 142 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 12 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 12 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 143 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 13 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 13 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 144 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 14 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 14 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 14 * ldG), gik);
      }

      for(; p_inner < npts_inner; p_inner += SCALAR_LENGTH) {
         double *Xik = (Xi + p_outer + p_inner);
         double *Gik = (Gi + p_outer + p_inner);

         SCALAR_TYPE tx, wg, xik, gik;
         tx  = SCALAR_LOAD((temp + 100 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 101 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 102 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 103 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 3 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 3 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 104 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 4 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 4 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 105 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 5 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 5 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 106 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 6 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 6 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 7 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 7 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 8 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 8 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 109 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 9 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 9 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 10 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 10 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 11 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 11 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 12 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 12 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 13 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 13 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 14 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 14 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 101 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 103 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 104 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 106 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 3 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 3 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 4 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 4 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 5 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 5 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 6 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 6 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 7 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 7 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 8 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 8 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 9 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 9 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 115 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 10 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 10 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 11 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 11 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 12 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 12 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 13 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 13 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 14 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 14 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 102 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 104 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 105 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 3 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 3 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 4 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 4 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 109 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 5 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 5 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 6 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 6 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 7 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 7 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 8 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 8 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 9 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 9 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 10 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 10 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 11 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 11 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 12 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 12 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 13 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 13 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 120 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 14 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 14 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 103 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 3 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 106 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 3 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 3 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 3 * ldX));
         gik = SCALAR_LOAD((Gik + 3 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 3 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 3 * ldX));
         gik = SCALAR_LOAD((Gik + 4 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 4 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 3 * ldX));
         gik = SCALAR_LOAD((Gik + 5 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 5 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 115 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 3 * ldX));
         gik = SCALAR_LOAD((Gik + 6 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 6 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 3 * ldX));
         gik = SCALAR_LOAD((Gik + 7 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 7 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 3 * ldX));
         gik = SCALAR_LOAD((Gik + 8 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 8 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 3 * ldX));
         gik = SCALAR_LOAD((Gik + 9 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 9 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 121 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 3 * ldX));
         gik = SCALAR_LOAD((Gik + 10 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 10 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 3 * ldX));
         gik = SCALAR_LOAD((Gik + 11 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 11 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 3 * ldX));
         gik = SCALAR_LOAD((Gik + 12 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 12 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 3 * ldX));
         gik = SCALAR_LOAD((Gik + 13 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 13 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 3 * ldX));
         gik = SCALAR_LOAD((Gik + 14 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 14 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 104 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 4 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 4 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 4 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 4 * ldX));
         gik = SCALAR_LOAD((Gik + 3 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 3 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 4 * ldX));
         gik = SCALAR_LOAD((Gik + 4 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 4 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 4 * ldX));
         gik = SCALAR_LOAD((Gik + 5 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 5 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 4 * ldX));
         gik = SCALAR_LOAD((Gik + 6 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 6 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 4 * ldX));
         gik = SCALAR_LOAD((Gik + 7 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 7 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 4 * ldX));
         gik = SCALAR_LOAD((Gik + 8 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 8 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 4 * ldX));
         gik = SCALAR_LOAD((Gik + 9 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 9 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 4 * ldX));
         gik = SCALAR_LOAD((Gik + 10 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 10 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 4 * ldX));
         gik = SCALAR_LOAD((Gik + 11 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 11 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 4 * ldX));
         gik = SCALAR_LOAD((Gik + 12 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 12 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 4 * ldX));
         gik = SCALAR_LOAD((Gik + 13 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 13 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 4 * ldX));
         gik = SCALAR_LOAD((Gik + 14 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 14 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 105 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 5 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 5 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 109 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 5 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 5 * ldX));
         gik = SCALAR_LOAD((Gik + 3 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 3 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 5 * ldX));
         gik = SCALAR_LOAD((Gik + 4 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 4 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 5 * ldX));
         gik = SCALAR_LOAD((Gik + 5 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 5 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 5 * ldX));
         gik = SCALAR_LOAD((Gik + 6 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 6 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 5 * ldX));
         gik = SCALAR_LOAD((Gik + 7 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 7 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 5 * ldX));
         gik = SCALAR_LOAD((Gik + 8 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 8 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 120 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 5 * ldX));
         gik = SCALAR_LOAD((Gik + 9 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 9 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 5 * ldX));
         gik = SCALAR_LOAD((Gik + 10 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 10 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 5 * ldX));
         gik = SCALAR_LOAD((Gik + 11 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 11 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 5 * ldX));
         gik = SCALAR_LOAD((Gik + 12 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 12 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 5 * ldX));
         gik = SCALAR_LOAD((Gik + 13 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 13 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 127 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 5 * ldX));
         gik = SCALAR_LOAD((Gik + 14 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 14 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 106 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 6 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 6 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 6 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 115 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 6 * ldX));
         gik = SCALAR_LOAD((Gik + 3 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 3 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 6 * ldX));
         gik = SCALAR_LOAD((Gik + 4 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 4 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 6 * ldX));
         gik = SCALAR_LOAD((Gik + 5 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 5 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 121 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 6 * ldX));
         gik = SCALAR_LOAD((Gik + 6 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 6 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 6 * ldX));
         gik = SCALAR_LOAD((Gik + 7 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 7 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 6 * ldX));
         gik = SCALAR_LOAD((Gik + 8 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 8 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 6 * ldX));
         gik = SCALAR_LOAD((Gik + 9 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 9 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 128 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 6 * ldX));
         gik = SCALAR_LOAD((Gik + 10 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 10 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 129 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 6 * ldX));
         gik = SCALAR_LOAD((Gik + 11 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 11 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 6 * ldX));
         gik = SCALAR_LOAD((Gik + 12 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 12 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 6 * ldX));
         gik = SCALAR_LOAD((Gik + 13 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 13 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 6 * ldX));
         gik = SCALAR_LOAD((Gik + 14 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 14 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 107 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 7 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 7 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 7 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 7 * ldX));
         gik = SCALAR_LOAD((Gik + 3 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 3 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 7 * ldX));
         gik = SCALAR_LOAD((Gik + 4 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 4 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 7 * ldX));
         gik = SCALAR_LOAD((Gik + 5 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 5 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 7 * ldX));
         gik = SCALAR_LOAD((Gik + 6 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 6 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 7 * ldX));
         gik = SCALAR_LOAD((Gik + 7 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 7 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 7 * ldX));
         gik = SCALAR_LOAD((Gik + 8 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 8 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 7 * ldX));
         gik = SCALAR_LOAD((Gik + 9 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 9 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 129 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 7 * ldX));
         gik = SCALAR_LOAD((Gik + 10 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 10 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 7 * ldX));
         gik = SCALAR_LOAD((Gik + 11 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 11 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 7 * ldX));
         gik = SCALAR_LOAD((Gik + 12 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 12 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 7 * ldX));
         gik = SCALAR_LOAD((Gik + 13 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 13 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 7 * ldX));
         gik = SCALAR_LOAD((Gik + 14 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 14 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 108 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 8 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 8 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 8 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 8 * ldX));
         gik = SCALAR_LOAD((Gik + 3 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 3 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 8 * ldX));
         gik = SCALAR_LOAD((Gik + 4 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 4 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 8 * ldX));
         gik = SCALAR_LOAD((Gik + 5 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 5 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 8 * ldX));
         gik = SCALAR_LOAD((Gik + 6 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 6 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 8 * ldX));
         gik = SCALAR_LOAD((Gik + 7 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 7 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 8 * ldX));
         gik = SCALAR_LOAD((Gik + 8 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 8 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 8 * ldX));
         gik = SCALAR_LOAD((Gik + 9 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 9 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 8 * ldX));
         gik = SCALAR_LOAD((Gik + 10 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 10 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 8 * ldX));
         gik = SCALAR_LOAD((Gik + 11 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 11 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 8 * ldX));
         gik = SCALAR_LOAD((Gik + 12 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 12 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 8 * ldX));
         gik = SCALAR_LOAD((Gik + 13 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 13 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 134 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 8 * ldX));
         gik = SCALAR_LOAD((Gik + 14 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 14 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 109 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 9 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 9 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 9 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 9 * ldX));
         gik = SCALAR_LOAD((Gik + 3 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 3 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 9 * ldX));
         gik = SCALAR_LOAD((Gik + 4 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 4 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 120 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 9 * ldX));
         gik = SCALAR_LOAD((Gik + 5 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 5 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 9 * ldX));
         gik = SCALAR_LOAD((Gik + 6 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 6 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 9 * ldX));
         gik = SCALAR_LOAD((Gik + 7 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 7 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 9 * ldX));
         gik = SCALAR_LOAD((Gik + 8 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 8 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 127 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 9 * ldX));
         gik = SCALAR_LOAD((Gik + 9 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 9 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 9 * ldX));
         gik = SCALAR_LOAD((Gik + 10 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 10 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 9 * ldX));
         gik = SCALAR_LOAD((Gik + 11 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 11 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 9 * ldX));
         gik = SCALAR_LOAD((Gik + 12 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 12 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 134 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 9 * ldX));
         gik = SCALAR_LOAD((Gik + 13 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 13 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 135 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 9 * ldX));
         gik = SCALAR_LOAD((Gik + 14 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 14 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 110 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 10 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 115 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 10 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 10 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 121 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 10 * ldX));
         gik = SCALAR_LOAD((Gik + 3 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 3 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 10 * ldX));
         gik = SCALAR_LOAD((Gik + 4 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 4 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 10 * ldX));
         gik = SCALAR_LOAD((Gik + 5 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 5 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 128 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 10 * ldX));
         gik = SCALAR_LOAD((Gik + 6 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 6 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 129 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 10 * ldX));
         gik = SCALAR_LOAD((Gik + 7 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 7 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 10 * ldX));
         gik = SCALAR_LOAD((Gik + 8 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 8 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 10 * ldX));
         gik = SCALAR_LOAD((Gik + 9 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 9 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 136 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 10 * ldX));
         gik = SCALAR_LOAD((Gik + 10 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 10 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 137 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 10 * ldX));
         gik = SCALAR_LOAD((Gik + 11 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 11 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 138 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 10 * ldX));
         gik = SCALAR_LOAD((Gik + 12 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 12 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 139 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 10 * ldX));
         gik = SCALAR_LOAD((Gik + 13 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 13 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 10 * ldX));
         gik = SCALAR_LOAD((Gik + 14 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 14 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 111 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 11 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 116 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 11 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 11 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 122 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 11 * ldX));
         gik = SCALAR_LOAD((Gik + 3 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 3 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 11 * ldX));
         gik = SCALAR_LOAD((Gik + 4 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 4 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 11 * ldX));
         gik = SCALAR_LOAD((Gik + 5 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 5 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 129 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 11 * ldX));
         gik = SCALAR_LOAD((Gik + 6 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 6 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 11 * ldX));
         gik = SCALAR_LOAD((Gik + 7 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 7 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 11 * ldX));
         gik = SCALAR_LOAD((Gik + 8 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 8 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 11 * ldX));
         gik = SCALAR_LOAD((Gik + 9 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 9 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 137 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 11 * ldX));
         gik = SCALAR_LOAD((Gik + 10 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 10 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 138 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 11 * ldX));
         gik = SCALAR_LOAD((Gik + 11 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 11 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 139 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 11 * ldX));
         gik = SCALAR_LOAD((Gik + 12 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 12 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 11 * ldX));
         gik = SCALAR_LOAD((Gik + 13 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 13 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 141 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 11 * ldX));
         gik = SCALAR_LOAD((Gik + 14 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 14 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 112 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 12 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 117 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 12 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 12 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 123 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 12 * ldX));
         gik = SCALAR_LOAD((Gik + 3 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 3 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 12 * ldX));
         gik = SCALAR_LOAD((Gik + 4 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 4 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 12 * ldX));
         gik = SCALAR_LOAD((Gik + 5 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 5 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 130 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 12 * ldX));
         gik = SCALAR_LOAD((Gik + 6 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 6 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 12 * ldX));
         gik = SCALAR_LOAD((Gik + 7 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 7 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 12 * ldX));
         gik = SCALAR_LOAD((Gik + 8 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 8 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 12 * ldX));
         gik = SCALAR_LOAD((Gik + 9 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 9 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 138 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 12 * ldX));
         gik = SCALAR_LOAD((Gik + 10 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 10 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 139 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 12 * ldX));
         gik = SCALAR_LOAD((Gik + 11 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 11 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 12 * ldX));
         gik = SCALAR_LOAD((Gik + 12 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 12 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 141 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 12 * ldX));
         gik = SCALAR_LOAD((Gik + 13 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 13 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 142 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 12 * ldX));
         gik = SCALAR_LOAD((Gik + 14 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 14 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 113 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 13 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 118 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 13 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 13 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 124 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 13 * ldX));
         gik = SCALAR_LOAD((Gik + 3 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 3 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 13 * ldX));
         gik = SCALAR_LOAD((Gik + 4 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 4 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 13 * ldX));
         gik = SCALAR_LOAD((Gik + 5 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 5 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 131 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 13 * ldX));
         gik = SCALAR_LOAD((Gik + 6 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 6 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 13 * ldX));
         gik = SCALAR_LOAD((Gik + 7 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 7 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 13 * ldX));
         gik = SCALAR_LOAD((Gik + 8 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 8 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 134 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 13 * ldX));
         gik = SCALAR_LOAD((Gik + 9 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 9 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 139 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 13 * ldX));
         gik = SCALAR_LOAD((Gik + 10 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 10 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 13 * ldX));
         gik = SCALAR_LOAD((Gik + 11 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 11 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 141 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 13 * ldX));
         gik = SCALAR_LOAD((Gik + 12 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 12 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 142 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 13 * ldX));
         gik = SCALAR_LOAD((Gik + 13 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 13 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 143 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 13 * ldX));
         gik = SCALAR_LOAD((Gik + 14 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 14 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 114 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 14 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 119 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 14 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 120 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 14 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 125 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 14 * ldX));
         gik = SCALAR_LOAD((Gik + 3 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 3 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 126 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 14 * ldX));
         gik = SCALAR_LOAD((Gik + 4 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 4 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 127 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 14 * ldX));
         gik = SCALAR_LOAD((Gik + 5 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 5 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 132 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 14 * ldX));
         gik = SCALAR_LOAD((Gik + 6 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 6 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 133 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 14 * ldX));
         gik = SCALAR_LOAD((Gik + 7 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 7 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 134 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 14 * ldX));
         gik = SCALAR_LOAD((Gik + 8 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 8 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 135 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 14 * ldX));
         gik = SCALAR_LOAD((Gik + 9 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 9 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 140 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 14 * ldX));
         gik = SCALAR_LOAD((Gik + 10 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 10 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 141 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 14 * ldX));
         gik = SCALAR_LOAD((Gik + 11 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 11 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 142 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 14 * ldX));
         gik = SCALAR_LOAD((Gik + 12 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 12 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 143 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 14 * ldX));
         gik = SCALAR_LOAD((Gik + 13 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 13 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 144 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 14 * ldX));
         gik = SCALAR_LOAD((Gik + 14 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 14 * ldG), gik);
      }
   }
}
}
