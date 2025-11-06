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
#include "integral_1.hpp"

#define PI 3.14159265358979323846

namespace XCPU {
void integral_1(size_t npts,
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
   __attribute__((__aligned__(64))) double buffer[9 * NPTS_LOCAL + 3 * NPTS_LOCAL];

   double *temp       = (buffer + 0);
   double *Tval       = (buffer + 9 * NPTS_LOCAL + 0 * NPTS_LOCAL);
   double *Tval_inv_e = (buffer + 9 * NPTS_LOCAL + 1 * NPTS_LOCAL);
   double *FmT        = (buffer + 9 * NPTS_LOCAL + 2 * NPTS_LOCAL);

   size_t npts_upper = NPTS_LOCAL * (npts / NPTS_LOCAL);
   size_t p_outer = 0;
   for(p_outer = 0; p_outer < npts_upper; p_outer += NPTS_LOCAL) {
      double *_point_outer = (_points + p_outer);

      double xA = rA.x;
      double yA = rA.y;
      double zA = rA.z;

      for(int i = 0; i < 9 * NPTS_LOCAL; i += SIMD_LENGTH) SIMD_ALIGNED_STORE((temp + i), SIMD_ZERO());

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
         boys_elements<2>(NPTS_LOCAL, Tval, Tval_inv_e, FmT, boys_table);

         // Evaluate VRR Buffer
         for(size_t p_inner = 0; p_inner < NPTS_LOCAL; p_inner += SIMD_LENGTH) {
            SIMD_TYPE xC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 0 * npts));
            SIMD_TYPE yC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 1 * npts));
            SIMD_TYPE zC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 2 * npts));

            SIMD_TYPE X_PC = SIMD_SUB(SIMD_DUPLICATE(&(xA)), xC);
            SIMD_TYPE Y_PC = SIMD_SUB(SIMD_DUPLICATE(&(yA)), yC);
            SIMD_TYPE Z_PC = SIMD_SUB(SIMD_DUPLICATE(&(zA)), zC);

            SIMD_TYPE tval, tval_inv_e, tx, ty, t00, t01, t02, t10, t11, t20;

            tval = SIMD_ALIGNED_LOAD((Tval + p_inner));
            tval_inv_e = SIMD_ALIGNED_LOAD((Tval_inv_e + p_inner));

            t02 = SIMD_ALIGNED_LOAD((FmT + p_inner));
            t01 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t02), tval_inv_e), SIMD_SET1(0.66666666666666662966));
            t00 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t01), tval_inv_e), SIMD_SET1(2.00000000000000000000));

            t00 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t00);
            t01 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t01);
            t02 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t02);
            t10 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t00);
            t10 = SIMD_FNMA(X_PC, t01, t10);
            t11 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t01);
            t11 = SIMD_FNMA(X_PC, t02, t11);
            tx = SIMD_ALIGNED_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t10);
            SIMD_ALIGNED_STORE((temp + 0 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t10);
            t20 = SIMD_FNMA(X_PC, t11, t20);
            tx = SIMD_SUB(t00, t01);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t20 = SIMD_FMA(tx, ty, t20);
            tx = SIMD_ALIGNED_LOAD((temp + 3 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t20);
            SIMD_ALIGNED_STORE((temp + 3 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t10);
            t20 = SIMD_FNMA(Y_PC, t11, t20);
            tx = SIMD_ALIGNED_LOAD((temp + 4 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t20);
            SIMD_ALIGNED_STORE((temp + 4 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t10);
            t20 = SIMD_FNMA(Z_PC, t11, t20);
            tx = SIMD_ALIGNED_LOAD((temp + 5 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t20);
            SIMD_ALIGNED_STORE((temp + 5 * NPTS_LOCAL + p_inner), tx);
            t10 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t00);
            t10 = SIMD_FNMA(Y_PC, t01, t10);
            t11 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t01);
            t11 = SIMD_FNMA(Y_PC, t02, t11);
            tx = SIMD_ALIGNED_LOAD((temp + 1 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t10);
            SIMD_ALIGNED_STORE((temp + 1 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t10);
            t20 = SIMD_FNMA(Y_PC, t11, t20);
            tx = SIMD_SUB(t00, t01);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t20 = SIMD_FMA(tx, ty, t20);
            tx = SIMD_ALIGNED_LOAD((temp + 6 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t20);
            SIMD_ALIGNED_STORE((temp + 6 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t10);
            t20 = SIMD_FNMA(Z_PC, t11, t20);
            tx = SIMD_ALIGNED_LOAD((temp + 7 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t20);
            SIMD_ALIGNED_STORE((temp + 7 * NPTS_LOCAL + p_inner), tx);
            t10 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t00);
            t10 = SIMD_FNMA(Z_PC, t01, t10);
            t11 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t01);
            t11 = SIMD_FNMA(Z_PC, t02, t11);
            tx = SIMD_ALIGNED_LOAD((temp + 2 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t10);
            SIMD_ALIGNED_STORE((temp + 2 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t10);
            t20 = SIMD_FNMA(Z_PC, t11, t20);
            tx = SIMD_SUB(t00, t01);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t20 = SIMD_FMA(tx, ty, t20);
            tx = SIMD_ALIGNED_LOAD((temp + 8 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t20);
            SIMD_ALIGNED_STORE((temp + 8 * NPTS_LOCAL + p_inner), tx);
         }
      }

      for(size_t p_inner = 0; p_inner < NPTS_LOCAL; p_inner += SIMD_LENGTH) {
         double *Xik = (Xi + p_outer + p_inner);
         double *Gik = (Gi + p_outer + p_inner);

         SIMD_TYPE tx, wg, xik, gik;
         tx  = SIMD_ALIGNED_LOAD((temp + 3 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 4 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 5 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 4 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 6 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 7 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 5 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 7 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 8 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
      }
   }

   // cleanup code
   for(; p_outer < npts; p_outer += NPTS_LOCAL) {
      size_t npts_inner = std::min((size_t) NPTS_LOCAL, npts - p_outer);
      double *_point_outer = (_points + p_outer);

      double xA = rA.x;
      double yA = rA.y;
      double zA = rA.z;

      for(int i = 0; i < 9 * NPTS_LOCAL; i += SIMD_LENGTH) SIMD_ALIGNED_STORE((temp + i), SIMD_ZERO());

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
         boys_elements<2>(npts_inner, Tval, Tval_inv_e, FmT, boys_table);

         // Evaluate VRR Buffer
         p_inner = 0;
         for(p_inner = 0; p_inner < npts_inner_upper; p_inner += SIMD_LENGTH) {
            SIMD_TYPE xC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 0 * npts));
            SIMD_TYPE yC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 1 * npts));
            SIMD_TYPE zC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 2 * npts));

            SIMD_TYPE X_PC = SIMD_SUB(SIMD_DUPLICATE(&(xA)), xC);
            SIMD_TYPE Y_PC = SIMD_SUB(SIMD_DUPLICATE(&(yA)), yC);
            SIMD_TYPE Z_PC = SIMD_SUB(SIMD_DUPLICATE(&(zA)), zC);

            SIMD_TYPE tval, tval_inv_e, tx, ty, t00, t01, t02, t10, t11, t20;

            tval = SIMD_ALIGNED_LOAD((Tval + p_inner));
            tval_inv_e = SIMD_ALIGNED_LOAD((Tval_inv_e + p_inner));

            t02 = SIMD_ALIGNED_LOAD((FmT + p_inner));
            t01 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t02), tval_inv_e), SIMD_SET1(0.66666666666666662966));
            t00 = SIMD_MUL(SIMD_ADD(SIMD_MUL(tval, t01), tval_inv_e), SIMD_SET1(2.00000000000000000000));

            t00 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t00);
            t01 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t01);
            t02 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t02);
            t10 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t00);
            t10 = SIMD_FNMA(X_PC, t01, t10);
            t11 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t01);
            t11 = SIMD_FNMA(X_PC, t02, t11);
            tx = SIMD_ALIGNED_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t10);
            SIMD_ALIGNED_STORE((temp + 0 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(X_PA)), t10);
            t20 = SIMD_FNMA(X_PC, t11, t20);
            tx = SIMD_SUB(t00, t01);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t20 = SIMD_FMA(tx, ty, t20);
            tx = SIMD_ALIGNED_LOAD((temp + 3 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t20);
            SIMD_ALIGNED_STORE((temp + 3 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t10);
            t20 = SIMD_FNMA(Y_PC, t11, t20);
            tx = SIMD_ALIGNED_LOAD((temp + 4 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t20);
            SIMD_ALIGNED_STORE((temp + 4 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t10);
            t20 = SIMD_FNMA(Z_PC, t11, t20);
            tx = SIMD_ALIGNED_LOAD((temp + 5 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t20);
            SIMD_ALIGNED_STORE((temp + 5 * NPTS_LOCAL + p_inner), tx);
            t10 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t00);
            t10 = SIMD_FNMA(Y_PC, t01, t10);
            t11 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t01);
            t11 = SIMD_FNMA(Y_PC, t02, t11);
            tx = SIMD_ALIGNED_LOAD((temp + 1 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t10);
            SIMD_ALIGNED_STORE((temp + 1 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Y_PA)), t10);
            t20 = SIMD_FNMA(Y_PC, t11, t20);
            tx = SIMD_SUB(t00, t01);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t20 = SIMD_FMA(tx, ty, t20);
            tx = SIMD_ALIGNED_LOAD((temp + 6 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t20);
            SIMD_ALIGNED_STORE((temp + 6 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t10);
            t20 = SIMD_FNMA(Z_PC, t11, t20);
            tx = SIMD_ALIGNED_LOAD((temp + 7 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t20);
            SIMD_ALIGNED_STORE((temp + 7 * NPTS_LOCAL + p_inner), tx);
            t10 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t00);
            t10 = SIMD_FNMA(Z_PC, t01, t10);
            t11 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t01);
            t11 = SIMD_FNMA(Z_PC, t02, t11);
            tx = SIMD_ALIGNED_LOAD((temp + 2 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t10);
            SIMD_ALIGNED_STORE((temp + 2 * NPTS_LOCAL + p_inner), tx);
            t20 = SIMD_MUL(SIMD_DUPLICATE(&(Z_PA)), t10);
            t20 = SIMD_FNMA(Z_PC, t11, t20);
            tx = SIMD_SUB(t00, t01);
            ty = SIMD_SET1(0.5 * 1);
            ty = SIMD_MUL(ty, SIMD_DUPLICATE(&(RHO_INV)));
            t20 = SIMD_FMA(tx, ty, t20);
            tx = SIMD_ALIGNED_LOAD((temp + 8 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t20);
            SIMD_ALIGNED_STORE((temp + 8 * NPTS_LOCAL + p_inner), tx);
         }

         for(; p_inner < npts_inner; p_inner += SCALAR_LENGTH) {
            SCALAR_TYPE xC = SCALAR_LOAD((_point_outer + p_inner + 0 * npts));
            SCALAR_TYPE yC = SCALAR_LOAD((_point_outer + p_inner + 1 * npts));
            SCALAR_TYPE zC = SCALAR_LOAD((_point_outer + p_inner + 2 * npts));

            SCALAR_TYPE X_PC = SCALAR_SUB(SCALAR_DUPLICATE(&(xA)), xC);
            SCALAR_TYPE Y_PC = SCALAR_SUB(SCALAR_DUPLICATE(&(yA)), yC);
            SCALAR_TYPE Z_PC = SCALAR_SUB(SCALAR_DUPLICATE(&(zA)), zC);

            SCALAR_TYPE tval, tval_inv_e, tx, ty, t00, t01, t02, t10, t11, t20;

            tval = SCALAR_LOAD((Tval + p_inner));
            tval_inv_e = SCALAR_LOAD((Tval_inv_e + p_inner));

            t02 = SCALAR_LOAD((FmT + p_inner));
            t01 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(tval, t02), tval_inv_e), SCALAR_SET1(0.66666666666666662966));
            t00 = SCALAR_MUL(SCALAR_ADD(SCALAR_MUL(tval, t01), tval_inv_e), SCALAR_SET1(2.00000000000000000000));

            t00 = SCALAR_MUL(SCALAR_DUPLICATE(&(eval)), t00);
            t01 = SCALAR_MUL(SCALAR_DUPLICATE(&(eval)), t01);
            t02 = SCALAR_MUL(SCALAR_DUPLICATE(&(eval)), t02);
            t10 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t00);
            t10 = SCALAR_FNMA(X_PC, t01, t10);
            t11 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t01);
            t11 = SCALAR_FNMA(X_PC, t02, t11);
            tx = SCALAR_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t10);
            SCALAR_STORE((temp + 0 * NPTS_LOCAL + p_inner), tx);
            t20 = SCALAR_MUL(SCALAR_DUPLICATE(&(X_PA)), t10);
            t20 = SCALAR_FNMA(X_PC, t11, t20);
            tx = SCALAR_SUB(t00, t01);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t20 = SCALAR_FMA(tx, ty, t20);
            tx = SCALAR_LOAD((temp + 3 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t20);
            SCALAR_STORE((temp + 3 * NPTS_LOCAL + p_inner), tx);
            t20 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t10);
            t20 = SCALAR_FNMA(Y_PC, t11, t20);
            tx = SCALAR_LOAD((temp + 4 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t20);
            SCALAR_STORE((temp + 4 * NPTS_LOCAL + p_inner), tx);
            t20 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t10);
            t20 = SCALAR_FNMA(Z_PC, t11, t20);
            tx = SCALAR_LOAD((temp + 5 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t20);
            SCALAR_STORE((temp + 5 * NPTS_LOCAL + p_inner), tx);
            t10 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t00);
            t10 = SCALAR_FNMA(Y_PC, t01, t10);
            t11 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t01);
            t11 = SCALAR_FNMA(Y_PC, t02, t11);
            tx = SCALAR_LOAD((temp + 1 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t10);
            SCALAR_STORE((temp + 1 * NPTS_LOCAL + p_inner), tx);
            t20 = SCALAR_MUL(SCALAR_DUPLICATE(&(Y_PA)), t10);
            t20 = SCALAR_FNMA(Y_PC, t11, t20);
            tx = SCALAR_SUB(t00, t01);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t20 = SCALAR_FMA(tx, ty, t20);
            tx = SCALAR_LOAD((temp + 6 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t20);
            SCALAR_STORE((temp + 6 * NPTS_LOCAL + p_inner), tx);
            t20 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t10);
            t20 = SCALAR_FNMA(Z_PC, t11, t20);
            tx = SCALAR_LOAD((temp + 7 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t20);
            SCALAR_STORE((temp + 7 * NPTS_LOCAL + p_inner), tx);
            t10 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t00);
            t10 = SCALAR_FNMA(Z_PC, t01, t10);
            t11 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t01);
            t11 = SCALAR_FNMA(Z_PC, t02, t11);
            tx = SCALAR_LOAD((temp + 2 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t10);
            SCALAR_STORE((temp + 2 * NPTS_LOCAL + p_inner), tx);
            t20 = SCALAR_MUL(SCALAR_DUPLICATE(&(Z_PA)), t10);
            t20 = SCALAR_FNMA(Z_PC, t11, t20);
            tx = SCALAR_SUB(t00, t01);
            ty = SCALAR_SET1(0.5 * 1);
            ty = SCALAR_MUL(ty, SCALAR_DUPLICATE(&(RHO_INV)));
            t20 = SCALAR_FMA(tx, ty, t20);
            tx = SCALAR_LOAD((temp + 8 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t20);
            SCALAR_STORE((temp + 8 * NPTS_LOCAL + p_inner), tx);
         }
      }

      size_t npts_inner_upper = SIMD_LENGTH * (npts_inner / SIMD_LENGTH);
      size_t p_inner = 0;
      for(p_inner = 0; p_inner < npts_inner_upper; p_inner += SIMD_LENGTH) {
         double *Xik = (Xi + p_outer + p_inner);
         double *Gik = (Gi + p_outer + p_inner);

         SIMD_TYPE tx, wg, xik, gik;
         tx  = SIMD_ALIGNED_LOAD((temp + 3 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 4 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 5 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 4 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 6 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 7 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 1 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 5 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 7 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 1 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 1 * ldG), gik);
         tx  = SIMD_ALIGNED_LOAD((temp + 8 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 2 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 2 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 2 * ldG), gik);
      }

      for(; p_inner < npts_inner; p_inner += SCALAR_LENGTH) {
         double *Xik = (Xi + p_outer + p_inner);
         double *Gik = (Gi + p_outer + p_inner);

         SCALAR_TYPE tx, wg, xik, gik;
         tx  = SCALAR_LOAD((temp + 3 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 4 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 5 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 4 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 6 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 7 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 1 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 5 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 7 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 1 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 1 * ldG), gik);
         tx  = SCALAR_LOAD((temp + 8 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 2 * ldX));
         gik = SCALAR_LOAD((Gik + 2 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 2 * ldG), gik);
      }
   }
}
}
