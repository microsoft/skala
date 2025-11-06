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
#include "integral_0_0.hpp"

#define PI 3.14159265358979323846

namespace XCPU {
void integral_0_0(size_t npts,
                  double *_points,
                  point /*rA*/,
                  point /*rB*/,
                  int nprim_pairs,
                  prim_pair *prim_pairs,
                  double *Xi,
                  double *Xj,
                  int ldX,
                  double *Gi,
                  double *Gj,
                  int ldG, 
                  double *weights,
                  double * /*boys_table*/) {
   __attribute__((__aligned__(64))) double buffer[1 * NPTS_LOCAL + 3 * NPTS_LOCAL];

   double *temp       = (buffer + 0);
   double *Tval       = (buffer + 1 * NPTS_LOCAL + 0 * NPTS_LOCAL);
  // double *Tval_inv_e = (buffer + 1 * NPTS_LOCAL + 1 * NPTS_LOCAL);
   double *FmT        = (buffer + 1 * NPTS_LOCAL + 2 * NPTS_LOCAL);

   size_t npts_upper = NPTS_LOCAL * (npts / NPTS_LOCAL);
   size_t p_outer = 0;
   for(p_outer = 0; p_outer < npts_upper; p_outer += NPTS_LOCAL) {
      double *_point_outer = (_points + p_outer);

      for(int i = 0; i < 1 * NPTS_LOCAL; i += SIMD_LENGTH) SIMD_ALIGNED_STORE((temp + i), SIMD_ZERO());

      for(int ij = 0; ij < nprim_pairs; ++ij) {
         double RHO = prim_pairs[ij].gamma;

         double xP = prim_pairs[ij].P.x;
         double yP = prim_pairs[ij].P.y;
         double zP = prim_pairs[ij].P.z;

         double eval = prim_pairs[ij].K_coeff_prod;
         if(std::abs(eval) < shpair_screen_tol) continue;

         // Evaluate T Values
         for(size_t p_inner = 0; p_inner < NPTS_LOCAL; p_inner += SIMD_LENGTH) {
            SIMD_TYPE xC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 0 * npts));
            SIMD_TYPE yC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 1 * npts));
            SIMD_TYPE zC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 2 * npts));

            SIMD_TYPE X_PC = SIMD_SUB(SIMD_DUPLICATE(&(xP)), xC);
            SIMD_TYPE Y_PC = SIMD_SUB(SIMD_DUPLICATE(&(yP)), yC);
            SIMD_TYPE Z_PC = SIMD_SUB(SIMD_DUPLICATE(&(zP)), zC);

            X_PC = SIMD_MUL(X_PC, X_PC);
            X_PC = SIMD_FMA(Y_PC, Y_PC, X_PC);
            X_PC = SIMD_FMA(Z_PC, Z_PC, X_PC);
            X_PC = SIMD_MUL(SIMD_DUPLICATE(&(RHO)), X_PC);
            SIMD_ALIGNED_STORE((Tval + p_inner), X_PC);
         }

         // Evaluate Boys function
         //boys_elements<0>(NPTS_LOCAL, Tval, Tval_inv_e, FmT, boys_table);
         boys_elements_0(NPTS_LOCAL,Tval,FmT); 

         // Evaluate VRR Buffer
         for(size_t p_inner = 0; p_inner < NPTS_LOCAL; p_inner += SIMD_LENGTH) {
            SIMD_TYPE tx, t00;

            t00 = SIMD_ALIGNED_LOAD((FmT + p_inner));

            t00 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t00);
            tx = SIMD_ALIGNED_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t00);
            SIMD_ALIGNED_STORE((temp + 0 * NPTS_LOCAL + p_inner), tx);
         }
      }

      for(size_t p_inner = 0; p_inner < NPTS_LOCAL; p_inner += SIMD_LENGTH) {
         double *Xik = (Xi + p_outer + p_inner);
         double *Xjk = (Xj + p_outer + p_inner);
         double *Gik = (Gi + p_outer + p_inner);
         double *Gjk = (Gj + p_outer + p_inner);

         SIMD_TYPE const_value_v = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         double const_value, X_ABp, Y_ABp, Z_ABp, comb_m_i, comb_n_j, comb_p_k;
         SIMD_TYPE const_value_w;
         SIMD_TYPE tx, ty, tz, tw, t0;

         X_ABp = 1.0; comb_m_i = 1.0;
         Y_ABp = 1.0; comb_n_j = 1.0;
         Z_ABp = 1.0; comb_p_k = 1.0;
         const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
         const_value_w = SIMD_MUL(const_value_v, SIMD_DUPLICATE(&(const_value)));
         tx = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         ty = SIMD_UNALIGNED_LOAD((Xjk + 0 * ldX));
         tz = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));
         tw = SIMD_UNALIGNED_LOAD((Gjk + 0 * ldG));
         t0 = SIMD_ALIGNED_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
         t0 = SIMD_MUL(t0, const_value_w);
         tz = SIMD_FMA(ty, t0, tz);
         tw = SIMD_FMA(tx, t0, tw);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), tz);
         SIMD_UNALIGNED_STORE((Gjk + 0 * ldG), tw);
      }
   }

   for(; p_outer < npts; p_outer += NPTS_LOCAL) {
     size_t npts_inner = std::min((size_t) NPTS_LOCAL, npts - p_outer);
      double *_point_outer = (_points + p_outer);

      for(int i = 0; i < 1 * NPTS_LOCAL; i += SIMD_LENGTH) SIMD_ALIGNED_STORE((temp + i), SIMD_ZERO());

      for(int ij = 0; ij < nprim_pairs; ++ij) {
         double RHO = prim_pairs[ij].gamma;

         double xP = prim_pairs[ij].P.x;
         double yP = prim_pairs[ij].P.y;
         double zP = prim_pairs[ij].P.z;

         double eval = prim_pairs[ij].K_coeff_prod;
         if(std::abs(eval) < shpair_screen_tol) continue;

         // Evaluate T Values
         size_t npts_inner_upper = SIMD_LENGTH * (npts_inner / SIMD_LENGTH);
         size_t p_inner = 0;
         for(p_inner = 0; p_inner < npts_inner_upper; p_inner += SIMD_LENGTH) {
            SIMD_TYPE xC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 0 * npts));
            SIMD_TYPE yC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 1 * npts));
            SIMD_TYPE zC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 2 * npts));

            SIMD_TYPE X_PC = SIMD_SUB(SIMD_DUPLICATE(&(xP)), xC);
            SIMD_TYPE Y_PC = SIMD_SUB(SIMD_DUPLICATE(&(yP)), yC);
            SIMD_TYPE Z_PC = SIMD_SUB(SIMD_DUPLICATE(&(zP)), zC);

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

            SCALAR_TYPE X_PC = SCALAR_SUB(SCALAR_DUPLICATE(&(xP)), xC);
            SCALAR_TYPE Y_PC = SCALAR_SUB(SCALAR_DUPLICATE(&(yP)), yC);
            SCALAR_TYPE Z_PC = SCALAR_SUB(SCALAR_DUPLICATE(&(zP)), zC);

            X_PC = SCALAR_MUL(X_PC, X_PC);
            X_PC = SCALAR_FMA(Y_PC, Y_PC, X_PC);
            X_PC = SCALAR_FMA(Z_PC, Z_PC, X_PC);
            X_PC = SCALAR_MUL(SCALAR_DUPLICATE(&(RHO)), X_PC);
            SCALAR_STORE((Tval + p_inner), X_PC);
         }

         // Evaluate Boys function
         //boys_elements<0>(npts_inner, Tval, Tval_inv_e, FmT, boys_table);
         boys_elements_0(NPTS_LOCAL,Tval,FmT); 

         // Evaluate VRR Buffer
         p_inner = 0;
         for(p_inner = 0; p_inner < npts_inner_upper; p_inner += SIMD_LENGTH) {
            SIMD_TYPE tx, t00;

            t00 = SIMD_ALIGNED_LOAD((FmT + p_inner));

            t00 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t00);
            tx = SIMD_ALIGNED_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t00);
            SIMD_ALIGNED_STORE((temp + 0 * NPTS_LOCAL + p_inner), tx);
         }

         for(; p_inner < npts_inner; p_inner += SCALAR_LENGTH) {
            SCALAR_TYPE tx, t00;

            t00 = SCALAR_LOAD((FmT + p_inner));

            t00 = SCALAR_MUL(SCALAR_DUPLICATE(&(eval)), t00);
            tx = SCALAR_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t00);
            SCALAR_STORE((temp + 0 * NPTS_LOCAL + p_inner), tx);
         }
      }

      size_t npts_inner_upper = SIMD_LENGTH * (npts_inner / SIMD_LENGTH);
      size_t p_inner = 0;
      for(p_inner = 0; p_inner < npts_inner_upper; p_inner += SIMD_LENGTH) {
         double *Xik = (Xi + p_outer + p_inner);
         double *Xjk = (Xj + p_outer + p_inner);
         double *Gik = (Gi + p_outer + p_inner);
         double *Gjk = (Gj + p_outer + p_inner);

         SIMD_TYPE const_value_v = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         double const_value, X_ABp, Y_ABp, Z_ABp, comb_m_i, comb_n_j, comb_p_k;
         SIMD_TYPE const_value_w;
         SIMD_TYPE tx, ty, tz, tw, t0;

         X_ABp = 1.0; comb_m_i = 1.0;
         Y_ABp = 1.0; comb_n_j = 1.0;
         Z_ABp = 1.0; comb_p_k = 1.0;
         const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
         const_value_w = SIMD_MUL(const_value_v, SIMD_DUPLICATE(&(const_value)));
         tx = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         ty = SIMD_UNALIGNED_LOAD((Xjk + 0 * ldX));
         tz = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));
         tw = SIMD_UNALIGNED_LOAD((Gjk + 0 * ldG));
         t0 = SIMD_ALIGNED_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
         t0 = SIMD_MUL(t0, const_value_w);
         tz = SIMD_FMA(ty, t0, tz);
         tw = SIMD_FMA(tx, t0, tw);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), tz);
         SIMD_UNALIGNED_STORE((Gjk + 0 * ldG), tw);
      }

      for(; p_inner < npts_inner; p_inner += SCALAR_LENGTH) {
         double *Xik = (Xi + p_outer + p_inner);
         double *Xjk = (Xj + p_outer + p_inner);
         double *Gik = (Gi + p_outer + p_inner);
         double *Gjk = (Gj + p_outer + p_inner);

         SCALAR_TYPE const_value_v = SCALAR_LOAD((weights + p_outer + p_inner));

         double const_value, X_ABp, Y_ABp, Z_ABp, comb_m_i, comb_n_j, comb_p_k;
         SCALAR_TYPE const_value_w;
         SCALAR_TYPE tx, ty, tz, tw, t0;

         X_ABp = 1.0; comb_m_i = 1.0;
         Y_ABp = 1.0; comb_n_j = 1.0;
         Z_ABp = 1.0; comb_p_k = 1.0;
         const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;
         const_value_w = SCALAR_MUL(const_value_v, SCALAR_DUPLICATE(&(const_value)));
         tx = SCALAR_LOAD((Xik + 0 * ldX));
         ty = SCALAR_LOAD((Xjk + 0 * ldX));
         tz = SCALAR_LOAD((Gik + 0 * ldG));
         tw = SCALAR_LOAD((Gjk + 0 * ldG));
         t0 = SCALAR_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
         t0 = SCALAR_MUL(t0, const_value_w);
         tz = SCALAR_FMA(ty, t0, tz);
         tw = SCALAR_FMA(tx, t0, tw);
         SCALAR_STORE((Gik + 0 * ldG), tz);
         SCALAR_STORE((Gjk + 0 * ldG), tw);
      }
   }
}
}
