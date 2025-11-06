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
#include <stdio.h>
#include <stdlib.h>
#include "../include/cpu/integral_data_types.hpp"
#include "../include/cpu/obara_saika_integrals.hpp"
#include "integral_0.hpp"
#include "integral_1.hpp"
#include "integral_2.hpp"
#include "integral_3.hpp"
#include "integral_4.hpp"
#include "integral_0_0.hpp"
#include "integral_1_0.hpp"
#include "integral_1_1.hpp"
#include "integral_2_0.hpp"
#include "integral_2_1.hpp"
#include "integral_2_2.hpp"
#include "integral_3_0.hpp"
#include "integral_3_1.hpp"
#include "integral_3_2.hpp"
#include "integral_3_3.hpp"
#include "integral_4_0.hpp"
#include "integral_4_1.hpp"
#include "integral_4_2.hpp"
#include "integral_4_3.hpp"
#include "integral_4_4.hpp"
namespace XCPU {
void generate_shell_pair( const shells& A, const shells& B, prim_pair *prim_pairs) {
   // L Values
   const auto xA = A.origin.x;
   const auto yA = A.origin.y;
   const auto zA = A.origin.z;

   const auto xB = B.origin.x;
   const auto yB = B.origin.y;
   const auto zB = B.origin.z;

   double rABx = xA - xB;
   double rABy = yA - yB;
   double rABz = zA - zB;

   const double dAB = rABx*rABx + rABy*rABy + rABz*rABz;

   const int nprim_A = A.m;
   const int nprim_B = B.m;
   for(int i = 0, ij = 0; i < nprim_A; ++i       )
   for(int j = 0        ; j < nprim_B; ++j, ++ij ) {
      auto& pair = prim_pairs[ij];
      const auto alpha_A = A.coeff[i].alpha;
      const auto alpha_B = B.coeff[j].alpha;

      pair.gamma = alpha_A + alpha_B;
      pair.gamma_inv = 1. / pair.gamma;

      pair.P.x = (alpha_A * xA + alpha_B * xB) * pair.gamma_inv;
      pair.P.y = (alpha_A * yA + alpha_B * yB) * pair.gamma_inv;
      pair.P.z = (alpha_A * zA + alpha_B * zB) * pair.gamma_inv;

      pair.PA.x = pair.P.x - xA;
      pair.PA.y = pair.P.y - yA;
      pair.PA.z = pair.P.z - zA;

      pair.PB.x = pair.P.x - xB;
      pair.PB.y = pair.P.y - yB;
      pair.PB.z = pair.P.z - zB;

      pair.K_coeff_prod = 2 * M_PI * pair.gamma_inv * std::exp( - alpha_A * alpha_B * dAB * pair.gamma_inv ) * A.coeff[i].coeff * B.coeff[j].coeff;
   }
}

void compute_integral_shell_pair(int is_diag,
                  size_t npts,
                  double *points,
                  int lA,
                  int lB,
                  point rA,
                  point rB,
                  int nprim_pairs,
                  prim_pair *prim_pairs,
                  double *Xi,
                  double *Xj,
                  int ldX,
                  double *Gi,
                  double *Gj,
                  int ldG, 
                  double *weights, 
                  double *boys_table) {
   if (is_diag) {
      if(lA == 0) {
         integral_0(npts,
                    points,
                    rA,
                    rB,
                    nprim_pairs,
                    prim_pairs,
                    Xi,
                    ldX,
                    Gi,
                    ldG, 
                    weights, 
                    boys_table);
      } else if(lA == 1) {
        integral_1(npts,
                    points,
                   rA,
                   rB,
                   nprim_pairs,
                   prim_pairs,
                   Xi,
                   ldX,
                   Gi,
                   ldG, 
                   weights, 
                   boys_table);
      } else if(lA == 2) {
        integral_2(npts,
                    points,
                   rA,
                   rB,
                   nprim_pairs,
                   prim_pairs,
                   Xi,
                   ldX,
                   Gi,
                   ldG, 
                   weights, 
                   boys_table);
      } else if(lA == 3) {
        integral_3(npts,
                    points,
                   rA,
                   rB,
                   nprim_pairs,
                   prim_pairs,
                   Xi,
                   ldX,
                   Gi,
                   ldG, 
                   weights, 
                   boys_table);
      } else if(lA == 4) {
        integral_4(npts,
                    points,
                   rA,
                   rB,
                   nprim_pairs,
                   prim_pairs,
                   Xi,
                   ldX,
                   Gi,
                   ldG, 
                   weights, 
                   boys_table);
      } else {
         printf("Type not defined!\n");
      }
   } else {
      if((lA == 0) && (lB == 0)) {
         integral_0_0(npts,
                      points,
                      rA,
                      rB,
                      nprim_pairs,
                      prim_pairs,
                      Xi,
                      Xj,
                      ldX,
                      Gi,
                      Gj,
                      ldG, 
                      weights, 
                      boys_table);
      } else if((lA == 1) && (lB == 0)) {
            integral_1_0(npts,
                         points,
                         rA,
                         rB,
                         nprim_pairs,
                         prim_pairs,
                         Xi,
                         Xj,
                         ldX,
                         Gi,
                         Gj,
                         ldG, 
                         weights, 
                         boys_table);
      } else if((lA == 0) && (lB == 1)) {
         integral_1_0(npts,
                      points,
                      rB,
                      rA,
                      nprim_pairs,
                      prim_pairs,
                      Xj,
                      Xi,
                      ldX,
                      Gj,
                      Gi,
                      ldG, 
                      weights, 
                      boys_table);
      } else if((lA == 1) && (lB == 1)) {
        integral_1_1(npts,
                     points,
                     rA,
                     rB,
                     nprim_pairs,
                     prim_pairs,
                     Xi,
                     Xj,
                     ldX,
                     Gi,
                     Gj,
                     ldG, 
                     weights, 
                     boys_table);
      } else if((lA == 2) && (lB == 0)) {
            integral_2_0(npts,
                         points,
                         rA,
                         rB,
                         nprim_pairs,
                         prim_pairs,
                         Xi,
                         Xj,
                         ldX,
                         Gi,
                         Gj,
                         ldG, 
                         weights, 
                         boys_table);
      } else if((lA == 0) && (lB == 2)) {
         integral_2_0(npts,
                      points,
                      rB,
                      rA,
                      nprim_pairs,
                      prim_pairs,
                      Xj,
                      Xi,
                      ldX,
                      Gj,
                      Gi,
                      ldG, 
                      weights, 
                      boys_table);
      } else if((lA == 2) && (lB == 1)) {
            integral_2_1(npts,
                         points,
                         rA,
                         rB,
                         nprim_pairs,
                         prim_pairs,
                         Xi,
                         Xj,
                         ldX,
                         Gi,
                         Gj,
                         ldG, 
                         weights, 
                         boys_table);
      } else if((lA == 1) && (lB == 2)) {
         integral_2_1(npts,
                      points,
                      rB,
                      rA,
                      nprim_pairs,
                      prim_pairs,
                      Xj,
                      Xi,
                      ldX,
                      Gj,
                      Gi,
                      ldG, 
                      weights, 
                      boys_table);
      } else if((lA == 2) && (lB == 2)) {
        integral_2_2(npts,
                     points,
                     rA,
                     rB,
                     nprim_pairs,
                     prim_pairs,
                     Xi,
                     Xj,
                     ldX,
                     Gi,
                     Gj,
                     ldG, 
                     weights, 
                     boys_table);
      } else if((lA == 3) && (lB == 0)) {
            integral_3_0(npts,
                         points,
                         rA,
                         rB,
                         nprim_pairs,
                         prim_pairs,
                         Xi,
                         Xj,
                         ldX,
                         Gi,
                         Gj,
                         ldG, 
                         weights, 
                         boys_table);
      } else if((lA == 0) && (lB == 3)) {
         integral_3_0(npts,
                      points,
                      rB,
                      rA,
                      nprim_pairs,
                      prim_pairs,
                      Xj,
                      Xi,
                      ldX,
                      Gj,
                      Gi,
                      ldG, 
                      weights, 
                      boys_table);
      } else if((lA == 3) && (lB == 1)) {
            integral_3_1(npts,
                         points,
                         rA,
                         rB,
                         nprim_pairs,
                         prim_pairs,
                         Xi,
                         Xj,
                         ldX,
                         Gi,
                         Gj,
                         ldG, 
                         weights, 
                         boys_table);
      } else if((lA == 1) && (lB == 3)) {
         integral_3_1(npts,
                      points,
                      rB,
                      rA,
                      nprim_pairs,
                      prim_pairs,
                      Xj,
                      Xi,
                      ldX,
                      Gj,
                      Gi,
                      ldG, 
                      weights, 
                      boys_table);
      } else if((lA == 3) && (lB == 2)) {
            integral_3_2(npts,
                         points,
                         rA,
                         rB,
                         nprim_pairs,
                         prim_pairs,
                         Xi,
                         Xj,
                         ldX,
                         Gi,
                         Gj,
                         ldG, 
                         weights, 
                         boys_table);
      } else if((lA == 2) && (lB == 3)) {
         integral_3_2(npts,
                      points,
                      rB,
                      rA,
                      nprim_pairs,
                      prim_pairs,
                      Xj,
                      Xi,
                      ldX,
                      Gj,
                      Gi,
                      ldG, 
                      weights, 
                      boys_table);
      } else if((lA == 3) && (lB == 3)) {
        integral_3_3(npts,
                     points,
                     rA,
                     rB,
                     nprim_pairs,
                     prim_pairs,
                     Xi,
                     Xj,
                     ldX,
                     Gi,
                     Gj,
                     ldG, 
                     weights, 
                     boys_table);
      } else if((lA == 4) && (lB == 0)) {
            integral_4_0(npts,
                         points,
                         rA,
                         rB,
                         nprim_pairs,
                         prim_pairs,
                         Xi,
                         Xj,
                         ldX,
                         Gi,
                         Gj,
                         ldG, 
                         weights, 
                         boys_table);
      } else if((lA == 0) && (lB == 4)) {
         integral_4_0(npts,
                      points,
                      rB,
                      rA,
                      nprim_pairs,
                      prim_pairs,
                      Xj,
                      Xi,
                      ldX,
                      Gj,
                      Gi,
                      ldG, 
                      weights, 
                      boys_table);
      } else if((lA == 4) && (lB == 1)) {
            integral_4_1(npts,
                         points,
                         rA,
                         rB,
                         nprim_pairs,
                         prim_pairs,
                         Xi,
                         Xj,
                         ldX,
                         Gi,
                         Gj,
                         ldG, 
                         weights, 
                         boys_table);
      } else if((lA == 1) && (lB == 4)) {
         integral_4_1(npts,
                      points,
                      rB,
                      rA,
                      nprim_pairs,
                      prim_pairs,
                      Xj,
                      Xi,
                      ldX,
                      Gj,
                      Gi,
                      ldG, 
                      weights, 
                      boys_table);
      } else if((lA == 4) && (lB == 2)) {
            integral_4_2(npts,
                         points,
                         rA,
                         rB,
                         nprim_pairs,
                         prim_pairs,
                         Xi,
                         Xj,
                         ldX,
                         Gi,
                         Gj,
                         ldG, 
                         weights, 
                         boys_table);
      } else if((lA == 2) && (lB == 4)) {
         integral_4_2(npts,
                      points,
                      rB,
                      rA,
                      nprim_pairs,
                      prim_pairs,
                      Xj,
                      Xi,
                      ldX,
                      Gj,
                      Gi,
                      ldG, 
                      weights, 
                      boys_table);
      } else if((lA == 4) && (lB == 3)) {
            integral_4_3(npts,
                         points,
                         rA,
                         rB,
                         nprim_pairs,
                         prim_pairs,
                         Xi,
                         Xj,
                         ldX,
                         Gi,
                         Gj,
                         ldG, 
                         weights, 
                         boys_table);
      } else if((lA == 3) && (lB == 4)) {
         integral_4_3(npts,
                      points,
                      rB,
                      rA,
                      nprim_pairs,
                      prim_pairs,
                      Xj,
                      Xi,
                      ldX,
                      Gj,
                      Gi,
                      ldG, 
                      weights, 
                      boys_table);
      } else if((lA == 4) && (lB == 4)) {
        integral_4_4(npts,
                     points,
                     rA,
                     rB,
                     nprim_pairs,
                     prim_pairs,
                     Xi,
                     Xj,
                     ldX,
                     Gi,
                     Gj,
                     ldG, 
                     weights, 
                     boys_table);
      } else {
         printf("Type not defined!\n");
      }
   }
}
}
