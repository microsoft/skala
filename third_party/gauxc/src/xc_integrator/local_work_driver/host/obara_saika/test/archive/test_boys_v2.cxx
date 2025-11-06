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
#include <sys/time.h>

#include <iostream>
#include <cmath>

#include "../include/chebyshev_boys_computation.hpp"

#define SQRT_PI_OVER_2 0.88622692545275801364

int main(int argc, char **argv) {
  
  int runs = atoi(argv[1]);
  double TVAL = atof(argv[2]);
  
  GauXC::gauxc_boys_init();

  struct timeval t0, t1;
  
  double f0, f1, f2, f3, f4, f5, f6, f7, f8;
  double g0, g1, g2, g3, g4, g5, g6, g7, g8;

  long long sum0 = 0;
  for(int r = 0; r < runs; ++r) {
    gettimeofday(&t0, NULL);
    f0 = GauXC::gauxc_boys_element<0>(TVAL);
    f1 = GauXC::gauxc_boys_element<1>(TVAL);
    f2 = GauXC::gauxc_boys_element<2>(TVAL);
    f3 = GauXC::gauxc_boys_element<3>(TVAL);
    f4 = GauXC::gauxc_boys_element<4>(TVAL);
    f5 = GauXC::gauxc_boys_element<5>(TVAL);
    f6 = GauXC::gauxc_boys_element<6>(TVAL);
    f7 = GauXC::gauxc_boys_element<7>(TVAL);
    f8 = GauXC::gauxc_boys_element<8>(TVAL);
    gettimeofday(&t1, NULL);

    sum0 += (t1.tv_sec-t0.tv_sec)*1000000LL + t1.tv_usec-t0.tv_usec;
  }

  long long sum1 = 0;
  for(int r = 0; r < runs; ++r) {
    gettimeofday(&t0, NULL);
    double e_13 = std::exp(-1.0 * TVAL / 3.0);
    double e_23 = e_13 * e_13;
    double e_11 = e_23 * e_13;

    double TVALr  = 1 / TVAL;
    double TVALrs = std::sqrt(TVALr);

    g0 = 0.0;
  
    if(TVAL < 28) {
      double n;
      double d;
    
      n = 0.101496827289892561636e-8 * TVAL + 0.100290453231804032913e-8;
      n = TVAL * n + 0.114315819494419468355e-5;
      n = TVAL * n + 0.488275205694491957804e-5;
      n = TVAL * n + 0.000399414007760856993659f;
      n = TVAL * n + 0.00173561872693777888307f;
      n = TVAL * n + 0.0447039731090986429495f;
      n = TVAL * n + 0.110797197786631479568f;
      n = TVAL * n + 0.999999999999995700241;

      d = -0.282456869187885785253e-12 * TVAL + 0.333637026721052224101e-10;
      d = TVAL * d - 0.128927743593561789607e-8;
      d = TVAL * d + 0.446404578868329808188e-8;
      d = TVAL * d + 0.76237111832045313941e-6;
      d = TVAL * d - 0.493272041137073499076e-5;
      d = TVAL * d - 0.000366831695730316839546f;
      d = TVAL * d + 0.000259528667137113921197f;
      d = TVAL * d + 0.110797197786377266728f;
      d = TVAL * d + 1.0f;

      double d_inv = 1 / d;
      g0 = e_13 * n * d_inv;
    } else {
      g0 = SQRT_PI_OVER_2 * TVALrs;
    }

    e_11 *= (-0.5) * TVALr;
    g1 = 0.5 * TVALr;
    g2 = 1.5 * TVALr;
    g3 = 2.5 * TVALr;
    g4 = 3.5 * TVALr;
    g5 = 4.5 * TVALr;
    g6 = 5.5 * TVALr;
    g7 = 6.5 * TVALr;
    g8 = 7.5 * TVALr;

    g1 = g1 * g0 + e_11;
    g2 = g2 * g1 + e_11;
    g3 = g3 * g2 + e_11;
    g4 = g4 * g3 + e_11;
    g5 = g5 * g4 + e_11;
    g6 = g6 * g5 + e_11;
    g7 = g7 * g6 + e_11;
    g8 = g8 * g7 + e_11;
    gettimeofday(&t1, NULL);
    
    sum1 += (t1.tv_sec-t0.tv_sec)*1000000LL + t1.tv_usec-t0.tv_usec;
  }

  printf("%lf - %lf = %e\n", f0, g0, (f0 - g0) / f0);
  printf("%lf - %lf = %e\n", f1, g1, (f1 - g1) / f1);
  printf("%lf - %lf = %e\n", f2, g2, (f2 - g2) / f2);
  printf("%lf - %lf = %e\n", f3, g3, (f3 - g3) / f3);
  printf("%lf - %lf = %e\n", f4, g4, (f4 - g4) / f4);
  printf("%lf - %lf = %e\n", f5, g5, (f5 - g5) / f5);
  printf("%lf - %lf = %e\n", f6, g6, (f6 - g6) / f6);
  printf("%lf - %lf = %e\n", f7, g7, (f7 - g7) / f7);
  printf("%lf - %lf = %e\n", f8, g8, (f8 - g8) / f8);

  printf("%lf\t%lf\n", sum0 / ((double) (1.0 * runs)), sum1 / ((double) (1.0 * runs)));
  
  GauXC::gauxc_boys_finalize();
  
  return 0;
}
