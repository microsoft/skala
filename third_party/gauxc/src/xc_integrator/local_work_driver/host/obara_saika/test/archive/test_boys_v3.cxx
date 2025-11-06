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
#include "../src/config_obara_saika.hpp"

#define SQRT_PI_OVER_2 0.88622692545275801364

int main(int argc, char **argv) {

  int runs = atoi(argv[1]);
  double TVAL = atof(argv[2]);
  
  double *boys_table = XCPU::boys_init();

  struct timeval t0, t1;

  double t;
  double f0, f1, f2, f3, f4, f5, f6, f7, f8;
  double g0, g1, g2, g3, g4, g5, g6, g7, g8;  
  
  long long sum0 = 0;
  for(int r = 0; r < runs; ++r) {
    gettimeofday(&t0, NULL);
    XCPU::boys_element<0>(&TVAL, &t, &f0, boys_table);
    XCPU::boys_element<1>(&TVAL, &t, &f1, boys_table);
    XCPU::boys_element<2>(&TVAL, &t, &f2, boys_table);
    XCPU::boys_element<3>(&TVAL, &t, &f3, boys_table);
    XCPU::boys_element<4>(&TVAL, &t, &f4, boys_table);
    XCPU::boys_element<5>(&TVAL, &t, &f5, boys_table);
    XCPU::boys_element<6>(&TVAL, &t, &f6, boys_table);
    XCPU::boys_element<7>(&TVAL, &t, &f7, boys_table);
    XCPU::boys_element<8>(&TVAL, &t, &f8, boys_table);
    gettimeofday(&t1, NULL);

    sum0 += (t1.tv_sec-t0.tv_sec)*1000000LL + t1.tv_usec-t0.tv_usec;
  }

  long long sum1 = 0;
  for(int r = 0; r < runs; ++r) {
    gettimeofday(&t0, NULL);

    XCPU::boys_element<8>(&TVAL, &t, &g8, boys_table);

    g7 = (TVAL * g8 + t) * (2.0 / 15.0);
    g6 = (TVAL * g7 + t) * (2.0 / 13.0);
    g5 = (TVAL * g6 + t) * (2.0 / 11.0);
    g4 = (TVAL * g5 + t) * (2.0 /  9.0);
    g3 = (TVAL * g4 + t) * (2.0 /  7.0);
    g2 = (TVAL * g3 + t) * (2.0 /  5.0);
    g1 = (TVAL * g2 + t) * (2.0 /  3.0);
    g0 = (TVAL * g1 + t) * (2.0 /  1.0);
    
    gettimeofday(&t1, NULL);
    
    sum1 += (t1.tv_sec-t0.tv_sec)*1000000LL + t1.tv_usec-t0.tv_usec;
  }

  printf("%e - %e = %e\n", f0, g0, (f0 - g0) / f0);
  printf("%e - %e = %e\n", f1, g1, (f1 - g1) / f1);
  printf("%e - %e = %e\n", f2, g2, (f2 - g2) / f2);
  printf("%e - %e = %e\n", f3, g3, (f3 - g3) / f3);
  printf("%e - %e = %e\n", f4, g4, (f4 - g4) / f4);
  printf("%e - %e = %e\n", f5, g5, (f5 - g5) / f5);
  printf("%e - %e = %e\n", f6, g6, (f6 - g6) / f6);
  printf("%e - %e = %e\n", f7, g7, (f7 - g7) / f7);
  printf("%e - %e = %e\n", f8, g8, (f8 - g8) / f8);

  printf("%lf\t%lf\n", sum0 / ((double) (1.0 * runs)), sum1 / ((double) (1.0 * runs)));
  
  XCPU::boys_finalize(boys_table);
  
  return 0;
}
