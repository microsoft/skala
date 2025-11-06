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
#pragma once
#include <cmath>
#include <gauxc/shell_pair.hpp>

namespace XCPU {

  typedef struct {
    double x, y, z;
  } point;

  typedef struct {
    double alpha, coeff;
  } coefficients;

  typedef struct {
    point origin;
    coefficients *coeff;
    int m, L;
  } shells;

#if 0
  typedef struct {
    point P;
    point PA;
    point PB;

    double K_coeff_prod;
    double gamma;
    double gamma_inv;
  } prim_pair;
#else
  using prim_pair = GauXC::PrimitivePair<double>;
#endif

}
