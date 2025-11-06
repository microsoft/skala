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
#include <gauxc/util/div_ceil.hpp>
#include "device/xc_device_task.hpp"
#include "device/common/shell_pair_to_task.hpp"

namespace XGPU {

  //typedef struct {
  //  double x, y, z;
  //} point;
  using point = GauXC::detail::cartesian_point;

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
  using shell_pair = GauXC::ShellPair<double>;
#endif

}

