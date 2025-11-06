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

#include <exchcxx/xc_functional.hpp>
#include <integratorxx/quadrature.hpp>
#include <integratorxx/batch/spherical_micro_batcher.hpp>

#include <gauxc/named_type.hpp>

namespace GauXC {

using functional_type = ExchCXX::XCFunctional;
//using quadrature_type = IntegratorXX::QuadratureBase<
//  std::vector<std::array<double,3>>,
//  std::vector<double>
//>;
using quadrature_type = IntegratorXX::SphericalQuadratureBase<
  std::vector<std::array<double,3>>,
  std::vector<double>
>;

using batcher_type = IntegratorXX::SphericalMicroBatcher<
  typename quadrature_type::point_container,
  typename quadrature_type::weight_container
>;

}

#include <gauxc/enums.hpp>
