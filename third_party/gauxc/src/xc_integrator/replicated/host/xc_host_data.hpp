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
#include <vector>
#include <cstdint>

#include <gauxc/gauxc_config.hpp>

namespace GauXC {

template <typename F>
struct XCHostData {

  std::vector<F> eps;
  std::vector<F> gamma;
  std::vector<F> tau;
  std::vector<F> lapl;
  std::vector<F> vrho;
  std::vector<F> vgamma;
  std::vector<F> vtau;
  std::vector<F> vlapl;
 
  std::vector<F> zmat;
  std::vector<F> gmat;
  std::vector<F> nbe_scr;
  std::vector<F> den_scr;
  std::vector<F> basis_eval;

  // Second order derivatives
  std::vector<F> v2rho2;
  std::vector<F> v2rhogamma;
  std::vector<F> v2rholapl;
  std::vector<F> v2rhotau;
  std::vector<F> v2gamma2;
  std::vector<F> v2gammalapl;
  std::vector<F> v2gammatau;
  std::vector<F> v2lapl2;
  std::vector<F> v2lapltau;
  std::vector<F> v2tau2;

  // For Fxc contraction
  std::vector<F> FXC_A;
  std::vector<F> FXC_B;
  std::vector<F> FXC_C;
  std::vector<F> tden_scr;
  std::vector<F> ttau;
  std::vector<F> tlapl;

   
  inline XCHostData() {}

};

}
