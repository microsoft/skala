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

#include "ut_common.hpp"
#include <gauxc/molgrid.hpp>
#include <gauxc/basisset.hpp>
#include <gauxc/load_balancer.hpp>
#include <gauxc/molgrid/defaults.hpp>
#include <fstream>
#include <string>

using namespace GauXC;

#define MAX_NPTS_CHECK 67

struct ref_collocation_data {
  std::vector<int32_t>              mask;
  std::vector<std::array<double,3>> pts;
  std::vector<double>               eval;
  std::vector<double>               deval_x;
  std::vector<double>               deval_y;
  std::vector<double>               deval_z;
  std::vector<double>               d2eval_xx;
  std::vector<double>               d2eval_xy;
  std::vector<double>               d2eval_xz;
  std::vector<double>               d2eval_yy;
  std::vector<double>               d2eval_yz;
  std::vector<double>               d2eval_zz;
  std::vector<double>               d2eval_lapl;
  std::vector<double>               d3eval_lapl_x;
  std::vector<double>               d3eval_lapl_y;
  std::vector<double>               d3eval_lapl_z;

  template <typename Archive>
  void serialize( Archive& ar ) {
    ar( mask, pts, eval, deval_x, deval_y, deval_z, d2eval_xx, d2eval_xy, d2eval_xz, 
        d2eval_yy, d2eval_yz, d2eval_zz, d2eval_lapl, d3eval_lapl_x, d3eval_lapl_y, d3eval_lapl_z);
  }

};

void check_collocation_transpose( int npts, int nbf, const double* ref_val, const double* comp_val, std::string msg = "" ) {

  // Check transpose
  for( int i = 0; i < nbf;  ++i )
  for( int j = 0; j < npts; ++j ) {
    INFO(msg << " IBF = " << i << " IPT = " << j);
    CHECK( ref_val[ i + j*nbf ] == Approx( comp_val[ i*npts + j ] ) );
  }

}

void check_collocation( int npts, int nbf, const double* ref_val, const double* comp_val ) {

  for( int i = 0; i < nbf;  ++i )
  for( int j = 0; j < npts; ++j ) {
    INFO("IBF = " << i << " IPT = " << j);
    CHECK( ref_val[ i + j*nbf ] == Approx( comp_val[ i + j*nbf ] ) );
  }

}
