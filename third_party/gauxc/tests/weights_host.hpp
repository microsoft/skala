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
#include "weights_generate.hpp"
#include <fstream>
#include <string>

#ifdef GAUXC_HAS_HOST
#include "host/reference/weights.hpp"
using namespace GauXC;

void test_host_weights( std::ifstream& in_file, XCWeightAlg weight_alg ) {

  ref_weights_data ref_data;
  {
    cereal::BinaryInputArchive ar( in_file );
    ar( ref_data );
  }

  switch(weight_alg) {
    case XCWeightAlg::Becke:
      reference_becke_weights_host( 
        ref_data.mol, *ref_data.meta, ref_data.tasks_unm.begin(), 
        ref_data.tasks_unm.end() );
      break;
    case XCWeightAlg::SSF:
      reference_ssf_weights_host( 
        ref_data.mol, *ref_data.meta, ref_data.tasks_unm.begin(), 
        ref_data.tasks_unm.end() );
      break;
    case XCWeightAlg::LKO:
      reference_lko_weights_host( 
        ref_data.mol, *ref_data.meta, ref_data.tasks_unm.begin(), 
        ref_data.tasks_unm.end() );
      break;
  }


  size_t ntasks = ref_data.tasks_unm.size();
  for( size_t itask = 0; itask < ntasks; ++itask ) {
    auto& task     = ref_data.tasks_unm.at(itask);
    auto& ref_task = ref_data.tasks_mod.at(itask);

    size_t npts = task.weights.size();
    for( size_t i = 0; i < npts; ++i ) {
      CHECK( task.weights.at(i) ==
             Approx(ref_task.weights.at(i)) );
    }
  }

}
#endif
