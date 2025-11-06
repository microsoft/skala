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

#include <gauxc/basisset_map.hpp>

namespace GauXC      {

std::tuple< std::vector< std::array<int32_t, 3> >, std::vector< int32_t > >
  gen_compressed_submat_map( const BasisSetMap&       basis_set,
                             const std::vector< int32_t >& shell_mask,
		             const int32_t LDA, const int32_t block_size ); 


}
