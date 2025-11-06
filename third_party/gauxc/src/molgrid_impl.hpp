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

#include <gauxc/molgrid.hpp>

namespace GauXC {
namespace detail {

class MolGridImpl {

  atomic_grid_map molgrid_;

public:

  MolGridImpl( const atomic_grid_map& );

  MolGridImpl( const MolGridImpl& );
  MolGridImpl( MolGridImpl&& ) noexcept;

  ~MolGridImpl() noexcept;

  size_t natoms_uniq() const;
  const Grid& get_grid( AtomicNumber ) const;
        Grid& get_grid( AtomicNumber )      ;

  size_t max_nbatches() const;

};

}
}
