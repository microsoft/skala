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

#include <gauxc/grid.hpp>

namespace GauXC {
namespace detail {

class GridImpl {

  std::shared_ptr< quadrature_type > quad_    = nullptr;
  std::shared_ptr< batcher_type    > batcher_ = nullptr;

  void generate_batcher(BatchSize);

public:

  GridImpl() = delete;

  GridImpl( std::shared_ptr<quadrature_type> q, BatchSize );

  GridImpl( const GridImpl& );
  GridImpl( GridImpl&& ) noexcept;

  GridImpl& operator=( const GridImpl& );
  GridImpl& operator=( GridImpl&& ) noexcept;

  ~GridImpl() noexcept;
  
  const batcher_type& batcher() const;
        batcher_type& batcher()      ;

};

}
}
