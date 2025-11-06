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
#include <gauxc/reduction_driver.hpp>

namespace GauXC  {
namespace detail {

class ReductionDriverImpl {

protected: 

  const RuntimeEnvironment& runtime_;

public:

  ReductionDriverImpl() = delete;
  ReductionDriverImpl( const RuntimeEnvironment& rt);

  virtual ~ReductionDriverImpl() noexcept;
  ReductionDriverImpl( const ReductionDriverImpl& );

  virtual void allreduce_typeerased( const void*, void*, size_t, ReductionOp, std::type_index, std::any ) = 0;
  virtual void allreduce_inplace_typeerased( void*, size_t, ReductionOp, std::type_index, std::any ) = 0;
  
  virtual bool takes_host_memory() const = 0;
  virtual bool takes_device_memory() const = 0;

  virtual std::unique_ptr<ReductionDriverImpl> clone() = 0;
};

}
}
