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
#include "host_reduction_driver.hpp"

namespace GauXC {

struct BasicMPIReductionDriver : public HostReductionDriver {

  BasicMPIReductionDriver(const RuntimeEnvironment& rt);
  virtual ~BasicMPIReductionDriver() noexcept;
  BasicMPIReductionDriver(const BasicMPIReductionDriver& );

  void allreduce_typeerased( const void*, void*, size_t, ReductionOp, std::type_index, std::any ) override;
  void allreduce_inplace_typeerased( void*, size_t, ReductionOp, std::type_index, std::any ) override;
  
  std::unique_ptr<detail::ReductionDriverImpl> clone() override;

};

}
