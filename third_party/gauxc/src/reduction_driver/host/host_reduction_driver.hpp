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
#include "reduction_driver_impl.hpp"


namespace GauXC {

struct HostReductionDriver : public detail::ReductionDriverImpl {

  bool takes_host_memory() const override; 
  bool takes_device_memory() const override;

  virtual ~HostReductionDriver() noexcept;

  HostReductionDriver(const RuntimeEnvironment& rt);

};

}
