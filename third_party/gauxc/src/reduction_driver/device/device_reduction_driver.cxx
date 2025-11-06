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
#include "device_reduction_driver.hpp"

namespace GauXC {

DeviceReductionDriver::DeviceReductionDriver(const RuntimeEnvironment& rt) :
  detail::ReductionDriverImpl(rt) { }


DeviceReductionDriver::~DeviceReductionDriver() noexcept = default;



bool DeviceReductionDriver::takes_host_memory() const {return false;}; 
bool DeviceReductionDriver::takes_device_memory() const {return true;};

}
