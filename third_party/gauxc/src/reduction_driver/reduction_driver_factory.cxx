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
#include "reduction_driver_impl.hpp"
#include "host/basic_mpi_reduction_driver.hpp"

#ifdef GAUXC_HAS_NCCL
#include "device/nccl_reduction_driver.hpp"
#endif


#include <algorithm>
#include <iostream>
#include <gauxc/exceptions.hpp>

namespace GauXC {

std::shared_ptr<ReductionDriver> ReductionDriverFactory::get_shared_instance(
  const RuntimeEnvironment& rt, std::string kernel_name ) {

  std::transform(kernel_name.begin(), kernel_name.end(), 
    kernel_name.begin(), ::toupper );

  std::unique_ptr<detail::ReductionDriverImpl> ptr = nullptr;

  if( kernel_name == "DEFAULT" ) kernel_name = "BASICMPI";

  if( kernel_name == "BASICMPI" )
    ptr = std::make_unique<BasicMPIReductionDriver>(rt);

  #ifdef GAUXC_HAS_NCCL
    if( kernel_name == "NCCL" )
      ptr = std::make_unique<NCCLReductionDriver>(rt);
  #endif

  if( !ptr ) GAUXC_GENERIC_EXCEPTION("Unknown Reduction Driver " + kernel_name);

  return std::make_shared<ReductionDriver>(std::move(ptr));

}

}
