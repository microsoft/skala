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
#include <gauxc/xc_integrator/replicated/replicated_xc_device_integrator.hpp>
#include "incore_replicated_xc_device_integrator.hpp"
#include "shell_batched_replicated_xc_device_integrator.hpp"
#include "device/local_device_work_driver.hpp"

namespace GauXC  {
namespace detail {

template <typename ValueType>
ReplicatedXCDeviceIntegrator<ValueType>::~ReplicatedXCDeviceIntegrator() noexcept = default;

template class ReplicatedXCDeviceIntegrator<double>;


template <typename ValueType>
typename ReplicatedXCDeviceIntegratorFactory<ValueType>::ptr_return_t
  ReplicatedXCDeviceIntegratorFactory<ValueType>::make_integrator_impl(
    std::string integrator_kernel,
    std::shared_ptr<functional_type> func,
    std::shared_ptr<LoadBalancer> lb, 
    std::unique_ptr<LocalWorkDriver>&& lwd,
    std::shared_ptr<ReductionDriver>   rd
    ) {

  // Make sure that the LWD is a valid LocalDeviceWorkDriver
  if(not dynamic_cast<LocalDeviceWorkDriver*>(lwd.get())) {
    GAUXC_GENERIC_EXCEPTION("Passed LWD Not valid for Device ExSpace");
  }

  std::transform(integrator_kernel.begin(), integrator_kernel.end(), 
    integrator_kernel.begin(), ::toupper );

  if( integrator_kernel == "DEFAULT" ) integrator_kernel = "INCORE";

  if( integrator_kernel == "INCORE" )
    return std::make_unique<IncoreReplicatedXCDeviceIntegrator<ValueType>>(
      func, lb, std::move(lwd), rd
    );
  else if( integrator_kernel == "SHELLBATCHED" )
    return std::make_unique<ShellBatchedReplicatedXCDeviceIntegrator<ValueType>>(
      func, lb, std::move(lwd), rd
    );

  else
    GAUXC_GENERIC_EXCEPTION("Integrator Kernel " + integrator_kernel + " Not Recognized");


}

template struct ReplicatedXCDeviceIntegratorFactory<double>;


}
}

