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
#include <gauxc/xc_integrator/replicated/replicated_xc_host_integrator.hpp>
#include "reference_replicated_xc_host_integrator.hpp"
#include "shell_batched_replicated_xc_host_integrator.hpp"
#include "host/local_host_work_driver.hpp"

namespace GauXC::detail {

template <typename ValueType>
ReplicatedXCHostIntegrator<ValueType>::~ReplicatedXCHostIntegrator() noexcept = default;

template class ReplicatedXCHostIntegrator<double>;


template <typename ValueType>
typename ReplicatedXCHostIntegratorFactory<ValueType>::ptr_return_t
  ReplicatedXCHostIntegratorFactory<ValueType>::make_integrator_impl(
    std::string integrator_kernel,
    std::shared_ptr<functional_type> func,
    std::shared_ptr<LoadBalancer> lb, 
    std::unique_ptr<LocalWorkDriver>&& lwd,
    std::shared_ptr<ReductionDriver>   rd
    ) {

  // Make sure that the LWD is a valid LocalHostWorkDriver
  if(not dynamic_cast<LocalHostWorkDriver*>(lwd.get())) {
    GAUXC_GENERIC_EXCEPTION("Passed LWD Not valid for Host ExSpace");
  }

  std::transform(integrator_kernel.begin(), integrator_kernel.end(), 
    integrator_kernel.begin(), ::toupper );

  if( integrator_kernel == "DEFAULT" ) integrator_kernel = "REFERENCE";

  if( integrator_kernel == "REFERENCE" )
    return std::make_unique<ReferenceReplicatedXCHostIntegrator<ValueType>>(
      func, lb, std::move(lwd), rd
    );

  else if( integrator_kernel == "SHELLBATCHED" )
    return std::make_unique<ShellBatchedReplicatedXCHostIntegrator<ValueType>>(
      func, lb, std::move(lwd), rd
    );

  else
    GAUXC_GENERIC_EXCEPTION("Integrator Kernel: " + integrator_kernel + " Not Recognized");

  return nullptr;


}

template struct ReplicatedXCHostIntegratorFactory<double>;


} // namespace GauXC::detail

