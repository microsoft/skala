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
#include "load_balancer_impl.hpp"
#include "load_balancer_device_factory.hpp"

#ifdef GAUXC_HAS_CUDA
#include "cuda/replicated_cuda_load_balancer.hpp"
#endif

#ifdef GAUXC_HAS_HIP
#include "hip/replicated_hip_load_balancer.hpp"
#endif

namespace GauXC {

std::shared_ptr<LoadBalancer> LoadBalancerDeviceFactory::get_shared_instance(
  std::string kernel_name, const RuntimeEnvironment& rt,
  const Molecule& mol, const MolGrid& mg, const BasisSet<double>& basis
) {

  std::transform(kernel_name.begin(), kernel_name.end(), 
    kernel_name.begin(), ::toupper );


  if( kernel_name == "DEFAULT" ) kernel_name = "REPLICATED";

  std::unique_ptr<detail::LoadBalancerImpl> ptr = nullptr;
  #ifdef GAUXC_HAS_DEVICE
  if( kernel_name == "REPLICATED" ) {
    ptr = std::make_unique<detail::DeviceReplicatedLoadBalancer>(
      rt, mol, mg, basis
    );
  }
  #endif

  if( ! ptr ) GAUXC_GENERIC_EXCEPTION("Load Balancer Kernel Not Recognized: " + kernel_name);

  return std::make_shared<LoadBalancer>(std::move(ptr));

}

}

