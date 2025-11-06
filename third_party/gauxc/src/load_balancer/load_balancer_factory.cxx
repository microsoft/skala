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
#include "host/load_balancer_host_factory.hpp"

#ifdef GAUXC_HAS_DEVICE
#include "device/load_balancer_device_factory.hpp"
#endif

namespace GauXC {

LoadBalancerFactory::LoadBalancerFactory( ExecutionSpace ex, std::string kernel_name ) :
  ex_(ex), kernel_name_(kernel_name) { }

std::shared_ptr<LoadBalancer> LoadBalancerFactory::get_shared_instance(
  const RuntimeEnvironment& rt,
  const Molecule& mol, const MolGrid& mg, const BasisSet<double>& basis
) {

  switch(ex_) {
    case ExecutionSpace::Host:
      using host_factory = LoadBalancerHostFactory;
      return host_factory::get_shared_instance(kernel_name_,
        rt, mol, mg, basis );
    #ifdef GAUXC_HAS_DEVICE
    case ExecutionSpace::Device:
      using device_factory = LoadBalancerDeviceFactory;
      return device_factory::get_shared_instance(kernel_name_,
        rt, mol, mg, basis );
    #endif
    default:
      GAUXC_GENERIC_EXCEPTION("Unrecognized Execution Space");
   }


}

LoadBalancer LoadBalancerFactory::get_instance(
  const RuntimeEnvironment& rt, 
  const Molecule& mol, const MolGrid& mg, const BasisSet<double>& basis
) {

  auto ptr = get_shared_instance(rt, mol, mg, basis);
  return LoadBalancer(std::move(*ptr));

}


}

