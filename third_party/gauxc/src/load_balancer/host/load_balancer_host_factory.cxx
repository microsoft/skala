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
#include "load_balancer_host_factory.hpp"
#include "petite_replicated_load_balancer.hpp"
#include "fillin_replicated_load_balancer.hpp"

namespace GauXC {

std::shared_ptr<LoadBalancer> LoadBalancerHostFactory::get_shared_instance(
  std::string kernel_name, const RuntimeEnvironment& rt,
  const Molecule& mol, const MolGrid& mg, const BasisSet<double>& basis
) {

  std::transform(kernel_name.begin(), kernel_name.end(), 
    kernel_name.begin(), ::toupper );


  if( kernel_name == "DEFAULT" or kernel_name == "REPLICATED" ) 
    kernel_name = "REPLICATED-PETITE";

  std::unique_ptr<detail::LoadBalancerImpl> ptr = nullptr;
  if( kernel_name == "REPLICATED-PETITE" )
    ptr = std::make_unique<detail::PetiteHostReplicatedLoadBalancer>(
      rt, mol, mg, basis
    );

  if( kernel_name == "REPLICATED-FILLIN" )
    ptr = std::make_unique<detail::FillInHostReplicatedLoadBalancer>(
      rt, mol, mg, basis
    );

  if( ! ptr ) GAUXC_GENERIC_EXCEPTION("Load Balancer Kernel Not Recognized: " + kernel_name);

  return std::make_shared<LoadBalancer>(std::move(ptr));

}

}

