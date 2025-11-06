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
#include "../molecular_weights_impl.hpp"
namespace GauXC::detail {

class HostMolecularWeights : public MolecularWeightsImpl {

public:

  HostMolecularWeights() = delete;
  virtual ~HostMolecularWeights() noexcept = default;
  HostMolecularWeights( const HostMolecularWeights& ) = delete;
  HostMolecularWeights( HostMolecularWeights&& ) noexcept = default;

  template <typename... Args>
  inline HostMolecularWeights(Args&&... args) :
    MolecularWeightsImpl(std::forward<Args>(args)...) {}

  void modify_weights(LoadBalancer&) const final;

};

template <typename... Args>
std::unique_ptr<MolecularWeightsImpl> 
  make_host_mol_weights_impl(Args&&... args) {
  return std::make_unique<HostMolecularWeights>(
    std::forward<Args>(args)...);
}

}
