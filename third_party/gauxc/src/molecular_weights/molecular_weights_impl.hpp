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
#include <gauxc/molecular_weights.hpp>
#include <gauxc/xc_integrator/local_work_driver.hpp>

namespace GauXC::detail {

class MolecularWeightsImpl {

protected:

  std::unique_ptr<LocalWorkDriver> local_work_driver_;
  MolecularWeightsSettings         settings_;
  util::Timer timer_;

public:

  MolecularWeightsImpl() = delete;
  virtual ~MolecularWeightsImpl() noexcept = default;
  MolecularWeightsImpl( const MolecularWeightsImpl& ) = delete;
  MolecularWeightsImpl( MolecularWeightsImpl&& ) noexcept = default;

  inline MolecularWeightsImpl(std::unique_ptr<LocalWorkDriver>&& lwd,
    MolecularWeightsSettings settings) :
    local_work_driver_(std::move(lwd)),
    settings_(settings) {}

  virtual void modify_weights(LoadBalancer&) const = 0;
  inline const util::Timer& get_timings() const {
    return timer_;
  };

  inline util::Timer& get_timer() {
    return timer_;
  };
};

}
