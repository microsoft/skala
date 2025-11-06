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
#include <gauxc/load_balancer.hpp>
#include <gauxc/util/timer.hpp>
#include <gauxc/enums.hpp>

namespace GauXC {

namespace detail {
  // Implementation base class for MolecularWeights
  class MolecularWeightsImpl;
}

struct MolecularWeightsSettings { 
    XCWeightAlg weight_alg = XCWeightAlg::SSF; ///< Weight partitioning scheme
    bool becke_size_adjustment = false; ///< Whether to use Becke size adjustments
};


/// A class which applies molecular partition weights to pre-generated quadrature
/// tasks.
class MolecularWeights {

public:

  using load_balancer_type = LoadBalancer;
  using load_balancer_reference = load_balancer_type&;

private:

  using pimpl_type = detail::MolecularWeightsImpl;
  using pimpl_ptr_type = std::unique_ptr<pimpl_type>;
  pimpl_ptr_type pimpl_; ///< Pointer to implementation instance

public:

  // Delete default ctor
  MolecularWeights() = delete;

  // Destructor (default)
  ~MolecularWeights() noexcept;

  /// Construct a MolecularWeights instance from preconstructed implementation
  MolecularWeights( pimpl_ptr_type&& ptr );

  // Delete copy ctor
  MolecularWeights( const MolecularWeights& ) = delete;

  // Move a MolecularWeights instance
  MolecularWeights( MolecularWeights&& ) noexcept;

  /// Apply weight partitioning scheme to pre-generated local quadrature tasks
  void modify_weights(load_balancer_reference lb) const;

  /// Return local timing tracker
  const util::Timer& get_timings() const;

}; // class MolecularWeights


/// A factory to generate MolecularWeights instances 
class MolecularWeightsFactory {

public:

    // Delete default ctor
    MolecularWeightsFactory() = delete;

    /**
     * @brief Construct a factory which generates a specific kind of MolecularWeights 
     *
     * @param[in] ex Execution space for the MolecularWeights phase. 
     *               Acceptable values:
     *               - Host: Run MolecularWeights on CPU
     *               - Device: Run MolecularWeights on GPU (if enabled)
     *
     * @param[in] local_work_kernel_name Specification of the LocalWorkDriver 
     *                                   kernel underlying the MolecularWeights
     *                                   instasnce. 
     *
     *                                   See documentation for LocalWorkDriver for
     *                                   details.
     *
     * @param[in] s Settings for the MolecularWeights calculation.
     */
    MolecularWeightsFactory( ExecutionSpace ex, 
                             std::string local_work_kernel_name,
                             MolecularWeightsSettings s);


    /// Generate a shared-pointer MolecularWeights instance 
    std::shared_ptr<MolecularWeights> get_shared_instance();

    /// Generate a MolecularWeights instance 
    inline MolecularWeights get_instance(){
      return MolecularWeights( std::move( *get_shared_instance() ) );
    };

private:

    ExecutionSpace ex_; ///< Execution space for the MolecularWeights phase
    std::string lwd_kernel_; ///< LocalWorkDriver kernel for the MolecularWeights phase
    MolecularWeightsSettings settings_; ///< Settings for the MolecualarWeights phase

}; // class MolecularWeightsSettings

}
