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

#include <gauxc/molgrid.hpp>
#include <gauxc/molmeta.hpp>
#include <gauxc/basisset.hpp>
#include <gauxc/basisset_map.hpp>
#include <gauxc/shell_pair.hpp>
#include <gauxc/xc_task.hpp>
#include <gauxc/util/timer.hpp>
#include <gauxc/runtime_environment.hpp>
#include <gauxc/enums.hpp>

namespace GauXC {

namespace detail {
  /// LoadBalancer Implementation class
  class LoadBalancerImpl;
}

/// State tracker for LoadBalancer instances 
struct LoadBalancerState {
  bool modified_weights_are_stored = false; 
    ///< Whether the load balancer currently stores partitioned weights
  XCWeightAlg weight_alg = XCWeightAlg::NOTPARTITIONED; 
    ///< Weight partitioning scheme used by this LoadBalancer
};


/** 
 *  @brief A class to distribute and manage local quadrature tasks for XCIntegraor
 *  operations
 */
class LoadBalancer {

  using pimpl_type = detail::LoadBalancerImpl;
  std::unique_ptr<pimpl_type> pimpl_; ///< Pointer to implementation instance

public:

  using basis_type      = BasisSet<double>;
  using basis_map_type  = BasisSetMap;
  using shell_pair_type = ShellPairCollection<double>;

  /// Construct default LoadBalancer instance with null internal state
  LoadBalancer();

  /// Construct LoadBalancer instance from preconstructed implementation
  LoadBalancer( std::unique_ptr<pimpl_type>&& pimpl );

  /// Copy a LoadBalancer instance
  LoadBalancer( const LoadBalancer& );

  /// Move a LoadBalancer instance 
  LoadBalancer( LoadBalancer&& ) noexcept;

  /// Destruct LoadBalancer instance (defaulted)
  ~LoadBalancer() noexcept;

  /// Get underlying (local) quadrature tasks for this process (cost)
  const std::vector<XCTask>& get_tasks() const;
  /// Get underlying (local) quadrature tasks for this process (non-cost)
        std::vector<XCTask>& get_tasks()      ;

  /// Rebalance quadrature batches according to weight-only cost
  void rebalance_weights();

  /// Rebalance quadrature batches according to exc-vxc cost
  void rebalance_exc_vxc();

  /// Rebalance quadrature batches according to exx cost 
  void rebalance_exx();
  
  /// Return internal timing tracker
  const util::Timer& get_timings() const;

  /// Return the total number of points for local tasks
  size_t total_npts() const;

  /// Return the maximum number of points for local tasks
  size_t max_npts()       const;

  /// Return the maximum effective basis dimention for local tasks
  size_t max_nbe()        const;

  /// Return the maximum npts x nde product for local tasks 
  size_t max_npts_x_nbe() const;

  /// Return the underlying molecule instance used to generate this LoadBalancer 
  const Molecule& molecule() const;

  /// Return the underlying MolMeta instance used to generate this LoadBalancer 
  const MolMeta&  molmeta()  const;

  /// Return the underlying BasisSet instance used to generate this LoadBalancer 
  const basis_type& basis()  const;

  /// Return BasisSetMap instance corresponding to basis/molecule
  const basis_map_type& basis_map() const;

  /// Return the number of non-negligible local shell pairs for this LoadBalancer
  const shell_pair_type& shell_pairs() const;
  const shell_pair_type& shell_pairs();

  /// Return the runtime handle used to construct this LoadBalancer
  const RuntimeEnvironment& runtime() const;
  
  /// Return the load balancer state (non-const)
  LoadBalancerState& state();

  /// Check equality of LoadBalancer instances
  bool operator==( const LoadBalancer& ) const;

}; // class LoadBalancer


/// A factory to generate LoadBalancer instances
class LoadBalancerFactory {

public:

  // Delete default ctor
  LoadBalancerFactory() = delete;

  /**
   * @brief Construct a factory which generates a specific kind of LoadBalancer
   *
   * @param[in] ex Execution space for the LoadBalancer phase. 
   *               Acceptable values:
   *               - Host: Run LoadBalancer on CPU
   *               - Device: Run LoadBalancer on GPU (if enabled)
   *
   * @param[in] kernel_name Specification of the LoadBalancer kernel. 
   *    Currently accepted values for Host execution space:
   *    - "DEFAULT": Read as "REPLICATED-PETITE"
   *    - "REPLICATED": Read as "REPLICATED-PETITE"
   *    - "REPLICATED-PETITE": Replicate the load balancer function, only keep
   *                           non negligible basis functions
   *    - "REPLICATED-FILLIN": Same as "REPLICATED-PETITE" except if two 
   *                           non-adjacent bfns are kept, the gaps are filled in.
   *                           This gurantees contiguous memory access but leads
   *                           to significantly more work. Not advised for general 
   *                           usage
   * 
   *    Currently accepted values for Device execution space:
   *      - "DEFAULT": Read as "REPLICATED"
   *      - "REPLICATED": Same as Host::REPLICATED-PETITE
   */
  LoadBalancerFactory( ExecutionSpace ex, std::string kernel_name );

  /** 
   *  @brief Generate a LoadBalancer instance per kernel and execution space
   *         specfication
   *
   *  @param[in] rt      Runtime handle defining the execution space across which
   *                     the quadrature tasks will be distributed.
   *  @param[in] mol     Molecule on which the quadrature is defined.
   *  @param[in] mg      The batched molecular quadrature
   *  @param[in] bs      The basis set whcih will be used for numerical integration
   *
   *  @returns A LoadBalancer instance constructed using the passed parameters.
   */
  LoadBalancer get_instance( const RuntimeEnvironment& rt, 
    const Molecule& mol, const MolGrid& mg, const BasisSet<double>& bs);

  /** 
   *  @brief Generate a shared pointer to a LoadBalancer instance per kernel and 
   *         execution space specfication
   *
   *  @param[in] rt      Runtime handle defining the execution space across which
   *                     the quadrature tasks will be distributed.
   *  @param[in] mol     Molecule on which the quadrature is defined.
   *  @param[in] mg      The batched molecular quadrature
   *  @param[in] bs      The basis set whcih will be used for numerical integration
   *
   *  @returns A shared pointer to a LoadBalancer instance constructed using 
   *           the passed parameters.
   */
  std::shared_ptr<LoadBalancer> get_shared_instance( 
    const RuntimeEnvironment& rt,
    const Molecule& mol, const MolGrid& mg, const BasisSet<double>&);

private:

  ExecutionSpace ex_; ///< Execution space for the generated LoadBalancer instances
  std::string    kernel_name_; ///< Kernel name of the generated Load Balancer instances 

}; // LoadBalancerFactory


}
