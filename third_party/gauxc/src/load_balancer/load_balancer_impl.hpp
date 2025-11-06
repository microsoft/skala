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

namespace GauXC  {
namespace detail {

class LoadBalancerImpl {

public:

  using basis_type      = BasisSet<double>;
  using basis_map_type  = BasisSetMap;
  using shell_pair_type = ShellPairCollection<double>;

protected:

  RuntimeEnvironment          runtime_;
  std::shared_ptr<Molecule>   mol_;
  std::shared_ptr<MolGrid>    mg_;
  std::shared_ptr<basis_type> basis_;
  std::shared_ptr<MolMeta>    molmeta_;
  std::shared_ptr<basis_map_type> basis_map_;
  std::shared_ptr<shell_pair_type> shell_pairs_;

  std::vector< XCTask >     local_tasks_;

  LoadBalancerState         state_;

  util::Timer               timer_;

  virtual std::vector< XCTask > create_local_tasks_() const = 0;

public:

  LoadBalancerImpl() = delete;

  LoadBalancerImpl( const RuntimeEnvironment&, const Molecule&, const MolGrid& mg,  
    const basis_type&);
  LoadBalancerImpl( const RuntimeEnvironment&, const Molecule&, const MolGrid& mg,  
    const basis_type&, const MolMeta& );
  LoadBalancerImpl( const RuntimeEnvironment&, const Molecule&, const MolGrid& mg,  
    const basis_type&, std::shared_ptr<MolMeta> );

  LoadBalancerImpl( const LoadBalancerImpl& );
  LoadBalancerImpl( LoadBalancerImpl&& ) noexcept;

  virtual ~LoadBalancerImpl() noexcept;

  const std::vector< XCTask >& get_tasks() const;
        std::vector< XCTask >& get_tasks()      ;

  void rebalance_weights();
  void rebalance_exc_vxc();
  void rebalance_exx();

  const util::Timer& get_timings() const;

  size_t total_npts()     const;
  size_t max_npts()       const;
  size_t max_nbe()        const;
  size_t max_npts_x_nbe() const;

  const Molecule& molecule() const;
  const MolMeta&  molmeta()  const;
  const basis_type& basis()  const;
  const RuntimeEnvironment& runtime() const;
  const basis_map_type& basis_map() const;
  const shell_pair_type& shell_pairs() const;
  const shell_pair_type& shell_pairs();

  LoadBalancerState& state();

  virtual std::unique_ptr<LoadBalancerImpl> clone() const = 0;

};

}
}
