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

#include <gauxc/xc_integrator/replicated/replicated_xc_integrator_impl.hpp>

namespace GauXC {
namespace detail {

/// Base class for ReplicatedXCIntegrator implentations on Host execution spaces
template <typename ValueType>
class ReplicatedXCHostIntegrator : public ReplicatedXCIntegratorImpl<ValueType> {

  using base_type  = ReplicatedXCIntegratorImpl<ValueType>;

public:

  using value_type = typename base_type::value_type;
  using basis_type = typename base_type::basis_type;

  template <typename... Args>
  ReplicatedXCHostIntegrator( Args&&... args) :
    base_type( std::forward<Args>(args)... ) { }

  virtual ~ReplicatedXCHostIntegrator() noexcept;

};

extern template class ReplicatedXCHostIntegrator<double>;



/// Factory to generate ReplicatedXCHostIntegrator instances
template <typename ValueType>
struct ReplicatedXCHostIntegratorFactory {

  using impl_type = ReplicatedXCIntegratorImpl<ValueType>;
  using ptr_return_t = std::unique_ptr<impl_type>;

  /** Generate a ReplicatedXCHostIntegrator instance
   *
   *  @param[in]  integration_kernel Name of integration scaffold to load ("Default", "Reference", etc)
   *  @param[in]  func               XC functional to integrate
   *  @param[in]  lb                 Pregenerated LoadBalancer instance
   *  @param[in]  lwd                Local Work Driver
   */
  static ptr_return_t make_integrator_impl( 
    std::string integrator_kernel,
    std::shared_ptr<functional_type>   func,
    std::shared_ptr<LoadBalancer>      lb,
    std::unique_ptr<LocalWorkDriver>&& lwd,
    std::shared_ptr<ReductionDriver>   rd
    );

};


extern template struct ReplicatedXCHostIntegratorFactory<double>;


}
}
