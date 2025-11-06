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
#include <gauxc/xc_integrator/impl.hpp>
#include <stdexcept>

#include <gauxc/xc_integrator/local_work_driver.hpp>
#include <gauxc/xc_integrator/replicated/replicated_xc_integrator_factory.hpp>
#include <gauxc/reduction_driver.hpp>

namespace GauXC {

/// Factory to generate XCIntegrator Instances
template <typename MatrixType>
class XCIntegratorFactory {

public:

  using integrator_type = XCIntegrator<MatrixType>;

  XCIntegratorFactory() = delete;

  /** Construct an XCIntegratorFactory instance 
   *
   *  @param[in] ex                      Execution space for the XCIntegrator instance
   *  @param[in] integrator_input_type   Input type for XC integration (e.g. "Replicated")
   *  @param[in] integrator_kernel_name  Name of Integraion scaffold kernel to load (e.g. "Reference" or "Default")
   *  @param[in] local_work_kerenl_name  Name of LWD to load (e.g. "Reference" or "Default")
   *  @param[in] setting                 Settings to pass to LWD (not currently used)
   */
  XCIntegratorFactory( ExecutionSpace ex, 
                       std::string integrator_input_type,
                       std::string integrator_kernel_name,
                       std::string local_work_kernel_name,
                       std::string reduction_kernel_name,
                       LocalWorkSettings settings = LocalWorkSettings() ) :
    ex_(ex), input_type_(integrator_input_type), 
    integrator_kernel_(integrator_kernel_name),
    lwd_kernel_(local_work_kernel_name), 
    rd_kernel_(reduction_kernel_name),
    local_work_settings_(settings) {}

 
  /** Generate XCIntegrator instance
   *
   *  @param[in] func  XC functional
   *  @param[in] lb    Preconstructed Load Balancer instance
   */
  std::shared_ptr<integrator_type> get_shared_instance( 
    std::shared_ptr<functional_type> func,
    std::shared_ptr<LoadBalancer>    lb ) {

    // Create Local Work Driver
    auto lwd = LocalWorkDriverFactory::make_local_work_driver( ex_, 
      lwd_kernel_, local_work_settings_ );

    // Create Reduction Driver
    auto rd = ReductionDriverFactory::get_shared_instance( 
      lb->runtime(), rd_kernel_ );

    // Create Integrator instance
    std::transform( input_type_.begin(), input_type_.end(), input_type_.begin(), 
      ::toupper );

    if( input_type_ == "REPLICATED" )
      return std::make_shared<integrator_type>( 
        ReplicatedXCIntegratorFactory<MatrixType>::make_integrator_impl(
          ex_, integrator_kernel_, func, lb, std::move(lwd), rd
        )
      );
    else GAUXC_GENERIC_EXCEPTION("INTEGRATOR TYPE NOT RECOGNIZED");

    return nullptr;

  }

  auto get_shared_instance( const functional_type& func, const LoadBalancer& lb ) {
    return get_shared_instance( std::make_shared<functional_type>(func),
                         std::make_shared<LoadBalancer>(lb) );
  }

  auto get_shared_instance( const functional_type& func,
                                       std::shared_ptr<LoadBalancer> lb ) {
    return get_shared_instance( std::make_shared<functional_type>(func), lb );
  }


  template <typename... Args>
  integrator_type get_instance( Args&&... args ) {

    return integrator_type( std::move(*get_shared_instance(std::forward<Args>(args)...) ));

  }

private:

  ExecutionSpace ex_;
  std::string input_type_, integrator_kernel_, lwd_kernel_, rd_kernel_;
  LocalWorkSettings local_work_settings_;

};

}
