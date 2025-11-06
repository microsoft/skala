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
#include <gauxc/xc_integrator/replicated/replicated_xc_device_integrator.hpp>
#include "device/xc_device_data.hpp"

namespace GauXC {
namespace detail {

template <typename ValueType>
class IncoreReplicatedXCDeviceIntegrator : 
  public ReplicatedXCDeviceIntegrator<ValueType> {

  using base_type  = ReplicatedXCDeviceIntegrator<ValueType>;

public:

  static constexpr bool is_device = true;
  using value_type = typename base_type::value_type;
  using basis_type = typename base_type::basis_type;

  using host_task_container = std::vector<XCTask>;
  using host_task_iterator  = typename host_task_container::iterator;

protected:

  void integrate_den_( int64_t m, int64_t n, const value_type* P,
                      int64_t ldp, value_type* N_EL ) override;

  void eval_exc_( int64_t m, int64_t n, const value_type* P, int64_t ldp, 
                  value_type* EXC, const IntegratorSettingsXC& ks_settings ) override;
  void eval_exc_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps, 
                  const value_type* Pz, int64_t ldpz,
                  value_type* EXC, const IntegratorSettingsXC& ks_settings ) override;
  void eval_exc_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps, 
                  const value_type* Pz, int64_t ldpz,
                  const value_type* Py, int64_t ldpy,
                  const value_type* Px, int64_t ldpx,
                  value_type* EXC, const IntegratorSettingsXC& ks_settings ) override;

  void eval_exc_vxc_( int64_t m, int64_t n, const value_type* P,
                      int64_t ldp, value_type* VXC, int64_t ldvxc,
                      value_type* EXC, const IntegratorSettingsXC& settings) override;

  void eval_exc_vxc_( int64_t m, int64_t n, const value_type* Ps,
                      int64_t ldps,
                      const value_type* Pz,
                      int64_t ldpz,
                      value_type* VXCs, int64_t ldvxcs,
                      value_type* VXCz, int64_t ldvxcz,
                      value_type* EXC, const IntegratorSettingsXC& settings ) override;

  void eval_exc_vxc_( int64_t m, int64_t n, const value_type* Ps,
                      int64_t ldps,
                      const value_type* Pz,
                      int64_t ldpz,
                      const value_type* Py,
                      int64_t ldpy,
                      const value_type* Px,
                      int64_t ldpx,
                      value_type* VXCs, int64_t ldvxcs,
                      value_type* VXCz, int64_t ldvxcz,
                      value_type* VXCy, int64_t ldvxcy,
                      value_type* VXCx, int64_t ldvxcx,
                      value_type* EXC, const IntegratorSettingsXC& settings ) override;


  void eval_exc_grad_( int64_t m, int64_t n, const value_type* P, int64_t ldp, 
                       value_type* EXC_GRAD, const IntegratorSettingsXC& settings ) override;
  void eval_exc_grad_( int64_t m, int64_t n, const value_type* Ps, int64_t ldps, 
                       const value_type* Pz, int64_t ldpz, value_type* EXC_GRAD, const IntegratorSettingsXC& settings ) override;

  void eval_exx_( int64_t m, int64_t n, const value_type* P,
                  int64_t ldp, value_type* K, int64_t ldk,
                  const IntegratorSettingsEXX& settings ) override;

  void eval_fxc_contraction_( int64_t m, int64_t n, 
                              const value_type* P, int64_t ldp,   
                              const value_type* tP, int64_t ldtp,
                              value_type* FXC, int64_t ldfxc,
                              const IntegratorSettingsXC& ks_settings ) override;

  void eval_fxc_contraction_( int64_t m, int64_t n, 
                              const value_type* Ps, int64_t ldps,   
                              const value_type* Pz, int64_t ldpz,
                              const value_type* tPs, int64_t ldtps,
                              const value_type* tPz, int64_t ldtpz,
                              value_type* FXCs, int64_t ldfxcs,
                              value_type* FXCz, int64_t ldfxcz,
                              const IntegratorSettingsXC& ks_settings ) override;

  void eval_dd_psi_( int64_t m, int64_t n, const value_type* P,
                     int64_t ldp, unsigned max_Ylm, value_type* ddPsi, 
                     int64_t ldPsi ) override;
  
  void eval_dd_psi_potential_( int64_t m, int64_t n, const value_type* X, 
                    unsigned max_Ylm, value_type* Vddx ) override;

  void eval_exc_vxc_onedft_(int64_t m, int64_t n, const value_type* Ps,
                    int64_t ldps,
                    const value_type* Pz,
                    int64_t ldpz,
                    value_type* VXCs, int64_t ldvxcs,
                    value_type* VXCz, int64_t ldvxcz,
                    value_type* EXC, const IntegratorSettingsXC& settings ) override;

  void integrate_den_local_work_( const basis_type& basis, const value_type* P, int64_t ldp, 
                            value_type *N_EL,
                            host_task_iterator task_begin, host_task_iterator task_end,
                            XCDeviceData& device_data );


  void exc_vxc_local_work_( const basis_type& basis, const value_type* Ps, int64_t ldps,
                            const value_type* Pz, int64_t ldpz,
                            const value_type* Py, int64_t ldpy,
                            const value_type* Px, int64_t ldpx,
                            host_task_iterator task_begin, host_task_iterator task_end,
                            XCDeviceData& device_data, bool do_vxc );

  void exc_vxc_local_work_( const basis_type& basis, const value_type* Ps, int64_t ldps,
                            const value_type* Pz, int64_t ldpz,
                            const value_type* Py, int64_t ldpy,
                            const value_type* Px, int64_t ldpx,
                            value_type* VXC, int64_t ldvxc,
                            value_type* VXCz, int64_t ldvxcz,
                            value_type* VXCy, int64_t ldvxcy,
                            value_type* VXCx, int64_t ldvxcx, value_type* EXC, value_type *N_EL,
                            host_task_iterator task_begin, host_task_iterator task_end,
                            XCDeviceData& device_data );

  void pre_onedft_local_work_( const basis_type& basis, const value_type* Ps, int64_t ldps,
                            const value_type* Pz, int64_t ldpz,
                            const value_type* Py, int64_t ldpy,
                            const value_type* Px, int64_t ldpx,
                            host_task_iterator task_begin, host_task_iterator task_end,
                            XCDeviceData& device_data, const integrator_term_tracker enabled_terms );

  void post_onedft_local_work_( const basis_type& basis, const value_type* Ps, int64_t ldps,
                            const value_type* Pz, int64_t ldpz,
                            const value_type* Py, int64_t ldpy,
                            const value_type* Px, int64_t ldpx,
                            host_task_iterator task_begin, host_task_iterator task_end,
                            XCDeviceData& device_data, const integrator_term_tracker enabled_terms );
  void fxc_contraction_local_work_( const basis_type& basis, const value_type* Ps, int64_t ldps,
                            const value_type* Pz, int64_t ldpz,
                            const value_type* tPs, int64_t ldtps,
                            const value_type* tPz, int64_t ldtpz,
                            host_task_iterator task_begin, host_task_iterator task_end,
                            XCDeviceData& device_data);

  void fxc_contraction_local_work_( const basis_type& basis, const value_type* Ps, int64_t ldps,
                            const value_type* Pz, int64_t ldpz,
                            const value_type* tPs, int64_t ldtps,
                            const value_type* tPz, int64_t ldtpz,
                            value_type *N_EL,
                            value_type* FXCs, int64_t ldfxcs,
                            value_type* FXCz, int64_t ldfxcz,
                            host_task_iterator task_begin, host_task_iterator task_end,
                            XCDeviceData& device_data );

  void eval_exc_grad_local_work_( const basis_type& basis, const value_type* Ps, int64_t ldps, 
                                  const value_type* Pz, int64_t ldpz,
                                  host_task_iterator task_begin, host_task_iterator task_end,
                                  XCDeviceData& device_data, const IntegratorSettingsXC& settings );

  void eval_exc_grad_local_work_( const basis_type& basis, const value_type* P, int64_t ldp, 
                                  const value_type* Pz, int64_t ldpz, value_type* EXC_GRAD, 
                                  host_task_iterator task_begin, host_task_iterator task_end,
                                  XCDeviceData& device_data, const IntegratorSettingsXC& settings );



  void exx_local_work_( const basis_type& basis, const value_type* P, int64_t ldp, 
                        host_task_iterator task_begin, host_task_iterator task_end,
                        XCDeviceData& device_data, 
                        const IntegratorSettingsEXX& settings);

  void exx_local_work_( const basis_type& basis, const value_type* P, int64_t ldp, 
                        value_type* K, int64_t ldk,
                        host_task_iterator task_begin, host_task_iterator task_end,
                        XCDeviceData& device_data, 
                        const IntegratorSettingsEXX& settings);

  void exx_ek_screening_local_work_( const basis_type& basis, 
                        const value_type* P, int64_t ldp, 
                        XCDeviceData& device_data, 
                        const IntegratorSettingsEXX& settings);

public:

  template <typename... Args>
  IncoreReplicatedXCDeviceIntegrator( Args&&... args ) :
    base_type( std::forward<Args>(args)... ) { }

  virtual ~IncoreReplicatedXCDeviceIntegrator() noexcept;

  
  template <typename... Args>
  void exc_vxc_local_work(Args&&... args) {
    exc_vxc_local_work_( std::forward<Args>(args)... );
  }

};

extern template class IncoreReplicatedXCDeviceIntegrator<double>;

}
}
