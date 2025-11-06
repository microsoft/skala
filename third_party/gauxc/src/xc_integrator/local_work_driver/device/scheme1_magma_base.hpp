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
#include "scheme1_base.hpp"

namespace GauXC {

struct AoSScheme1MAGMABase : public AoSScheme1Base {

  void eval_xmat( double fac, XCDeviceData*, bool do_grad, density_id den ) override final;
  void eval_exx_fmat( XCDeviceData* ) override final;
  void inc_vxc( XCDeviceData*, density_id den, bool ) override final;
  void inc_exx_k( XCDeviceData* ) override final;

  struct Data;

  virtual ~AoSScheme1MAGMABase() = default;
};

struct AoSScheme1MAGMABase::Data : public AoSScheme1Base::Data {

  using base_type = AoSScheme1Base::Data;
  using base_type::host_task_type;
  using base_type::device_buffer_t;

  struct magma_data {
    double** xdmat_array_device = nullptr;
    double** fdmat_array_device = nullptr;
    double** vmat_array_device  = nullptr;
    double** kmat_array_device  = nullptr;
    double** zmat_array_device  = nullptr;
    double** fmat_array_device  = nullptr;
    double** gmat_array_device  = nullptr;
    double** bf_array_device    = nullptr;

    int32_t* xmat_m_array_device   = nullptr;
    int32_t* xmat_n_array_device   = nullptr;
    int32_t* xmat_k_array_device   = nullptr;
    int32_t* ld_xdmat_array_device = nullptr;

    int32_t* fmat_m_array_device   = nullptr;
    int32_t* fmat_n_array_device   = nullptr;
    int32_t* fmat_k_array_device   = nullptr;
    int32_t* ld_fdmat_array_device = nullptr;

    int32_t* ld_vmat_array_device  = nullptr;
    int32_t* ld_zmat_array_device  = nullptr;
    int32_t* ld_fmat_array_device  = nullptr;
    int32_t* ld_bf_array_device    = nullptr;

    inline void reset(){ std::memset(this,0,sizeof(magma_data)); }
  };

  magma_data magma_stack;

  template <typename... Args>
  Data( Args&&... args ) : base_type( std::forward<Args>(args)... ) { }

  virtual ~Data() = default;

  size_t get_mem_req( integrator_term_tracker, 
    const host_task_type&) override final;
  size_t get_static_mem_requirement() override final; 
  void reset_allocations() override final;
  device_buffer_t allocate_dynamic_stack( integrator_term_tracker terms,
    host_task_iterator begin, host_task_iterator end, device_buffer_t buf )
    override final;
  void pack_and_send( integrator_term_tracker terms,
    host_task_iterator begin, host_task_iterator end, 
    const BasisSetMap& basis_map ) override final;


  void pack_and_send_xmat( host_task_iterator, host_task_iterator ); 
  void pack_and_send_fmat( host_task_iterator, host_task_iterator );
};

}
