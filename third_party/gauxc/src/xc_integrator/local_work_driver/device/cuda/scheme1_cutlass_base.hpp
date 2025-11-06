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
#include <gauxc/gauxc_config.hpp>
#ifdef GAUXC_HAS_CUTLASS
#include "device/scheme1_base.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>

namespace GauXC {

struct AoSScheme1CUTLASSBase : public AoSScheme1Base {

  template<bool is_trial>
  void eval_xmat_impl(double fac, XCDeviceData*, bool do_grad, density_id );
  template<bool is_fxc>
  void inc_potential_impl(XCDeviceData*, density_id, bool do_m);


  void eval_xmat(double fac, XCDeviceData*, bool do_grad, density_id ) override final;
  void eval_xmat_trial(double fac, XCDeviceData*, bool do_grad, density_id ) override final;
  void inc_vxc( XCDeviceData*, density_id, bool ) override final;
  void inc_fxc( XCDeviceData*, density_id, bool ) override final;

  struct Data;

  virtual ~AoSScheme1CUTLASSBase() = default;
};

struct AoSScheme1CUTLASSBase::Data : public AoSScheme1Base::Data {

  using base_type = AoSScheme1Base::Data;
  using base_type::host_task_type;
  using base_type::device_buffer_t;

  struct cutlass_data {
    double** dmat_s_array_device = nullptr;
    double** dmat_z_array_device = nullptr;
    double** dmat_y_array_device = nullptr;
    double** dmat_x_array_device = nullptr;
    double** vmat_array_device = nullptr;
    double** zmat_array_device = nullptr;
    double** bf_array_device   = nullptr;
    double** bfx_array_device   = nullptr;
    double** bfy_array_device   = nullptr;
    double** bfz_array_device   = nullptr;
    double** xmat_x_array_device   = nullptr;
    double** xmat_y_array_device   = nullptr;
    double** xmat_z_array_device   = nullptr;

    double** tdmat_s_array_device = nullptr;
    double** tdmat_z_array_device = nullptr;
    double** tdmat_y_array_device = nullptr;
    double** tdmat_x_array_device = nullptr;

    inline double** dmat_array(density_id id) {
      switch(id) {
        case DEN_S: return dmat_s_array_device;
        case DEN_Z: return dmat_z_array_device;
        case DEN_Y: return dmat_y_array_device;
        case DEN_X: return dmat_x_array_device;
        default: GAUXC_GENERIC_EXCEPTION("dmat_array: density_id not recognized");
      }
    }

    inline double** tdmat_array(density_id id) {
      switch(id) {
        case DEN_S: return tdmat_s_array_device;
        case DEN_Z: return tdmat_z_array_device;
        case DEN_Y: return tdmat_y_array_device;
        case DEN_X: return tdmat_x_array_device;
        default: GAUXC_GENERIC_EXCEPTION("dmat_array: density_id not recognized");
      }
    }
                           
    cutlass::gemm::GemmCoord* problem_sizes_device = nullptr;
    cutlass::gemm::GemmCoord* syr2k_sizes_device = nullptr;
      
    int64_t* ld64_dmat_array_device = nullptr;
    int64_t* ld64_vmat_array_device = nullptr;
    int64_t* ld64_zmat_array_device = nullptr;
    int64_t* ld64_bf_array_device   = nullptr;

    inline void reset(){ std::memset(this,0,sizeof(cutlass_data)); }
  };

                           
  std::vector<cutlass::gemm::GemmCoord> syr2k_sizes_host;
  std::vector<cutlass::gemm::GemmCoord> problem_sizes_host;
      

  cutlass_data cutlass_stack;

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

};

}
#endif
