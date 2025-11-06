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

#include "xc_device_stack_data.hpp"
#include "xc_device_task.hpp"

namespace GauXC {


/// Base type for XCDeviceData instances that address task batches as AoS
struct XCDeviceAoSData : public XCDeviceStackData {

  size_t total_nbe_bfn_task_batch      = 0; ///< Sum of nbe_bfn for task batch
  size_t total_nbe_scr_task_batch      = 0; ///< Sum of max(nbe,...) * nbe_bfn for task batch
  size_t total_nbe_bfn_npts_task_batch = 0; ///< Sum of npts*nbe_bfn for task batch
  size_t total_ncut_bfn_task_batch     = 0; ///< Sum of ncut_bfn for task batch
  size_t total_nblock_bfn_task_batch   = 0; ///< Sum of nblock_bfn for task batch
  size_t total_nbe_cou_npts_task_batch = 0; ///< Sum of npts*nbe_cou for task batch
  size_t total_ncut_cou_task_batch     = 0; ///< Sum of ncut_cou for task batch
  size_t total_nblock_cou_task_batch   = 0; ///< Sum of nblock_cou for task batch

  // Collocation buffers
  struct aos_stack_data {
    double* bf_eval_device    = nullptr; 
      ///< Contiguous batch local collocation for task batch
    double* dbf_x_eval_device = nullptr; ///< Derivative of `bf_eval_device` wrt x
    double* dbf_y_eval_device = nullptr; ///< Derivative of `bf_eval_device` wrt y
    double* dbf_z_eval_device = nullptr; ///< Derivative of `bf_eval_device` wrt z

    double* d2bf_xx_eval_device = nullptr; ///< 2nd Derivative of `bf_eval_device` wrt x+x
    double* d2bf_xy_eval_device = nullptr; ///< 2nd Derivative of `bf_eval_device` wrt x+y
    double* d2bf_xz_eval_device = nullptr; ///< 2nd Derivative of `bf_eval_device` wrt x+z
    double* d2bf_yy_eval_device = nullptr; ///< 2nd Derivative of `bf_eval_device` wrt y+y
    double* d2bf_yz_eval_device = nullptr; ///< 2nd Derivative of `bf_eval_device` wrt y+z
    double* d2bf_zz_eval_device = nullptr; ///< 2nd Derivative of `bf_eval_device` wrt z+z

    double* d2bf_lapl_eval_device = nullptr; ///< Laplacian of `bf_eval_device`
    double* d3bf_lapgrad_x_eval_device = nullptr; ///< Laplacian derivative of bf_eval_device wrt x
    double* d3bf_lapgrad_y_eval_device = nullptr; ///< Laplacian derivative of bf_eval_device wrt y
    double* d3bf_lapgrad_z_eval_device = nullptr; ///< Laplacian derivative of bf_eval_device wrt z

    // VXC Z Matrix
    double* zmat_vxc_device = nullptr;
      ///< Contiguous batch local Z matrix for LDA/GGA VXC for task batch

    // X mat gradients
    double* xmat_dx_device = nullptr;
    double* xmat_dy_device = nullptr;
    double* xmat_dz_device = nullptr;

    // Persistent X mat
    double* xmatS_device    = nullptr;
    double* xmatS_dx_device = nullptr;
    double* xmatS_dy_device = nullptr;
    double* xmatS_dz_device = nullptr;
    double* xmatZ_device    = nullptr;
    double* xmatZ_dx_device = nullptr;
    double* xmatZ_dy_device = nullptr;
    double* xmatZ_dz_device = nullptr;

    // EXX Intermediates
    double* fmat_exx_device = nullptr;
    double* gmat_exx_device = nullptr;

    // EXX EK Maps
    int32_t* bfn_shell_indirection_device = nullptr;

    // Scratch buffer
    double* nbe_scr_device = nullptr; ///< nbe*nbe scratch allocated for task batch

    // AoS Buffers
    int32_t* submat_cut_bfn_device   = nullptr;
      ///< Contiguous batch local submatrix cuts for task batch (bfn)
    int32_t* submat_block_bfn_device = nullptr;
      ///< Contiguous batch local submatrix blocking factors for task batch (bfn)
    int32_t* submat_cut_cou_device   = nullptr;
      ///< Contiguous batch local submatrix cuts for task batch (cou)
    int32_t* submat_block_cou_device = nullptr;
      ///< Contiguous batch local submatrix blocking factors for task batch (cou)

    // Indirection
    XCDeviceTask* device_tasks = nullptr; ///< Task indirection in device memory

    inline void reset() { std::memset(this,0,sizeof(aos_stack_data)); }
  };

  std::vector<XCDeviceTask> host_device_tasks; ///< Task indirection in host memory
  aos_stack_data aos_stack;


  XCDeviceAoSData() = delete;
  inline XCDeviceAoSData( const DeviceRuntimeEnvironment& rt ) :
    XCDeviceStackData( rt ) { }

  // Make it polymorphic
  virtual ~XCDeviceAoSData() noexcept = default;

  // AoS Specific API
 
  /** Get L2 compatiable submatrix block size for a specified matrix dimension
   *
   *  @param[in] LDA Leading dimension large matrix which is extracted from
   *  @param[in] dev_id ID of device to query memory information
   *
   *  @returns Submatrix blocking factor
   */
  virtual size_t get_submat_chunk_size( int32_t LDA, int32_t dev_id ) = 0;

  // Overridable API overrides
  virtual size_t get_mem_req( integrator_term_tracker, const host_task_type&) override;
  virtual device_buffer_t allocate_dynamic_stack( integrator_term_tracker terms,
    host_task_iterator begin, host_task_iterator end, device_buffer_t buf) override;
  virtual void pack_and_send( integrator_term_tracker terms,
    host_task_iterator begin, host_task_iterator end, 
    const BasisSetMap& basis_map ) override;

  inline virtual void 
    add_extra_to_indirection(integrator_term_tracker, std::vector<XCDeviceTask>&) { };

  virtual void reset_allocations() override;

  void populate_submat_maps( size_t, host_task_iterator, host_task_iterator, const BasisSetMap& ) override;

};

}
