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
#include "device/xc_device_aos_data.hpp"
#include "device/common/shell_to_task.hpp"
#include "device/common/shell_pair_to_task.hpp"

namespace GauXC {

struct Scheme1DataBase : public XCDeviceAoSData {

  using base_type = XCDeviceAoSData;
  using base_type::host_task_type;
  using base_type::device_buffer_t;
  using shell_pair = ShellPair<double>;

  struct scheme1_data {
    double*  dist_scratch_device = nullptr;
    double*  dist_nearest_device = nullptr;
    int32_t* iparent_device      = nullptr;
    
    inline void reset(){ std::memset(this,0,sizeof(scheme1_data)); }
  };

  struct collocation_data {
    size_t*  shell_list_device   = nullptr;
      ///< Contiguous batch local shell left for task batch
    size_t*  shell_offs_device   = nullptr;
      ///< Contiguous batch local shell offsets for task batch
    inline void reset(){ std::memset(this,0,sizeof(collocation_data)); }
  };

  struct shell_to_task_data {
    ShellToTaskDevice* shell_to_task_device;

    int32_t* shell_to_task_idx_device = nullptr;
    int32_t* shell_to_task_off_device = nullptr;

    inline void reset(){ std::memset(this,0,sizeof(shell_to_task_data)); }
  };

  struct shell_pair_to_task_data {
    ShellPairToTaskDevice* shell_pair_to_task_device;

    int32_t* shell_pair_to_task_idx_device = nullptr;
    int32_t* shell_pair_to_task_row_off_device = nullptr;
    int32_t* shell_pair_to_task_col_off_device = nullptr;

    inline void reset(){ std::memset(this,0,sizeof(shell_pair_to_task_data)); }
  };

  struct task_to_shell_pair_data {
    TaskToShellPairDevice* task_to_shell_pair_device;

    // Each task has their own copy
    int32_t* task_shell_linear_idx_device = nullptr;
    int32_t* task_shell_off_row_device = nullptr;
    int32_t* task_shell_off_col_device = nullptr;

    std::array<int32_t, 4>* subtask_device = nullptr;

    // Reused for all tasks. Indexed by linear idx
    int32_t* nprim_pairs_device = nullptr;
    GauXC::PrimitivePair<double>** pp_ptr_device = nullptr;
    double* sp_X_AB_device = nullptr;
    double* sp_Y_AB_device = nullptr;
    double* sp_Z_AB_device = nullptr;

    inline void reset(){ std::memset(this,0,sizeof(task_to_shell_pair_device)); }
  };

  size_t total_nshells_bfn_task_batch  = 0; ///< Sum of nshells for task batch (bfn)
  scheme1_data       scheme1_stack;
  collocation_data   collocation_stack;
  shell_to_task_data shell_to_task_stack;
  std::vector<AngularMomentumShellToTaskBatch> l_batched_shell_to_task;

  //size_t total_nshells_cou_task_batch  = 0; ///< Sum of nshells for task batch (cou)
  size_t total_nshells_cou_sqlt_task_batch  = 0; ///< Sum of nshells for task batch (cou)
  //collocation_data   coulomb_stack;
  shell_pair_to_task_data shell_pair_to_task_stack;
  std::vector<ShellPairToTaskHost> shell_pair_to_task;
  std::vector<AngularMomentumShellPairToTaskBatch> 
    l_batched_shell_pair_to_task_diag,
    l_batched_shell_pair_to_task_off_diag;

  std::vector<TaskToShellPairHost> task_to_shell_pair;
  std::vector<AngularMomentumTaskToShellPairBatchHost> l_batch_task_to_shell_pair;
  std::vector<AngularMomentumTaskToShellPairBatch> l_batch_task_to_shell_pair_device;

  std::vector<AngularMomentumTaskToShellPairBatchHost> l_batch_diag_task_to_shell_pair;
  std::vector<AngularMomentumTaskToShellPairBatch> l_batch_diag_task_to_shell_pair_device;
  task_to_shell_pair_data task_to_shell_pair_stack;

  std::vector<std::array<int32_t, 4>> subtask;
  std::vector<int32_t> nprim_pairs_host;
  std::vector<GauXC::PrimitivePair<double>*> pp_ptr_host;
  std::vector<double> sp_X_AB_host;
  std::vector<double> sp_Y_AB_host;
  std::vector<double> sp_Z_AB_host;

  virtual ~Scheme1DataBase() noexcept;
  Scheme1DataBase(const DeviceRuntimeEnvironment& rt);

  // Final overrides
  void add_extra_to_indirection(integrator_term_tracker, 
    std::vector<XCDeviceTask>& ) override final;

  // Overrideable API's
  virtual size_t get_mem_req( integrator_term_tracker, 
    const host_task_type&) override;
  virtual size_t get_static_mem_requirement() override; 
  virtual void reset_allocations() override;
  virtual device_buffer_t allocate_dynamic_stack( integrator_term_tracker terms,
    host_task_iterator begin, host_task_iterator end, device_buffer_t buf )
    override;
  virtual void pack_and_send( integrator_term_tracker terms,
    host_task_iterator begin, host_task_iterator end, 
    const BasisSetMap& basis_map ) override;

};

}
