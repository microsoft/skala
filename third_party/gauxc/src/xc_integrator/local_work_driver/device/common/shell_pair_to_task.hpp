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
#include <gauxc/shell_pair.hpp>

namespace GauXC {

struct ShellPairToTaskHost {
  using shell_pair = ShellPair<double>;
  using point      = detail::cartesian_point;
  std::vector<int32_t> task_idx;
  std::vector<int32_t> task_shell_off_row;
  std::vector<int32_t> task_shell_off_col;
  shell_pair* shell_pair_device;

  int32_t idx_bra, idx_ket;
  int32_t lA, lB;
  point rA, rB;

  void clear() {
    task_idx.clear();
    task_shell_off_row.clear();
    task_shell_off_col.clear();
    shell_pair_device = nullptr;
  }
};

struct ShellPairToTaskDevice {
  using shell_pair = ShellPair<double>;
  using point      = detail::cartesian_point;
  int32_t* task_idx_device;
  int32_t* task_shell_off_row_device;
  int32_t* task_shell_off_col_device;
  int32_t nprim_pairs;
  GauXC::PrimitivePair<double>* prim_pairs_device;
  int32_t ntask;

  double X_AB, Y_AB, Z_AB;
};

struct AngularMomentumShellPairToTaskBatch {
  size_t ntask_average;
  size_t npts_average;
  size_t nshells_in_batch;
  ShellPairToTaskDevice* shell_pair_to_task_device;

  int lA, lB;
};

struct TaskToShellPairHost {
  using shell_pair = ShellPair<double>;
  using point      = detail::cartesian_point;
  std::vector<int32_t> shell_pair_linear_idx;
  std::vector<int32_t> task_shell_off_row;
  std::vector<int32_t> task_shell_off_col;
  int32_t nsp;
  int32_t nsp_filled;

  void clear() {
    shell_pair_linear_idx.clear();
    task_shell_off_row.clear();
    task_shell_off_col.clear();
  };
};

struct TaskToShellPairDevice {
  using shell_pair = ShellPair<double>;

  int32_t* shell_pair_linear_idx_device;
  int32_t* task_shell_off_row_device;
  int32_t* task_shell_off_col_device;

  int32_t nsp;
};

struct AngularMomentumTaskToShellPairBatchHost {
  std::vector<TaskToShellPairHost> task_to_shell_pair;

  int lA, lB;
  int max_prim_pairs;

  void clear() {
    task_to_shell_pair.clear();
  }
};


struct AngularMomentumTaskToShellPairBatch {
  TaskToShellPairDevice* task_to_shell_pair_device;

  int max_prim_pairs;
  int lA, lB;
};

}
