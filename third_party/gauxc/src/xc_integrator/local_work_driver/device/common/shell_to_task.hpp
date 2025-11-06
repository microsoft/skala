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
#include <gauxc/shell.hpp>

namespace GauXC {

struct ShellToTaskDevice {
  int32_t  ntask;
  int32_t  center_idx;
  int32_t  true_idx;
  int32_t*  task_idx_device;
  int32_t*  task_shell_offs_device;
  Shell<double>*   shell_device;
};

struct AngularMomentumShellToTaskBatch {
  size_t             ntask_average;
  size_t             npts_average;
  size_t             nshells_in_batch;
  ShellToTaskDevice* shell_to_task_device;
  uint32_t           pure;
};

}
