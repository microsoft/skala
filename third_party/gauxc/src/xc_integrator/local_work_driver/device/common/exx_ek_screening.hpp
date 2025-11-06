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
#include "device/xc_device_task.hpp"
#include "device/device_queue.hpp"
#include "device/device_blas_handle.hpp"
#include <gauxc/shell.hpp>
#include <gauxc/shell_pair.hpp>
#include <gauxc/xc_task.hpp>

namespace GauXC {
using host_task_iterator = std::vector<XCTask>::iterator;

void exx_ek_screening_bfn_stats( size_t        ntasks,
                                 XCDeviceTask* tasks_device,
                                 double      * max_bfn_sum_device,
                                 double      * bfn_max_device,
                                 size_t        LDBFM,
                                 device_queue queue );

void exx_ek_shellpair_collision(
  int32_t       ntasks,
  int32_t       nshells,
  int32_t       nbf,
  const double* abs_dmat_device,
  size_t        LDP,
  const double* V_max_sparse_device,
  const size_t* sp_row_ind_device,
  const size_t* sp_col_ind_device,
  const double* max_bf_sum_device,
  const double* bfn_max_device,
  size_t        LDBM,
  const Shell<double>* shells_device,
  const int32_t* shell_to_bf_device,
  const int32_t* shell_sizes_device,
  double        eps_E,
  double        eps_K,
  void*         dyn_stack,
  size_t        dyn_size,
  host_task_iterator tb,
  host_task_iterator te,
  const ShellPairCollection<double>& shpairs,
  device_queue  queue,
  device_blas_handle handle
);

}
