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
#include "device/xc_device_task.hpp"
#include "device/device_queue.hpp"
#include "shell_to_task.hpp"

namespace GauXC {


template <typename T>
void eval_collocation_masked(
  size_t            nshells,
  size_t            nbf,
  size_t            npts,
  const Shell<T>*   shells_device,
  const size_t*     mask_device,
  const size_t*     offs_device,
  const T*          pts_device,
  T*                eval_device,
  device_queue queue );


template <typename T>
void eval_collocation_masked_deriv1(
  size_t            nshells,
  size_t            nbf,
  size_t            npts,
  const Shell<T>*   shells_device,
  const size_t*     mask_device,
  const size_t*     offs_device,
  const T*          pts_device,
  T*                eval_device,
  T*                deval_device_x,
  T*                deval_device_y,
  T*                deval_device_z,
  device_queue queue );


template <typename T>
void eval_collocation_masked_combined(
  size_t            ntasks,
  size_t            npts_max,
  size_t            nshells_max,
  Shell<T>*         shells_device,
  XCDeviceTask*     device_tasks,
  device_queue queue );




template <typename T>
void eval_collocation_masked_combined_deriv1(
  size_t        ntasks,
  size_t        npts_max,
  size_t        nshells_max,
  Shell<T>*     shells_device,
  XCDeviceTask* device_tasks,
  device_queue queue );








void eval_collocation_shell_to_task(
  uint32_t                    max_l,
  AngularMomentumShellToTaskBatch* l_batched_shell_to_task,
  XCDeviceTask*               device_tasks,
  device_queue           queue );

void eval_collocation_shell_to_task_gradient(
  uint32_t                    max_l,
  AngularMomentumShellToTaskBatch* l_batched_shell_to_task,
  XCDeviceTask*               device_tasks,
  device_queue           queue );

void eval_collocation_shell_to_task_hessian(
  uint32_t                    max_l,
  AngularMomentumShellToTaskBatch* l_batched_shell_to_task,
  XCDeviceTask*               device_tasks,
  device_queue           queue );

void eval_collocation_shell_to_task_laplacian(
  uint32_t                    max_l,
  AngularMomentumShellToTaskBatch* l_batched_shell_to_task,
  XCDeviceTask*               device_tasks,
  device_queue           queue );

void eval_collocation_shell_to_task_lapgrad(
  uint32_t                    max_l,
  AngularMomentumShellToTaskBatch* l_batched_shell_to_task,
  XCDeviceTask*               device_tasks,
  device_queue           queue );

} // namespace GauXC
