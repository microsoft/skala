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
#include <gauxc/xc_task.hpp>
#include <host/local_host_work_driver.hpp>
#ifdef GAUXC_HAS_DEVICE
#include <device/local_device_work_driver.hpp>
#endif

namespace GauXC {
namespace exx_detail {
  using host_task_container = std::vector<XCTask>;
  using host_task_iterator  = typename host_task_container::iterator;
}

void exx_ek_screening( 
  const BasisSet<double>& basis, const BasisSetMap& basis_map,
  const ShellPairCollection<double>& shpairs,
  const double* P_abs, size_t ldp, const double* V_shell_max, size_t ldv,
  double eps_E, double eps_K, LocalHostWorkDriver* lwd, 
  exx_detail::host_task_iterator task_begin,
  exx_detail::host_task_iterator task_end );

#ifdef GAUXC_HAS_DEVICE
void exx_ek_screening( 
  const BasisSet<double>& basis, const BasisSetMap& basis_map,
  const ShellPairCollection<double>& shpairs,
  const double* P_abs, size_t ldp, const double* V_shell_max, size_t ldv,
  double eps_E, double eps_K, XCDeviceData& device_data, 
  LocalDeviceWorkDriver* lwd, 
  exx_detail::host_task_iterator task_begin,
  exx_detail::host_task_iterator task_end );
#endif
 
}
