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
#include <gauxc/basisset.hpp>
#include <gauxc/xc_task.hpp>

namespace GauXC {
namespace detail {

struct ShellBatchedXCIntegratorBase {

  using basis_type = BasisSet<double>;

  using host_task_container = std::vector<XCTask>;
  using host_task_iterator  = typename host_task_container::iterator;

  // Struct to manage data associated with task subset to execute in batch
  struct incore_task_data {
    host_task_iterator   task_begin;
    host_task_iterator   task_end;
    std::vector<int32_t> shell_list;
  };

  incore_task_data generate_incore_task( 
    uint32_t nbf_threshold, const basis_type& basis,
    host_task_iterator task_begin, host_task_iterator task_end );

  virtual ~ShellBatchedXCIntegratorBase() noexcept = default;

};

}
}
