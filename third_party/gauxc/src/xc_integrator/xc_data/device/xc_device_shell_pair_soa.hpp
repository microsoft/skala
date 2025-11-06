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
struct XCDeviceShellPairSoA {
  using shell_pair = ShellPair<double>;
  using point      = detail::cartesian_point;
  std::vector<GauXC::PrimitivePair<double>*> prim_pair_dev_ptr;
  std::vector<int32_t>                shell_pair_nprim_pairs;
  std::vector<std::pair<int,int>>     shell_pair_shidx;
  std::vector<std::pair<int,int>>     shell_pair_ls;
  std::vector<std::pair<point,point>> shell_pair_centers;

  std::vector<size_t> sp_row_ptr;
  std::vector<size_t> sp_col_ind;

  inline void reset() {
    shell_pair_nprim_pairs.clear();
    prim_pair_dev_ptr.clear();
    shell_pair_ls.clear();
    shell_pair_centers.clear();
  }
};
}
