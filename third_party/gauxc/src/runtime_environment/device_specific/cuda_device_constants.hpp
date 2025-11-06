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
#include <cstdint>

namespace GauXC {
namespace cuda  {

static constexpr uint32_t warp_size = 32;
static constexpr uint32_t max_threads_per_thread_block = 1024;
static constexpr uint32_t max_warps_per_thread_block = 
  max_threads_per_thread_block / warp_size;

}

}
