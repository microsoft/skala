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
#include <gauxc/gauxc_config.hpp>
#include <array>

namespace GauXC  {
namespace detail {

template <typename T, size_t N>
inline HOST_DEVICE_ACCESSIBLE T* contiguous_data( const std::array<T,N>& arr ) {
  return reinterpret_cast<T*>( &const_cast<std::array<T,N>&>(arr) );
}

}
}
