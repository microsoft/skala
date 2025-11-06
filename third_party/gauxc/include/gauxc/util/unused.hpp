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

#include <utility>

namespace GauXC::util {

inline static void unused() { }

template <typename T, typename... Args>
inline static void unused( const T& t, Args&&... args ) {
  (void)(t);
  unused( std::forward<Args>(args)... );
}


}
