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
namespace util  {

template <typename T>
T max_coulomb( const Shell<T>& bra, const Shell<T>& ket);

extern template double max_coulomb( const Shell<double>&, const Shell<double>& );

}
}
