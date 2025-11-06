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
#include <type_traits>
#include <cstdlib>
#include <cstdint>
#include <cassert>

#include <gauxc/gauxc_config.hpp>

namespace GauXC  {
namespace util  {

namespace detail {

template <typename... Args>
struct are_integral;

template <typename Head, typename... Tail>
struct are_integral<Head, Tail...> {
  static constexpr bool value = std::is_integral<Head>::value and 
                         are_integral<Tail...>::value;
};

template <typename T>
struct are_integral<T> {
  static constexpr bool value = std::is_integral<T>::value;
};


template <typename... Args>
struct largest;

template <typename Head, typename... Tail>
struct largest< Head, Tail... > {
private:
  using tail_type = typename largest<Tail...>::type;
public:
  using type = std::conditional_t< 
                 (sizeof(Head) > sizeof(tail_type)),
                 Head, tail_type >;
};

template <typename T>
struct largest<T> {
  using type = T;
};

template <typename... Args>
using largest_t = typename largest<Args...>::type;

}

inline HOST_DEVICE_ACCESSIBLE uintmax_t div_ceil( uintmax_t i, uintmax_t j ) {
  return (i + j - 1) / j;
}


}
}
