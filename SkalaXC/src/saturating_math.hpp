#pragma once

#include <cstdint>
#include <limits>

namespace SkalaXC::detail {

constexpr std::uint64_t saturating_add(std::uint64_t lhs,
                                       std::uint64_t rhs) noexcept {
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  return rhs > maximum - lhs ? maximum : lhs + rhs;
}

constexpr std::uint64_t saturating_multiply(std::uint64_t lhs,
                                            std::uint64_t rhs) noexcept {
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  if (lhs == 0 || rhs == 0) return 0;
  return lhs > maximum / rhs ? maximum : lhs * rhs;
}

}  // namespace SkalaXC::detail