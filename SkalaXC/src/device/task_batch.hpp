#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>

namespace SkalaXC::detail {

/** Bound a candidate tile before GauXC applies its independent memory limit. */
template <typename Iterator>
Iterator task_batch_end(Iterator begin, Iterator end, std::size_t task_limit) {
  if (task_limit == 0)
    throw std::invalid_argument("CUDA task tile limit must be positive");
  const auto remaining = static_cast<std::size_t>(std::distance(begin, end));
  return std::next(
      begin,
      static_cast<typename std::iterator_traits<Iterator>::difference_type>(
          std::min(remaining, task_limit)));
}

}  // namespace SkalaXC::detail
