#include "device/task_batch.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <numeric>
#include <vector>

TEST_CASE("CUDA task tiles respect launch and memory limits",
          "[skala][task-batching]") {
  constexpr std::size_t launch_limit = 65535;
  const auto task_count = GENERATE(65534U, 65535U, 65536U, 131071U);
  const auto memory_limit = GENERATE(17U, 65535U, 131072U);
  std::vector<std::size_t> tasks(task_count);
  std::iota(tasks.begin(), tasks.end(), 0);
  std::vector<unsigned> visits(task_count, 0);
  auto begin = tasks.begin();
  std::size_t batches = 0;
  while (begin != tasks.end()) {
    const auto candidate_end =
        SkalaXC::detail::task_batch_end(begin, tasks.end(), launch_limit);
    REQUIRE(static_cast<std::size_t>(candidate_end - begin) <= launch_limit);
    // Model GauXC accepting only the prefix that fits its memory arena.
    const auto end =
        SkalaXC::detail::task_batch_end(begin, candidate_end, memory_limit);
    REQUIRE(end > begin);
    for (auto task = begin; task != end; ++task) ++visits[*task];
    begin = end;
    ++batches;
  }
  const auto capacity = std::min<std::size_t>(launch_limit, memory_limit);
  CHECK(batches == (task_count + capacity - 1) / capacity);
  CHECK(std::all_of(visits.begin(), visits.end(),
                    [](unsigned count) { return count == 1; }));
  CHECK(SkalaXC::detail::task_batch_end(tasks.end(), tasks.end(),
                                        launch_limit) == tasks.end());
  CHECK_THROWS_AS(
      SkalaXC::detail::task_batch_end(tasks.begin(), tasks.end(), 0),
      std::invalid_argument);
}
