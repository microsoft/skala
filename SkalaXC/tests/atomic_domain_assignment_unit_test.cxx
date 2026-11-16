#include "atomic_domain_assignment.hpp"
#include "saturating_math.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

std::vector<SkalaXC::types::WorkEstimate> work_estimates(
    std::initializer_list<std::uint64_t> values) {
  std::vector<SkalaXC::types::WorkEstimate> result;
  result.reserve(values.size());
  for (const auto value : values)
    result.push_back(SkalaXC::types::WorkEstimate{value});
  return result;
}

std::vector<SkalaXC::types::CommunicatorRank> communicator_ranks(
    std::initializer_list<int> values) {
  std::vector<SkalaXC::types::CommunicatorRank> result;
  result.reserve(values.size());
  for (const auto value : values)
    result.push_back(SkalaXC::types::CommunicatorRank{value});
  return result;
}

}  // namespace

TEST_CASE("unsigned arithmetic saturates instead of wrapping",
          "[saturating-math]") {
  const auto maximum = std::numeric_limits<std::uint64_t>::max();

  CHECK(SkalaXC::detail::saturating_add(3, 4) == 7);
  CHECK(SkalaXC::detail::saturating_add(maximum, 1) == maximum);
  CHECK(SkalaXC::detail::saturating_multiply(3, 4) == 12);
  CHECK(SkalaXC::detail::saturating_multiply(maximum, 0) == 0);
  CHECK(SkalaXC::detail::saturating_multiply(maximum, 2) == maximum);
}

TEST_CASE("atomic domains use deterministic least-loaded ownership",
          "[atomic-domain-assignment]") {
  const auto assignment = SkalaXC::detail::assign_atomic_domains(
      work_estimates({8, 7, 6, 5}), SkalaXC::types::CommunicatorSize{2});

  CHECK(assignment.owner_by_atom == communicator_ranks({0, 1, 1, 0}));
  CHECK(assignment.rank_costs == work_estimates({13, 13}));
}

TEST_CASE("atomic domain assignment has stable tie breaking",
          "[atomic-domain-assignment]") {
  SECTION("equal atom costs retain atom index order") {
    const auto assignment = SkalaXC::detail::assign_atomic_domains(
        work_estimates({4, 4, 4, 4, 4}), SkalaXC::types::CommunicatorSize{3});

    CHECK(assignment.owner_by_atom == communicator_ranks({0, 1, 2, 0, 1}));
    CHECK(assignment.rank_costs == work_estimates({8, 8, 4}));
  }

  SECTION("lowest rank wins equal load") {
    const auto assignment = SkalaXC::detail::assign_atomic_domains(
        work_estimates({0, 0, 0}), SkalaXC::types::CommunicatorSize{2});

    CHECK(assignment.owner_by_atom == communicator_ranks({0, 0, 0}));
  }

  SECTION("excess ranks and empty inputs are supported") {
    const auto one_atom = SkalaXC::detail::assign_atomic_domains(
        work_estimates({9}), SkalaXC::types::CommunicatorSize{3});
    const auto no_atoms = SkalaXC::detail::assign_atomic_domains(
        work_estimates({}), SkalaXC::types::CommunicatorSize{2});

    CHECK(one_atom.owner_by_atom == communicator_ranks({0}));
    CHECK(one_atom.rank_costs == work_estimates({9, 0, 0}));
    CHECK(no_atoms.owner_by_atom.empty());
    CHECK(no_atoms.rank_costs == work_estimates({0, 0}));
  }

  SECTION("rank costs saturate instead of wrapping") {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    const auto assignment = SkalaXC::detail::assign_atomic_domains(
        work_estimates({maximum, maximum, 1}),
        SkalaXC::types::CommunicatorSize{2});

    CHECK(assignment.owner_by_atom == communicator_ranks({0, 1, 0}));
    CHECK(assignment.rank_costs == work_estimates({maximum, maximum}));
  }

  CHECK_THROWS_AS(SkalaXC::detail::assign_atomic_domains(
                      work_estimates({1}), SkalaXC::types::CommunicatorSize{0}),
                  std::invalid_argument);
}