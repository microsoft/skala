#include <catch2/catch_test_macros.hpp>

#include "model_grid_exchange.hpp"
#include "skala_util.hpp"

#include <skalaxc/skalaxc.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename Left, typename Right, typename = void>
struct IsAddable : std::false_type {};

template <typename Left, typename Right>
struct IsAddable<
    Left, Right,
    std::void_t<decltype(std::declval<Left>() + std::declval<Right>())>>
    : std::true_type {};

static_assert(SkalaXC::types::GridPointCount::is_additive);
static_assert(!SkalaXC::types::AtomIndex::is_additive);
static_assert(
    std::is_same_v<
        decltype(std::declval<const SkalaXC::types::GridPointCount&>().raw()),
        std::int64_t>);
static_assert(
    !std::is_constructible_v<std::int64_t, SkalaXC::types::GridPointCount>);
static_assert(
    !std::is_convertible_v<SkalaXC::types::GridPointCount, std::int64_t>);
static_assert(IsAddable<SkalaXC::types::GridPointCount,
                        SkalaXC::types::GridPointCount>::value);
static_assert(IsAddable<SkalaXC::types::GridPointOffset,
                        SkalaXC::types::GridPointCount>::value);
static_assert(!IsAddable<SkalaXC::types::GridPointOffset,
                         SkalaXC::types::GridPointOffset>::value);
static_assert(
    !IsAddable<SkalaXC::types::AtomIndex, SkalaXC::types::AtomIndex>::value);
static_assert(!IsAddable<SkalaXC::types::CommunicatorRank,
                         SkalaXC::types::CommunicatorRank>::value);
static_assert(!IsAddable<SkalaXC::types::AtomCount,
                         SkalaXC::types::GridPointCount>::value);

std::vector<SkalaXC::types::PermutationIndex> permutation(
    std::initializer_list<std::int64_t> values) {
  std::vector<SkalaXC::types::PermutationIndex> result;
  result.reserve(values.size());
  for (const auto value : values)
    result.push_back(SkalaXC::types::PermutationIndex{value});
  return result;
}

std::vector<SkalaXC::types::GridPointCount> point_counts(
    std::initializer_list<std::int64_t> values) {
  std::vector<SkalaXC::types::GridPointCount> result;
  result.reserve(values.size());
  for (const auto value : values)
    result.push_back(SkalaXC::types::GridPointCount{value});
  return result;
}

}  // namespace

TEST_CASE("Collective layout keeps MPI metadata consistent",
          "[skala][mpi][layout]") {
  const std::vector<int> counts = {2, 3};
  const std::vector<int> mismatched_displacements = {0};
  CHECK_THROWS_AS(
      SkalaXC::mpi::CollectiveLayout(counts, mismatched_displacements),
      std::invalid_argument);

  const SkalaXC::mpi::CollectiveLayout large_layout(
      {std::numeric_limits<int>::max()});
  CHECK_THROWS_AS(large_layout.scaled(2), std::invalid_argument);
}

TEST_CASE("Model grid layout caches stable atom-ordered task blocks",
          "[skala][model-grid][layout]") {
  std::vector<GauXC::XCTask> tasks(3);
  tasks[0].iParent = 1;
  tasks[0].points.resize(2);
  tasks[1].iParent = 0;
  tasks[1].points.resize(1);
  tasks[2].iParent = 1;
  tasks[2].points.resize(3);
  GauXC::RuntimeEnvironment rt{GAUXC_MPI_CODE(MPI_COMM_SELF)};

  const SkalaXC::ModelGridLayout layout(tasks, SkalaXC::types::AtomCount{2},
                                        rt);
  const auto& blocks = layout.task_blocks();

  STATIC_REQUIRE_FALSE(std::is_convertible_v<SkalaXC::types::TaskIndex,
                                             SkalaXC::types::AtomIndex>);
  STATIC_REQUIRE_FALSE(std::is_convertible_v<SkalaXC::types::GridPointOffset,
                                             SkalaXC::types::GridPointCount>);
  REQUIRE(blocks.size() == 3);
  CHECK(blocks[0].task_index == SkalaXC::types::TaskIndex{1});
  CHECK(blocks[0].point_offset == SkalaXC::types::GridPointOffset{0});
  CHECK(blocks[0].point_count == SkalaXC::types::GridPointCount{1});
  CHECK(blocks[1].task_index == SkalaXC::types::TaskIndex{0});
  CHECK(blocks[1].point_offset == SkalaXC::types::GridPointOffset{1});
  CHECK(blocks[1].point_count == SkalaXC::types::GridPointCount{2});
  CHECK(blocks[2].task_index == SkalaXC::types::TaskIndex{2});
  CHECK(blocks[2].point_offset == SkalaXC::types::GridPointOffset{3});
  CHECK(blocks[2].point_count == SkalaXC::types::GridPointCount{3});
  CHECK(layout.local_point_count() == SkalaXC::types::GridPointCount{6});
  CHECK(layout.global_point_count() == SkalaXC::types::GridPointCount{6});
  CHECK(layout.point_layout().counts() == std::vector<int>{6});
  const std::vector<SkalaXC::types::GridPointCount> expected_atom_point_counts{
      SkalaXC::types::GridPointCount{1}, SkalaXC::types::GridPointCount{5}};
  CHECK(layout.global_atom_point_counts() == expected_atom_point_counts);
  CHECK(layout.rank_to_atom_points().empty());
  CHECK(layout.atom_to_rank_points().empty());
}

TEST_CASE("Model domain batches preserve exact atomic grid sizes",
          "[skala][model-grid][batching]") {
  std::vector<GauXC::XCTask> tasks(4);
  tasks[0].iParent = 1;
  tasks[0].points.resize(5);
  tasks[1].iParent = 0;
  tasks[1].points.resize(2);
  tasks[2].iParent = 2;
  tasks[2].points.resize(3);
  tasks[3].iParent = 0;
  tasks[3].points.resize(3);
  GauXC::RuntimeEnvironment rt{GAUXC_MPI_CODE(MPI_COMM_SELF)};

  const SkalaXC::ModelGridExchange conservative(
      tasks, SkalaXC::types::AtomCount{3}, rt,
      SkalaXC::DomainBatchMode::Conservative);
  const auto& conservative_batches = conservative.local_batches();
  REQUIRE(conservative_batches.size() == 3);
  CHECK(conservative_batches[0].atoms ==
        std::vector<SkalaXC::types::AtomIndex>{SkalaXC::types::AtomIndex{0}});
  CHECK(conservative_batches[0].grid_size == SkalaXC::types::GridPointCount{5});
  CHECK(conservative_batches[0].point_count ==
        SkalaXC::types::GridPointCount{5});
  CHECK(conservative_batches[1].atoms ==
        std::vector<SkalaXC::types::AtomIndex>{SkalaXC::types::AtomIndex{1}});
  CHECK(conservative_batches[1].grid_size == SkalaXC::types::GridPointCount{5});
  CHECK(conservative_batches[2].atoms ==
        std::vector<SkalaXC::types::AtomIndex>{SkalaXC::types::AtomIndex{2}});
  CHECK(conservative_batches[2].grid_size == SkalaXC::types::GridPointCount{3});

  const SkalaXC::ModelGridExchange aggressive(
      tasks, SkalaXC::types::AtomCount{3}, rt,
      SkalaXC::DomainBatchMode::Aggressive);
  const auto& aggressive_batches = aggressive.local_batches();
  REQUIRE(aggressive_batches.size() == 2);
  CHECK(aggressive_batches[0].atoms ==
        std::vector<SkalaXC::types::AtomIndex>{SkalaXC::types::AtomIndex{2}});
  CHECK(aggressive_batches[0].grid_size == SkalaXC::types::GridPointCount{3});
  CHECK(aggressive_batches[0].point_count == SkalaXC::types::GridPointCount{3});
  CHECK((aggressive_batches[1].atoms ==
         std::vector<SkalaXC::types::AtomIndex>{SkalaXC::types::AtomIndex{0},
                                                SkalaXC::types::AtomIndex{1}}));
  CHECK(aggressive_batches[1].grid_size == SkalaXC::types::GridPointCount{5});
  CHECK(aggressive_batches[1].point_count ==
        SkalaXC::types::GridPointCount{10});
  REQUIRE(aggressive_batches[1].task_blocks.size() == 3);
  CHECK(aggressive_batches[1].task_blocks[0].point_offset ==
        SkalaXC::types::GridPointOffset{0});
  CHECK(aggressive_batches[1].task_blocks[1].point_offset ==
        SkalaXC::types::GridPointOffset{2});
  CHECK(aggressive_batches[1].task_blocks[2].point_offset ==
        SkalaXC::types::GridPointOffset{5});
}

TEST_CASE("Model grid layout throws the SkalaXC exception type",
          "[skala][model-grid][exceptions]") {
  std::vector<GauXC::XCTask> tasks(1);
  tasks.front().iParent = 0;
  GauXC::RuntimeEnvironment rt{GAUXC_MPI_CODE(MPI_COMM_SELF)};
  try {
    SkalaXC::ModelGridLayout layout(tasks, SkalaXC::types::AtomCount{0}, rt);
    FAIL("Expected an invalid parent atom to throw");
  } catch (const SkalaXC::Exception& error) {
    const std::string message = error.what();
    REQUIRE(message.find("SkalaXC Exception (Invalid task parent atom)") !=
            std::string::npos);
    REQUIRE(message.find("model_grid_layout.cxx") != std::string::npos);
    REQUIRE(message.find("  Function ") != std::string::npos);
    REQUIRE(message.find("  Line     ") != std::string::npos);
  }
}

TEST_CASE("Point reorder validates dimensions at its boundary",
          "[skala][reorder]") {
  SkalaXC::Vector weights(2);
  SkalaXC::AlphaBetaMatrix density(3, 2);
  SkalaXC::CartesianMatrix coordinates(2, 3);
  SkalaXC::AlphaBetaMatrix kinetic;
  const auto point_permutation = permutation({0, 1});
  CHECK_THROWS(SkalaXC::reorder_to_atom_order(
      weights, density, coordinates, kinetic, point_permutation,
      SkalaXC::types::GridPointCount{2}));

  density.resize(2, 2);
  const auto short_permutation = permutation({0});
  CHECK_THROWS(SkalaXC::reorder_to_rank_order(
      density, kinetic, short_permutation, SkalaXC::types::GridPointCount{2}));
}

TEST_CASE("Atom reorder uses shaped point records", "[skala][reorder]") {
  const SkalaXC::types::AtomCount atom_count{3};
  const SkalaXC::types::CommunicatorSize communicator_size{2};
  const auto all_rank_atom_sizes = point_counts({2, 3, 1, 1, 0, 2});
  const SkalaXC::mpi::CollectiveLayout point_layout({6, 3});
  const std::vector<int> expected_displacements = {0, 6};
  const std::vector<int> expected_scaled_counts = {12, 6};

  CHECK(point_layout.displacements() == expected_displacements);
  CHECK(point_layout.extent() == 9);
  CHECK(point_layout.scaled(2).counts() == expected_scaled_counts);

  auto [permutation, inverse] = SkalaXC::build_atom_reorder_perm(
      all_rank_atom_sizes, point_layout, atom_count, communicator_size);

  REQUIRE(permutation.size() == 9);
  REQUIRE(inverse.size() == 9);
  for (std::int64_t point = 0; point < 9; ++point)
    CHECK(inverse[static_cast<std::size_t>(permutation[point].raw())] ==
          SkalaXC::types::PermutationIndex{point});

  SkalaXC::Vector weights(9);
  SkalaXC::AlphaBetaMatrix density(9, 2);
  SkalaXC::CartesianMatrix coordinates(9, 3);
  SkalaXC::AlphaBetaMatrix kinetic(9, 2);
  for (Eigen::Index point = 0; point < 9; ++point) {
    weights(point) = 10.0 + point;
    for (Eigen::Index spin = 0; spin < 2; ++spin) {
      density(point, spin) = 100.0 * spin + point;
      kinetic(point, spin) = 200.0 * spin + point;
    }
    for (Eigen::Index direction = 0; direction < 3; ++direction)
      coordinates(point, direction) = 300.0 * direction + point;
  }

  const SkalaXC::Vector original_weights = weights;
  const SkalaXC::AlphaBetaMatrix original_density = density;
  const SkalaXC::CartesianMatrix original_coordinates = coordinates;
  const SkalaXC::AlphaBetaMatrix original_kinetic = kinetic;

  SkalaXC::reorder_to_atom_order(weights, density, coordinates, kinetic,
                                 permutation,
                                 SkalaXC::types::GridPointCount{9});
  SkalaXC::reorder_to_rank_order(density, kinetic, inverse,
                                 SkalaXC::types::GridPointCount{9});

  SkalaXC::Vector restored_weights(9);
  SkalaXC::CartesianMatrix restored_coordinates(9, 3);
  for (Eigen::Index atom_point = 0; atom_point < 9; ++atom_point) {
    const auto rank_point = inverse[atom_point].raw();
    restored_weights(rank_point) = weights(atom_point);
    restored_coordinates.row(rank_point) = coordinates.row(atom_point);
  }

  CHECK(density.isApprox(original_density));
  CHECK(kinetic.isApprox(original_kinetic));
  CHECK(restored_weights.isApprox(original_weights));
  CHECK(restored_coordinates.isApprox(original_coordinates));
}
