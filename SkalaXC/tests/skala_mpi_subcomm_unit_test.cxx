#include <catch2/catch.hpp>

#include "model_grid_exchange.hpp"
#include "mpi_wrapper.hpp"
#include "skala_util.hpp"
#include "spin_gradient.hpp"

#include <gauxc/runtime_environment.hpp>
#include <gauxc/util/mpi.hpp>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

TEST_CASE("Eigen MPI collectives use runtime communicator",
          "[skala][mpi][subcomm][mpi-wrapper][mpi-only]") {
#ifdef GAUXC_HAS_MPI
  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (world_size < 3) {
    SUCCEED("Requires at least 3 MPI ranks");
    return;
  }

  MPI_Comm subcomm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, world_rank % 2, world_rank, &subcomm);
  GauXC::RuntimeEnvironment rt(GAUXC_MPI_CODE(subcomm));
  const int rank = rt.comm_rank();
  const int size = rt.comm_size();

  std::vector<int> counts;
  int total_count = 0;
  if (rank == 0) {
    counts.resize(size);
    for (int source_rank = 0; source_rank < size; ++source_rank) {
      counts[source_rank] = source_rank + 1;
      total_count += counts[source_rank];
    }
  }
  const SkalaXC::mpi::CollectiveLayout layout(std::move(counts));

  std::vector<double> local(rank + 1, static_cast<double>(rank));
  std::vector<double> gathered(rank == 0 ? total_count : 0);
  SkalaXC::mpi::gatherv(local, gathered, layout, rt);
  if (rank == 0)
    for (int source_rank = 0; source_rank < size; ++source_rank)
      for (int index = 0; index < layout.counts()[source_rank]; ++index)
        CHECK(gathered[layout.displacements()[source_rank] + index] ==
              source_rank);

  std::vector<double> scattered(rank + 1);
  SkalaXC::mpi::scatterv(gathered, layout, scattered, rt);
  CHECK(scattered == local);

  SkalaXC::RowMajorMatrix reduced =
      SkalaXC::RowMajorMatrix::Constant(2, 2, rank + 1.0);
  SkalaXC::mpi::allreduce_sum(reduced, rt);
  const double expected_sum = 0.5 * size * (size + 1);
  CHECK(
      reduced.isApprox(SkalaXC::RowMajorMatrix::Constant(2, 2, expected_sum)));

  std::string payload;
  if (rank == 0) payload = std::string("model\0archive", 13);
  SkalaXC::mpi::broadcast_string(payload, rt);
  CHECK(payload == std::string("model\0archive", 13));

  MPI_Comm_free(&subcomm);
#else
  SUCCEED("MPI disabled");
#endif
}

TEST_CASE("MPI gradient wrapper transports semantic point records",
          "[skala][mpi][subcomm][gradient-wrapper][mpi-only]") {
#ifdef GAUXC_HAS_MPI
  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (world_size < 3) {
    SUCCEED("Requires at least 3 MPI ranks");
    return;
  }

  MPI_Comm subcomm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, world_rank % 2, world_rank, &subcomm);
  GauXC::RuntimeEnvironment rt(GAUXC_MPI_CODE(subcomm));
  const int rank = rt.comm_rank();
  const int size = rt.comm_size();
  const int local_points = rank;
  const int source_offset = rank * (rank - 1) / 2;

  SkalaXC::SpinGradient local(local_points);
  for (int point = 0; point < local_points; ++point)
    for (int spin = 0; spin < SkalaXC::spin_dimension; ++spin)
      for (int direction = 0; direction < SkalaXC::direction_dimension;
           ++direction)
        local(static_cast<SkalaXC::Direction>(direction), point,
              static_cast<SkalaXC::SpinChannel>(spin)) =
            1000.0 * direction + 100.0 * spin + source_offset + point;

  std::vector<int> counts;
  std::vector<SkalaXC::types::PermutationIndex> reverse_permutation;
  int global_points = 0;
  if (rank == 0) {
    counts.resize(size);
    for (int source_rank = 0; source_rank < size; ++source_rank) {
      counts[source_rank] = source_rank;
      global_points += counts[source_rank];
    }
    reverse_permutation.resize(global_points);
    for (int source = 0; source < global_points; ++source)
      reverse_permutation[source] =
          SkalaXC::types::PermutationIndex{global_points - source - 1};
  }
  const SkalaXC::mpi::CollectiveLayout point_layout(std::move(counts));

  at::Tensor gathered = SkalaXC::mpi::gather_torch_gradient(
      local, point_layout, reverse_permutation, rt);
  if (rank == 0) {
    REQUIRE(gathered.sizes() ==
            at::IntArrayRef({SkalaXC::spin_dimension,
                             SkalaXC::direction_dimension, global_points}));
    for (int source = 0; source < global_points; ++source)
      for (int spin = 0; spin < SkalaXC::spin_dimension; ++spin)
        for (int direction = 0; direction < SkalaXC::direction_dimension;
             ++direction)
          CHECK(gathered.index({spin, direction, global_points - source - 1})
                    .item<double>() ==
                1000.0 * direction + 100.0 * spin + source);
  }

  at::Tensor root_gradient;
  if (rank == 0) root_gradient = gathered.detach() + 5.0;
  SkalaXC::SpinGradient restored = SkalaXC::mpi::scatter_torch_gradient(
      root_gradient, local_points, point_layout, reverse_permutation, rt);
  for (int point = 0; point < local_points; ++point)
    for (int spin = 0; spin < SkalaXC::spin_dimension; ++spin)
      for (int direction = 0; direction < SkalaXC::direction_dimension;
           ++direction)
        CHECK(restored(static_cast<SkalaXC::Direction>(direction), point,
                       static_cast<SkalaXC::SpinChannel>(spin)) ==
              local(static_cast<SkalaXC::Direction>(direction), point,
                    static_cast<SkalaXC::SpinChannel>(spin)) +
                  5.0);

  MPI_Comm_free(&subcomm);
#else
  SUCCEED("MPI disabled");
#endif
}

TEST_CASE("Model grid layout caches subcommunicator ordering metadata",
          "[skala][mpi][subcomm][model-grid][mpi-only]") {
#ifdef GAUXC_HAS_MPI
  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (world_size < 3) {
    SUCCEED("Requires at least 3 MPI ranks");
    return;
  }

  MPI_Comm subcomm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, world_rank % 2, world_rank, &subcomm);
  GauXC::RuntimeEnvironment rt(GAUXC_MPI_CODE(subcomm));
  const int rank = rt.comm_rank();
  const int size = rt.comm_size();
  std::vector<GauXC::XCTask> tasks(2);
  tasks[0].iParent = 1;
  tasks[0].points.resize(rank + 1);
  tasks[1].iParent = 0;
  tasks[1].points.resize(1);

  const SkalaXC::ModelGridLayout layout(tasks, SkalaXC::types::AtomCount{2},
                                        rt);
  const auto& blocks = layout.task_blocks();
  REQUIRE(blocks.size() == 2);
  CHECK(blocks[0].task_index == SkalaXC::types::TaskIndex{1});
  CHECK(blocks[0].point_offset == SkalaXC::types::GridPointOffset{0});
  CHECK(blocks[0].point_count == SkalaXC::types::GridPointCount{1});
  CHECK(blocks[1].task_index == SkalaXC::types::TaskIndex{0});
  CHECK(blocks[1].point_offset == SkalaXC::types::GridPointOffset{1});
  CHECK(blocks[1].point_count == SkalaXC::types::GridPointCount{rank + 1});
  CHECK(layout.local_point_count() == SkalaXC::types::GridPointCount{rank + 2});

  if (rank == 0) {
    const int expected_points = size * (size + 3) / 2;
    CHECK(layout.global_point_count() ==
          SkalaXC::types::GridPointCount{expected_points});
    CHECK(layout.point_layout().extent() == expected_points);
    REQUIRE(layout.point_layout().counts().size() ==
            static_cast<std::size_t>(size));
    for (int source_rank = 0; source_rank < size; ++source_rank)
      CHECK(layout.point_layout().counts()[source_rank] == source_rank + 2);
    const std::vector<SkalaXC::types::GridPointCount>
        expected_atom_point_counts{
            SkalaXC::types::GridPointCount{size},
            SkalaXC::types::GridPointCount{size * (size + 1) / 2}};
    CHECK(layout.global_atom_point_counts() == expected_atom_point_counts);
    if (size == 1) {
      CHECK(layout.rank_to_atom_points().empty());
      CHECK(layout.atom_to_rank_points().empty());
    } else {
      REQUIRE(layout.rank_to_atom_points().size() ==
              static_cast<std::size_t>(expected_points));
      REQUIRE(layout.atom_to_rank_points().size() ==
              static_cast<std::size_t>(expected_points));
      for (int rank_point = 0; rank_point < expected_points; ++rank_point)
        CHECK(layout.atom_to_rank_points()[static_cast<std::size_t>(
                  layout.rank_to_atom_points()[rank_point].raw())] ==
              SkalaXC::types::PermutationIndex{rank_point});
    }
  } else {
    CHECK(layout.global_atom_point_counts().empty());
    CHECK(layout.rank_to_atom_points().empty());
    CHECK(layout.atom_to_rank_points().empty());
  }

  MPI_Comm_free(&subcomm);
#else
  SUCCEED("MPI disabled");
#endif
}
