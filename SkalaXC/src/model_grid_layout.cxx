#include "model_grid_layout.hpp"

#include "exceptions.hpp"
#include "host/skala_util.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>

namespace SkalaXC {

ModelGridLayout::ModelGridLayout(
    const std::vector<GauXC::XCTask>& tasks, types::AtomCount atom_count,
    const GauXC::RuntimeEnvironment& rt,
    [[maybe_unused]] bool build_collective_metadata) {
  if (atom_count.raw() >
      static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    SKALAXC_EXCEPTION("Atom count exceeds supported size");
  const auto atom_count_value = static_cast<std::size_t>(atom_count.raw());

  std::vector<std::size_t> task_order(tasks.size());
  std::iota(task_order.begin(), task_order.end(), std::size_t{0});
  std::stable_sort(task_order.begin(), task_order.end(),
                   [&](std::size_t left, std::size_t right) {
                     return tasks[left].iParent < tasks[right].iParent;
                   });

  local_atom_point_counts_.assign(atom_count_value, types::GridPointCount{});
  task_blocks_.reserve(tasks.size());
  for (const std::size_t task_index : task_order) {
    const auto& task = tasks[task_index];
    if (task.iParent < 0 ||
        static_cast<std::uint64_t>(task.iParent) >= atom_count.raw())
      SKALAXC_EXCEPTION("Invalid task parent atom");
    if (task.points.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max() -
                                 local_point_count_.raw()))
      SKALAXC_EXCEPTION("Local model grid exceeds supported size");

    const auto point_count =
        types::GridPointCount{static_cast<std::int64_t>(task.points.size())};
    const auto parent_atom =
        types::AtomIndex{static_cast<std::size_t>(task.iParent)};
    task_blocks_.push_back({types::TaskIndex{task_index},
                            types::GridPointOffset{local_point_count_.raw()},
                            point_count, parent_atom});
    local_point_count_ += point_count;
    local_atom_point_counts_[parent_atom.raw()] += point_count;
  }

  for (std::size_t atom = 0; atom < atom_count_value; ++atom)
    if (local_atom_point_counts_[atom].raw() > 0)
      local_atoms_.push_back(types::AtomIndex{atom});

  global_point_count_ = local_point_count_;
  global_atom_point_counts_ = local_atom_point_counts_;
  point_layout_ =
      mpi::CollectiveLayout({static_cast<int>(local_point_count_.raw())});

#ifdef GAUXC_HAS_MPI
  if (rt.comm_size() > 1 && build_collective_metadata) {
    const types::CommunicatorRank rank{rt.comm_rank()};
    const types::CommunicatorSize rank_count{rt.comm_size()};
    std::vector<int> local_count{static_cast<int>(local_point_count_.raw())};
    std::vector<int> rank_point_counts(
        rank == types::CommunicatorRank{0} ? rank_count.raw() : 0);
    mpi::gather(local_count, rank_point_counts, rt);

    std::vector<std::int64_t> local_atom_point_counts;
    local_atom_point_counts.reserve(local_atom_point_counts_.size());
    for (const auto count : local_atom_point_counts_)
      local_atom_point_counts.push_back(count.raw());
    std::vector<std::int64_t> all_rank_atom_point_count_values(
        rank == types::CommunicatorRank{0}
            ? static_cast<std::size_t>(rank_count.raw()) * atom_count_value
            : 0);
    mpi::gather(local_atom_point_counts, all_rank_atom_point_count_values, rt);

    point_layout_ = {};
    global_atom_point_counts_.clear();
    global_point_count_ = types::GridPointCount{};
    if (rank == types::CommunicatorRank{0}) {
      point_layout_ = mpi::CollectiveLayout(std::move(rank_point_counts));
      global_point_count_ = types::GridPointCount{point_layout_.extent()};
      global_atom_point_counts_.assign(atom_count_value,
                                       types::GridPointCount{});
      for (int source_rank = 0; source_rank < rank_count.raw(); ++source_rank)
        for (std::size_t atom = 0; atom < atom_count_value; ++atom)
          global_atom_point_counts_[atom] += types::GridPointCount{
              all_rank_atom_point_count_values
                  [static_cast<std::size_t>(source_rank) * atom_count_value +
                   atom]};

      std::vector<types::GridPointCount> all_rank_atom_point_counts;
      all_rank_atom_point_counts.reserve(
          all_rank_atom_point_count_values.size());
      for (const auto count : all_rank_atom_point_count_values)
        all_rank_atom_point_counts.push_back(types::GridPointCount{count});

      auto permutations = build_atom_reorder_perm(
          all_rank_atom_point_counts, point_layout_, atom_count, rank_count);
      rank_to_atom_points_ = std::move(permutations.first);
      atom_to_rank_points_ = std::move(permutations.second);
    }
  }
#else
  if (rt.comm_size() > 1)
    SKALAXC_EXCEPTION("MPI runtime used by a non-MPI SkalaXC build");
#endif
}

std::vector<ModelDomainBatch> ModelGridLayout::make_local_batches(
    DomainBatchMode batch_mode) const {
  std::vector<std::vector<types::AtomIndex>> batch_atoms;
  if (batch_mode == DomainBatchMode::Conservative) {
    batch_atoms.reserve(local_atoms_.size());
    for (const auto atom : local_atoms_) batch_atoms.push_back({atom});
  } else {
    std::map<std::int64_t, std::vector<types::AtomIndex>> atoms_by_grid_size;
    for (const auto atom : local_atoms_)
      atoms_by_grid_size[local_atom_point_counts_[atom.raw()].raw()].push_back(
          atom);
    batch_atoms.reserve(atoms_by_grid_size.size());
    for (auto& [grid_size, atoms] : atoms_by_grid_size) {
      (void)grid_size;
      batch_atoms.push_back(std::move(atoms));
    }
  }

  std::vector<ModelDomainBatch> batches;
  batches.reserve(batch_atoms.size());
  for (auto& atoms : batch_atoms) {
    ModelDomainBatch batch;
    batch.atoms = std::move(atoms);
    batch.grid_size = local_atom_point_counts_[batch.atoms.front().raw()];
    for (const auto atom : batch.atoms) {
      if (local_atom_point_counts_[atom.raw()] != batch.grid_size)
        SKALAXC_EXCEPTION("Model batch mixes atomic grid sizes");
      for (const auto& block : task_blocks_) {
        if (block.parent_atom != atom) continue;
        batch.task_blocks.push_back(
            {block.task_index, types::GridPointOffset{batch.point_count.raw()},
             block.point_count, atom});
        batch.point_count += block.point_count;
      }
    }
    batches.push_back(std::move(batch));
  }
  return batches;
}

}  // namespace SkalaXC