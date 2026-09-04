#pragma once

#include "host/mpi_wrapper.hpp"
#include "index_types.hpp"

#include <gauxc/runtime_environment.hpp>
#include <gauxc/xc_task.hpp>
#include <skalaxc/skalaxc.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace SkalaXC {

/** @brief Stable mapping from one GauXC task to a packed point interval. */
struct ModelTaskBlock {
  types::TaskIndex task_index;          ///< Rank-local GauXC task index.
  types::GridPointOffset point_offset;  ///< First packed point.
  types::GridPointCount point_count;    ///< Points in this task.
  types::AtomIndex parent_atom;         ///< Global owning atom index.
};

/** @brief Exact-size batch of complete atomic domains owned by one rank. */
struct ModelDomainBatch {
  std::vector<types::AtomIndex> atoms;      ///< Global atom indices.
  std::vector<ModelTaskBlock> task_blocks;  ///< Batch-local task layout.
  types::GridPointCount point_count{};      ///< Total points in the batch.
  types::GridPointCount grid_size{};        ///< Points in each domain.
};

/** @brief Cache task, atom, point, and optional MPI reorder metadata. */
class ModelGridLayout {
 public:
  /**
   * @brief Build fixed local layout and optional communicator-wide reorder
   * metadata.
   * @param tasks Sorted rank-local GauXC tasks.
   * @param atom_count Global atom count.
   * @param rt Runtime environment defining the communicator.
   * @param build_collective_metadata Whether to gather global point metadata.
   */
  ModelGridLayout(const std::vector<GauXC::XCTask>& tasks,
                  types::AtomCount atom_count,
                  const GauXC::RuntimeEnvironment& rt,
                  bool build_collective_metadata = true);

  /** @return Rank-local task blocks in packed point order. */
  const std::vector<ModelTaskBlock>& task_blocks() const noexcept {
    return task_blocks_;
  }
  /** @return Total number of rank-local points. */
  types::GridPointCount local_point_count() const noexcept {
    return local_point_count_;
  }
  /** @return Total number of points across the runtime communicator. */
  types::GridPointCount global_point_count() const noexcept {
    return global_point_count_;
  }
  /** @return Rank-local point count for every global atom. */
  const std::vector<types::GridPointCount>& local_atom_point_counts()
      const noexcept {
    return local_atom_point_counts_;
  }
  /** @return Global indices of complete atomic domains owned locally. */
  const std::vector<types::AtomIndex>& local_atoms() const noexcept {
    return local_atoms_;
  }
  /**
   * @brief Group complete local domains according to a batching policy.
   * @param batch_mode Exact-size domain batching policy.
   * @return Rank-local model batches.
   */
  std::vector<ModelDomainBatch> make_local_batches(
      DomainBatchMode batch_mode) const;
  /** @return Per-rank global point counts and displacements. */
  const mpi::CollectiveLayout& point_layout() const noexcept {
    return point_layout_;
  }
  /** @return Communicator-wide point count for every global atom. */
  const std::vector<types::GridPointCount>& global_atom_point_counts()
      const noexcept {
    return global_atom_point_counts_;
  }
  /** @return Destination atom-order index for each rank-order point. */
  const std::vector<types::PermutationIndex>& rank_to_atom_points()
      const noexcept {
    return rank_to_atom_points_;
  }
  /** @return Destination rank-order index for each atom-order point. */
  const std::vector<types::PermutationIndex>& atom_to_rank_points()
      const noexcept {
    return atom_to_rank_points_;
  }

 private:
  std::vector<ModelTaskBlock> task_blocks_;
  types::GridPointCount local_point_count_{};
  types::GridPointCount global_point_count_{};
  std::vector<types::GridPointCount> local_atom_point_counts_;
  std::vector<types::AtomIndex> local_atoms_;
  mpi::CollectiveLayout point_layout_;
  std::vector<types::GridPointCount> global_atom_point_counts_;
  std::vector<types::PermutationIndex> rank_to_atom_points_;
  std::vector<types::PermutationIndex> atom_to_rank_points_;
};

}  // namespace SkalaXC
