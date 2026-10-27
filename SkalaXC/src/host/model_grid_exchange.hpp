#pragma once

#include "model_grid_layout.hpp"
#include "skala_util.hpp"
#include "task_data.hpp"

#include <gauxc/load_balancer.hpp>
#include <gauxc/molecule.hpp>
#include <gauxc/runtime_environment.hpp>
#include <skalaxc/skalaxc.hpp>

#include <vector>

namespace SkalaXC {

/** @brief Exchanges task features and model derivatives in cached grid order.
 */
class ModelGridExchange {
 public:
  /** @brief Build exchange metadata for a fixed task decomposition. */
  /**
   * @param tasks Sorted rank-local integration tasks.
   * @param atom_count Global atom count.
   * @param rt Runtime environment defining the communicator.
   * @param batch_mode Exact-size complete-domain batching policy.
   */
  ModelGridExchange(const std::vector<GauXC::XCTask>& tasks,
                    types::AtomCount atom_count,
                    const GauXC::RuntimeEnvironment& rt,
                    DomainBatchMode batch_mode = DomainBatchMode::Conservative);

  /** @return Exact-size batches of complete locally owned domains. */
  const std::vector<ModelDomainBatch>& local_batches() const noexcept {
    return local_batches_;
  }

  /**
   * @brief Pack one batch of complete locally owned atomic domains.
   * @param batch Batch metadata.
   * @param tasks Sorted rank-local tasks.
   * @param task_features Per-task feature buffers.
   * @param raw_weights Per-task pre-partition weights.
   * @param molecule Molecular coordinates.
   * @param feature_keys Requested model feature keys.
   * @return Model feature dictionary for the batch.
   */
  FeatureDict prepare_local_features(
      const ModelDomainBatch& batch, const std::vector<GauXC::XCTask>& tasks,
      const std::vector<TaskFeatureData>& task_features,
      const std::vector<std::vector<double>>& raw_weights,
      const GauXC::Molecule& molecule,
      const std::vector<std::string>& feature_keys) const;

  /**
   * @brief Map one local batch's derivatives into local task buffers.
   * @param batch Batch metadata.
   * @param has_density_gradient Whether density-gradient derivatives exist.
   * @param has_kinetic Whether kinetic-density derivatives exist.
   * @param feature_dict Model derivative tensors.
   * @param task_potentials Destination per-task buffers.
   */
  void distribute_local_potentials(
      const ModelDomainBatch& batch, bool has_density_gradient,
      bool has_kinetic, const FeatureDict& feature_dict,
      std::vector<TaskPotentialData>& task_potentials) const;

  /**
   * @brief Map one local batch's `dE/dw` values into local tasks.
   * @param batch Batch metadata.
   * @param atom_ordered_values Batch values in atom-major point order.
   * @param task_potentials Destination per-task buffers.
   */
  void distribute_local_dE_dw(
      const ModelDomainBatch& batch, std::vector<double> atom_ordered_values,
      std::vector<TaskPotentialData>& task_potentials) const;

  /**
   * @brief Accumulate one local batch's point derivatives by parent atom.
   * @param batch Batch metadata.
   * @param point_gradient Point-coordinate derivatives.
   * @param atom_gradient Destination atom-major Cartesian gradient.
   */
  void accumulate_local_point_gradient(const ModelDomainBatch& batch,
                                       const at::Tensor& point_gradient,
                                       RowMajorMatrixMap atom_gradient) const;

  /**
   * @brief Accumulate one batch's coordinate derivatives by parent atom.
   * @param batch Batch metadata.
   * @param coordinate_gradient Explicit atomic-coordinate derivatives.
   * @param atom_gradient Destination atom-major Cartesian gradient.
   */
  void accumulate_local_coordinate_gradient(
      const ModelDomainBatch& batch, const at::Tensor& coordinate_gradient,
      RowMajorMatrixMap atom_gradient) const;

  /** @return Number of complete atomic domains owned by this rank. */
  std::size_t local_domain_count() const noexcept {
    return layout_.local_atoms().size();
  }

 private:
  ModelGridLayout layout_;
  std::vector<ModelDomainBatch> local_batches_;
};

}  // namespace SkalaXC
