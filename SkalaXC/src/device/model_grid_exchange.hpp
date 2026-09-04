#pragma once
/**
 * @file
 * @brief Exchange data between GauXC CUDA tasks and Skala model tensors.
 */

#include "host/skala_util.hpp"
#include "model_grid_layout.hpp"

#include <c10/core/Device.h>
#include <gauxc/molecule.hpp>
#include <gauxc/runtime_environment.hpp>
#include <gauxc/xc_task.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace at {
class Tensor;
}

namespace GauXC {
struct XCDeviceAoSData;
}

namespace SkalaXC {

/**
 * @brief Translate task-ordered device data to atom-ordered model tensors.
 *
 * Complete atomic domains are packed into exact-size rank-local model batches.
 */
class DeviceModelGridExchange {
 public:
  /**
   * @brief Build fixed task, point, atom, and MPI exchange metadata.
   * @param tasks Sorted local integration tasks.
   * @param atom_count Number of atoms in the molecular system.
   * @param rt Runtime environment whose communicator defines rank layout.
   * @param raw_weights Per-task quadrature weights before partitioning.
   * @param device_id CUDA device that owns persistent exchange metadata.
   * @param batch_mode Exact-size complete-domain batching policy.
   */
  DeviceModelGridExchange(const std::vector<GauXC::XCTask>& tasks,
                          types::AtomCount atom_count,
                          const GauXC::RuntimeEnvironment& rt,
                          const std::vector<std::vector<double>>& raw_weights,
                          types::DeviceId device_id,
                          DomainBatchMode batch_mode);

  /** @brief Release persistent SkalaXC-owned CUDA exchange storage. */
  ~DeviceModelGridExchange() noexcept;

  /**
   * @brief Allocate rank-local model feature tensors on a device.
   * @param feature_keys Model input keys to allocate.
   * @param device LibTorch device on which tensors are allocated.
   * @return Feature dictionary sized for the local grid.
   */
  FeatureDict prepare_features(const std::vector<std::string>& feature_keys,
                               const c10::Device& device) const;

  /** @return Exact-size batches of complete locally owned domains. */
  const std::vector<ModelDomainBatch>& local_batches() const noexcept {
    return local_batches_;
  }

  /**
   * @brief Slice one exact-size local domain batch into model inputs.
   * @param batch Batch metadata.
   * @param local_features Full rank-local feature tensors.
   * @param molecule Molecular coordinates.
   * @param feature_keys Requested model feature keys.
   * @param device LibTorch device owning the tensors.
   * @param geometry_gradients Whether coordinate inputs require gradients.
   * @return Batch feature dictionary.
   */
  FeatureDict prepare_local_batch_features(
      const ModelDomainBatch& batch, const FeatureDict& local_features,
      const GauXC::Molecule& molecule,
      const std::vector<std::string>& feature_keys, const c10::Device& device,
      bool geometry_gradients) const;

  /**
   * @brief Allocate full-grid rank-local derivative tensors.
   * @param has_density_gradient Whether to allocate density-gradient storage.
   * @param has_kinetic Whether to allocate kinetic-density storage.
   * @param device LibTorch device owning the tensors.
   * @return Rank-local potential dictionary.
   */
  FeatureDict prepare_local_potentials(bool has_density_gradient,
                                       bool has_kinetic,
                                       const c10::Device& device) const;

  /**
   * @brief Copy one local batch's autograd derivatives into full-grid tensors.
   * @param batch Batch metadata.
   * @param has_density_gradient Whether density-gradient derivatives exist.
   * @param has_kinetic Whether kinetic-density derivatives exist.
   * @param batch_features Batch tensors carrying autograd derivatives.
   * @param local_potentials Destination full-grid tensors.
   */
  void store_local_batch_potentials(const ModelDomainBatch& batch,
                                    bool has_density_gradient, bool has_kinetic,
                                    const FeatureDict& batch_features,
                                    const FeatureDict& local_potentials) const;

  /**
   * @brief Allocate a full-grid rank-local `dE/dw` tensor.
   * @param device Owning LibTorch device.
   * @return Allocated tensor.
   */
  at::Tensor prepare_local_dE_dw(const c10::Device& device) const;

  /**
   * @brief Copy one batch's `dE/dw` values into the full local grid.
   * @param batch Batch metadata.
   * @param batch_dE_dw Source batch tensor.
   * @param local_dE_dw Destination full-grid tensor.
   */
  void store_local_batch_dE_dw(const ModelDomainBatch& batch,
                               const at::Tensor& batch_dE_dw,
                               const at::Tensor& local_dE_dw) const;

  /**
   * @brief Add one local batch's point and coordinate derivatives on device.
   * @param batch_index Index into the vector returned by @c local_batches().
   * @param batch_features Model inputs containing autograd derivatives.
   * @param device_data GauXC device data whose queue orders the operation.
   */
  void accumulate_local_geometry_gradients(
      std::size_t batch_index, const FeatureDict& batch_features,
      GauXC::XCDeviceAoSData& device_data) const;

  /**
   * @brief Pack one post-U-variable task batch into local model tensors.
   *
   * GauXC's UKS `eval_uvars_*` must have completed for the active batch.
   * @param device_data Device data containing the active task batch.
   * @param first_task Batch's first index in the load balancer task list.
   * @param feature_dict Destination rank-local device tensors.
   */
  void pack_post_uvars_features(GauXC::XCDeviceAoSData& device_data,
                                types::TaskIndex first_task,
                                const FeatureDict& feature_dict) const;

  /**
   * @brief Unpack rank-local model potentials into one GauXC task batch.
   * @param device_data Device data containing the active task batch.
   * @param first_task Batch's first index in the load balancer task list.
   * @param feature_dict Source rank-local potential tensors.
   */
  void unpack_potentials(GauXC::XCDeviceAoSData& device_data,
                         types::TaskIndex first_task,
                         const FeatureDict& feature_dict) const;

  /**
   * @brief Prepare a task batch for GauXC molecular-weight derivatives.
   * @param device_data Device data containing the active task batch.
   * @param first_task Batch's first index in the load balancer task list.
   * @param dE_dw Rank-local contiguous CUDA `dE/dw` values.
   */
  void prepare_weight_derivatives(GauXC::XCDeviceAoSData& device_data,
                                  types::TaskIndex first_task,
                                  const at::Tensor& dE_dw) const;

 private:
  /** @brief Opaque owner of persistent CUDA transport metadata. */
  struct DeviceStorage;

  /** @brief Fixed local task, point, and atom ordering. */
  ModelGridLayout layout_;
  /** @brief Exact-size batches of complete domains owned by this rank. */
  std::vector<ModelDomainBatch> local_batches_;
  /** @brief Model-grid point offset for each load-balancer task. */
  std::vector<types::GridPointOffset> point_offset_by_task_;
  /** @brief Start offset of each batch in persistent device atom indices. */
  std::vector<std::size_t> local_batch_atom_offsets_;
  /** @brief Local raw weights arranged in atom-major model-grid order. */
  std::vector<double> atom_ordered_raw_weights_;
  /** @brief Persistent non-Torch CUDA metadata used by exchange kernels. */
  std::unique_ptr<DeviceStorage> device_storage_;
};

}  // namespace SkalaXC
