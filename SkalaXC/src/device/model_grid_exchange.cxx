/**
 * @file
 * @brief Implementation of rank-local CUDA task and model tensor exchange.
 */
#include "device/model_grid_exchange.hpp"

#include "device/common/model_grid.hpp"
#include "device/xc_device_aos_data.hpp"
#include "exceptions.hpp"
#include "host/skala_util.hpp"

#include <torch/torch.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace SkalaXC {

namespace {

/**
 * @brief Test whether a dictionary contains a known model feature.
 * @param features Feature dictionary to inspect.
 * @param feature Feature identifier to find.
 * @return `true` when the mapped feature key is present.
 */
bool has_feature(const FeatureDict& features, SKALA_FEATURE feature) {
  return features.find(feat_map().at(feature)) != features.end();
}

/**
 * @brief Access mutable storage for an optional feature tensor.
 * @param features Feature dictionary containing device tensors.
 * @param feature Feature identifier to access.
 * @return Tensor data pointer, or `nullptr` when absent.
 */
double* feature_data(const FeatureDict& features, SKALA_FEATURE feature,
                     const c10::Device& device, std::int64_t point_count) {
  if (!has_feature(features, feature)) return nullptr;
  const auto& tensor = features.at(feat_map().at(feature));
  std::vector<std::int64_t> sizes;
  switch (feature) {
    case SKALA_FEATURE::DEN:
    case SKALA_FEATURE::TAU:
      sizes = {spin_dimension, point_count};
      break;
    case SKALA_FEATURE::DDEN:
      sizes = {spin_dimension, direction_dimension, point_count};
      break;
    case SKALA_FEATURE::POINTS:
      sizes = {point_count, direction_dimension};
      break;
    case SKALA_FEATURE::WEIGHTS:
      sizes = {point_count};
      break;
    default:
      SKALAXC_EXCEPTION("Unsupported CUDA model feature storage");
  }
  validate_model_tensor(tensor,
                        "CUDA model feature '" + feat_map().at(feature) + "'",
                        device, torch::kFloat64, sizes, true);
  return tensor.data_ptr<double>();
}

/**
 * @brief Validate and access a model potential tensor.
 * @param features Potential dictionary containing device tensors.
 * @param feature Feature derivative to access.
 * @return Contiguous tensor data, or `nullptr` when absent.
 */
const double* potential_data(const FeatureDict& features, SKALA_FEATURE feature,
                             const c10::Device& device,
                             std::int64_t point_count) {
  if (!has_feature(features, feature)) return nullptr;
  const auto& potential = features.at(feat_map().at(feature));
  const auto sizes =
      feature == SKALA_FEATURE::DDEN
          ? std::vector<std::int64_t>{spin_dimension, direction_dimension,
                                      point_count}
          : std::vector<std::int64_t>{spin_dimension, point_count};
  validate_model_tensor(potential,
                        "CUDA model potential '" + feat_map().at(feature) + "'",
                        device, torch::kFloat64, sizes, true);
  return potential.data_ptr<double>();
}

/**
 * @brief Return the scalar width of one point record.
 * @param feature Feature whose point layout is queried.
 * @return Record width, or zero for a non-point-indexed feature.
 */
int point_components(SKALA_FEATURE feature) {
  switch (feature) {
    case SKALA_FEATURE::DEN:
    case SKALA_FEATURE::TAU:
      return spin_dimension;
    case SKALA_FEATURE::DDEN:
      return spin_dimension * direction_dimension;
    case SKALA_FEATURE::POINTS:
      return direction_dimension;
    case SKALA_FEATURE::WEIGHTS:
    case SKALA_FEATURE::ATOMIC_GRID_WEIGHTS:
      return 1;
    default:
      return 0;
  }
}

/** @brief Return the point-indexed dimension for a model-facing feature. */
int point_dimension(SKALA_FEATURE feature) {
  switch (feature) {
    case SKALA_FEATURE::DEN:
    case SKALA_FEATURE::TAU:
      return 1;
    case SKALA_FEATURE::DDEN:
      return 2;
    case SKALA_FEATURE::POINTS:
    case SKALA_FEATURE::WEIGHTS:
    case SKALA_FEATURE::ATOMIC_GRID_WEIGHTS:
      return 0;
    default:
      SKALAXC_EXCEPTION("Feature is not point-indexed");
  }
}

/** @brief Copy a point interval between tensors in model-facing layout. */
void copy_point_interval(const at::Tensor& source, std::int64_t source_offset,
                         const at::Tensor& destination,
                         std::int64_t destination_offset,
                         std::int64_t point_count, SKALA_FEATURE feature) {
  const int dimension = point_dimension(feature);
  destination.narrow(dimension, destination_offset, point_count)
      .copy_(source.narrow(dimension, source_offset, point_count));
}

/** @brief Persistent CUDA allocation with value-count-based access. */
template <typename T>
class CudaBuffer {
 public:
  /**
   * @brief Allocate persistent storage on the current CUDA device.
   * @param size Number of values to allocate.
   * @param host_values Optional host values copied into the allocation.
   * @param device_id Required current CUDA device ordinal.
   */
  CudaBuffer(std::size_t size, const T* host_values, int device_id)
      : device_id_(device_id) {
    int current_device = -1;
    auto status = cudaGetDevice(&current_device);
    if (status != cudaSuccess)
      SKALAXC_EXCEPTION(std::string("Failed to query the CUDA device: ") +
                        cudaGetErrorString(status));
    if (current_device != device_id_)
      SKALAXC_EXCEPTION(
          "CUDA exchange metadata device does not match the active device");
    if (size == 0) return;

    status = cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T));
    if (status != cudaSuccess)
      SKALAXC_EXCEPTION(
          std::string("Failed to allocate CUDA exchange storage: ") +
          cudaGetErrorString(status));
    if (!host_values) return;

    status = cudaMemcpy(data_, host_values, size * sizeof(T),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      cudaFree(data_);
      data_ = nullptr;
      SKALAXC_EXCEPTION(
          std::string("Failed to initialize CUDA exchange storage: ") +
          cudaGetErrorString(status));
    }
  }

  /** @brief Release the allocation on its owning CUDA device. */
  ~CudaBuffer() noexcept {
    if (!data_) return;
    int previous_device = device_id_;
    const bool restore_device =
        cudaGetDevice(&previous_device) == cudaSuccess &&
        previous_device != device_id_;
    if (restore_device) cudaSetDevice(device_id_);
    cudaFree(data_);
    if (restore_device) cudaSetDevice(previous_device);
  }

  /** @brief CUDA allocations cannot be copied. */
  CudaBuffer(const CudaBuffer&) = delete;
  /** @brief CUDA allocations cannot be copy-assigned. */
  CudaBuffer& operator=(const CudaBuffer&) = delete;

  /** @return Mutable device pointer, or null for an empty allocation. */
  T* data() noexcept { return data_; }
  /** @return Read-only device pointer, or null for an empty allocation. */
  const T* data() const noexcept { return data_; }
  /** @return CUDA device ordinal that owns the allocation. */
  int device_id() const noexcept { return device_id_; }

 private:
  T* data_ = nullptr;
  int device_id_ = 0;
};

}  // namespace

/** @brief Persistent non-Torch CUDA metadata used by model-grid exchange. */
struct DeviceModelGridExchange::DeviceStorage {
  DeviceStorage(const std::vector<std::int64_t>& point_offsets,
                const std::vector<std::int64_t>& batch_atom_indices,
                types::DeviceId device_id)
      : point_offsets(point_offsets.size(), point_offsets.data(),
                      device_id.raw()),
        batch_atom_indices(batch_atom_indices.size(), batch_atom_indices.data(),
                           device_id.raw()) {}

  CudaBuffer<std::int64_t> point_offsets;
  CudaBuffer<std::int64_t> batch_atom_indices;
};

DeviceModelGridExchange::DeviceModelGridExchange(
    const std::vector<GauXC::XCTask>& tasks, types::AtomCount atom_count,
    const GauXC::RuntimeEnvironment& rt,
    const std::vector<std::vector<double>>& raw_weights,
    types::DeviceId device_id, DomainBatchMode batch_mode)
    : layout_(tasks, atom_count, rt, false),
      local_batches_(layout_.make_local_batches(batch_mode)),
      point_offset_by_task_(tasks.size(), types::GridPointOffset{-1}) {
  if (raw_weights.size() != tasks.size())
    SKALAXC_EXCEPTION("Raw grid weights do not match CUDA tasks");

  std::vector<std::int64_t> point_offsets(tasks.size(), -1);
  atom_ordered_raw_weights_.resize(
      static_cast<std::size_t>(layout_.local_point_count().raw()));
  for (const auto& block : layout_.task_blocks()) {
    const auto task_index = block.task_index.raw();
    const auto point_offset = block.point_offset.raw();
    const auto point_count = block.point_count.raw();
    point_offset_by_task_[task_index] = block.point_offset;
    point_offsets[task_index] = point_offset;
    if (raw_weights[task_index].size() != static_cast<std::size_t>(point_count))
      SKALAXC_EXCEPTION("Invalid CUDA raw grid-weight dimensions");
    std::copy(raw_weights[task_index].begin(), raw_weights[task_index].end(),
              atom_ordered_raw_weights_.begin() + point_offset);
  }

  std::vector<std::int64_t> local_batch_atom_indices;
  local_batch_atom_offsets_.reserve(local_batches_.size() + 1);
  for (const auto& batch : local_batches_) {
    local_batch_atom_offsets_.push_back(local_batch_atom_indices.size());
    for (const auto atom : batch.atoms)
      local_batch_atom_indices.push_back(static_cast<std::int64_t>(atom.raw()));
  }
  local_batch_atom_offsets_.push_back(local_batch_atom_indices.size());

  device_storage_ = std::make_unique<DeviceStorage>(
      point_offsets, local_batch_atom_indices, device_id);
}

DeviceModelGridExchange::~DeviceModelGridExchange() noexcept = default;

FeatureDict DeviceModelGridExchange::prepare_features(
    const std::vector<std::string>& feature_keys,
    const c10::Device& device) const {
  const std::int64_t point_count = layout_.local_point_count().raw();
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat64).device(device);
  FeatureDict features;

  for (const auto& key : feature_keys) {
    at::Tensor tensor;
    switch (reverse_feat_map().at(key)) {
      case SKALA_FEATURE::DEN:
        tensor = torch::full({2, point_count},
                             std::numeric_limits<double>::quiet_NaN(), options)
                     .contiguous();
        break;
      case SKALA_FEATURE::DDEN:
        tensor = torch::full({2, 3, point_count},
                             std::numeric_limits<double>::quiet_NaN(), options)
                     .contiguous();
        break;
      case SKALA_FEATURE::TAU:
        tensor = torch::full({2, point_count},
                             std::numeric_limits<double>::quiet_NaN(), options)
                     .contiguous();
        break;
      case SKALA_FEATURE::POINTS:
        tensor = torch::full({point_count, 3},
                             std::numeric_limits<double>::quiet_NaN(), options);
        break;
      case SKALA_FEATURE::WEIGHTS:
        tensor = torch::full({point_count},
                             std::numeric_limits<double>::quiet_NaN(), options);
        break;
      case SKALA_FEATURE::COORDS:
      case SKALA_FEATURE::ATOMIC_GRID_SIZES:
      case SKALA_FEATURE::ATOMIC_GRID_SIZE_BOUND_SHAPE:
        continue;
      case SKALA_FEATURE::ATOMIC_GRID_WEIGHTS:
        tensor =
            torch::from_blob(
                const_cast<double*>(atom_ordered_raw_weights_.data()),
                {point_count}, torch::TensorOptions().dtype(torch::kFloat64))
                .clone()
                .to(device);
        break;
      default:
        SKALAXC_EXCEPTION("Feature Key Not Implemented: " + key);
    }
    features.insert(key, tensor);
  }
  return features;
}

FeatureDict DeviceModelGridExchange::prepare_local_batch_features(
    const ModelDomainBatch& batch, const FeatureDict& local_features,
    const GauXC::Molecule& molecule,
    const std::vector<std::string>& feature_keys, const c10::Device& device,
    bool geometry_gradients) const {
  FeatureDict batch_features;
  for (const auto& key : feature_keys) {
    const auto feature = reverse_feat_map().at(key);
    if (point_components(feature) != 0) {
      const auto local = local_features.find(key);
      if (local == local_features.end())
        SKALAXC_EXCEPTION("Missing local CUDA feature: " + key);
      std::vector<at::Tensor> task_tensors;
      task_tensors.reserve(batch.task_blocks.size());
      const int dimension = point_dimension(feature);
      for (const auto& block : batch.task_blocks) {
        const auto offset = point_offset_by_task_[block.task_index.raw()].raw();
        task_tensors.push_back(
            local->value().narrow(dimension, offset, block.point_count.raw()));
      }
      auto tensor = task_tensors.size() == 1
                        ? task_tensors.front().clone()
                        : torch::cat(task_tensors, dimension);
      const bool requires_gradient =
          feature == SKALA_FEATURE::DEN || feature == SKALA_FEATURE::DDEN ||
          feature == SKALA_FEATURE::TAU ||
          (geometry_gradients && (feature == SKALA_FEATURE::POINTS ||
                                  feature == SKALA_FEATURE::WEIGHTS));
      if (requires_gradient) tensor.requires_grad_(true);
      batch_features.insert(key, tensor);
      continue;
    }

    at::Tensor tensor;
    switch (feature) {
      case SKALA_FEATURE::COORDS: {
        std::vector<double> coordinates(direction_dimension *
                                        batch.atoms.size());
        for (std::size_t local_atom = 0; local_atom < batch.atoms.size();
             ++local_atom) {
          const auto atom = batch.atoms[local_atom].raw();
          coordinates[direction_dimension * local_atom] = molecule[atom].x;
          coordinates[direction_dimension * local_atom + 1] = molecule[atom].y;
          coordinates[direction_dimension * local_atom + 2] = molecule[atom].z;
        }
        tensor =
            torch::from_blob(coordinates.data(),
                             {static_cast<std::int64_t>(batch.atoms.size()),
                              direction_dimension},
                             torch::TensorOptions().dtype(torch::kFloat64))
                .clone()
                .to(device);
        if (geometry_gradients) tensor.requires_grad_(true);
        break;
      }
      case SKALA_FEATURE::ATOMIC_GRID_SIZES: {
        std::vector<std::int64_t> grid_sizes(batch.atoms.size(),
                                             batch.grid_size.raw());
        tensor =
            torch::from_blob(grid_sizes.data(),
                             {static_cast<std::int64_t>(grid_sizes.size())},
                             torch::TensorOptions().dtype(torch::kInt64))
                .clone()
                .to(device);
        break;
      }
      case SKALA_FEATURE::ATOMIC_GRID_SIZE_BOUND_SHAPE:
        tensor = torch::zeros(
            {batch.grid_size.raw(), 0},
            torch::TensorOptions().dtype(torch::kInt64).device(device));
        break;
      default:
        SKALAXC_EXCEPTION("Feature Key Not Implemented: " + key);
    }
    batch_features.insert(key, tensor);
  }
  return batch_features;
}

FeatureDict DeviceModelGridExchange::prepare_local_potentials(
    bool has_density_gradient, bool has_kinetic,
    const c10::Device& device) const {
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat64).device(device);
  const auto point_count = layout_.local_point_count().raw();
  FeatureDict potentials;
  potentials.insert(feat_map().at(SKALA_FEATURE::DEN),
                    torch::zeros({spin_dimension, point_count}, options));
  if (has_density_gradient)
    potentials.insert(
        feat_map().at(SKALA_FEATURE::DDEN),
        torch::zeros({spin_dimension, direction_dimension, point_count},
                     options));
  if (has_kinetic)
    potentials.insert(feat_map().at(SKALA_FEATURE::TAU),
                      torch::zeros({spin_dimension, point_count}, options));
  return potentials;
}

void DeviceModelGridExchange::store_local_batch_potentials(
    const ModelDomainBatch& batch, bool has_density_gradient, bool has_kinetic,
    const FeatureDict& batch_features,
    const FeatureDict& local_potentials) const {
  const std::vector<SKALA_FEATURE> potential_features{
      SKALA_FEATURE::DEN, SKALA_FEATURE::DDEN, SKALA_FEATURE::TAU};
  at::NoGradGuard no_grad;
  for (const auto feature : potential_features) {
    if ((feature == SKALA_FEATURE::DDEN && !has_density_gradient) ||
        (feature == SKALA_FEATURE::TAU && !has_kinetic))
      continue;
    const auto& key = feat_map().at(feature);
    const auto gradient = validated_model_gradient(
        batch_features.at(key), "CUDA model potential '" + key + "'");
    const auto& destination = local_potentials.at(key);
    validate_model_tensor(destination, "CUDA full-grid potential '" + key + "'",
                          gradient.device(), gradient.scalar_type(),
                          destination.sizes(), true);
    for (const auto& block : batch.task_blocks)
      copy_point_interval(gradient, block.point_offset.raw(), destination,
                          point_offset_by_task_[block.task_index.raw()].raw(),
                          block.point_count.raw(), feature);
  }
}

at::Tensor DeviceModelGridExchange::prepare_local_dE_dw(
    const c10::Device& device) const {
  return torch::zeros(
      {layout_.local_point_count().raw()},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));
}

void DeviceModelGridExchange::store_local_batch_dE_dw(
    const ModelDomainBatch& batch, const at::Tensor& batch_dE_dw,
    const at::Tensor& local_dE_dw) const {
  const auto device = c10::Device(c10::DeviceType::CUDA,
                                  device_storage_->point_offsets.device_id());
  validate_model_tensor(batch_dE_dw, "local CUDA model dE/dw", device,
                        torch::kFloat64, {batch.point_count.raw()});
  validate_model_tensor(local_dE_dw, "full-grid CUDA model dE/dw", device,
                        torch::kFloat64, {layout_.local_point_count().raw()},
                        true);

  const auto contiguous_dE_dw = batch_dE_dw.contiguous();
  at::NoGradGuard no_grad;
  for (const auto& block : batch.task_blocks)
    copy_point_interval(contiguous_dE_dw, block.point_offset.raw(), local_dE_dw,
                        point_offset_by_task_[block.task_index.raw()].raw(),
                        block.point_count.raw(), SKALA_FEATURE::WEIGHTS);
}

void DeviceModelGridExchange::accumulate_local_geometry_gradients(
    std::size_t batch_index, const FeatureDict& batch_features,
    GauXC::XCDeviceAoSData& device_data) const {
  const auto& batch = local_batches_.at(batch_index);
  const double* point_gradient_data = nullptr;
  const double* coordinate_gradient_data = nullptr;
  at::Tensor point_gradient;
  at::Tensor coordinate_gradient;

  const auto points = batch_features.find(feat_map().at(SKALA_FEATURE::POINTS));
  if (points != batch_features.end() && points->value().grad().defined()) {
    point_gradient = validated_model_gradient(
        points->value(), "local CUDA model point gradient");
    validate_model_tensor(
        point_gradient, "local CUDA model point gradient",
        c10::Device(c10::DeviceType::CUDA,
                    device_storage_->point_offsets.device_id()),
        torch::kFloat64, {batch.point_count.raw(), direction_dimension}, true);
    point_gradient_data = point_gradient.data_ptr<double>();
  }

  const auto coordinates =
      batch_features.find(feat_map().at(SKALA_FEATURE::COORDS));
  if (coordinates != batch_features.end() &&
      coordinates->value().grad().defined()) {
    coordinate_gradient = validated_model_gradient(
        coordinates->value(), "local CUDA model coordinate gradient");
    validate_model_tensor(
        coordinate_gradient, "local CUDA model coordinate gradient",
        c10::Device(c10::DeviceType::CUDA,
                    device_storage_->point_offsets.device_id()),
        torch::kFloat64,
        {static_cast<std::int64_t>(batch.atoms.size()), direction_dimension},
        true);
    coordinate_gradient_data = coordinate_gradient.data_ptr<double>();
  }

  if (!device_data.static_stack.exc_grad_device)
    SKALAXC_EXCEPTION("Missing GauXC CUDA XC-gradient storage");
  const auto atom_offset = local_batch_atom_offsets_.at(batch_index);
  accumulate_model_geometry_gradient(
      batch.atoms.size(), batch.grid_size.raw(),
      device_storage_->batch_atom_indices.data() + atom_offset,
      point_gradient_data, coordinate_gradient_data,
      device_data.static_stack.exc_grad_device, device_data.queue());
}

void DeviceModelGridExchange::pack_post_uvars_features(
    GauXC::XCDeviceAoSData& device_data, types::TaskIndex first_task,
    const FeatureDict& feature_dict) const {
  const std::size_t task_count = device_data.host_device_tasks.size();
  const auto first_task_value = first_task.raw();
  if (first_task_value + task_count > point_offset_by_task_.size())
    SKALAXC_EXCEPTION("CUDA task batch exceeds model-grid layout");
  const auto device = c10::Device(c10::DeviceType::CUDA,
                                  device_storage_->point_offsets.device_id());
  std::int32_t max_points = 0;
  for (std::size_t task = 0; task < task_count; ++task) {
    if (device_data.host_device_tasks[task].npts >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
      SKALAXC_EXCEPTION("CUDA task point count exceeds supported size");
    max_points = std::max(
        max_points,
        static_cast<std::int32_t>(device_data.host_device_tasks[task].npts));
  }

  pack_post_uvars_model_grid_features(
      task_count, max_points, device_data.aos_stack.device_tasks,
      device_storage_->point_offsets.data() + first_task_value,
      layout_.local_point_count().raw(),
      feature_data(feature_dict, SKALA_FEATURE::DEN, device,
                   layout_.local_point_count().raw()),
      feature_data(feature_dict, SKALA_FEATURE::DDEN, device,
                   layout_.local_point_count().raw()),
      feature_data(feature_dict, SKALA_FEATURE::TAU, device,
                   layout_.local_point_count().raw()),
      feature_data(feature_dict, SKALA_FEATURE::POINTS, device,
                   layout_.local_point_count().raw()),
      feature_data(feature_dict, SKALA_FEATURE::WEIGHTS, device,
                   layout_.local_point_count().raw()),
      device_data.queue());
}

void DeviceModelGridExchange::unpack_potentials(
    GauXC::XCDeviceAoSData& device_data, types::TaskIndex first_task,
    const FeatureDict& feature_dict) const {
  const std::size_t task_count = device_data.host_device_tasks.size();
  const auto first_task_value = first_task.raw();
  if (first_task_value + task_count > point_offset_by_task_.size())
    SKALAXC_EXCEPTION("CUDA task batch exceeds model-grid layout");
  const auto device = c10::Device(c10::DeviceType::CUDA,
                                  device_storage_->point_offsets.device_id());
  std::int32_t max_points = 0;
  for (std::size_t task = 0; task < task_count; ++task) {
    max_points = std::max(
        max_points,
        static_cast<std::int32_t>(device_data.host_device_tasks[task].npts));
  }

  unpack_model_grid_potentials(
      task_count, max_points, device_data.aos_stack.device_tasks,
      device_storage_->point_offsets.data() + first_task_value,
      layout_.local_point_count().raw(),
      potential_data(feature_dict, SKALA_FEATURE::DEN, device,
                     layout_.local_point_count().raw()),
      potential_data(feature_dict, SKALA_FEATURE::DDEN, device,
                     layout_.local_point_count().raw()),
      potential_data(feature_dict, SKALA_FEATURE::TAU, device,
                     layout_.local_point_count().raw()),
      device_data.queue());
}

void DeviceModelGridExchange::prepare_weight_derivatives(
    GauXC::XCDeviceAoSData& device_data, types::TaskIndex first_task,
    const at::Tensor& dE_dw) const {
  const std::size_t task_count = device_data.host_device_tasks.size();
  const auto first_task_value = first_task.raw();
  if (first_task_value + task_count > point_offset_by_task_.size())
    SKALAXC_EXCEPTION("CUDA task batch exceeds model-grid layout");
  validate_model_tensor(dE_dw, "CUDA model dE/dw tensor",
                        c10::Device(c10::DeviceType::CUDA,
                                    device_storage_->point_offsets.device_id()),
                        torch::kFloat64, {layout_.local_point_count().raw()},
                        true);

  std::int32_t max_points = 0;
  for (std::size_t task = 0; task < task_count; ++task) {
    max_points = std::max(
        max_points,
        static_cast<std::int32_t>(device_data.host_device_tasks[task].npts));
  }
  prepare_model_grid_weight_derivatives(
      task_count, max_points, device_data.aos_stack.device_tasks,
      device_storage_->point_offsets.data() + first_task_value,
      layout_.local_point_count().raw(), dE_dw.data_ptr<double>(),
      device_data.queue());
}

}  // namespace SkalaXC
