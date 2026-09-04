#include "model_grid_exchange.hpp"
#include "exceptions.hpp"

#include <algorithm>

namespace SkalaXC {

ModelGridExchange::ModelGridExchange(const std::vector<GauXC::XCTask>& tasks,
                                     types::AtomCount atom_count,
                                     const GauXC::RuntimeEnvironment& rt,
                                     DomainBatchMode batch_mode)
    : layout_(tasks, atom_count, rt, false),
      local_batches_(layout_.make_local_batches(batch_mode)) {}

FeatureDict ModelGridExchange::prepare_local_features(
    const ModelDomainBatch& batch, const std::vector<GauXC::XCTask>& tasks,
    const std::vector<TaskFeatureData>& task_features,
    const std::vector<std::vector<double>>& raw_weights,
    const GauXC::Molecule& molecule,
    const std::vector<std::string>& feature_keys) const {
  const bool has_density_gradient =
      std::find(feature_keys.begin(), feature_keys.end(),
                feat_map().at(SKALA_FEATURE::DDEN)) != feature_keys.end();
  const bool has_kinetic =
      std::find(feature_keys.begin(), feature_keys.end(),
                feat_map().at(SKALA_FEATURE::TAU)) != feature_keys.end();
  const Eigen::Index point_count =
      static_cast<Eigen::Index>(batch.point_count.raw());

  AlphaBetaMatrix density(point_count, spin_dimension);
  AlphaBetaMatrix kinetic(has_kinetic ? point_count : 0, spin_dimension);
  CartesianMatrix grid_coordinates(point_count, direction_dimension);
  Vector grid_weights(point_count);
  Vector raw_grid_weights(point_count);
  SpinGradient density_gradient(has_density_gradient ? point_count : 0);

  for (const auto& block : batch.task_blocks) {
    const auto task_index = block.task_index.raw();
    const Eigen::Index block_offset =
        static_cast<Eigen::Index>(block.point_offset.raw());
    const Eigen::Index block_point_count =
        static_cast<Eigen::Index>(block.point_count.raw());
    const auto& task = tasks[task_index];
    const auto& features = task_features[task_index];
    if (raw_weights[task_index].size() !=
        static_cast<std::size_t>(block_point_count))
      SKALAXC_EXCEPTION("Invalid raw grid-weight dimensions");

    for (Eigen::Index point = 0; point < block_point_count; ++point)
      for (Eigen::Index direction = 0; direction < direction_dimension;
           ++direction)
        grid_coordinates(block_offset + point, direction) =
            task.points[point][direction];
    grid_weights.segment(block_offset, block_point_count) =
        ConstVectorMap(task.weights.data(), block_point_count);
    raw_grid_weights.segment(block_offset, block_point_count) =
        ConstVectorMap(raw_weights[task_index].data(), block_point_count);
    density.middleRows(block_offset, block_point_count) = features.density;
    if (has_kinetic)
      kinetic.middleRows(block_offset, block_point_count) = features.kinetic;

    if (has_density_gradient) {
      if (features.density_gradient.points() != block_point_count)
        SKALAXC_EXCEPTION("Invalid task density-gradient dimensions");
      for (Eigen::Index point = 0; point < block_point_count; ++point)
        for (Eigen::Index direction = 0; direction < direction_dimension;
             ++direction)
          for (Eigen::Index spin = 0; spin < spin_dimension; ++spin)
            density_gradient(static_cast<Direction>(direction),
                             block_offset + point,
                             static_cast<SpinChannel>(spin)) =
                features.density_gradient(static_cast<Direction>(direction),
                                          point,
                                          static_cast<SpinChannel>(spin));
    }
  }

  const int atom_count = static_cast<int>(batch.atoms.size());
  std::vector<std::int64_t> atom_point_counts(batch.atoms.size(),
                                              batch.grid_size.raw());
  CartesianMatrix atomic_coordinates(atom_count, direction_dimension);
  for (int local_atom = 0; local_atom < atom_count; ++local_atom) {
    const auto atom = batch.atoms[static_cast<std::size_t>(local_atom)].raw();
    atomic_coordinates(local_atom, X) = molecule[atom].x;
    atomic_coordinates(local_atom, Y) = molecule[atom].y;
    atomic_coordinates(local_atom, Z) = molecule[atom].z;
  }
  const std::int64_t max_grid_size = batch.grid_size.raw();

  const auto options =
      torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  FeatureDict feature_dict;
  for (const auto& key : feature_keys) {
    at::Tensor tensor;
    switch (reverse_feat_map().at(key)) {
      case SKALA_FEATURE::DEN:
        tensor = torch::from_blob(density.data(), {spin_dimension, point_count},
                                  {1, spin_dimension}, options)
                     .clone()
                     .requires_grad_(true);
        break;
      case SKALA_FEATURE::DDEN:
        tensor = mpi::spin_gradient_to_torch(density_gradient);
        break;
      case SKALA_FEATURE::TAU:
        tensor = torch::from_blob(kinetic.data(), {spin_dimension, point_count},
                                  {1, spin_dimension}, options)
                     .clone()
                     .requires_grad_(true);
        break;
      case SKALA_FEATURE::POINTS:
        tensor = torch::from_blob(grid_coordinates.data(),
                                  {point_count, direction_dimension}, options)
                     .clone();
        break;
      case SKALA_FEATURE::WEIGHTS:
        tensor = torch::from_blob(grid_weights.data(), {point_count}, options)
                     .clone();
        break;
      case SKALA_FEATURE::COORDS:
        tensor = torch::from_blob(atomic_coordinates.data(),
                                  {atom_count, direction_dimension}, options)
                     .clone();
        break;
      case SKALA_FEATURE::ATOMIC_GRID_WEIGHTS:
        tensor =
            torch::from_blob(raw_grid_weights.data(), {point_count}, options)
                .clone();
        break;
      case SKALA_FEATURE::ATOMIC_GRID_SIZES: {
        const auto sizes_options =
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        tensor = torch::from_blob(atom_point_counts.data(), {atom_count},
                                  sizes_options)
                     .clone();
        break;
      }
      case SKALA_FEATURE::ATOMIC_GRID_SIZE_BOUND_SHAPE: {
        const auto sizes_options =
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        tensor = torch::zeros({max_grid_size, 0}, sizes_options);
        break;
      }
      default:
        SKALAXC_EXCEPTION("Feature Key Not Implemented: " + key);
    }
    if (tensor.isnan().any().item<bool>())
      SKALAXC_EXCEPTION("NaN detected in feature tensor: " + key);
    feature_dict.insert(key, tensor);
  }
  return feature_dict;
}

void ModelGridExchange::distribute_local_potentials(
    const ModelDomainBatch& batch, bool has_density_gradient, bool has_kinetic,
    const FeatureDict& feature_dict,
    std::vector<TaskPotentialData>& task_potentials) const {
  const auto density_tensor = validated_model_gradient(
      feature_dict.at(feat_map().at(SKALA_FEATURE::DEN)),
      "host density potential");
  validate_model_tensor_finite(density_tensor, "host density potential");
  const Eigen::Map<const AlphaBetaByPointMatrix> density_channels(
      density_tensor.data_ptr<double>(), spin_dimension,
      static_cast<Eigen::Index>(batch.point_count.raw()));
  const AlphaBetaMatrix density_potential = density_channels.transpose();

  AlphaBetaMatrix kinetic_potential;
  if (has_kinetic) {
    const auto kinetic_tensor = validated_model_gradient(
        feature_dict.at(feat_map().at(SKALA_FEATURE::TAU)),
        "host kinetic potential");
    validate_model_tensor_finite(kinetic_tensor, "host kinetic potential");
    const Eigen::Map<const AlphaBetaByPointMatrix> kinetic_channels(
        kinetic_tensor.data_ptr<double>(), spin_dimension,
        static_cast<Eigen::Index>(batch.point_count.raw()));
    kinetic_potential = kinetic_channels.transpose();
  }

  SpinGradient density_gradient;
  if (has_density_gradient) {
    const auto gradient_tensor = validated_model_gradient(
        feature_dict.at(feat_map().at(SKALA_FEATURE::DDEN)),
        "host density-gradient potential");
    validate_model_tensor_finite(gradient_tensor,
                                 "host density-gradient potential");
    density_gradient =
        mpi::torch_to_spin_gradient(gradient_tensor, batch.point_count.raw());
  }

  for (const auto& block : batch.task_blocks) {
    const auto task_index = block.task_index.raw();
    const Eigen::Index block_offset =
        static_cast<Eigen::Index>(block.point_offset.raw());
    const Eigen::Index block_point_count =
        static_cast<Eigen::Index>(block.point_count.raw());
    auto& potentials = task_potentials[task_index];
    potentials.density =
        density_potential.middleRows(block_offset, block_point_count);
    if (has_density_gradient) {
      potentials.density_gradient.resize(block_point_count);
      for (Eigen::Index point = 0; point < block_point_count; ++point)
        for (Eigen::Index direction = 0; direction < direction_dimension;
             ++direction)
          for (Eigen::Index spin = 0; spin < spin_dimension; ++spin)
            potentials.density_gradient(static_cast<Direction>(direction),
                                        point, static_cast<SpinChannel>(spin)) =
                density_gradient(static_cast<Direction>(direction),
                                 block_offset + point,
                                 static_cast<SpinChannel>(spin));
    }
    if (has_kinetic)
      potentials.kinetic =
          kinetic_potential.middleRows(block_offset, block_point_count);
  }
}

void ModelGridExchange::distribute_local_dE_dw(
    const ModelDomainBatch& batch, std::vector<double> atom_ordered_values,
    std::vector<TaskPotentialData>& task_potentials) const {
  if (atom_ordered_values.size() !=
      static_cast<std::size_t>(batch.point_count.raw()))
    SKALAXC_EXCEPTION("Mismatch in number of model dE/dw values");
  for (const auto& block : batch.task_blocks)
    task_potentials[block.task_index.raw()].dE_dw =
        ConstVectorMap(atom_ordered_values.data() + block.point_offset.raw(),
                       static_cast<Eigen::Index>(block.point_count.raw()));
}

void ModelGridExchange::accumulate_local_point_gradient(
    const ModelDomainBatch& batch, const at::Tensor& point_gradient,
    RowMajorMatrixMap atom_gradient) const {
  if (!point_gradient.defined()) return;
  if (atom_gradient.rows() !=
          static_cast<Eigen::Index>(layout_.local_atom_point_counts().size()) ||
      atom_gradient.cols() != direction_dimension)
    SKALAXC_EXCEPTION("Invalid local model point-gradient dimensions");

  validate_model_tensor(point_gradient, "host model point gradient",
                        c10::Device(c10::DeviceType::CPU), torch::kFloat64,
                        {batch.point_count.raw(), direction_dimension});
  const auto contiguous_gradient = point_gradient.contiguous();
  validate_model_tensor_finite(contiguous_gradient,
                               "host model point gradient");
  const Eigen::Map<const CartesianMatrix> point_gradients(
      contiguous_gradient.data_ptr<double>(),
      static_cast<Eigen::Index>(batch.point_count.raw()), direction_dimension);
  Eigen::Index point_offset = 0;
  for (const auto atom : batch.atoms) {
    atom_gradient.row(static_cast<Eigen::Index>(atom.raw())) +=
        point_gradients
            .middleRows(point_offset,
                        static_cast<Eigen::Index>(batch.grid_size.raw()))
            .colwise()
            .sum();
    point_offset += static_cast<Eigen::Index>(batch.grid_size.raw());
  }
}

void ModelGridExchange::accumulate_local_coordinate_gradient(
    const ModelDomainBatch& batch, const at::Tensor& coordinate_gradient,
    RowMajorMatrixMap atom_gradient) const {
  if (!coordinate_gradient.defined()) return;
  if (atom_gradient.rows() !=
          static_cast<Eigen::Index>(layout_.local_atom_point_counts().size()) ||
      atom_gradient.cols() != direction_dimension)
    SKALAXC_EXCEPTION("Invalid local model coordinate-gradient dimensions");

  validate_model_tensor(
      coordinate_gradient, "host model coordinate gradient",
      c10::Device(c10::DeviceType::CPU), torch::kFloat64,
      {static_cast<std::int64_t>(batch.atoms.size()), direction_dimension});
  const auto contiguous_gradient = coordinate_gradient.contiguous();
  validate_model_tensor_finite(contiguous_gradient,
                               "host model coordinate gradient");
  const Eigen::Map<const CartesianMatrix> local_gradient(
      contiguous_gradient.data_ptr<double>(), batch.atoms.size(),
      direction_dimension);
  for (std::size_t local_atom = 0; local_atom < batch.atoms.size();
       ++local_atom)
    atom_gradient.row(
        static_cast<Eigen::Index>(batch.atoms[local_atom].raw())) +=
        local_gradient.row(static_cast<Eigen::Index>(local_atom));
}

}  // namespace SkalaXC
