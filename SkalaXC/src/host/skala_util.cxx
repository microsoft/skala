#include "skala_util.hpp"
#include "exceptions.hpp"
#include "mpi_wrapper.hpp"
#include "skala_model.hpp"
#include <gauxc/gauxc_config.hpp>
#ifdef GAUXC_HAS_CUDA
#include <cuda_runtime.h>
#endif
#include <gauxc/util/mpi.hpp>
#include <iostream>
namespace SkalaXC {

namespace {

template <typename Matrix>
void permute_rows(
    Matrix& values,
    const std::vector<types::PermutationIndex>& destination_for_source) {
  if (values.size() == 0) return;
  Matrix permuted(values.rows(), values.cols());
  for (Eigen::Index source = 0; source < values.rows(); ++source) {
    const auto destination = destination_for_source[source].raw();
    if (destination < 0 || destination >= values.rows())
      SKALAXC_EXCEPTION("Invalid point-record permutation");
    permuted.row(destination) = values.row(source);
  }
  values = std::move(permuted);
}

void permute_values(
    Vector& values,
    const std::vector<types::PermutationIndex>& destination_for_source) {
  if (values.size() == 0) return;
  Vector permuted(values.size());
  for (Eigen::Index source = 0; source < values.size(); ++source) {
    const auto destination = destination_for_source[source].raw();
    if (destination < 0 || destination >= values.size())
      SKALAXC_EXCEPTION("Invalid point permutation");
    permuted(destination) = values(source);
  }
  values = std::move(permuted);
}

}  // namespace

bool valueExists(const std::string& value) {
  for (const auto& pair : feat_map()) {
    if (pair.second == value) {
      return true;
    }
  }
  return false;
}

at::Tensor get_exc(const torch::jit::Method& exc_func, FeatureDict features) {
  IValueList args;
  IValueMap kwargs;
  kwargs["mol"] = features;
  auto output = exc_func(args, kwargs);
  if (!output.isTensor())
    SKALAXC_EXCEPTION("Integrated model energy must be a tensor");
  return output.toTensor();
}

void validate_model_tensor(const at::Tensor& tensor,
                           std::string_view description,
                           const c10::Device& expected_device,
                           c10::ScalarType expected_type,
                           at::IntArrayRef expected_sizes,
                           bool require_contiguous, bool require_gradient) {
  const std::string label(description);
  if (!tensor.defined()) SKALAXC_EXCEPTION("Undefined " + label);
  if (tensor.device() != expected_device)
    SKALAXC_EXCEPTION(label + " is on the wrong device");
  if (tensor.scalar_type() != expected_type)
    SKALAXC_EXCEPTION(label + " has the wrong dtype");
  if (!tensor.sizes().equals(expected_sizes))
    SKALAXC_EXCEPTION(label + " has invalid dimensions");
  if (require_contiguous && !tensor.is_contiguous())
    SKALAXC_EXCEPTION(label + " must be contiguous");
  if (require_gradient && !tensor.requires_grad())
    SKALAXC_EXCEPTION(label + " is not connected to autograd");
}

at::Tensor validated_model_gradient(const at::Tensor& feature,
                                    std::string_view description) {
  if (!feature.defined())
    SKALAXC_EXCEPTION("Undefined model feature for " +
                      std::string(description));
  const auto& gradient = feature.grad();
  validate_model_tensor(gradient, description, feature.device(),
                        feature.scalar_type(), feature.sizes());
  return gradient.contiguous();
}

at::Tensor model_tensor_finite_check(const at::Tensor& tensor) {
  if (!tensor.defined()) SKALAXC_EXCEPTION("Cannot check undefined tensor");
  return tensor.isfinite().all();
}

void validate_model_tensor_finite(const at::Tensor& tensor,
                                  std::string_view description) {
  if (!model_tensor_finite_check(tensor).item<bool>())
    SKALAXC_EXCEPTION("Non-finite " + std::string(description));
}

at::Tensor evaluate_model_energy(const SkalaModel& model,
                                 const FeatureDict& features,
                                 const c10::Device& expected_device) {
  auto energy = get_exc(model.energy_function(), features);
  validate_model_tensor(energy, "integrated model energy", expected_device,
                        torch::kFloat64, {}, false, true);
  return energy;
}

std::pair<std::vector<types::PermutationIndex>,
          std::vector<types::PermutationIndex>>
    build_atom_reorder_perm(
        const std::vector<types::GridPointCount>& all_rank_atom_sizes,
        const mpi::CollectiveLayout& point_layout, types::AtomCount atom_count,
        types::CommunicatorSize communicator_size) {
  const auto& displacements = point_layout.displacements();
  const auto total_points = types::GridPointCount{point_layout.extent()};
  const auto atom_count_value = static_cast<std::size_t>(atom_count.raw());
  const auto communicator_size_value =
      static_cast<std::size_t>(communicator_size.raw());
  if (all_rank_atom_sizes.size() != atom_count_value * communicator_size_value)
    SKALAXC_EXCEPTION("Invalid per-rank atom point counts");

  std::vector<types::PermutationIndex> perm(
      static_cast<std::size_t>(total_points.raw()));
  std::vector<types::PermutationIndex> inv_perm(
      static_cast<std::size_t>(total_points.raw()));
  const auto atom_size = [&](std::size_t rank, std::size_t atom) {
    return all_rank_atom_sizes[rank * atom_count_value + atom].raw();
  };

  // Precompute per-rank per-atom offsets within each rank's chunk
  // src_off[r][a] = rank displacement + sizes of earlier atoms on that rank.
  std::vector<std::vector<std::int64_t>> source_offsets(
      communicator_size_value, std::vector<std::int64_t>(atom_count_value));
  for (std::size_t rank = 0; rank < communicator_size_value; ++rank) {
    std::int64_t offset = displacements[rank];
    for (std::size_t atom = 0; atom < atom_count_value; ++atom) {
      source_offsets[rank][atom] = offset;
      offset += atom_size(rank, atom);
    }
  }

  // Precompute global atom offsets (destination start for each atom)
  std::vector<std::int64_t> global_atom_offsets(atom_count_value);
  {
    std::int64_t offset = 0;
    for (std::size_t atom = 0; atom < atom_count_value; ++atom) {
      global_atom_offsets[atom] = offset;
      for (std::size_t rank = 0; rank < communicator_size_value; ++rank)
        offset += atom_size(rank, atom);
    }
  }

  // Build perm: for each atom, concatenate contributions from all ranks in rank
  // order dst_cursor tracks the next write position for each atom
  std::vector<std::int64_t> destination_cursors = global_atom_offsets;
  for (std::size_t atom = 0; atom < atom_count_value; ++atom) {
    for (std::size_t rank = 0; rank < communicator_size_value; ++rank) {
      const auto count = atom_size(rank, atom);
      const auto source = source_offsets[rank][atom];
      for (std::int64_t point = 0; point < count; ++point) {
        perm[static_cast<std::size_t>(source + point)] =
            types::PermutationIndex{destination_cursors[atom] + point};
      }
      destination_cursors[atom] += count;
    }
  }

  // Build inverse: inv_perm[perm[i]] = i
  for (std::int64_t point = 0; point < total_points.raw(); ++point)
    inv_perm[static_cast<std::size_t>(perm[point].raw())] =
        types::PermutationIndex{point};

  return {std::move(perm), std::move(inv_perm)};
}

void reorder_to_atom_order(Vector& grid_weights, AlphaBetaMatrix& density,
                           CartesianMatrix& grid_coords,
                           AlphaBetaMatrix& kinetic,
                           const std::vector<types::PermutationIndex>& perm,
                           types::GridPointCount total_points) {
  const auto point_count = total_points.raw();
  if (grid_weights.size() != point_count || density.rows() != point_count ||
      grid_coords.rows() != point_count ||
      (kinetic.size() != 0 && kinetic.rows() != point_count) ||
      perm.size() != static_cast<std::size_t>(point_count))
    SKALAXC_EXCEPTION("Invalid atom-order permutation dimensions");
  permute_values(grid_weights, perm);
  permute_rows(density, perm);
  permute_rows(grid_coords, perm);
  permute_rows(kinetic, perm);
}

void reorder_to_rank_order(AlphaBetaMatrix& density, AlphaBetaMatrix& kinetic,
                           const std::vector<types::PermutationIndex>& inv_perm,
                           types::GridPointCount total_points) {
  const auto point_count = total_points.raw();
  if (density.rows() != point_count ||
      (kinetic.size() != 0 && kinetic.rows() != point_count) ||
      inv_perm.size() != static_cast<std::size_t>(point_count))
    SKALAXC_EXCEPTION("Invalid rank-order permutation dimensions");
  permute_rows(density, inv_perm);
  permute_rows(kinetic, inv_perm);
}

}  // namespace SkalaXC
