#include "mpi_wrapper.hpp"

#include <torch/torch.h>

#include <stdexcept>

namespace SkalaXC::mpi {
namespace {

SpinGradientRecordMatrix pack_point_records(const SpinGradient& gradient) {
  SpinGradientRecordMatrix records(gradient.points(), spin_gradient_dimension);
  for (Eigen::Index point = 0; point < gradient.points(); ++point)
    for (Eigen::Index spin = 0; spin < spin_dimension; ++spin)
      for (Eigen::Index direction = 0; direction < direction_dimension;
           ++direction)
        records(point, spin * direction_dimension + direction) =
            gradient(static_cast<Direction>(direction), point,
                     static_cast<SpinChannel>(spin));
  return records;
}

SpinGradient unpack_point_records(const SpinGradientRecordMatrix& records) {
  SpinGradient gradient(records.rows());
  for (Eigen::Index point = 0; point < records.rows(); ++point)
    for (Eigen::Index spin = 0; spin < spin_dimension; ++spin)
      for (Eigen::Index direction = 0; direction < direction_dimension;
           ++direction)
        gradient(static_cast<Direction>(direction), point,
                 static_cast<SpinChannel>(spin)) =
            records(point, spin * direction_dimension + direction);
  return gradient;
}

SpinGradientRecordMatrix permute_point_records(
    const SpinGradientRecordMatrix& source,
    const std::vector<SkalaXC::types::PermutationIndex>&
        destination_for_source) {
  if (destination_for_source.empty()) return source;
  if (destination_for_source.size() != static_cast<std::size_t>(source.rows()))
    throw std::invalid_argument("Gradient permutation size mismatch");

  SpinGradientRecordMatrix result(source.rows(), spin_gradient_dimension);
  std::vector<bool> assigned(destination_for_source.size(), false);
  for (Eigen::Index source_point = 0; source_point < source.rows();
       ++source_point) {
    const auto destination = destination_for_source[source_point].raw();
    if (destination < 0 || destination >= source.rows() ||
        assigned[destination])
      throw std::invalid_argument("Gradient permutation is invalid");
    assigned[destination] = true;
    result.row(destination) = source.row(source_point);
  }
  return result;
}

at::Tensor point_records_to_torch(const SpinGradientRecordMatrix& records) {
  auto options =
      torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  if (records.rows() == 0)
    return torch::empty({spin_dimension, direction_dimension, 0}, options)
        .requires_grad_(true);
  return torch::from_blob(const_cast<double*>(records.data()),
                          {spin_dimension, direction_dimension, records.rows()},
                          {direction_dimension, 1, spin_gradient_dimension},
                          options)
      .clone()
      .requires_grad_(true);
}

void validate_torch_gradient(const at::Tensor& tensor,
                             Eigen::Index expected_points) {
  if (!tensor.defined() || tensor.scalar_type() != torch::kFloat64 ||
      tensor.dim() != 3 || tensor.size(0) != spin_dimension ||
      tensor.size(1) != direction_dimension ||
      tensor.size(2) != expected_points)
    throw std::invalid_argument(
        "Torch gradient must be a double tensor shaped [2, 3, points]");
}

}  // namespace

at::Tensor spin_gradient_to_torch(const SpinGradient& gradient) {
  return point_records_to_torch(pack_point_records(gradient));
}

SpinGradient torch_to_spin_gradient(const at::Tensor& tensor,
                                    Eigen::Index expected_points) {
  validate_torch_gradient(tensor, expected_points);
  auto point_major = tensor.detach().cpu().permute({2, 0, 1}).contiguous();
  Eigen::Map<const SpinGradientRecordMatrix> records(
      point_major.data_ptr<double>(), expected_points, spin_gradient_dimension);
  return unpack_point_records(records);
}

void broadcast_string(std::string& value, const GauXC::RuntimeEnvironment& rt,
                      int root) {
  if (root < 0 || root >= rt.comm_size())
    throw std::invalid_argument("Invalid broadcast root");
  if (rt.comm_size() == 1) return;

#ifdef GAUXC_HAS_MPI
  std::uint64_t size = rt.comm_rank() == root ? value.size() : 0;
  MPI_Bcast(&size, 1, MPI_UINT64_T, root, rt.comm());
  if (size > value.max_size())
    throw std::length_error("Broadcast string exceeds local capacity");
  if (rt.comm_rank() != root) value.resize(static_cast<std::size_t>(size));

  std::uint64_t offset = 0;
  while (offset < size) {
    const auto remaining = size - offset;
    const auto chunk = static_cast<int>(
        std::min<std::uint64_t>(remaining, std::numeric_limits<int>::max()));
    MPI_Bcast(value.data() + static_cast<std::size_t>(offset), chunk, MPI_CHAR,
              root, rt.comm());
    offset += static_cast<std::uint64_t>(chunk);
  }
#else
  (void)value;
  throw std::logic_error("MPI broadcast requested without MPI support");
#endif
}

at::Tensor gather_torch_gradient(
    const SpinGradient& local_gradient,
    [[maybe_unused]] const CollectiveLayout& point_layout,
    const std::vector<SkalaXC::types::PermutationIndex>&
        rank_to_atom_permutation,
    const GauXC::RuntimeEnvironment& rt, [[maybe_unused]] int root) {
  auto local_records = pack_point_records(local_gradient);
  if (rt.comm_size() == 1) {
    return point_records_to_torch(
        permute_point_records(local_records, rank_to_atom_permutation));
  }

#ifdef GAUXC_HAS_MPI
  SpinGradientRecordMatrix gathered_records;
  const auto component_layout = point_layout.scaled(spin_gradient_dimension);
  if (rt.comm_rank() == root) {
    gathered_records.resize(point_layout.extent(), spin_gradient_dimension);
  }

  gatherv(local_records, gathered_records, component_layout, rt, root);

  if (rt.comm_rank() != root) return {};
  return point_records_to_torch(
      permute_point_records(gathered_records, rank_to_atom_permutation));
#else
  throw std::logic_error("MPI gradient gather requested without MPI support");
#endif
}

SpinGradient scatter_torch_gradient(
    const at::Tensor& root_tensor, Eigen::Index local_points,
    [[maybe_unused]] const CollectiveLayout& point_layout,
    const std::vector<SkalaXC::types::PermutationIndex>&
        atom_to_rank_permutation,
    const GauXC::RuntimeEnvironment& rt, [[maybe_unused]] int root) {
  if (local_points < 0)
    throw std::invalid_argument("Local gradient point count is invalid");
  if (rt.comm_size() == 1) {
    validate_torch_gradient(root_tensor, local_points);
    auto point_major =
        root_tensor.detach().cpu().permute({2, 0, 1}).contiguous();
    Eigen::Map<const SpinGradientRecordMatrix> records(
        point_major.data_ptr<double>(), local_points, spin_gradient_dimension);
    return unpack_point_records(
        permute_point_records(records, atom_to_rank_permutation));
  }

#ifdef GAUXC_HAS_MPI
  SpinGradientRecordMatrix rank_ordered_records;
  at::Tensor point_major;
  const auto component_layout = point_layout.scaled(spin_gradient_dimension);
  if (rt.comm_rank() == root) {
    const auto global_points = point_layout.extent();
    validate_torch_gradient(root_tensor, global_points);
    point_major = root_tensor.detach().cpu().permute({2, 0, 1}).contiguous();
    Eigen::Map<const SpinGradientRecordMatrix> atom_ordered_records(
        point_major.data_ptr<double>(), global_points, spin_gradient_dimension);
    rank_ordered_records =
        permute_point_records(atom_ordered_records, atom_to_rank_permutation);
  }

  SpinGradientRecordMatrix local_records(local_points, spin_gradient_dimension);
  scatterv(rank_ordered_records, component_layout, local_records, rt, root);
  return unpack_point_records(local_records);
#else
  throw std::logic_error("MPI gradient scatter requested without MPI support");
#endif
}

}  // namespace SkalaXC::mpi