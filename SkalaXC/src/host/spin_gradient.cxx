#include "spin_gradient.hpp"

#include <torch/torch.h>

#include <stdexcept>

namespace SkalaXC {

SpinGradient spin_gradient_from_torch(const at::Tensor& tensor) {
  if (!tensor.defined())
    throw std::invalid_argument("SpinGradient tensor must be defined");
  if (tensor.scalar_type() != torch::kFloat64)
    throw std::invalid_argument("SpinGradient tensor must contain doubles");
  if (tensor.dim() != 3 || tensor.size(0) != spin_dimension ||
      tensor.size(1) != direction_dimension)
    throw std::invalid_argument(
        "SpinGradient tensor shape must be [2, 3, points]");

  auto contiguous = tensor.detach().cpu().contiguous();
  auto source = contiguous.accessor<double, 3>();
  SpinGradient result(contiguous.size(2));
  for (Eigen::Index point = 0; point < result.points(); ++point)
    for (Eigen::Index direction = 0; direction < direction_dimension;
         ++direction)
      for (Eigen::Index spin = 0; spin < spin_dimension; ++spin)
        result(static_cast<Direction>(direction), point,
               static_cast<SpinChannel>(spin)) = source[spin][direction][point];
  return result;
}

at::Tensor spin_gradient_to_torch(const SpinGradient& gradient,
                                  bool requires_grad) {
  auto tensor = torch::empty(
      {spin_dimension, direction_dimension, gradient.points()},
      torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
  auto destination = tensor.accessor<double, 3>();
  for (Eigen::Index point = 0; point < gradient.points(); ++point)
    for (Eigen::Index direction = 0; direction < direction_dimension;
         ++direction)
      for (Eigen::Index spin = 0; spin < spin_dimension; ++spin)
        destination[spin][direction][point] =
            gradient(static_cast<Direction>(direction), point,
                     static_cast<SpinChannel>(spin));
  return tensor.requires_grad_(requires_grad);
}

void convert_scalar_z_to_alpha_beta(const ScalarZGradient& scalar_z,
                                    SpinGradient& alpha_beta) {
  alpha_beta.resize(scalar_z.points());
  for (Eigen::Index direction = 0; direction < direction_dimension;
       ++direction) {
    const auto source = scalar_z.direction(static_cast<Direction>(direction));
    auto destination = alpha_beta.direction(static_cast<Direction>(direction));
    destination.col(SpinChannel::Alpha) =
        0.5 *
        (source.col(PauliChannel::Scalar) + source.col(PauliChannel::SpinZ));
    destination.col(SpinChannel::Beta) =
        0.5 *
        (source.col(PauliChannel::Scalar) - source.col(PauliChannel::SpinZ));
  }
}

ScalarZChannels alpha_beta_to_scalar_z(ConstAlphaBetaMatrixRef alpha_beta) {
  ScalarZChannels scalar_z(alpha_beta.rows(), spin_dimension);
  scalar_z.col(PauliChannel::Scalar) =
      0.5 *
      (alpha_beta.col(SpinChannel::Alpha) + alpha_beta.col(SpinChannel::Beta));
  scalar_z.col(PauliChannel::SpinZ) =
      0.5 *
      (alpha_beta.col(SpinChannel::Alpha) - alpha_beta.col(SpinChannel::Beta));
  return scalar_z;
}

}  // namespace SkalaXC