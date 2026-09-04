#pragma once

#include "eigen_types.hpp"
#include "index_types.hpp"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace at {
class Tensor;
}

namespace SkalaXC {

/** @brief Collinear alpha/beta spin channel. */
enum SpinChannel { Alpha, Beta };
/** @brief Scalar and Pauli-z channel. */
enum PauliChannel { Scalar, SpinZ };
/** @brief Cartesian direction. */
enum Direction { X, Y, Z };

/** @brief Point-major scalar/spin-z values. */
using ScalarZChannels =
    Eigen::Matrix<double, Eigen::Dynamic, spin_dimension, Eigen::RowMajor>;

/** @brief Pointwise Cartesian gradients with two semantic channels. */
template <typename Channel>
class ChannelGradient final {
 public:
  /** @brief Contiguous direction-major storage. */
  using Storage = Eigen::Matrix<double, direction_dimension, Eigen::Dynamic,
                                Eigen::RowMajor>;
  /** @brief Point-major two-channel values for one direction. */
  using DirectionMatrix =
      Eigen::Matrix<double, Eigen::Dynamic, spin_dimension, Eigen::RowMajor>;
  /** @brief Mutable one-direction view. */
  using DirectionMap = Eigen::Map<DirectionMatrix>;
  /** @brief Read-only one-direction view. */
  using ConstDirectionMap = Eigen::Map<const DirectionMatrix>;

  static_assert(DirectionMatrix::ColsAtCompileTime == spin_dimension);

  /** @brief Construct empty gradient storage. */
  ChannelGradient() = default;
  /** @brief Construct storage for points. @param points Number of points. */
  explicit ChannelGradient(Eigen::Index points) { resize(points); }

  /** @brief Resize and discard values. @param points Number of points. */
  void resize(Eigen::Index points) {
    if (points < 0)
      throw std::invalid_argument(
          "ChannelGradient point count must be nonnegative");
    values_.resize(direction_dimension, points * spin_dimension);
  }

  /** @return Number of represented points. */
  Eigen::Index points() const noexcept {
    return values_.cols() / spin_dimension;
  }

  /**
   * @brief Access one value.
   * @param direction Cartesian direction.
   * @param point Point index.
   * @param channel Semantic channel.
   * @return Mutable value reference.
   */
  double& operator()(Direction direction, Eigen::Index point, Channel channel) {
    return values_(direction, column(point, channel));
  }

  /**
   * @brief Access one value.
   * @param direction Cartesian direction.
   * @param point Point index.
   * @param channel Semantic channel.
   * @return Read-only value reference.
   */
  const double& operator()(Direction direction, Eigen::Index point,
                           Channel channel) const {
    return values_(direction, column(point, channel));
  }

  /**
   * @brief View one Cartesian direction.
   * @param direction Direction index.
   * @return Mutable point-major view.
   */
  DirectionMap direction(Direction direction) {
    return DirectionMap(direction_data(direction), points(), spin_dimension);
  }

  /**
   * @brief View one Cartesian direction.
   * @param direction Direction index.
   * @return Read-only point-major view.
   */
  ConstDirectionMap direction(Direction direction) const {
    return ConstDirectionMap(direction_data(direction), points(),
                             spin_dimension);
  }

  /**
   * @brief Get one direction's storage.
   * @param direction Direction index.
   * @return Mutable storage pointer.
   */
  double* direction_data(Direction direction) {
    return values_.row(direction).data();
  }

  /**
   * @brief Get one direction's storage.
   * @param direction Direction index.
   * @return Read-only storage pointer.
   */
  const double* direction_data(Direction direction) const {
    return values_.row(direction).data();
  }

  /**
   * @brief Reorder points using a destination index for each source point.
   * @param destination_for_source Destination index for every source point.
   */
  void permute_points(
      const std::vector<types::PermutationIndex>& destination_for_source) {
    if (destination_for_source.size() != static_cast<std::size_t>(points()))
      throw std::invalid_argument("ChannelGradient permutation size mismatch");

    Storage permuted(direction_dimension, values_.cols());
    std::vector<bool> assigned(destination_for_source.size(), false);
    for (Eigen::Index source = 0; source < points(); ++source) {
      const auto destination = destination_for_source[source].raw();
      if (destination < 0 || destination >= points() || assigned[destination])
        throw std::invalid_argument("ChannelGradient permutation is invalid");
      assigned[destination] = true;
      permuted.middleCols(destination * spin_dimension, spin_dimension) =
          values_.middleCols(source * spin_dimension, spin_dimension);
    }
    values_ = std::move(permuted);
  }

 private:
  Eigen::Index column(Eigen::Index point, Channel channel) const {
    if (point < 0 || point >= points())
      throw std::out_of_range("ChannelGradient point index out of range");
    if (channel < 0 || channel >= spin_dimension)
      throw std::out_of_range("ChannelGradient channel index out of range");
    return point * spin_dimension + channel;
  }

  Storage values_;
};

/** @brief Pointwise Cartesian density gradients in scalar/spin-z form. */
using ScalarZGradient = ChannelGradient<PauliChannel>;

/** @brief Pointwise Cartesian density gradients for alpha and beta spin. */
using SpinGradient = ChannelGradient<SpinChannel>;

/**
 * @brief Convert a double tensor shaped `[spin, direction, points]`.
 * @param tensor Source tensor.
 * @return Converted gradient.
 */
SpinGradient spin_gradient_from_torch(const at::Tensor& tensor);

/**
 * @brief Convert to a tensor shaped `[spin, direction, points]`.
 * @param gradient Source gradient.
 * @param requires_grad Whether the tensor tracks gradients.
 * @return Converted tensor.
 */
at::Tensor spin_gradient_to_torch(const SpinGradient& gradient,
                                  bool requires_grad = false);

/**
 * @brief Convert scalar/spin-z gradients to alpha/beta gradients.
 * @param scalar_z Source channels.
 * @param alpha_beta Destination channels.
 */
void convert_scalar_z_to_alpha_beta(const ScalarZGradient& scalar_z,
                                    SpinGradient& alpha_beta);

/**
 * @brief Convert alpha/beta point values to scalar/spin-z channels.
 * @param alpha_beta Source values.
 * @return Converted values.
 */
ScalarZChannels alpha_beta_to_scalar_z(ConstAlphaBetaMatrixRef alpha_beta);

}  // namespace SkalaXC
