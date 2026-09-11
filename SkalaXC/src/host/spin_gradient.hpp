#pragma once

#include "eigen_types.hpp"
#include "index_types.hpp"

#include <cstdint>
#include <stdexcept>
#include <type_traits>
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
  /** @brief Mutable direction-by-interleaved-channel point block. */
  using PointSlice = Eigen::Block<Storage, direction_dimension, Eigen::Dynamic>;
  /** @brief Read-only point block. */
  using ConstPointSlice =
      Eigen::Block<const Storage, direction_dimension, Eigen::Dynamic>;

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
   * @brief Borrow a point range without changing its direction stride.
   * @param offset First point in the owner.
   * @param count Number of points, possibly zero.
   * @return Mutable view valid until the owner invalidates its storage.
   */
  PointSlice point_slice(Eigen::Index offset, Eigen::Index count) & {
    validate_point_slice(offset, count);
    return values_.middleCols(offset * spin_dimension, count * spin_dimension);
  }

  /**
   * @brief Borrow a point range read-only.
   * @param offset First point in the owner.
   * @param count Number of points, possibly zero.
   * @return Read-only view valid until the owner invalidates its storage.
   */
  ConstPointSlice point_slice(Eigen::Index offset, Eigen::Index count) const& {
    validate_point_slice(offset, count);
    return values_.middleCols(offset * spin_dimension, count * spin_dimension);
  }
  PointSlice point_slice(Eigen::Index, Eigen::Index) && = delete;
  ConstPointSlice point_slice(Eigen::Index, Eigen::Index) const&& = delete;

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
  void validate_point_slice(Eigen::Index offset, Eigen::Index count) const {
    if (offset < 0 || count < 0 || offset > points() ||
        count > points() - offset)
      throw std::out_of_range("ChannelGradient point slice out of range");
  }

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
 * @brief Copy equal-sized, nonoverlapping point blocks without allocation.
 * @param source Point block from point_slice; channel semantics must match.
 * @param destination Writable point block; its owner is never resized.
 * @throws std::invalid_argument If extents differ or nonempty ranges overlap.
 */
template <typename Matrix>
void copy_points(
    const Eigen::Block<Matrix, direction_dimension, Eigen::Dynamic>& source,
    SpinGradient::PointSlice destination) {
  static_assert(
      std::is_same_v<std::remove_const_t<Matrix>, SpinGradient::Storage>);
  if (source.cols() != destination.cols() ||
      source.cols() % spin_dimension != 0)
    throw std::invalid_argument("Gradient point block dimensions mismatch");
  if (source.cols() == 0) return;

  const bool same_storage =
      source.nestedExpression().data() == destination.nestedExpression().data();
  const auto source_start = source.startCol();
  const auto destination_start = destination.startCol();
  const bool overlap = same_storage &&
                       source_start < destination_start + destination.cols() &&
                       destination_start < source_start + source.cols();
  if (overlap) throw std::invalid_argument("Gradient point blocks overlap");
  destination = source;
}

/** @brief Copy all source points into a selected destination range. */
template <typename Channel>
void copy_points(const ChannelGradient<Channel>& source,
                 SpinGradient::PointSlice destination) {
  copy_points(source.point_slice(0, source.points()), destination);
}

/** @brief Copy a selected source range into a pre-sized destination. */
template <typename Matrix, typename Channel>
void copy_points(
    const Eigen::Block<Matrix, direction_dimension, Eigen::Dynamic>& source,
    ChannelGradient<Channel>& destination) {
  copy_points(source, destination.point_slice(0, destination.points()));
}

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
