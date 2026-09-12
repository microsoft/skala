#pragma once

#include "component_matrix_map.hpp"
#include "spin_gradient.hpp"

#include <array>
#include <stdexcept>
#include <utility>

namespace SkalaXC {

namespace detail {

inline Eigen::Index first_derivative_component(Direction direction) {
  if (direction < X || direction > Z)
    throw std::out_of_range("Derivative direction out of range");
  return direction + 1;
}

inline Eigen::Index hessian_component(Direction first, Direction second) {
  first_derivative_component(first);
  first_derivative_component(second);
  constexpr std::array<std::array<Eigen::Index, 3>, 3> components{
      {{{4, 5, 6}}, {{5, 7, 8}}, {{6, 8, 9}}}};
  return components[first][second];
}

}  // namespace detail

/** @brief Borrow GauXC basis values, gradients, and symmetric Hessian blocks.
 */
template <typename Map>
class BasisComponentView {
 public:
  /**
   * @brief Borrow an existing map with 1, 4, or 10 basis components.
   * @param components Component map to borrow.
   */
  explicit BasisComponentView(Map& components) : components_(components) {
    const auto count = components.components();
    if (count != 1 && count != 4 && count != 10)
      throw std::invalid_argument("Invalid basis component count");
  }

  /** @return Non-owning basis-value matrix. */
  auto value() { return components_.component(0); }
  /** @return Read-only basis-value matrix. */
  auto value() const { return std::as_const(components_).component(0); }

  /**
   * @param direction Requested Cartesian direction.
   * @return Non-owning derivative matrix.
   */
  auto first_derivative(Direction direction) {
    return components_.component(detail::first_derivative_component(direction));
  }
  /**
   * @param direction Requested Cartesian direction.
   * @return Read-only derivative matrix.
   */
  auto first_derivative(Direction direction) const {
    return std::as_const(components_)
        .component(detail::first_derivative_component(direction));
  }

  /**
   * @param first First Cartesian direction.
   * @param second Second Cartesian direction.
   * @return Non-owning Hessian matrix, symmetric in its direction arguments.
   */
  auto hessian(Direction first, Direction second) {
    return components_.component(detail::hessian_component(first, second));
  }
  /**
   * @param first First Cartesian direction.
   * @param second Second Cartesian direction.
   * @return Read-only Hessian matrix.
   */
  auto hessian(Direction first, Direction second) const {
    return std::as_const(components_)
        .component(detail::hessian_component(first, second));
  }

 private:
  Map& components_;
};

/** @brief Borrow scalar/spin-z matrices with optional Cartesian derivatives. */
template <typename Map>
class PauliComponentView {
 public:
  /**
   * @brief Borrow an existing map with 1 or 4 components per channel.
   * @param components Component map to borrow.
   */
  explicit PauliComponentView(Map& components) : components_(components) {
    if (components.components() != 2 && components.components() != 8)
      throw std::invalid_argument("Invalid Pauli component count");
  }

  /**
   * @param channel Requested Pauli channel.
   * @return Non-owning value matrix.
   */
  auto value(PauliChannel channel) {
    return components_.component(channel_offset(channel));
  }
  /**
   * @param channel Requested Pauli channel.
   * @return Read-only value matrix.
   */
  auto value(PauliChannel channel) const {
    return std::as_const(components_).component(channel_offset(channel));
  }

  /**
   * @param channel Requested Pauli channel.
   * @param direction Requested Cartesian direction.
   * @return Non-owning derivative matrix.
   */
  auto first_derivative(PauliChannel channel, Direction direction) {
    return components_.component(derivative_component(channel, direction));
  }
  /**
   * @param channel Requested Pauli channel.
   * @param direction Requested Cartesian direction.
   * @return Read-only derivative matrix.
   */
  auto first_derivative(PauliChannel channel, Direction direction) const {
    return std::as_const(components_)
        .component(derivative_component(channel, direction));
  }

 private:
  Eigen::Index channel_offset(PauliChannel channel) const {
    if (channel < Scalar || channel > SpinZ)
      throw std::out_of_range("Pauli channel out of range");
    return channel * (components_.components() / spin_dimension);
  }

  Eigen::Index derivative_component(PauliChannel channel,
                                    Direction direction) const {
    if (components_.components() != 8)
      throw std::out_of_range("Pauli derivative components are absent");
    return channel_offset(channel) +
           detail::first_derivative_component(direction);
  }

  Map& components_;
};

}  // namespace SkalaXC