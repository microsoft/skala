#pragma once

#include "eigen_types.hpp"

#include <stdexcept>

namespace SkalaXC {

/** @brief Checked view of component-major matrices stored contiguously. */
class ComponentMatrixMap {
 public:
  /**
   * @brief Construct a component matrix view.
   * @param data Mutable contiguous storage.
   * @param components Number of components.
   * @param rows Rows per component.
   * @param points Columns per component.
   */
  ComponentMatrixMap(double* data, Eigen::Index components, Eigen::Index rows,
                     Eigen::Index points)
      : values_(data, rows, points * components),
        components_(components),
        points_(points) {
    if (components < 0 || rows < 0 || points < 0)
      throw std::invalid_argument(
          "ComponentMatrixMap dimensions must be nonnegative");
  }

  /**
   * @brief Access one value.
   * @param component Component index.
   * @param row Row index.
   * @param point Point index.
   * @return Mutable value reference.
   */
  double& operator()(Eigen::Index component, Eigen::Index row,
                     Eigen::Index point) {
    validate(component, row, point);
    return values_(row, component * points_ + point);
  }

  /**
   * @brief Access one value.
   * @param component Component index.
   * @param row Row index.
   * @param point Point index.
   * @return Read-only value reference.
   */
  const double& operator()(Eigen::Index component, Eigen::Index row,
                           Eigen::Index point) const {
    validate(component, row, point);
    return values_(row, component * points_ + point);
  }

  /**
   * @brief View one component matrix.
   * @param component Component index.
   * @return Mutable matrix block.
   */
  auto component(Eigen::Index component) {
    validate_component(component);
    return values_.middleCols(component * points_, points_);
  }

  /**
   * @brief View one component matrix.
   * @param component Component index.
   * @return Read-only matrix block.
   */
  auto component(Eigen::Index component) const {
    validate_component(component);
    return values_.middleCols(component * points_, points_);
  }

  /**
   * @brief Get one component's contiguous storage.
   * @param component Component index.
   * @return Mutable storage pointer.
   */
  double* component_data(Eigen::Index component) {
    validate_component(component);
    return values_.data() + component * values_.rows() * points_;
  }

  /**
   * @brief Get one component's contiguous storage.
   * @param component Component index.
   * @return Read-only storage pointer.
   */
  const double* component_data(Eigen::Index component) const {
    validate_component(component);
    return values_.data() + component * values_.rows() * points_;
  }

  /** @return Number of components. */
  Eigen::Index components() const noexcept { return components_; }
  /** @return Rows per component. */
  Eigen::Index rows() const noexcept { return values_.rows(); }
  /** @return Points per component. */
  Eigen::Index points() const noexcept { return points_; }

 private:
  void validate_component(Eigen::Index component) const {
    if (component < 0 || component >= components_)
      throw std::out_of_range("ComponentMatrixMap component out of range");
  }

  void validate(Eigen::Index component, Eigen::Index row,
                Eigen::Index point) const {
    validate_component(component);
    if (row < 0 || row >= values_.rows())
      throw std::out_of_range("ComponentMatrixMap row out of range");
    if (point < 0 || point >= points_)
      throw std::out_of_range("ComponentMatrixMap point out of range");
  }

  Eigen::Map<ColMajorMatrix> values_;
  Eigen::Index components_;
  Eigen::Index points_;
};

}  // namespace SkalaXC
