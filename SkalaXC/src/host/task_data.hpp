#pragma once

#include "eigen_types.hpp"
#include "spin_gradient.hpp"

namespace SkalaXC {

/** @brief ML feature inputs associated with one GauXC grid task. */
struct TaskFeatureData {
  AlphaBetaMatrix density;        ///< Alpha/beta density values.
  SpinGradient density_gradient;  ///< Alpha/beta Cartesian density gradients.
  AlphaBetaMatrix kinetic;        ///< Alpha/beta kinetic-density values.
};

/** @brief ML-derived potentials associated with one GauXC grid task. */
struct TaskPotentialData {
  AlphaBetaMatrix density;        ///< Density-feature derivatives.
  SpinGradient density_gradient;  ///< Density-gradient derivatives.
  AlphaBetaMatrix kinetic;        ///< Kinetic-feature derivatives.
  Vector dE_dw;                   ///< Integrated-energy weight derivatives.
};

}  // namespace SkalaXC
