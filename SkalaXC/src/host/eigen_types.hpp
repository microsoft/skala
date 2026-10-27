#pragma once

#include <Eigen/Core>

namespace SkalaXC {

/** @brief Number of collinear spin channels. */
inline constexpr Eigen::Index spin_dimension = 2;
/** @brief Number of Cartesian directions. */
inline constexpr Eigen::Index direction_dimension = 3;
/** @brief Number of flattened spin-gradient components. */
inline constexpr Eigen::Index spin_gradient_dimension =
    spin_dimension * direction_dimension;

using Vector = Eigen::VectorXd;  ///< Dynamic vector of doubles.
/** @brief Dynamic row-major matrix of doubles. */
using RowMajorMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
/** @brief Dynamic column-major matrix of doubles. */
using ColMajorMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
/** @brief Point-major alpha/beta matrix. */
using AlphaBetaMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, spin_dimension, Eigen::RowMajor>;
/** @brief Alpha/beta-by-point matrix. */
using AlphaBetaByPointMatrix =
    Eigen::Matrix<double, spin_dimension, Eigen::Dynamic, Eigen::RowMajor>;
/** @brief Point-major Cartesian matrix. */
using CartesianMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, direction_dimension, Eigen::RowMajor>;
/** @brief Point-major flattened spin-gradient records. */
using SpinGradientRecordMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, spin_gradient_dimension,
                  Eigen::RowMajor>;

using VectorMap = Eigen::Map<Vector>;             ///< Mutable vector view.
using ConstVectorMap = Eigen::Map<const Vector>;  ///< Read-only vector view.
/** @brief Mutable strided vector view. */
using StridedVectorMap =
    Eigen::Map<Vector, Eigen::Unaligned, Eigen::InnerStride<Eigen::Dynamic>>;
/** @brief Read-only strided vector view. */
using ConstStridedVectorMap = Eigen::Map<const Vector, Eigen::Unaligned,
                                         Eigen::InnerStride<Eigen::Dynamic>>;

using RowMajorMatrixMap =
    Eigen::Map<RowMajorMatrix>;  ///< Mutable row-major view.
using ConstRowMajorMatrixMap =
    Eigen::Map<const RowMajorMatrix>;  ///< Read-only row-major view.
using ColMajorMatrixMap =
    Eigen::Map<ColMajorMatrix>;  ///< Mutable column-major view.
using ConstColMajorMatrixMap =
    Eigen::Map<const ColMajorMatrix>;  ///< Read-only column-major view.

/** @brief Read-only reference to point-major alpha/beta values. */
using ConstAlphaBetaMatrixRef = Eigen::Ref<const AlphaBetaMatrix>;

static_assert(AlphaBetaMatrix::ColsAtCompileTime == spin_dimension);
static_assert(AlphaBetaByPointMatrix::RowsAtCompileTime == spin_dimension);
static_assert(CartesianMatrix::ColsAtCompileTime == direction_dimension);
static_assert(SpinGradientRecordMatrix::ColsAtCompileTime ==
              spin_gradient_dimension);

}  // namespace SkalaXC
