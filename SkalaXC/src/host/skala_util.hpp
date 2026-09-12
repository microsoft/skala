#pragma once
#include "eigen_types.hpp"
#include "index_types.hpp"
#include "mpi_wrapper.hpp"

/**
 * @file
 * @brief SkalaXC ML utility declarations.
 *
 * Ported from GauXC/skala onedft_util.hpp (onedft -> skala).
 *
 * Internal header: freely uses GauXC + LibTorch types and must never be
 * included by any SkalaXC public header (ABI isolation).
 */
#include <gauxc/gauxc_config.hpp>
#include <gauxc/runtime_environment.hpp>
#include <gauxc/xc_integrator/local_work_driver.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#include <string_view>

using IValueList = std::vector<c10::IValue>;  ///< Ordered TorchScript values.
using IValueMap =
    std::unordered_map<std::string, c10::IValue>;  ///< Named TorchScript
                                                   ///< values.
using FeatureDict =
    c10::Dict<std::string, at::Tensor>;  ///< Named model tensors.

namespace SkalaXC {

class SkalaModel;

/// Canonical feature keys used by the Skala model inputs/outputs.
enum SKALA_FEATURE {
  DEN,                          ///< Density.
  DDEN,                         ///< Density gradient.
  TAU,                          ///< Kinetic density.
  POINTS,                       ///< Grid coordinates.
  WEIGHTS,                      ///< Partitioned quadrature weights.
  COORDS,                       ///< Atomic coordinates.
  ATOMIC_GRID_WEIGHTS,          ///< Per-domain raw weights.
  ATOMIC_GRID_SIZES,            ///< Per-domain point counts.
  ATOMIC_GRID_SIZE_BOUND_SHAPE  ///< Bound-shape metadata.
};

/// @return Mapping from feature enum to model key string.
inline const std::map<SKALA_FEATURE, std::string>& feat_map() {
  static const std::map<SKALA_FEATURE, std::string> map = {
      {DEN, "density"},
      {DDEN, "grad"},
      {TAU, "kin"},
      {POINTS, "grid_coords"},
      {WEIGHTS, "grid_weights"},
      {COORDS, "coarse_0_atomic_coords"},
      {ATOMIC_GRID_WEIGHTS, "atomic_grid_weights"},
      {ATOMIC_GRID_SIZES, "atomic_grid_sizes"},
      {ATOMIC_GRID_SIZE_BOUND_SHAPE, "atomic_grid_size_bound_shape"}};
  return map;
}

/// @return Reverse mapping from model key string to feature enum.
inline const std::map<std::string, SKALA_FEATURE>& reverse_feat_map() {
  static const std::map<std::string, SKALA_FEATURE> map = {
      {"density", DEN},
      {"grad", DDEN},
      {"kin", TAU},
      {"grid_coords", POINTS},
      {"grid_weights", WEIGHTS},
      {"coarse_0_atomic_coords", COORDS},
      {"atomic_grid_weights", ATOMIC_GRID_WEIGHTS},
      {"atomic_grid_sizes", ATOMIC_GRID_SIZES},
      {"atomic_grid_size_bound_shape", ATOMIC_GRID_SIZE_BOUND_SHAPE}};
  return map;
}

/// @brief Check whether a feature key exists in the known feature map.
/// @param value Feature key.
/// @return Whether the key is known.
bool valueExists(const std::string& value);

/// @brief Run the model forward pass and return EXC tensor output.
/// @param exc_func Integrated-energy TorchScript method.
/// @param features Model input tensors.
/// @return Raw model output tensor.
at::Tensor get_exc(const torch::jit::Method& exc_func, FeatureDict features);

/**
 * @brief Validate a tensor before model-boundary use or raw storage access.
 * @param tensor Tensor to validate.
 * @param description Diagnostic description of the tensor.
 * @param expected_device Required device and CUDA device index.
 * @param expected_type Required scalar type.
 * @param expected_sizes Required exact dimensions.
 * @param require_contiguous Whether storage must be contiguous.
 * @param require_gradient Whether the tensor must participate in autograd.
 * @throws Exception If any required tensor property is absent.
 */
void validate_model_tensor(const at::Tensor& tensor,
                           std::string_view description,
                           const c10::Device& expected_device,
                           c10::ScalarType expected_type,
                           at::IntArrayRef expected_sizes,
                           bool require_contiguous = false,
                           bool require_gradient = false);

/**
 * @brief Return a validated contiguous feature gradient.
 * @param feature Differentiable model input whose gradient is required.
 * @param description Diagnostic description of the gradient.
 * @return Gradient with the feature's device, dtype, and exact shape.
 */
at::Tensor validated_model_gradient(const at::Tensor& feature,
                                    std::string_view description);

/**
 * @brief Produce a scalar finite-value check without reading it on the host.
 * @param tensor Tensor to inspect.
 * @return Scalar boolean tensor on the same device.
 */
at::Tensor model_tensor_finite_check(const at::Tensor& tensor);

/**
 * @brief Validate finite values immediately, synchronizing if required.
 * @param tensor Tensor to inspect.
 * @param description Diagnostic description of the tensor.
 * @throws Exception If any value is non-finite.
 */
void validate_model_tensor_finite(const at::Tensor& tensor,
                                  std::string_view description);

/**
 * @brief Evaluate a model's integrated XC energy.
 *
 * @param model Loaded model whose method is invoked.
 * @param features Model feature dictionary.
 * @param expected_device Device on which output is required.
 * @return Scalar integrated XC energy tensor.
 */
at::Tensor evaluate_model_energy(const SkalaModel& model,
                                 const FeatureDict& features,
                                 const c10::Device& expected_device);

/**
 * @brief Build rank-order to atom-order point permutations.
 * @param all_rank_atom_sizes Atom sizes for all ranks in row-major rank order.
 * @param point_layout Per-rank point counts and displacements.
 * @param atom_count Number of atoms.
 * @param communicator_size Number of MPI ranks.
 * @return Pair (perm, inv_perm) with perm[rank_ordered_idx] = atom_ordered_idx.
 */
std::pair<std::vector<types::PermutationIndex>,
          std::vector<types::PermutationIndex>>
    build_atom_reorder_perm(
        const std::vector<types::GridPointCount>& all_rank_atom_sizes,
        const mpi::CollectiveLayout& point_layout, types::AtomCount atom_count,
        types::CommunicatorSize communicator_size);

/**
 * @brief Reorder ordinary point records from rank-order to atom-order.
 * @param grid_weights Point weights to reorder.
 * @param density Point densities to reorder.
 * @param grid_coords Point coordinates to reorder.
 * @param kinetic Point kinetic densities to reorder.
 * @param perm Destination atom-order index for each source point.
 * @param total_points Number of point records.
 */
void reorder_to_atom_order(Vector& grid_weights, AlphaBetaMatrix& density,
                           CartesianMatrix& grid_coords,
                           AlphaBetaMatrix& kinetic,
                           const std::vector<types::PermutationIndex>& perm,
                           types::GridPointCount total_points);

/**
 * @brief Reorder ordinary point records from atom-order to rank-order.
 * @param density Point densities to reorder.
 * @param kinetic Point kinetic densities to reorder.
 * @param inv_perm Destination rank-order index for each source point.
 * @param total_points Number of point records.
 */
void reorder_to_rank_order(AlphaBetaMatrix& density, AlphaBetaMatrix& kinetic,
                           const std::vector<types::PermutationIndex>& inv_perm,
                           types::GridPointCount total_points);
}  // namespace SkalaXC
