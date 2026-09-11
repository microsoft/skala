#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "xc_integrator/xc_data/device/xc_device_task.hpp"

#include <cuda_runtime.h>

namespace SkalaXC::cuda {

enum class SpinChannel { Alpha, Beta };
enum class PauliChannel { Scalar, SpinZ };
enum class Direction { X, Y, Z };

/**
 * @brief Select a model-gradient potential buffer before its transform.
 * @param task GauXC task during model-potential unpacking or before conversion
 * to GauXC gradient inputs, which overwrites the beta buffers with
 * coefficients.
 * @param channel Alpha or beta model channel.
 * @param direction Cartesian derivative direction.
 * @return Non-owning pointer, read-only for a const task; null for invalid
 * selectors.
 */
template <typename Task>
__host__ __device__ __forceinline__ auto gradient_potential(Task& task,
                                                            SpinChannel channel,
                                                            Direction direction)
    -> std::conditional_t<std::is_const_v<Task>, const double*, double*> {
  static_assert(std::is_same_v<std::remove_const_t<Task>, GauXC::XCDeviceTask>);
  if (channel != SpinChannel::Alpha && channel != SpinChannel::Beta)
    return nullptr;
  switch (direction) {
    case Direction::X:
      return channel == SpinChannel::Alpha ? task.gamma_pp : task.vgamma_pp;
    case Direction::Y:
      return channel == SpinChannel::Alpha ? task.gamma_pm : task.vgamma_pm;
    case Direction::Z:
      return channel == SpinChannel::Alpha ? task.gamma_mm : task.vgamma_mm;
  }
  return nullptr;
}

/** @brief Read a basis derivative buffer; null for an invalid direction. */
__host__ __device__ __forceinline__ const double* basis_derivative(
    const GauXC::XCDeviceTask& task, Direction direction) {
  switch (direction) {
    case Direction::X:
      return task.dbfx;
    case Direction::Y:
      return task.dbfy;
    case Direction::Z:
      return task.dbfz;
  }
  return nullptr;
}

/**
 * @brief Access GauXC scalar/spin-z density-gradient scratch buffers.
 * @param task Task holding density gradients during feature packing, or
 * transformed gradient potentials after conversion to GauXC gradient inputs.
 * @param channel Scalar or spin-z channel.
 * @param direction Cartesian direction.
 * @return Non-owning pointer, read-only for a const task; null for invalid
 * selectors.
 */
template <typename Task>
__host__ __device__ __forceinline__ auto density_gradient(Task& task,
                                                          PauliChannel channel,
                                                          Direction direction)
    -> std::conditional_t<std::is_const_v<Task>, const double*, double*> {
  static_assert(std::is_same_v<std::remove_const_t<Task>, GauXC::XCDeviceTask>);
  if (channel != PauliChannel::Scalar && channel != PauliChannel::SpinZ)
    return nullptr;
  switch (direction) {
    case Direction::X:
      return channel == PauliChannel::Scalar ? task.dden_sx : task.dden_zx;
    case Direction::Y:
      return channel == PauliChannel::Scalar ? task.dden_sy : task.dden_zy;
    case Direction::Z:
      return channel == PauliChannel::Scalar ? task.dden_sz : task.dden_zz;
  }
  return nullptr;
}

struct AlphaBetaChannels {
  double alpha;
  double beta;
};

struct ScalarZChannels {
  double scalar;
  double spin_z;
};

__device__ __forceinline__ AlphaBetaChannels
    scalar_z_to_alpha_beta(double scalar, double spin_z) {
  return {0.5 * (scalar + spin_z), 0.5 * (scalar - spin_z)};
}

__device__ __forceinline__ ScalarZChannels alpha_beta_to_scalar_z(double alpha,
                                                                  double beta) {
  return {0.5 * (alpha + beta), 0.5 * (alpha - beta)};
}

}  // namespace SkalaXC::cuda