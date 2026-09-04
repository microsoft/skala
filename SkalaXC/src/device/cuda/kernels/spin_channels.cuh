#pragma once

namespace SkalaXC::cuda {

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