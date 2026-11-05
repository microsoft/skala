/**
 * @file
 * @brief Non-synchronizing validation for SkalaXC-owned CUDA kernel launches.
 */
#pragma once

#include "exceptions.hpp"

#include <cuda_runtime_api.h>

#include <string>

namespace SkalaXC::cuda {

/**
 * @brief Throw with operation context if the preceding kernel launch failed.
 * @param operation Description of the kernel operation.
 */
inline void check_launch(const char* operation) {
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    SKALAXC_EXCEPTION(std::string(operation) + ": " +
                      cudaGetErrorString(status));
}

}  // namespace SkalaXC::cuda
