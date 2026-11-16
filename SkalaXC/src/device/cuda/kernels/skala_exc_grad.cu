/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 *
 * All rights reserved.
 *
 * See LICENSE.txt for details
 *
 * ---------------------------------------------------------------------------
 * SkalaXC adaptation: renamed from GauXC skala-branch
 * cuda/kernels/onedft_exc_grad.cu (onedft -> skala). Compiled only when
 * SKALAXC_ENABLE_CUDA is ON (requires a CUDA toolchain).
 * ---------------------------------------------------------------------------
 */
/**
 * @file
 * @brief CUDA adaptation of Skala derivatives for GauXC gradient kernels.
 */
#include "common/skala_exc_grad.hpp"
#include "device/cuda_launch_check.hpp"
#include "device_specific/cuda_device_constants.hpp"
#include "device_specific/cuda_util.hpp"
#include "spin_channels.cuh"
#include <gauxc/util/div_ceil.hpp>
#include <initializer_list>

namespace SkalaXC {

using GauXC::device_queue;
using GauXC::XCDeviceTask;

/**
 * @brief Convert per-component Skala VXC values to GauXC gradient inputs.
 * @param ntasks Number of tasks in @p tasks_device.
 * @param tasks_device Tasks indexed by the z grid dimension.
 *
 * Alpha directional derivatives occupy the `gamma` buffers and beta
 * derivatives occupy the corresponding `vgamma` buffers. The kernel writes
 * scalar/z derivatives to `dden` and sets GauXC's coupling coefficients so its
 * standard GGA and meta-GGA gradient kernels reproduce the Skala expression.
 */
__global__ void transform_skala_vxc_for_grad_kernel(
    uint32_t ntasks, XCDeviceTask* __restrict__ tasks_device) {
  using cuda::Direction;
  using cuda::PauliChannel;
  using cuda::SpinChannel;

  const int batch_idx = blockIdx.z;
  if (batch_idx >= ntasks) return;

  auto& task = tasks_device[batch_idx];
  const auto npts = task.npts;

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npts) return;

  // Read per-direction SkalaXC derivatives (alpha/beta)
  for (const auto direction : {Direction::X, Direction::Y, Direction::Z}) {
    const auto potential = cuda::alpha_beta_to_scalar_z(
        cuda::gradient_potential(task, SpinChannel::Alpha, direction)[tid],
        cuda::gradient_potential(task, SpinChannel::Beta, direction)[tid]);
    cuda::density_gradient(task, PauliChannel::Scalar, direction)[tid] =
        potential.scalar;
    cuda::density_gradient(task, PauliChannel::SpinZ, direction)[tid] =
        potential.spin_z;
  }

  // Set vgamma coefficients so the standard kernel reproduces the SkalaXC
  // formula
  task.vgamma_pp[tid] = 1.0;
  task.vgamma_pm[tid] = 0.0;
  task.vgamma_mm[tid] = 1.0;
}

/**
 * @brief Launch conversion of Skala VXC buffers to GauXC gradient inputs.
 * @param ntasks Number of active tasks.
 * @param max_npts Maximum point count among tasks.
 * @param tasks_device Device pointer to active tasks.
 * @param queue Queue on which conversion is enqueued.
 */
void transform_skala_vxc_for_grad(size_t ntasks, int32_t max_npts,
                                  XCDeviceTask* tasks_device,
                                  device_queue queue) {

  cudaStream_t stream = queue.queue_as<GauXC::util::cuda_stream>();

  dim3 threads(256);
  dim3 blocks(GauXC::util::div_ceil((uint32_t)max_npts, threads.x), 1, ntasks);

  transform_skala_vxc_for_grad_kernel<<<blocks, threads, 0, stream>>>(
      ntasks, tasks_device);
  cuda::check_launch("Failed to launch Skala gradient-potential transform");
}

}  // namespace SkalaXC
