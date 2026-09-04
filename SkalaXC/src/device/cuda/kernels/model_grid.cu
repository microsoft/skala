/**
 * @file
 * @brief CUDA kernels for GauXC task and Skala model-grid data transfer.
 */
#include "common/model_grid.hpp"
#include "device/cuda_launch_check.hpp"
#include "spin_channels.cuh"

#include "device_specific/cuda_util.hpp"

#include <gauxc/util/div_ceil.hpp>

#include <cuda_runtime.h>

namespace SkalaXC {

namespace {

__device__ cuda::AlphaBetaChannels post_uvars_density(
    const GauXC::XCDeviceTask& task, std::size_t point) {
  return {task.den_s[point], task.den_z[point]};
}

__device__ cuda::ScalarZChannels post_uvars_density_gradient(
    const GauXC::XCDeviceTask& task, std::size_t point,
    std::int64_t direction) {
  switch (direction) {
    case 0:
      return {task.dden_sx[point], task.dden_zx[point]};
    case 1:
      return {task.dden_sy[point], task.dden_zy[point]};
    default:
      return {task.dden_sz[point], task.dden_zz[point]};
  }
}

__device__ cuda::AlphaBetaChannels post_uvars_kinetic(
    const GauXC::XCDeviceTask& task, std::size_t point) {
  return {task.tau_s[point], task.tau_z[point]};
}

/**
 * @brief Pack one post-U-variable task point into model feature tensors.
 * @param task_count Number of tasks in @p tasks.
 * @param tasks Device tasks indexed by the y grid dimension.
 * @param point_offsets Device task-to-model-grid offsets.
 * @param total_points Number of points in each tensor channel.
 * @param density Optional two-channel density output.
 * @param density_gradient Optional six-channel gradient output.
 * @param kinetic Optional two-channel kinetic-density output.
 * @param grid_coordinates Optional interleaved xyz output.
 * @param grid_weights Optional quadrature-weight output.
 */
__global__ void pack_post_uvars_model_grid_features_kernel(
    std::size_t task_count, const GauXC::XCDeviceTask* tasks,
    const std::int64_t* point_offsets, std::int64_t total_points,
    double* density, double* density_gradient, double* kinetic,
    double* grid_coordinates, double* grid_weights) {
  const std::size_t task_index = blockIdx.y;
  if (task_index >= task_count) return;

  const auto& task = tasks[task_index];
  const std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= task.npts) return;

  const std::int64_t destination = point_offsets[task_index] + point;
  if (destination < 0 || destination >= total_points) return;

  if (density) {
    const auto density_alpha_beta = post_uvars_density(task, point);
    density[destination] = density_alpha_beta.alpha;
    density[total_points + destination] = density_alpha_beta.beta;
  }
  if (density_gradient) {
    for (std::int64_t direction = 0; direction < 3; ++direction) {
      const auto gradient_scalar_z =
          post_uvars_density_gradient(task, point, direction);
      const auto gradient_alpha_beta = cuda::scalar_z_to_alpha_beta(
          gradient_scalar_z.scalar, gradient_scalar_z.spin_z);
      density_gradient[direction * total_points + destination] =
          gradient_alpha_beta.alpha;
      density_gradient[(3 + direction) * total_points + destination] =
          gradient_alpha_beta.beta;
    }
  }
  if (kinetic) {
    const auto kinetic_alpha_beta = post_uvars_kinetic(task, point);
    kinetic[destination] = kinetic_alpha_beta.alpha;
    kinetic[total_points + destination] = kinetic_alpha_beta.beta;
  }
  if (grid_coordinates) {
    grid_coordinates[3 * destination] = task.points_x[point];
    grid_coordinates[3 * destination + 1] = task.points_y[point];
    grid_coordinates[3 * destination + 2] = task.points_z[point];
  }
  if (grid_weights) grid_weights[destination] = task.weights[point];
}

/**
 * @brief Unpack one model-grid point's potentials into a GauXC task.
 * @param task_count Number of tasks in @p tasks.
 * @param tasks Device tasks indexed by the y grid dimension.
 * @param point_offsets Device task-to-model-grid offsets.
 * @param total_points Number of points in each tensor channel.
 * @param density_potential Two-channel density-potential input.
 * @param density_gradient_potential Optional six-channel gradient input.
 * @param kinetic_potential Optional two-channel kinetic-potential input.
 */
__global__ void unpack_model_grid_potentials_kernel(
    std::size_t task_count, GauXC::XCDeviceTask* tasks,
    const std::int64_t* point_offsets, std::int64_t total_points,
    const double* density_potential, const double* density_gradient_potential,
    const double* kinetic_potential) {
  const std::size_t task_index = blockIdx.y;
  if (task_index >= task_count) return;

  auto& task = tasks[task_index];
  const std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= task.npts) return;

  const std::int64_t source = point_offsets[task_index] + point;
  if (source < 0 || source >= total_points) return;

  task.vrho_pos[point] = density_potential[source];
  task.vrho_neg[point] = density_potential[total_points + source];
  if (density_gradient_potential) {
    task.gamma_pp[point] = density_gradient_potential[source];
    task.gamma_pm[point] = density_gradient_potential[total_points + source];
    task.gamma_mm[point] =
        density_gradient_potential[2 * total_points + source];
    task.vgamma_pp[point] =
        density_gradient_potential[3 * total_points + source];
    task.vgamma_pm[point] =
        density_gradient_potential[4 * total_points + source];
    task.vgamma_mm[point] =
        density_gradient_potential[5 * total_points + source];
  }
  if (kinetic_potential) {
    task.vtau_pos[point] = kinetic_potential[source];
    task.vtau_neg[point] = kinetic_potential[total_points + source];
  }
}

/**
 * @brief Prepare one task point for a contracted weight derivative.
 * @param task_count Number of tasks in @p tasks.
 * @param tasks Device tasks indexed by the y grid dimension.
 * @param point_offsets Device task-to-model-grid offsets.
 * @param total_points Number of entries in @p dE_dw.
 * @param dE_dw Model energy derivatives with respect to grid weights.
 */
__global__ void prepare_model_grid_weight_derivatives_kernel(
    std::size_t task_count, GauXC::XCDeviceTask* tasks,
    const std::int64_t* point_offsets, std::int64_t total_points,
    const double* dE_dw) {
  const std::size_t task_index = blockIdx.y;
  if (task_index >= task_count) return;

  auto& task = tasks[task_index];
  const std::size_t point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= task.npts) return;

  const std::int64_t source = point_offsets[task_index] + point;
  if (source < 0 || source >= total_points) return;

  // GauXC's contracted partition derivative expects w_i * f_i. The model
  // boundary cotangent is f_i = dE/dw_i.
  task.eps[point] = dE_dw[source] * task.weights[point];
  task.den_s[point] = 1.0;
  task.den_z[point] = 0.0;
}

/** @brief Reduce one batch's model coordinate derivatives by atom. */
__global__ void accumulate_model_geometry_gradient_kernel(
    std::size_t atom_count, std::int64_t grid_size,
    const std::int64_t* atom_indices, const double* point_gradient,
    const double* coordinate_gradient, double* xc_gradient) {
  const std::size_t local_atom = blockIdx.x;
  const std::size_t direction = blockIdx.y;
  if (local_atom >= atom_count || direction >= 3) return;

  double value = 0.0;
  if (point_gradient) {
    for (std::int64_t point = threadIdx.x; point < grid_size;
         point += blockDim.x) {
      const auto batch_point =
          static_cast<std::int64_t>(local_atom) * grid_size + point;
      value += point_gradient[3 * batch_point + direction];
    }
  }

  __shared__ double partial_sums[256];
  partial_sums[threadIdx.x] = value;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    value = partial_sums[0];
    if (coordinate_gradient)
      value += coordinate_gradient[3 * local_atom + direction];
    const auto atom = atom_indices[local_atom];
    atomicAdd(xc_gradient + 3 * atom + direction, value);
  }
}

}  // namespace

void pack_post_uvars_model_grid_features(
    std::size_t task_count, std::int32_t max_points,
    GauXC::XCDeviceTask* tasks_device, const std::int64_t* point_offsets_device,
    std::int64_t total_points, double* density, double* density_gradient,
    double* kinetic, double* grid_coordinates, double* grid_weights,
    GauXC::device_queue queue) {
  if (task_count == 0) return;
  cudaStream_t stream = queue.queue_as<GauXC::util::cuda_stream>();
  constexpr int threads = 256;
  const dim3 blocks(GauXC::util::div_ceil(max_points, threads), task_count);
  pack_post_uvars_model_grid_features_kernel<<<blocks, threads, 0, stream>>>(
      task_count, tasks_device, point_offsets_device, total_points, density,
      density_gradient, kinetic, grid_coordinates, grid_weights);
  cuda::check_launch("Failed to launch model feature packing");
}

void unpack_model_grid_potentials(
    std::size_t task_count, std::int32_t max_points,
    GauXC::XCDeviceTask* tasks_device, const std::int64_t* point_offsets_device,
    std::int64_t total_points, const double* density_potential,
    const double* density_gradient_potential, const double* kinetic_potential,
    GauXC::device_queue queue) {
  if (task_count == 0) return;
  cudaStream_t stream = queue.queue_as<GauXC::util::cuda_stream>();
  constexpr int threads = 256;
  const dim3 blocks(GauXC::util::div_ceil(max_points, threads), task_count);
  unpack_model_grid_potentials_kernel<<<blocks, threads, 0, stream>>>(
      task_count, tasks_device, point_offsets_device, total_points,
      density_potential, density_gradient_potential, kinetic_potential);
  cuda::check_launch("Failed to launch model potential unpacking");
}

void prepare_model_grid_weight_derivatives(
    std::size_t task_count, std::int32_t max_points,
    GauXC::XCDeviceTask* tasks_device, const std::int64_t* point_offsets_device,
    std::int64_t total_points, const double* dE_dw, GauXC::device_queue queue) {
  if (task_count == 0) return;
  cudaStream_t stream = queue.queue_as<GauXC::util::cuda_stream>();
  constexpr int threads = 256;
  const dim3 blocks(GauXC::util::div_ceil(max_points, threads), task_count);
  prepare_model_grid_weight_derivatives_kernel<<<blocks, threads, 0, stream>>>(
      task_count, tasks_device, point_offsets_device, total_points, dE_dw);
  cuda::check_launch("Failed to launch model weight derivatives");
}

void accumulate_model_geometry_gradient(std::size_t atom_count,
                                        std::int64_t grid_size,
                                        const std::int64_t* atom_indices_device,
                                        const double* point_gradient,
                                        const double* coordinate_gradient,
                                        double* xc_gradient,
                                        GauXC::device_queue queue) {
  if (atom_count == 0 || (!point_gradient && !coordinate_gradient)) return;
  cudaStream_t stream = queue.queue_as<GauXC::util::cuda_stream>();
  constexpr unsigned int threads = 256;
  const dim3 blocks(atom_count, 3);
  accumulate_model_geometry_gradient_kernel<<<blocks, threads, 0, stream>>>(
      atom_count, grid_size, atom_indices_device, point_gradient,
      coordinate_gradient, xc_gradient);
  cuda::check_launch("Failed to launch model geometry-gradient accumulation");
}

}  // namespace SkalaXC
