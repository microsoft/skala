#pragma once
/**
 * @file
 * @brief CUDA entry points for GauXC task and Skala model-grid transfers.
 */

#include "device/device_queue.hpp"
#include "device/xc_device_task.hpp"

#include <cstddef>
#include <cstdint>

namespace SkalaXC {

/**
 * @brief Pack post-U-variable GauXC task data into model-grid tensors.
 *
 * This entry point must be called after GauXC's UKS `eval_uvars_*`. At that
 * point `den_s`/`den_z` and `tau_s`/`tau_z` contain alpha/beta values, while
 * the directional `dden_s*`/`dden_z*` buffers remain scalar/z values.
 * @param task_count Number of tasks in the active batch.
 * @param max_points Maximum point count among the active tasks.
 * @param tasks_device Device pointer to the active GauXC task array.
 * @param point_offsets_device Device offsets from each task into model-grid
 *        order.
 * @param total_points Number of points in each model tensor channel.
 * @param density Optional device density tensor with two point channels.
 * @param density_gradient Optional six-channel device density-gradient tensor.
 * @param kinetic Optional two-channel device kinetic-density tensor.
 * @param grid_coordinates Optional interleaved xyz device tensor.
 * @param grid_weights Optional device quadrature-weight vector.
 * @param queue Device execution queue on which transfer work is enqueued.
 */
void pack_post_uvars_model_grid_features(
    std::size_t task_count, std::int32_t max_points,
    GauXC::XCDeviceTask* tasks_device, const std::int64_t* point_offsets_device,
    std::int64_t total_points, double* density, double* density_gradient,
    double* kinetic, double* grid_coordinates, double* grid_weights,
    GauXC::device_queue queue);

/**
 * @brief Unpack contiguous model potentials into GauXC task buffers.
 * @param task_count Number of tasks in the active batch.
 * @param max_points Maximum point count among the active tasks.
 * @param tasks_device Device pointer to the active GauXC task array.
 * @param point_offsets_device Device offsets from each task into model-grid
 *        order.
 * @param total_points Number of points in each model tensor channel.
 * @param density_potential Device density-potential tensor with two channels.
 * @param density_gradient_potential Optional six-channel device gradient
 *        potential tensor.
 * @param kinetic_potential Optional two-channel device kinetic potential.
 * @param queue Device execution queue on which transfer work is enqueued.
 */
void unpack_model_grid_potentials(
    std::size_t task_count, std::int32_t max_points,
    GauXC::XCDeviceTask* tasks_device, const std::int64_t* point_offsets_device,
    std::int64_t total_points, const double* density_potential,
    const double* density_gradient_potential, const double* kinetic_potential,
    GauXC::device_queue queue);

/**
 * @brief Populate task buffers for contracted molecular-weight derivatives.
 * @param task_count Number of tasks in the active batch.
 * @param max_points Maximum point count among the active tasks.
 * @param tasks_device Device pointer to the active GauXC task array.
 * @param point_offsets_device Device offsets from each task into model-grid
 *        order.
 * @param total_points Number of entries in @p dE_dw.
 * @param dE_dw Device `dE/dw` vector in model-grid order.
 * @param queue Device execution queue on which transfer work is enqueued.
 */
void prepare_model_grid_weight_derivatives(
    std::size_t task_count, std::int32_t max_points,
    GauXC::XCDeviceTask* tasks_device, const std::int64_t* point_offsets_device,
    std::int64_t total_points, const double* dE_dw, GauXC::device_queue queue);

/**
 * @brief Accumulate model point and coordinate derivatives by atom.
 * @param atom_count Number of atoms in the exact-size model batch.
 * @param grid_size Number of grid points belonging to each batch atom.
 * @param atom_indices_device Device global-atom indices for the batch.
 * @param point_gradient Optional contiguous device point derivatives shaped
 *        `(atom_count * grid_size, 3)`.
 * @param coordinate_gradient Optional contiguous device atom-coordinate
 *        derivatives shaped `(atom_count, 3)`.
 * @param xc_gradient Device atom-major XC gradient to update.
 * @param queue Device execution queue on which the operation is enqueued.
 */
void accumulate_model_geometry_gradient(std::size_t atom_count,
                                        std::int64_t grid_size,
                                        const std::int64_t* atom_indices_device,
                                        const double* point_gradient,
                                        const double* coordinate_gradient,
                                        double* xc_gradient,
                                        GauXC::device_queue queue);

}  // namespace SkalaXC
