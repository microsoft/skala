/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 *
 * ---------------------------------------------------------------------------
 * SkalaXC adaptation: renamed from GauXC skala-branch
 * cuda/kernels/onedft.cu (onedft -> skala). SkalaXC ML VXC device kernels.
 * Compiled only when SKALAXC_ENABLE_CUDA is ON (requires a CUDA toolchain).
 * GauXC device headers/utilities are supplied by the in-tree `gauxc` target.
 * ---------------------------------------------------------------------------
 */
#include "common/skala.hpp"
#include "device/cuda_launch_check.hpp"
#include "device_specific/cuda_device_constants.hpp"
#include "device_specific/cuda_util.hpp"
#include "exceptions.hpp"
#include "spin_channels.cuh"
#include <gauxc/util/div_ceil.hpp>

namespace SkalaXC {

using GauXC::DEN_S;
using GauXC::DEN_Z;
using GauXC::density_id;
using GauXC::device_queue;
using GauXC::GGA;
using GauXC::integrator_xc_approx;
using GauXC::LDA;
using GauXC::MGGA_TAU;
using GauXC::XCDeviceTask;

/**
 * @brief Assemble one LDA Z-matrix element per CUDA thread.
 * @tparam den_selector Scalar or z-spin density channel.
 * @param ntasks Number of tasks in @p tasks_device.
 * @param tasks_device Tasks indexed by the z grid dimension.
 */
template <density_id den_selector>
__global__ void zmat_lda_vxc_skala_kernel(size_t ntasks,
                                          XCDeviceTask* tasks_device) {

  const int batch_idx = blockIdx.z;
  if (batch_idx >= ntasks) return;

  auto& task = tasks_device[batch_idx];
  const auto npts = task.npts;
  const auto nbf = task.bfn_screening.nbe;
  const auto* basis_eval_device = task.bf;

  auto* z_matrix_device = task.zmat;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (tid_x < npts and tid_y < nbf) {

    const size_t ibfoff = tid_y * npts + tid_x;
    const auto density_scalar_z = cuda::alpha_beta_to_scalar_z(
        task.vrho_pos[tid_x], task.vrho_neg[tid_x]);
    double density_potential = density_scalar_z.scalar;
    if constexpr (den_selector == DEN_Z)
      density_potential = density_scalar_z.spin_z;

    z_matrix_device[ibfoff] =
        0.5 * density_potential * basis_eval_device[ibfoff];
  }
}

/**
 * @brief Assemble one GGA Z-matrix element per CUDA thread.
 * @tparam den_selector Scalar or z-spin density channel.
 * @param ntasks Number of tasks in @p tasks_device.
 * @param tasks_device Tasks indexed by the z grid dimension.
 */
template <density_id den_selector>
__global__ void zmat_gga_vxc_skala_kernel(size_t ntasks,
                                          XCDeviceTask* tasks_device) {

  const int batch_idx = blockIdx.z;
  if (batch_idx >= ntasks) return;

  auto& task = tasks_device[batch_idx];
  const auto npts = task.npts;
  const auto nbf = task.bfn_screening.nbe;

  const double* dden_x_grad_a = task.gamma_pp;
  const double* dden_x_grad_b = task.vgamma_pp;
  const double* dden_y_grad_a = task.gamma_pm;
  const double* dden_y_grad_b = task.vgamma_pm;
  const double* dden_z_grad_a = task.gamma_mm;
  const double* dden_z_grad_b = task.vgamma_mm;

  const auto* basis_eval_device = task.bf;
  const auto* dbasis_x_eval_device = task.dbfx;
  const auto* dbasis_y_eval_device = task.dbfy;
  const auto* dbasis_z_eval_device = task.dbfz;

  auto* z_matrix_device = task.zmat;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (tid_x < npts and tid_y < nbf) {

    const size_t ibfoff = tid_y * npts + tid_x;

    const auto density_scalar_z = cuda::alpha_beta_to_scalar_z(
        task.vrho_pos[tid_x], task.vrho_neg[tid_x]);
    const auto x_scalar_z = cuda::alpha_beta_to_scalar_z(dden_x_grad_a[tid_x],
                                                         dden_x_grad_b[tid_x]);
    const auto y_scalar_z = cuda::alpha_beta_to_scalar_z(dden_y_grad_a[tid_x],
                                                         dden_y_grad_b[tid_x]);
    const auto z_scalar_z = cuda::alpha_beta_to_scalar_z(dden_z_grad_a[tid_x],
                                                         dden_z_grad_b[tid_x]);

    double density_potential = density_scalar_z.scalar;
    double x_fact = x_scalar_z.scalar;
    double y_fact = y_scalar_z.scalar;
    double z_fact = z_scalar_z.scalar;

    if constexpr (den_selector == DEN_Z) {
      density_potential = density_scalar_z.spin_z;
      x_fact = x_scalar_z.spin_z;
      y_fact = y_scalar_z.spin_z;
      z_fact = z_scalar_z.spin_z;
    }

    z_matrix_device[ibfoff] =
        x_fact * dbasis_x_eval_device[ibfoff] +
        y_fact * dbasis_y_eval_device[ibfoff] +
        z_fact * dbasis_z_eval_device[ibfoff] +
        0.5 * density_potential * basis_eval_device[ibfoff];
  }
}

/**
 * @brief Assemble one meta-GGA Z-matrix element per CUDA thread.
 * @tparam need_lapl Whether to include Laplacian contributions.
 * @tparam den_selector Scalar or z-spin density channel.
 * @param ntasks Number of tasks in @p tasks_device.
 * @param tasks_device Tasks indexed by the z grid dimension.
 */
template <bool need_lapl, density_id den_selector>
__global__ void zmat_mgga_vxc_skala_kernel(size_t ntasks,
                                           XCDeviceTask* tasks_device) {

  const int batch_idx = blockIdx.z;
  if (batch_idx >= ntasks) return;

  auto& task = tasks_device[batch_idx];
  const auto npts = task.npts;
  const auto nbf = task.bfn_screening.nbe;

  const double* vlapl_pos_device = task.vlapl_pos;
  const double* vlapl_neg_device = task.vlapl_neg;

  const double* dden_x_grad_a = task.gamma_pp;
  const double* dden_x_grad_b = task.vgamma_pp;
  const double* dden_y_grad_a = task.gamma_pm;
  const double* dden_y_grad_b = task.vgamma_pm;
  const double* dden_z_grad_a = task.gamma_mm;
  const double* dden_z_grad_b = task.vgamma_mm;

  const auto* basis_eval_device = task.bf;
  const auto* dbasis_x_eval_device = task.dbfx;
  const auto* dbasis_y_eval_device = task.dbfy;
  const auto* dbasis_z_eval_device = task.dbfz;
  const auto* d2basis_lapl_eval_device = task.d2bflapl;

  auto* z_matrix_device = task.zmat;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (tid_x < npts and tid_y < nbf) {

    const size_t ibfoff = tid_y * npts + tid_x;

    const auto density_scalar_z = cuda::alpha_beta_to_scalar_z(
        task.vrho_pos[tid_x], task.vrho_neg[tid_x]);
    const auto x_scalar_z = cuda::alpha_beta_to_scalar_z(dden_x_grad_a[tid_x],
                                                         dden_x_grad_b[tid_x]);
    const auto y_scalar_z = cuda::alpha_beta_to_scalar_z(dden_y_grad_a[tid_x],
                                                         dden_y_grad_b[tid_x]);
    const auto z_scalar_z = cuda::alpha_beta_to_scalar_z(dden_z_grad_a[tid_x],
                                                         dden_z_grad_b[tid_x]);

    double density_potential = density_scalar_z.scalar;
    double x_fact = x_scalar_z.scalar;
    double y_fact = y_scalar_z.scalar;
    double z_fact = z_scalar_z.scalar;

    if constexpr (den_selector == DEN_Z) {
      density_potential = density_scalar_z.spin_z;
      x_fact = x_scalar_z.spin_z;
      y_fact = y_scalar_z.spin_z;
      z_fact = z_scalar_z.spin_z;
    }

    auto val = x_fact * dbasis_x_eval_device[ibfoff] +
               y_fact * dbasis_y_eval_device[ibfoff] +
               z_fact * dbasis_z_eval_device[ibfoff] +
               0.5 * density_potential * basis_eval_device[ibfoff];

    if constexpr (need_lapl) {
      const auto laplacian_scalar_z = cuda::alpha_beta_to_scalar_z(
          vlapl_pos_device[tid_x], vlapl_neg_device[tid_x]);
      double laplacian_potential = laplacian_scalar_z.scalar;
      if constexpr (den_selector == DEN_Z)
        laplacian_potential = laplacian_scalar_z.spin_z;
      val += laplacian_potential * d2basis_lapl_eval_device[ibfoff];
    }

    z_matrix_device[ibfoff] = val;
  }
}

/**
 * @brief Launch Z-matrix assembly for one density channel.
 * @param ntasks Number of active tasks.
 * @param max_nbf Maximum screened basis size among tasks.
 * @param max_npts Maximum point count among tasks.
 * @param tasks_device Device pointer to active tasks.
 * @param scheme Density approximation controlling kernel selection.
 * @param sel Scalar or z-spin density channel.
 * @param queue Queue on which assembly is enqueued.
 */
void zmat_skala_vxc(size_t ntasks, int32_t max_nbf, int32_t max_npts,
                    XCDeviceTask* tasks_device, integrator_xc_approx scheme,
                    density_id sel, device_queue queue) {

  cudaStream_t stream = queue.queue_as<GauXC::util::cuda_stream>();

  dim3 threads(GauXC::cuda::warp_size, GauXC::cuda::max_warps_per_thread_block,
               1);
  dim3 blocks(GauXC::util::div_ceil(max_npts, threads.x),
              GauXC::util::div_ceil(max_nbf, threads.y), ntasks);
  if (scheme == LDA) {
    switch (sel) {
      case DEN_S:
        zmat_lda_vxc_skala_kernel<DEN_S>
            <<<blocks, threads, 0, stream>>>(ntasks, tasks_device);
        break;
      case DEN_Z:
        zmat_lda_vxc_skala_kernel<DEN_Z>
            <<<blocks, threads, 0, stream>>>(ntasks, tasks_device);
        break;
      default:
        SKALAXC_EXCEPTION("Skala VXC requires scalar or z-spin density");
    }
  } else if (scheme == GGA) {
    switch (sel) {
      case DEN_S:
        zmat_gga_vxc_skala_kernel<DEN_S>
            <<<blocks, threads, 0, stream>>>(ntasks, tasks_device);
        break;
      case DEN_Z:
        zmat_gga_vxc_skala_kernel<DEN_Z>
            <<<blocks, threads, 0, stream>>>(ntasks, tasks_device);
        break;
      default:
        SKALAXC_EXCEPTION("Skala VXC requires scalar or z-spin density");
    }
  } else if (scheme == MGGA_TAU) {
    switch (sel) {
      case DEN_S:
        zmat_mgga_vxc_skala_kernel<false, DEN_S>
            <<<blocks, threads, 0, stream>>>(ntasks, tasks_device);
        break;
      case DEN_Z:
        zmat_mgga_vxc_skala_kernel<false, DEN_Z>
            <<<blocks, threads, 0, stream>>>(ntasks, tasks_device);
        break;
      default:
        SKALAXC_EXCEPTION("Skala VXC requires scalar or z-spin density");
    }
  } else {
    SKALAXC_EXCEPTION("SKALA NYI for this scheme");
  }
  cuda::check_launch("Failed to launch Skala VXC assembly");
}

}  // namespace SkalaXC
