/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 *
 * ---------------------------------------------------------------------------
 * SkalaXC adaptation: renamed from GauXC skala-branch
 * device/common/onedft.hpp (onedft -> skala). Declares the SkalaXC ML VXC
 * device kernels. Operates on GauXC-master XCDeviceTask buffers; compiled only
 * when SKALAXC_ENABLE_CUDA is ON (requires a CUDA toolchain).
 * ---------------------------------------------------------------------------
 */
#pragma once
#include "device/device_queue.hpp"
#include "device/xc_device_data.hpp"
#include "device/xc_device_task.hpp"

namespace SkalaXC {

/**
 * @brief Assemble the SkalaXC Z-matrix from per-point ML XC derivatives.
 * @param ntasks Number of tasks.
 * @param max_nbf Maximum basis size across tasks.
 * @param max_npts Maximum number of grid points across tasks.
 * @param tasks_device Device pointer to task array.
 * @param scheme Density approximation scheme.
 * @param sel Density selector.
 * @param queue Device execution queue.
 */
void zmat_skala_vxc(size_t ntasks, int32_t max_nbf, int32_t max_npts,
                    GauXC::XCDeviceTask* tasks_device,
                    GauXC::integrator_xc_approx scheme, GauXC::density_id sel,
                    GauXC::device_queue queue);
}  // namespace SkalaXC
