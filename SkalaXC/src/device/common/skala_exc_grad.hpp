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
 * device/common/onedft_exc_grad.hpp (onedft -> skala). Compiled only when
 * SKALAXC_ENABLE_CUDA is ON (requires a CUDA toolchain).
 * ---------------------------------------------------------------------------
 */
#pragma once
#include "device/device_queue.hpp"
#include "device/xc_device_task.hpp"

namespace SkalaXC {

/**
 * @brief Convert SkalaXC directional VXC buffers into inc_exc_grad format.
 * @param ntasks Number of tasks.
 * @param max_npts Maximum number of grid points across tasks.
 * @param tasks_device Device pointer to task array.
 * @param queue Device execution queue.
 *
 * After this call, dden_sx/sy/sz contain vds_x/y/z and
 * vgamma_pp/pm/mm are set to 1/0/1.
 */
void transform_skala_vxc_for_grad(size_t ntasks, int32_t max_npts,
                                  GauXC::XCDeviceTask* tasks_device,
                                  GauXC::device_queue queue);

}  // namespace SkalaXC
