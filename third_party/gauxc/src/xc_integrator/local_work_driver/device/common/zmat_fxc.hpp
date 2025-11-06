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
 */
#include "device/xc_device_task.hpp"
#include "device/xc_device_data.hpp"
#include "device/device_queue.hpp"

namespace GauXC {

void zmat_lda_fxc( size_t        ntasks,
                   int32_t       max_nbf,
                   int32_t       max_npts,
                   XCDeviceTask* tasks_device,
                   density_id sel,
                   device_queue queue );

void zmat_gga_fxc( size_t        ntasks,
                   int32_t       max_nbf,
                   int32_t       max_npts,
                   XCDeviceTask* tasks_device,
                   density_id sel,
                   device_queue queue );

void zmat_mgga_fxc( size_t        ntasks,
                    int32_t       max_nbf,
                    int32_t       max_npts,
                    XCDeviceTask* tasks_device,
                    bool          do_lapl,
                    density_id sel,
                    device_queue queue );

void mmat_mgga_fxc( size_t        ntasks,
                    int32_t       max_nbf,
                    int32_t       max_npts,
                    XCDeviceTask* tasks_device,
                    bool          do_lapl,
                    density_id sel,
                    device_queue queue );
}
