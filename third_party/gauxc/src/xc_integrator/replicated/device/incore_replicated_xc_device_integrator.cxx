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
#include "incore_replicated_xc_device_integrator_integrate_den.hpp"
#include "incore_replicated_xc_device_integrator_exc.hpp"
#include "incore_replicated_xc_device_integrator_exc_vxc.hpp"
#include "incore_replicated_xc_device_integrator_exc_grad.hpp"
#include "incore_replicated_xc_device_integrator_exx.hpp"
#include "incore_replicated_xc_device_integrator_fxc_contraction.hpp"
#include "incore_replicated_xc_device_integrator_dd.hpp"
#include "incore_replicated_xc_device_integrator_onedft.hpp"
namespace GauXC  {
namespace detail {

template <typename ValueType>
IncoreReplicatedXCDeviceIntegrator<ValueType>::~IncoreReplicatedXCDeviceIntegrator() noexcept = default;



template class IncoreReplicatedXCDeviceIntegrator<double>;

}
}
