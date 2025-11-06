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
#include "shell_batched_replicated_xc_device_integrator.hpp"
#include "shell_batched_replicated_xc_integrator_integrate_den.hpp"
#include "shell_batched_replicated_xc_integrator_exc.hpp"
#include "shell_batched_replicated_xc_integrator_exc_vxc.hpp"
#include "shell_batched_replicated_xc_integrator_exc_grad.hpp"
#include "shell_batched_replicated_xc_integrator_exx.hpp"
#include "shell_batched_replicated_xc_integrator_fxc_contraction.hpp"
#include "shell_batched_replicated_xc_integrator_dd_psi.hpp"
#include "shell_batched_replicated_xc_integrator_dd_psi_potential.hpp"

namespace GauXC  {
namespace detail {

template <typename ValueType>
ShellBatchedReplicatedXCDeviceIntegrator<ValueType>::~ShellBatchedReplicatedXCDeviceIntegrator() noexcept = default;

template class ShellBatchedReplicatedXCDeviceIntegrator<double>;

}
}
