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
#include "reference_replicated_xc_host_integrator_integrate_den.hpp"
#include "reference_replicated_xc_host_integrator_exc.hpp"
#include "reference_replicated_xc_host_integrator_exc_vxc.hpp"
#include "reference_replicated_xc_host_integrator_exc_grad.hpp"
#include "reference_replicated_xc_host_integrator_exx.hpp"
#include "reference_replicated_xc_host_integrator_fxc_contraction.hpp"
#include "reference_replicated_xc_host_integrator_dd_psi.hpp"
#include "reference_replicated_xc_host_integrator_dd_psi_potential.hpp"
#include "reference_replicated_xc_host_integrator_onedft.hpp"

namespace GauXC::detail {

template <typename ValueType>
ReferenceReplicatedXCHostIntegrator<ValueType>::~ReferenceReplicatedXCHostIntegrator() noexcept = default;

template class ReferenceReplicatedXCHostIntegrator<double>;

}
