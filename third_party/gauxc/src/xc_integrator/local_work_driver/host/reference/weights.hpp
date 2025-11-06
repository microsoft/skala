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
#include "host/local_host_work_driver_pimpl.hpp"

namespace GauXC {

using task_iterator = detail::LocalHostWorkDriverPIMPL::task_iterator;

void reference_ssf_weights_host(
  const Molecule&        mol,
  const MolMeta&         meta,
  task_iterator          task_begin,
  task_iterator          task_end
);

void reference_becke_weights_host(
  const Molecule&        mol,
  const MolMeta&         meta,
  task_iterator          task_begin,
  task_iterator          task_end
);

void reference_lko_weights_host(
  const Molecule&        mol,
  const MolMeta&         meta,
  task_iterator          task_begin,
  task_iterator          task_end
);

void reference_becke_weights_1st_derivative_host(
  const Molecule&        mol,
  const MolMeta&         meta,
  const XCTask& task,
  double* weight_deri
);

void reference_ssf_weights_1st_derivative_host(
  const Molecule&        mol,
  const MolMeta&         meta,
  const XCTask& task,
  double* weight_deri
);

// Becke weights 1st derivative contracted with integrator
void reference_becke_weights_1std_contraction_host(
  const Molecule&        mol,
  const MolMeta&         meta,
  const XCTask& task,
  const double* w_times_f,
  double* exc_grad_w
);

// SSF weights 1st derivative contracted with integrator
void reference_ssf_weights_1std_contraction_host(
  const Molecule&        mol,
  const MolMeta&         meta,
  const XCTask& task,
  const double* w_times_f,
  double* exc_grad_w
);

}
