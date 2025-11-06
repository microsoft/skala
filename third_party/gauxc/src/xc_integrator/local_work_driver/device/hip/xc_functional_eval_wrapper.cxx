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
#include "device/common/xc_functional_eval_wrapper.hpp"
#include "device_specific/hip_util.hpp"

namespace GauXC {

void eval_kern_exc_vxc_lda( const functional_type& func, size_t npts,
  const double* rho, double* eps, double* vrho, device_queue queue ) {

  hipStream_t stream = queue.queue_as<util::hip_stream>();
  func.eval_exc_vxc_device( npts, rho, eps, vrho, stream );

}

void eval_kern_exc_vxc_gga( const functional_type& func, size_t npts,
  const double* rho, const double* gamma, double* eps, double* vrho,
  double* vgamma, device_queue queue ) {

  hipStream_t stream = queue.queue_as<util::hip_stream>();
  func.eval_exc_vxc_device( npts, rho, gamma, eps, vrho, vgamma, stream );

}

}
