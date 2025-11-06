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
#include "hip/hip_runtime.h"
#include "hip_ssf_1d.hpp"
#include "device_specific/hip_device_constants.hpp"
#include "common/integrator_constants.hpp"
#include <gauxc/util/div_ceil.hpp>
#include <numeric>
#include <iostream>
#include "exceptions/hip_exception.hpp"

static constexpr auto eps_d = std::numeric_limits<double>::epsilon();

namespace GauXC {


// SIMT over points: 1D kernel
__global__ void modify_weights_ssf_kernel_1d(
        size_t                            npts,
        size_t                            natoms,
  const double*                           RAB,
        size_t                            ldRAB,
  const double*                           coords,
  const double*                           dist_scratch,
        size_t                            lddist,
  const int32_t*                          iparent_device,
  const double*                           dist_nearest_device,
        double*                           weights_device
) {

  // Frisch partition functions
  auto gFrisch = [](double x) {

    const double s_x  = x / integrator::magic_ssf_factor<>;
    const double s_x2 = s_x  * s_x;
    const double s_x3 = s_x  * s_x2;
    const double s_x5 = s_x3 * s_x2;
    const double s_x7 = s_x5 * s_x2;

    return (35.*(s_x - s_x3) + 21.*s_x5 - 5.*s_x7) / 16.;
  };
  
  auto sFrisch = [&] (double x) {
    if( fabs(x) < integrator::magic_ssf_factor<> ) return 0.5 * (1. - gFrisch(x));
    else if( x >= integrator::magic_ssf_factor<> ) return 0.;
    else                               return 1.;
  };

  constexpr double weight_tol = 1e-10;

  const int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int nt_x  = blockDim.x  * gridDim.x;

  for( int ipt = tid_x; ipt < npts; ipt += nt_x ) {

    const auto iParent = iparent_device[ipt];

    double sum = 0.; 
    double parent_weight = 0.;

    const double* const local_dist_scratch = dist_scratch + ipt * lddist;
    const double dist_cutoff = 0.5 * (1 - integrator::magic_ssf_factor<> ) * 
      dist_nearest_device[ipt];
    if( local_dist_scratch[iParent] < dist_cutoff ) continue;


    // Do iParent First
    {

      const double ri = local_dist_scratch[ iParent ];
      const double* const local_rab = RAB + iParent * ldRAB;

      parent_weight = 1.;
      for( int jCenter = 0; jCenter < natoms; jCenter++ ) 
      if( parent_weight > weight_tol ) {
      if( iParent != jCenter ) {
      
        const double rj = local_dist_scratch[ jCenter ];

        const double mu = (ri - rj) * local_rab[ jCenter ]; // XXX: RAB is symmetric
        parent_weight *= sFrisch( mu );

      }
      } else break;

      //__syncwarp();
      sum += parent_weight;

    }

    if( parent_weight < eps_d ) {
      weights_device[ipt] = 0.;
      continue;
    }

    for( int iCenter = 0; iCenter < natoms; iCenter++ ) 
    if( iParent != iCenter ) {

      const double ri = local_dist_scratch[ iCenter ];

      const double* const local_rab = RAB + iCenter * natoms;

      double ps = 1.;
      for( int jCenter = 0; jCenter < natoms; jCenter++ ) 
      if( ps > weight_tol ) {
      if( iCenter != jCenter ) {
      
        const double rj = local_dist_scratch[ jCenter ];

        const double mu = (ri - rj) * local_rab[ jCenter ]; // XXX: RAB is symmetric
        ps *= sFrisch( mu );

      }
      } else break;

      //__syncwarp();
      sum += ps;

    }
    weights_device[ipt] *= parent_weight / sum;
  }

}

void partition_weights_ssf_1d( int32_t npts, int32_t natoms, const double* RAB,
  int32_t ldRAB, const double* coords, const double* dist, int32_t lddist,
  const int32_t* iparent, const double* dist_nearest, double* weights,
  hipStream_t stream ) {

  dim3 threads( hip::max_threads_per_thread_block );
  dim3 blocks ( util::div_ceil( npts, threads.x ) );
  hipLaunchKernelGGL(modify_weights_ssf_kernel_1d, dim3(blocks), dim3(threads), 0, stream, 
    npts, natoms, RAB, ldRAB, coords, dist, lddist, iparent, dist_nearest, weights
  );

}

}
