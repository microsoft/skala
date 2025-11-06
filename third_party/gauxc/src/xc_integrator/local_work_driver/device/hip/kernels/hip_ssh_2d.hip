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
#include "hip_extensions.hpp"
#include "device_specific/hip_device_constants.hpp"
#include "device/hip/hip_aos_scheme1.hpp"
#include "common/integrator_constants.hpp"
#include <numeric>

inline constexpr static auto eps_d = std::numeric_limits<double>::epsilon();

namespace GauXC {

__device__ __inline__ double gFrisch(double x) {
  // Frisch partition functions
//  const double s_x  = x / integrator::magic_ssf_factor<>;
  const double s_x  = x * 1.5625;
  const double s_x2 = s_x  * s_x;
  const double s_x3 = s_x  * s_x2;
  const double s_x5 = s_x3 * s_x2;
  const double s_x7 = s_x5 * s_x2;

  return ((35.) *(s_x - s_x3) + (21.) *s_x5 - (5.) *s_x7);
}


__device__ __inline__ double sFrisch(double x) {
    //double frisch_val = (0.5 - (0.5/ 16.0) * gFrisch(x));

    if( fabs(x) < integrator::magic_ssf_factor<> ) return (0.5 - (0.5/ 16.0) * gFrisch(x));
    else if( x >= integrator::magic_ssf_factor<> ) return 0.;
    else                               return 1.;
}

template <uint32_t weight_unroll,             // Unrolling factor
          uint32_t weight_thread_block,       // Number of threads / thread block
          uint32_t weight_thread_block_per_sm // Thread blocks / SM
          >
__global__ __launch_bounds__(weight_thread_block, weight_thread_block_per_sm)
void modify_weights_ssf_kernel_2d( int32_t npts, int32_t natoms, 
  const double* RAB, size_t ldRAB, const double* coords, const double* dist,
  size_t lddist, const int32_t* iparent, const double* dist_nearest,
  double* weights ) {

  constexpr uint32_t warps_per_thread_block = weight_thread_block / hip::warp_size;
  static_assert( weight_unroll == 4 );

  constexpr double weight_tol = 1e-10;
  int natom_block = ((natoms + blockDim.x - 1) / blockDim.x) * blockDim.x;

  const int tid_x = threadIdx.y + blockIdx.y * blockDim.y;
  const int nt_x  = blockDim.y  * gridDim.y;


  __shared__ int jCounter_sm[hip::max_warps_per_thread_block];
  int* jCounter = reinterpret_cast<int *>(jCounter_sm) + threadIdx.y;

  // Each warp will work together on a point
  for( int ipt = tid_x; ipt < npts; ipt += nt_x ) {
  #if 1

    const auto iParent = iparent[ipt];

    double sum = 0.; 
    double parent_weight = 0.;

    const double* const local_dist_scratch = dist + ipt * lddist;
    const double dist_cutoff = 0.5 * (1 - integrator::magic_ssf_factor<> ) * 
      dist_nearest[ipt];
    if( local_dist_scratch[iParent] < dist_cutoff ) continue;

    // Do iParent First
    {

      const double ri = local_dist_scratch[ iParent ];
      const double* const local_rab = RAB + iParent * ldRAB;

      parent_weight = 1.;
      for( int jCenter = threadIdx.x; jCenter < natom_block; jCenter+=blockDim.x ) {
        double contribution = 1.0;
        if (jCenter < natoms && iParent != jCenter) {
          const double rj = local_dist_scratch[ jCenter ];
          const double mu = (ri - rj) * local_rab[ jCenter ]; // XXX: RAB is symmetric
          contribution = sFrisch( mu );
        }
        contribution = hip::warp_reduce_prod<hip::warp_size>(contribution);

        parent_weight *= contribution;

        if (parent_weight < weight_tol) break;
      }
    }

    if( parent_weight < eps_d ) {
      if (threadIdx.x == 0)
        weights[ipt] = 0.;
      continue;
    }

    // Initialize each counter to 0
    if (threadIdx.x == 0) {
      jCounter[0] = 0;
    }

    // Each thread will process an iCenter. Atomic operations are used to assign
    // an iCenter value to each thread.
    int iCenter = atomicAdd(jCounter, 1);
    if (iCenter >= iParent) iCenter++; // iCenter == iParent is skipped

    // The entire warp processes the same jCenter value at the same time
    int jCenter = 0;

    const double* local_rab = RAB + iCenter * ldRAB;
    double ri = local_dist_scratch[ iCenter ];
    double ps = 1.;
    int iCount = 0; 
    int cont = (iCenter < natoms);

    // We will continue iterating until all of the threads have cont set to 0
    while (__any(cont)) {
      if (cont) {
        double2 rj[weight_unroll/2];
        double2 rab_val[weight_unroll/2];
        double mu[weight_unroll];
        iCount += weight_unroll;

        #pragma unroll
        for (int k = 0; k < weight_unroll/2; k++) {
          rj[k]      = *((double2*)(local_dist_scratch + jCenter) + k);
          rab_val[k] = *((double2*)(local_rab          + jCenter) + k); 
        }

        #pragma unroll
        for (int k = 0; k < weight_unroll/2; k++) {
          mu[2*k+0] = (ri - rj[k].x) * rab_val[k].x; // XXX: RAB is symmetric
          mu[2*k+1] = (ri - rj[k].y) * rab_val[k].y; 
        }

        #pragma unroll
        for (int k = 0; k < weight_unroll; k++) {
          if((iCenter != jCenter + k) && (jCenter + k < natoms)) {
            mu[k] = sFrisch( mu[k] );
            ps *= mu[k];
          }
        }

        // A thread is done with a iCenter based on 2 conditions. Weight tolerance
        // Or if it has seen all of the jCenters
        if( !(ps > weight_tol && iCount < lddist )) {
          // In the case were the thread is done, it begins processing another iCenter
          sum += ps;
          iCenter = atomicAdd(jCounter, 1);
          if (iCenter >= iParent) iCenter++;

          // If there are no more iCenters left to process, it signals it is ready to exit
          cont = (iCenter < natoms);
          ri = local_dist_scratch[ iCenter ];
          local_rab = RAB + iCenter * ldRAB;
          ps = 1.;
          iCount = 0;
        }
      }
      // Wraps jCenter around. This was faster than modulo
      jCenter += weight_unroll;
      jCenter = (jCenter < ldRAB) ? jCenter : 0;
    }

    // All of the threads then sum their contributions. Only thread 0 needs to add the parent
    // contribution.
    sum = hip::warp_reduce_sum<hip::warp_size>(sum);
    if (threadIdx.x == 0) {
      sum += parent_weight;
      weights[ipt] *= parent_weight / sum;
    }

#endif
  }

}

void partition_weights_ssf_2d( int32_t npts, int32_t natoms, const double* RAB,
  int32_t ldRAB, const double* coords, const double* dist, int32_t lddist,
  const int32_t* iparent, const double* dist_nearest, double* weights,
  hipStream_t stream ) {

  constexpr auto weight_unroll =
    alg_constants::HipAoSScheme1::weight_unroll;
  constexpr auto weight_thread_block =
    alg_constants::HipAoSScheme1::weight_thread_block;
  constexpr auto weight_thread_block_per_sm =
    alg_constants::HipAoSScheme1::weight_thread_block_per_sm;

  // Get the number of CUs on the device
  int num_sm;
  int dev_id = 0;
  hipDeviceGetAttribute(&num_sm, hipDeviceAttributeMultiprocessorCount, dev_id);

  // Modify weights
  dim3 threads( hip::warp_size, weight_thread_block / hip::warp_size );
  dim3 blocks ( 1, num_sm * weight_thread_block_per_sm );
  modify_weights_ssf_kernel_2d
    <weight_unroll, weight_thread_block, weight_thread_block_per_sm>
    <<< blocks, threads, 0, stream >>> (
   npts, natoms, RAB, ldRAB, coords, dist, lddist, iparent, dist_nearest, weights);

}


}
