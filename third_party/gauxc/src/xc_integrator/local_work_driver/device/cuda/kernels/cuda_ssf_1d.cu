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
#include "cuda_ssf_1d.hpp"
#include "device_specific/cuda_device_constants.hpp"
#include "common/integrator_constants.hpp"
#include <gauxc/util/div_ceil.hpp>
#include <numeric>

static constexpr auto eps_d = std::numeric_limits<double>::epsilon();

namespace GauXC {


// SIMT over points: 1D kernel
__global__ void modify_weights_ssf_kernel_1d(
        size_t                            npts,
        size_t                            natoms,
  const double*                           RAB,
        int32_t                           ldRAB,
  const double*                           coords,
  const double*                           dist_scratch,
        int32_t                           lddist,
  const int32_t*                          iparent_device,
  const double*                           dist_nearest_device,
        double*                           weights_device
) {

  // Frisch partition functions
  auto gFrisch = [](double x) {

    const double s_x  = x  * 1.5625; // / integrator::magic_ssf_factor<>;
    const double s_x2 = s_x  * s_x;
    const double s_x3 = s_x  * s_x2;
    const double s_x5 = s_x3 * s_x2;
    const double s_x7 = s_x5 * s_x2;

    //return (35.*(s_x - s_x3) + 21.*s_x5 - 5.*s_x7) / 16.;
    return ((35.)*(s_x - s_x3) + (21.)*s_x5 - (5.)*s_x7);
  };
  
  auto sFrisch = [&] (double x) {
    if( fabs(x) < integrator::magic_ssf_factor<> ) return (0.5 - (0.5/16.) * gFrisch(x));
    else if( x >= integrator::magic_ssf_factor<> ) return 0.;
    else                               return 1.;
  };

  constexpr double weight_tol = integrator::ssf_weight_tol;

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

      const double* const local_rab = RAB + iCenter * ldRAB;

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
  cudaStream_t stream ) {

  dim3 threads( cuda::max_threads_per_thread_block );
  dim3 blocks ( util::div_ceil( npts, threads.x ) );
  modify_weights_ssf_kernel_1d<<<blocks, threads, 0, stream>>>(
    npts, natoms, RAB, ldRAB, coords, dist, lddist, iparent, dist_nearest, weights
  );

}

__global__ void eval_weight_1st_deriv_contracted_ssf_kernel_1d(
        size_t                            npts,
        size_t                            natoms,
  const double*                           RAB,
        int32_t                           ldRAB,
  const double*                           coords,
  const double*                           points_x,
  const double*                           points_y,
  const double*                           points_z,
  const double*                           dist_scratch,
        int32_t                           lddist,
  const int32_t*                          iparent_device,
  const double*                           dist_nearest_device,
  const double*       __restrict__        w_times_f_device,
        double*       __restrict__        exc_grad_w_device
) {

  // Frisch partition functions
  auto gFrisch = [](double x) {

    const double s_x  = x  * 1.5625; // / integrator::magic_ssf_factor<>;
    const double s_x2 = s_x  * s_x;
    const double s_x3 = s_x  * s_x2;
    const double s_x5 = s_x3 * s_x2;
    const double s_x7 = s_x5 * s_x2;

    return ((35.)*(s_x - s_x3) + (21.)*s_x5 - (5.)*s_x7);
  };
  
  auto sFrisch = [&] (double x) {
    if( fabs(x) < integrator::magic_ssf_factor<> ) return (0.5 - (0.5/16.) * gFrisch(x));
    else if( x >= integrator::magic_ssf_factor<> ) return 0.;
    else                               return 1.;
  };
  
  auto tFrisch = [&](double x) {
    const double s_x  = x * 1.5625; // / integrator::magic_ssf_factor<>;
    const double s_x2 = s_x  * s_x;
    const double s_x3 = s_x  * s_x2;
    const double numerator = (35.) * (s_x3 + (3.) * s_x2 + (3.) * s_x + (1.));
    const double denominator = (x - integrator::magic_ssf_factor<>) * ((5.)*s_x3 + (20.)*s_x2 + (29.)*s_x + (16.));
    return numerator / denominator ;
  };

  constexpr double safe_magic_ssf_bound = integrator::magic_ssf_factor<> - 1e-4;
  constexpr double weight_tol = integrator::ssf_weight_tol;
  constexpr double w_times_f_thresh = 1.e-12;

  const int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int nt_x  = blockDim.x  * gridDim.x;

  for( int ipt = tid_x; ipt < npts; ipt += nt_x ) {

    const auto w_times_f_i = w_times_f_device[ipt];
    if (fabs(w_times_f_i) < w_times_f_thresh) continue; // weight derivative = 0 when p_A = 0
    const auto iParent = iparent_device[ipt];

    double sum = 0.; 
    double parent_weight = 0.;

    const double* const local_dist_scratch = dist_scratch + ipt * lddist;
    const double dist_cutoff = 0.18 * dist_nearest_device[ipt]; // 0.5 * (1-integrator::magic_ssf_factor<>) * task.dist_nearest
    if( local_dist_scratch[iParent] < dist_cutoff ) continue; //weight derivative = 0 when p_A = 1

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

      sum += parent_weight;
    }

    // caculate sum
    for( int iCenter = 0; iCenter < natoms; iCenter++ ) 
    if ( iParent != iCenter ) {
      const double ri = local_dist_scratch[ iCenter ];
      const double* const local_rab = RAB + iCenter * ldRAB;
      double ps = 1.;
      for( int jCenter = 0; jCenter < natoms; jCenter++ ) 
      if( ps > weight_tol ) {
        if( iCenter != jCenter ) {
        
          const double rj = local_dist_scratch[ jCenter ];
          const double mu = (ri - rj) * local_rab[ jCenter ]; // XXX: RAB is symmetric
          ps *= sFrisch( mu );
        }
      } else break;

      sum += ps;

    }

    double sum_inv = 1. / sum;

    const double point_x = points_x[ipt];
    const double point_y = points_y[ipt];
    const double point_z = points_z[ipt];

    // Now do derivative
    for( int iB = 0; iB < natoms; iB++ ) 
    if( iParent != iB ) 
    {
      double exc_grad_w_iBx = 0.0, exc_grad_w_iBy = 0.0, exc_grad_w_iBz = 0.0;

      const double* const local_Rinv_B = RAB + iB * ldRAB;
      const double rB = local_dist_scratch[ iB ];
      const double coords_B_x = coords[3*iB + 0];
      const double coords_B_y = coords[3*iB + 1];
      const double coords_B_z = coords[3*iB + 2];

      // first term
      const double rA = local_dist_scratch[ iParent ];
      const double rAB_inv = local_Rinv_B[ iParent ];
      const double mu_AB = (rA - rB) * rAB_inv; 
      if( fabs(mu_AB) < safe_magic_ssf_bound) {
        // first term is tFrisch(mu_AB) * (PA-Z)/Z * w_times_f_i * nabla_B mu_BA 
        double coef1 = tFrisch(mu_AB) * rAB_inv * (parent_weight - sum) * sum_inv * w_times_f_i / rB;
        exc_grad_w_iBx = coef1 * (coords_B_x - point_x + mu_AB * ( coords_B_x - coords[3*iParent + 0]) * rAB_inv * rB);
        exc_grad_w_iBy = coef1 * (coords_B_y - point_y + mu_AB * ( coords_B_y - coords[3*iParent + 1]) * rAB_inv * rB);
        exc_grad_w_iBz = coef1 * (coords_B_z - point_z + mu_AB * ( coords_B_z - coords[3*iParent + 2]) * rAB_inv * rB);
      }

      // second term and third term
      // first need to calculate PB
      double PB = 1.;
      for( int jCenter = 0; jCenter < natoms; jCenter++ )
      if( PB > weight_tol ) {
        if( iB != jCenter ) {
          const double rj = local_dist_scratch[ jCenter ];
          const double mu = (rB - rj) * local_Rinv_B[ jCenter ]; 
          PB *= sFrisch( mu );
        }
      } else break;

      if( PB >  weight_tol ) 
        for( int iC = 0; iC < natoms; iC++ ) {
          if (iB == iC) continue;
          const double rBC_inv = local_Rinv_B[iC];
          const double rC = local_dist_scratch[iC];
          const double mu_BC = (rB - rC) * rBC_inv;
          
          if(fabs(mu_BC) < safe_magic_ssf_bound){
            const double t_BC = tFrisch(mu_BC);
            const double coef = PB * t_BC * rBC_inv * sum_inv * w_times_f_i;

            const double coords_C_x = coords[3*iC + 0];
            const double coords_C_y = coords[3*iC + 1];
            const double coords_C_z = coords[3*iC + 2];

            // second term
            {
              const double rB_inv = 1. / rB;
              exc_grad_w_iBx -= coef * ((coords_B_x - point_x) * rB_inv - mu_BC * (coords_B_x - coords_C_x) * rBC_inv);
              exc_grad_w_iBy -= coef * ((coords_B_y - point_y) * rB_inv - mu_BC * (coords_B_y - coords_C_y) * rBC_inv);
              exc_grad_w_iBz -= coef * ((coords_B_z - point_z) * rB_inv - mu_BC * (coords_B_z - coords_C_z) * rBC_inv);
            }

            if(iC != iParent) {
              // third term
              const double rC_inv = 1. / rC;
              const double C_x = coef * ((coords_C_x - point_x) * rC_inv + mu_BC * (coords_C_x - coords_B_x) * rBC_inv);
              const double C_y = coef * ((coords_C_y - point_y) * rC_inv + mu_BC * (coords_C_y - coords_B_y) * rBC_inv);
              const double C_z = coef * ((coords_C_z - point_z) * rC_inv + mu_BC * (coords_C_z - coords_B_z) * rBC_inv);
              
              atomicAdd(exc_grad_w_device + 3*iC + 0, C_x);
              atomicAdd(exc_grad_w_device + 3*iC + 1, C_y);
              atomicAdd(exc_grad_w_device + 3*iC + 2, C_z);

              // Update parent atom
              atomicAdd(exc_grad_w_device + 3*iParent + 0, -C_x);
              atomicAdd(exc_grad_w_device + 3*iParent + 1, -C_y);
              atomicAdd(exc_grad_w_device + 3*iParent + 2, -C_z);
            }
          }
        }

        atomicAdd(exc_grad_w_device + 3*iB + 0, exc_grad_w_iBx);
        atomicAdd(exc_grad_w_device + 3*iB + 1, exc_grad_w_iBy);
        atomicAdd(exc_grad_w_device + 3*iB + 2, exc_grad_w_iBz);

        // Update parent atom
        atomicAdd(exc_grad_w_device + 3*iParent + 0, -exc_grad_w_iBx);
        atomicAdd(exc_grad_w_device + 3*iParent + 1, -exc_grad_w_iBy);
        atomicAdd(exc_grad_w_device + 3*iParent + 2, -exc_grad_w_iBz);

    }

  }

}



void eval_weight_1st_deriv_contracted_ssf_1d( int32_t npts, int32_t natoms, const double* RAB,
  int32_t ldRAB, const double* coords, 
  const double* points_x, const double* points_y, const double* points_z,
  const double* dist, int32_t lddist,
  const int32_t* iparent, const double* dist_nearest, const double* w_times_f,
  double* exc_grad_w, cudaStream_t stream){

  dim3 threads( cuda::max_threads_per_thread_block/4 );
  dim3 blocks ( util::div_ceil( npts, threads.x ) );
  eval_weight_1st_deriv_contracted_ssf_kernel_1d<<<blocks, threads, 0, stream>>>(
    npts, natoms, RAB, ldRAB, coords, points_x, points_y, points_z, dist, lddist, iparent, dist_nearest,
    w_times_f, exc_grad_w
  );

}


}
