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
#include "device/common/zmat_vxc.hpp"
#include <gauxc/util/div_ceil.hpp>
#include "device_specific/cuda_util.hpp"
#include "device_specific/cuda_device_constants.hpp"

namespace GauXC {


__global__ void zmat_lda_vxc_rks_kernel( size_t        ntasks,
                                     XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];
  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;
  const auto* vrho_device    = task.vrho;

  const auto* basis_eval_device = task.bf;

  auto* z_matrix_device = task.zmat;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nbf ) {

    const size_t ibfoff = tid_y * npts + tid_x;
    const double fact = 0.5 * vrho_device[tid_x];

    z_matrix_device[ ibfoff ] = fact * basis_eval_device[ ibfoff ];

  }

}













template<density_id den_selector>
__global__ void zmat_lda_vxc_uks_kernel( size_t        ntasks,
                                     XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];
  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;
  const double* vrho_pos_device    = task.vrho_pos;
  const double* vrho_neg_device    = task.vrho_neg;


  const auto* basis_eval_device = task.bf;


  auto* z_matrix_device = task.zmat;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nbf ) {

    const size_t ibfoff = tid_y * npts + tid_x;
    const double factp = 0.5 * vrho_pos_device[tid_x];
    const double factm = 0.5 * vrho_neg_device[tid_x];
    double sign = 1.0;
    if constexpr ( den_selector == DEN_Z )  sign = -1.0;
    
    z_matrix_device[ ibfoff ] = 0.5*(factp * basis_eval_device[ ibfoff ] + sign * factm * basis_eval_device[ ibfoff ]);
  }

}

template<density_id den_selector>
__global__ void zmat_lda_vxc_gks_kernel( size_t        ntasks,
                                     XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];
  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;
  const double* vrho_pos_device    = task.vrho_pos;
  const double* vrho_neg_device    = task.vrho_neg;

  double* K_device;
  if constexpr ( den_selector == DEN_Z ) K_device = task.K_z;
  if constexpr ( den_selector == DEN_Y ) K_device = task.K_y;
  if constexpr ( den_selector == DEN_X ) K_device = task.K_x;
  


  const auto* basis_eval_device = task.bf;


  auto* z_matrix_device = task.zmat;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nbf ) {
    const size_t ibfoff = tid_y * npts + tid_x;
    const double factp = 0.5 * vrho_pos_device[tid_x];
    const double factm = 0.5 * vrho_neg_device[tid_x];

    if constexpr ( den_selector == DEN_S ) {
      z_matrix_device[ ibfoff ] = 0.5*(factp * basis_eval_device[ ibfoff ] + factm * basis_eval_device[ ibfoff ]);
    }
    else {
      const double factk = 0.5 * (factp - factm);
      z_matrix_device[ ibfoff ] = K_device[ ibfoff ] * factk * basis_eval_device[ ibfoff ];
    }
  }

}









__global__ void zmat_gga_vxc_rks_kernel( size_t        ntasks,
                                     XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];
  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;
  const auto* vrho_device    = task.vrho;
  const auto* vgamma_device  = task.vgamma;
  const auto* den_x_eval_device = task.dden_sx;
  const auto* den_y_eval_device = task.dden_sy;
  const auto* den_z_eval_device = task.dden_sz;

  const auto* basis_eval_device = task.bf;
  const auto* dbasis_x_eval_device = task.dbfx;
  const auto* dbasis_y_eval_device = task.dbfy;
  const auto* dbasis_z_eval_device = task.dbfz;

  auto* z_matrix_device = task.zmat;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nbf ) {

    const size_t ibfoff = tid_y * npts + tid_x;
    const double fact_1 = 0.5 * vrho_device[tid_x]  ;
    const double fact_2 = 2.0 * vgamma_device[tid_x];

    const double dx = den_x_eval_device[ tid_x ] * dbasis_x_eval_device[ ibfoff ];
    const double dy = den_y_eval_device[ tid_x ] * dbasis_y_eval_device[ ibfoff ];
    const double dz = den_z_eval_device[ tid_x ] * dbasis_z_eval_device[ ibfoff ];

    z_matrix_device[ ibfoff ] = 
      fact_1 * basis_eval_device[ ibfoff ] + fact_2 * ( dx + dy + dz ); 

  }
}








template<density_id den_selector>
__global__ void zmat_gga_vxc_uks_kernel( size_t        ntasks,
                                     XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];
  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;

  const double* vrho_pos_device    = task.vrho_pos;
  const double* vrho_neg_device    = task.vrho_neg;
  const double* vgamma_pp_device   = task.vgamma_pp;
  const double* vgamma_pm_device   = task.vgamma_pm;
  const double* vgamma_mm_device   = task.vgamma_mm;

  const auto* den_pos_x_eval_device = task.dden_sx;
  const auto* den_pos_y_eval_device = task.dden_sy;
  const auto* den_pos_z_eval_device = task.dden_sz;
  const auto* den_neg_x_eval_device = task.dden_zx;
  const auto* den_neg_y_eval_device = task.dden_zy;
  const auto* den_neg_z_eval_device = task.dden_zz;


  const auto* basis_eval_device = task.bf;
  const auto* dbasis_x_eval_device = task.dbfx;
  const auto* dbasis_y_eval_device = task.dbfy;
  const auto* dbasis_z_eval_device = task.dbfz;

  auto* z_matrix_device = task.zmat;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nbf ) {

    const size_t ibfoff = tid_y * npts + tid_x;

    const double factp = 0.25 * vrho_pos_device[tid_x];
    const double factm = 0.25 * vrho_neg_device[tid_x];
    
    const auto gga_fact_pp  = vgamma_pp_device[tid_x];
    const auto gga_fact_pm  = vgamma_pm_device[tid_x];
    const auto gga_fact_mm  = vgamma_mm_device[tid_x];
    
    const auto gga_fact_1 = 0.5*(gga_fact_pp + gga_fact_pm + gga_fact_mm);
    const auto gga_fact_2 = 0.5*(gga_fact_pp - gga_fact_mm);
    const auto gga_fact_3 = 0.5*(gga_fact_pp - gga_fact_pm + gga_fact_mm);

    double sign = 1.0;

    double x_fact, y_fact, z_fact;

    if constexpr ( den_selector == DEN_S ) {
       x_fact = gga_fact_1 * den_pos_x_eval_device[ tid_x ] + gga_fact_2 * den_neg_x_eval_device[ tid_x ];
       y_fact = gga_fact_1 * den_pos_y_eval_device[ tid_x ] + gga_fact_2 * den_neg_y_eval_device[ tid_x ];
       z_fact = gga_fact_1 * den_pos_z_eval_device[ tid_x ] + gga_fact_2 * den_neg_z_eval_device[ tid_x ];
      

    }
    if constexpr ( den_selector == DEN_Z ) {
       sign = -1.0;
       x_fact = gga_fact_3 * den_neg_x_eval_device[ tid_x ] + gga_fact_2 * den_pos_x_eval_device[ tid_x ];
       y_fact = gga_fact_3 * den_neg_y_eval_device[ tid_x ] + gga_fact_2 * den_pos_y_eval_device[ tid_x ];
       z_fact = gga_fact_3 * den_neg_z_eval_device[ tid_x ] + gga_fact_2 * den_pos_z_eval_device[ tid_x ];

    }

    z_matrix_device[ ibfoff ] =   x_fact * dbasis_x_eval_device[ ibfoff ]      
                                + y_fact * dbasis_y_eval_device[ ibfoff ]
                                + z_fact * dbasis_z_eval_device[ ibfoff ] 
                                + (factp + sign * factm) * basis_eval_device[ ibfoff ];
  }
}




template<density_id den_selector>
__global__ void zmat_gga_vxc_gks_kernel( size_t        ntasks,
                                     XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];
  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;

  const double* vrho_pos_device    = task.vrho_pos;
  const double* vrho_neg_device    = task.vrho_neg;
  const double* vgamma_pp_device   = task.vgamma_pp;
  const double* vgamma_pm_device   = task.vgamma_pm;
  const double* vgamma_mm_device   = task.vgamma_mm;

  
  // for non-DEN_S
  double* K_device;
  double* H_device;
  if constexpr ( den_selector == DEN_Z ) { K_device = task.K_z; H_device = task.H_z; }
  if constexpr ( den_selector == DEN_Y ) { K_device = task.K_y; H_device = task.H_y; }
  if constexpr ( den_selector == DEN_X ) { K_device = task.K_x; H_device = task.H_x; }

  const auto* dden_sx_eval_device = task.dden_sx;
  const auto* dden_sy_eval_device = task.dden_sy;
  const auto* dden_sz_eval_device = task.dden_sz;
  const auto* dden_zx_eval_device = task.dden_zx;
  const auto* dden_zy_eval_device = task.dden_zy;
  const auto* dden_zz_eval_device = task.dden_zz;
  const auto* dden_yx_eval_device = task.dden_yx;
  const auto* dden_yy_eval_device = task.dden_yy;
  const auto* dden_yz_eval_device = task.dden_yz;
  const auto* dden_xx_eval_device = task.dden_xx;
  const auto* dden_xy_eval_device = task.dden_xy;
  const auto* dden_xz_eval_device = task.dden_xz;


  const auto* basis_eval_device = task.bf;
  const auto* dbasis_x_eval_device = task.dbfx;
  const auto* dbasis_y_eval_device = task.dbfy;
  const auto* dbasis_z_eval_device = task.dbfz;

  auto* z_matrix_device = task.zmat;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nbf ) {

    const size_t ibfoff = tid_y * npts + tid_x;

    const double fact_p =  0.5*vrho_pos_device[tid_x];
    const double fact_m =  0.5*vrho_neg_device[tid_x];
    
    const auto gga_fact_pp  = vgamma_pp_device[tid_x];
    const auto gga_fact_pm  = vgamma_pm_device[tid_x];
    const auto gga_fact_mm  = vgamma_mm_device[tid_x];
    
    const auto gga_fact_1 = 0.5*(gga_fact_pp + gga_fact_pm + gga_fact_mm);
    const auto gga_fact_2 = 0.5*(gga_fact_pp - gga_fact_mm);
    const auto gga_fact_3 = 0.5*(gga_fact_pp - gga_fact_pm + gga_fact_mm);

    double s_fact, x_fact, y_fact, z_fact;

    if constexpr ( den_selector == DEN_S ) {
      const double* Hz_device          = task.H_z;
      const double* Hy_device          = task.H_y;
      const double* Hx_device          = task.H_x;
      
      s_fact = 0.5 * (fact_p + fact_m);

      x_fact = gga_fact_1 * dden_sx_eval_device[ tid_x ]
             + gga_fact_2 * (Hz_device[ tid_x ] * dden_zx_eval_device[ tid_x ]
                          +  Hy_device[ tid_x ] * dden_yx_eval_device[ tid_x ]
                          +  Hx_device[ tid_x ] * dden_xx_eval_device[ tid_x ] );
      y_fact = gga_fact_1 * dden_sy_eval_device[ tid_x ]
             + gga_fact_2 * (Hz_device[ tid_x ] * dden_zy_eval_device[ tid_x ]
                          +  Hy_device[ tid_x ] * dden_yy_eval_device[ tid_x ]
                          +  Hx_device[ tid_x ] * dden_xy_eval_device[ tid_x ] );
      z_fact = gga_fact_1 * dden_sz_eval_device[ tid_x ]
                        + gga_fact_2 * (Hz_device[ tid_x ] * dden_zz_eval_device[ tid_x ]
                                     +  Hy_device[ tid_x ] * dden_yz_eval_device[ tid_x ]
                                     +  Hx_device[ tid_x ] * dden_xz_eval_device[ tid_x ] );
    }

    if constexpr ( den_selector == DEN_Z ) {
      s_fact  = K_device[ tid_x ] * 0.5 * (fact_p - fact_m);
      x_fact  = gga_fact_3 * dden_zx_eval_device[ tid_x ]
             +  gga_fact_2 * H_device[ tid_x ] * dden_sx_eval_device[ tid_x ];
      y_fact  = gga_fact_3 * dden_zy_eval_device[ tid_x ]
             +  gga_fact_2 * H_device[ tid_x ] * dden_sy_eval_device[ tid_x ];
      z_fact  = gga_fact_3 * dden_zz_eval_device[ tid_x ]
             +  gga_fact_2 * H_device[ tid_x ] * dden_sz_eval_device[ tid_x ];
    }

    if constexpr ( den_selector == DEN_Y ) {
      s_fact  = K_device[ tid_x ] * 0.5 * (fact_p - fact_m);
      x_fact  = gga_fact_3 * dden_yx_eval_device[ tid_x ]
             +  gga_fact_2 * H_device[ tid_x ] * dden_sx_eval_device[ tid_x ];
      y_fact  = gga_fact_3 * dden_yy_eval_device[ tid_x ]
             +  gga_fact_2 * H_device[ tid_x ] * dden_sy_eval_device[ tid_x ];
      z_fact  = gga_fact_3 * dden_yz_eval_device[ tid_x ]
             +  gga_fact_2 * H_device[ tid_x ] * dden_sz_eval_device[ tid_x ];
    }

    if constexpr ( den_selector == DEN_X ) {
      s_fact  = K_device[ tid_x ] * 0.5 * (fact_p - fact_m);
      x_fact  = gga_fact_3 * dden_xx_eval_device[ tid_x ]
             +  gga_fact_2 * H_device[ tid_x ] * dden_sx_eval_device[ tid_x ];
      y_fact  = gga_fact_3 * dden_xy_eval_device[ tid_x ]
             +  gga_fact_2 * H_device[ tid_x ] * dden_sy_eval_device[ tid_x ];
      z_fact  = gga_fact_3 * dden_xz_eval_device[ tid_x ]
             +  gga_fact_2 * H_device[ tid_x ] * dden_sz_eval_device[ tid_x ];
    }

    z_matrix_device[ ibfoff ] =   x_fact * dbasis_x_eval_device[ ibfoff ]      
                                + y_fact * dbasis_y_eval_device[ ibfoff ]
                                + z_fact * dbasis_z_eval_device[ ibfoff ] 
                                + s_fact *  basis_eval_device[ ibfoff ];

  }
}




template <bool need_lapl>
__global__ void zmat_mgga_vxc_rks_kernel( size_t        ntasks,
                                     XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];
  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;
  const auto* vrho_device    = task.vrho;
  const auto* vgamma_device  = task.vgamma;
  const double* vlapl_device = need_lapl ? task.vlapl : nullptr;
  const auto* den_x_eval_device = task.dden_sx;
  const auto* den_y_eval_device = task.dden_sy;
  const auto* den_z_eval_device = task.dden_sz;

  const auto* basis_eval_device = task.bf;
  const auto* dbasis_x_eval_device = task.dbfx;
  const auto* dbasis_y_eval_device = task.dbfy;
  const auto* dbasis_z_eval_device = task.dbfz;
  const double* d2basis_lapl_eval_device = 
    need_lapl ? task.d2bflapl : nullptr;
  

  auto* z_matrix_device = task.zmat;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nbf ) {

    const size_t ibfoff = tid_y * npts + tid_x;
    const double fact_1 = 0.5 * vrho_device[tid_x]  ;
    const double fact_2 = 2.0 * vgamma_device[tid_x];

    const double dx = den_x_eval_device[ tid_x ] * dbasis_x_eval_device[ ibfoff ];
    const double dy = den_y_eval_device[ tid_x ] * dbasis_y_eval_device[ ibfoff ];
    const double dz = den_z_eval_device[ tid_x ] * dbasis_z_eval_device[ ibfoff ];

    double val = 
      fact_1 * basis_eval_device[ ibfoff ] + fact_2 * ( dx + dy + dz ); 

    if constexpr (need_lapl) {
      val += vlapl_device[tid_x] * d2basis_lapl_eval_device[ibfoff];
    }

    z_matrix_device[ ibfoff ] = val;
  }
}

template<bool need_lapl, density_id den_selector>
__global__ void zmat_mgga_vxc_uks_kernel( size_t        ntasks,
                                     XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];
  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;

  const double* vrho_pos_device    = task.vrho_pos;
  const double* vrho_neg_device    = task.vrho_neg;
  const double* vlapl_pos_device    = task.vlapl_pos;
  const double* vlapl_neg_device    = task.vlapl_neg;
  const double* vgamma_pp_device   = task.vgamma_pp;
  const double* vgamma_pm_device   = task.vgamma_pm;
  const double* vgamma_mm_device   = task.vgamma_mm;

  const auto* den_pos_x_eval_device = task.dden_sx;
  const auto* den_pos_y_eval_device = task.dden_sy;
  const auto* den_pos_z_eval_device = task.dden_sz;
  const auto* den_neg_x_eval_device = task.dden_zx;
  const auto* den_neg_y_eval_device = task.dden_zy;
  const auto* den_neg_z_eval_device = task.dden_zz;


  const auto* basis_eval_device = task.bf;
  const auto* dbasis_x_eval_device = task.dbfx;
  const auto* dbasis_y_eval_device = task.dbfy;
  const auto* dbasis_z_eval_device = task.dbfz;
  const auto* d2basis_lapl_eval_device = task.d2bflapl;

  auto* z_matrix_device = task.zmat;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nbf ) {

    const size_t ibfoff = tid_y * npts + tid_x;

    const double factp = 0.25 * vrho_pos_device[tid_x];
    const double factm = 0.25 * vrho_neg_device[tid_x];
    
    const auto gga_fact_pp  = vgamma_pp_device[tid_x];
    const auto gga_fact_pm  = vgamma_pm_device[tid_x];
    const auto gga_fact_mm  = vgamma_mm_device[tid_x];
    
    const auto gga_fact_1 = 0.5*(gga_fact_pp + gga_fact_pm + gga_fact_mm);
    const auto gga_fact_2 = 0.5*(gga_fact_pp - gga_fact_mm);
    const auto gga_fact_3 = 0.5*(gga_fact_pp - gga_fact_pm + gga_fact_mm);

    double sign = 1.0;

    double x_fact, y_fact, z_fact;

    if constexpr ( den_selector == DEN_S ) {
       x_fact = gga_fact_1 * den_pos_x_eval_device[ tid_x ] + gga_fact_2 * den_neg_x_eval_device[ tid_x ];
       y_fact = gga_fact_1 * den_pos_y_eval_device[ tid_x ] + gga_fact_2 * den_neg_y_eval_device[ tid_x ];
       z_fact = gga_fact_1 * den_pos_z_eval_device[ tid_x ] + gga_fact_2 * den_neg_z_eval_device[ tid_x ];
    }
    if constexpr ( den_selector == DEN_Z ) {
       sign = -1.0;
       x_fact = gga_fact_3 * den_neg_x_eval_device[ tid_x ] + gga_fact_2 * den_pos_x_eval_device[ tid_x ];
       y_fact = gga_fact_3 * den_neg_y_eval_device[ tid_x ] + gga_fact_2 * den_pos_y_eval_device[ tid_x ];
       z_fact = gga_fact_3 * den_neg_z_eval_device[ tid_x ] + gga_fact_2 * den_pos_z_eval_device[ tid_x ];
    }

    auto val = x_fact * dbasis_x_eval_device[ ibfoff ]      
             + y_fact * dbasis_y_eval_device[ ibfoff ]
             + z_fact * dbasis_z_eval_device[ ibfoff ] 
             + (factp + sign * factm) * basis_eval_device[ ibfoff ];

    if constexpr (need_lapl) {
      const double lfactp = vlapl_pos_device[tid_x];
      const double lfactm = vlapl_neg_device[tid_x];

      val += 0.5 * (lfactp + sign * lfactm) * d2basis_lapl_eval_device[ ibfoff ];
    }

    z_matrix_device[ ibfoff ] = val;
  }
}



#define ZMAT_VXC_KERN(xc_approx) \
  cudaStream_t stream = queue.queue_as<util::cuda_stream>(); \
  dim3 threads(cuda::warp_size,cuda::max_warps_per_thread_block,1); \
  dim3 blocks( util::div_ceil( max_npts, threads.x ), \
               util::div_ceil( max_nbf,  threads.y ), \
               ntasks ); \
  switch( scheme ) { \
    case RKS: \
      zmat_##xc_approx##_vxc_rks_kernel<<< blocks, threads, 0, stream >>>( ntasks, tasks_device ); \
      break; \
    case UKS: \
      if ( sel == DEN_S )       zmat_##xc_approx##_vxc_uks_kernel<DEN_S><<< blocks, threads, 0, stream >>>( ntasks, tasks_device ); \
      else if ( sel == DEN_Z )  zmat_##xc_approx##_vxc_uks_kernel<DEN_Z><<< blocks, threads, 0, stream >>>( ntasks, tasks_device ); \
      else GAUXC_GENERIC_EXCEPTION( "zmat_##xc_approx##_vxc invalid density" ); \
      break; \
    case GKS: \
      if ( sel == DEN_S )       zmat_##xc_approx##_vxc_gks_kernel<DEN_S><<< blocks, threads, 0, stream >>>( ntasks, tasks_device ); \
      else if ( sel == DEN_Z )  zmat_##xc_approx##_vxc_gks_kernel<DEN_Z><<< blocks, threads, 0, stream >>>( ntasks, tasks_device ); \
      else if ( sel == DEN_Y )  zmat_##xc_approx##_vxc_gks_kernel<DEN_Y><<< blocks, threads, 0, stream >>>( ntasks, tasks_device ); \
      else if ( sel == DEN_X )  zmat_##xc_approx##_vxc_gks_kernel<DEN_X><<< blocks, threads, 0, stream >>>( ntasks, tasks_device ); \
      else GAUXC_GENERIC_EXCEPTION( "zmat_##xc_approx##_vxc invalid density" ); \
      break; \
    default: \
      GAUXC_GENERIC_EXCEPTION( "zmat_##xc_approx##_vxc invalid KS scheme" ); \
  }



void zmat_lda_vxc( size_t            ntasks,
                   int32_t           max_nbf,
                   int32_t           max_npts,
                   XCDeviceTask*     tasks_device,
                   integrator_ks_scheme scheme,
                   density_id sel,
                   device_queue queue ) {
ZMAT_VXC_KERN(lda)
}



void zmat_gga_vxc( size_t            ntasks,
                   int32_t           max_nbf,
                   int32_t           max_npts,
                   XCDeviceTask*     tasks_device,
                   integrator_ks_scheme scheme,
                   density_id sel,
                   device_queue queue ) {
ZMAT_VXC_KERN(gga)
}



void zmat_mgga_vxc( size_t            ntasks,
                    int32_t           max_nbf,
                    int32_t           max_npts,
                    XCDeviceTask*     tasks_device,
                    bool              do_lapl,
                    integrator_ks_scheme scheme,
                    density_id sel,
                    device_queue queue ) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>() ;


  dim3 threads(cuda::warp_size,cuda::max_warps_per_thread_block,1);
  dim3 blocks( util::div_ceil( max_npts, threads.x ),
               util::div_ceil( max_nbf,  threads.y ),
               ntasks );

  if(scheme == RKS) {
    if(do_lapl)
      zmat_mgga_vxc_rks_kernel<true><<< blocks, threads, 0, stream >>>( ntasks, tasks_device );
    else
      zmat_mgga_vxc_rks_kernel<false><<< blocks, threads, 0, stream >>>( ntasks, tasks_device );
  } else if(scheme == UKS) {
    switch(sel) {
      case DEN_S:
        if(do_lapl)
          zmat_mgga_vxc_uks_kernel<true, DEN_S><<< blocks, threads, 0, stream >>>( ntasks, tasks_device );
        else
          zmat_mgga_vxc_uks_kernel<false, DEN_S><<< blocks, threads, 0, stream >>>( ntasks, tasks_device );
        break;
      case DEN_Z:
        if(do_lapl)
          zmat_mgga_vxc_uks_kernel<true, DEN_Z><<< blocks, threads, 0, stream >>>( ntasks, tasks_device );
        else
          zmat_mgga_vxc_uks_kernel<false, DEN_Z><<< blocks, threads, 0, stream >>>( ntasks, tasks_device );
        break;
    }
  } else {
    GAUXC_GENERIC_EXCEPTION("MGGA + DEVICE + GKS NYI");
  }

}















template <bool need_lapl>
__global__ void mmat_mgga_vxc_rks_kernel( size_t        ntasks,
                                     XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];
  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;
  const auto* vtau_device    = task.vtau;
  const double* vlapl_device = need_lapl ? task.vlapl : nullptr;

  const auto* dbasis_x_eval_device = task.dbfx;
  const auto* dbasis_y_eval_device = task.dbfy;
  const auto* dbasis_z_eval_device = task.dbfz;

  auto* mmat_x = task.xmat_x;
  auto* mmat_y = task.xmat_y;
  auto* mmat_z = task.xmat_z;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nbf ) {

    const size_t ibfoff = tid_y * npts + tid_x;
    const double fact_1 = 0.25 * vtau_device[tid_x] + 
      (need_lapl ? vlapl_device[tid_x] : 0.0);

    mmat_x[ ibfoff ] = fact_1 * dbasis_x_eval_device[ ibfoff ]; 
    mmat_y[ ibfoff ] = fact_1 * dbasis_y_eval_device[ ibfoff ]; 
    mmat_z[ ibfoff ] = fact_1 * dbasis_z_eval_device[ ibfoff ]; 
  }
}

template <bool need_lapl, density_id id>
__global__ void mmat_mgga_vxc_uks_kernel( size_t        ntasks,
                                     XCDeviceTask* tasks_device ) {

  const int batch_idx = blockIdx.z;
  if( batch_idx >= ntasks ) return;

  auto& task = tasks_device[ batch_idx ];
  const auto npts            = task.npts;
  const auto nbf             = task.bfn_screening.nbe;
  const auto* vtau_pos_device    = task.vtau_pos;
  const auto* vtau_neg_device    = task.vtau_neg;
  const double* vlapl_pos_device = need_lapl ? task.vlapl_pos : nullptr;
  const double* vlapl_neg_device = need_lapl ? task.vlapl_neg : nullptr;

  const auto* dbasis_x_eval_device = task.dbfx;
  const auto* dbasis_y_eval_device = task.dbfy;
  const auto* dbasis_z_eval_device = task.dbfz;

  auto* mmat_x = task.xmat_x;
  auto* mmat_y = task.xmat_y;
  auto* mmat_z = task.xmat_z;

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < npts and tid_y < nbf ) {

    double sign = 1.0;
    if(id == DEN_Z) sign = -1;

    const size_t ibfoff = tid_y * npts + tid_x;
    const auto tfactp = 0.25 * vtau_pos_device[tid_x];
    const auto tfactm = 0.25 * vtau_neg_device[tid_x];
    const double fact_tau = 0.5 * (tfactp + sign * tfactm);
    double fact_lapl = 0.0;
    if(need_lapl) {
      const auto lfactp = vlapl_pos_device[tid_x];
      const auto lfactm = vlapl_neg_device[tid_x];
      fact_lapl = 0.5 * (lfactp + sign * lfactm);
    }
    const double fact_1 = fact_tau + fact_lapl;

    mmat_x[ ibfoff ] = fact_1 * dbasis_x_eval_device[ ibfoff ]; 
    mmat_y[ ibfoff ] = fact_1 * dbasis_y_eval_device[ ibfoff ]; 
    mmat_z[ ibfoff ] = fact_1 * dbasis_z_eval_device[ ibfoff ]; 
  }
}

//__global__ void print_zmat_stats( size_t            ntasks,
//                    XCDeviceTask*     tasks_device) {
//
//  for(size_t iT = 0; iT < ntasks; ++iT) {
//    auto& task = tasks_device[iT];
//    const auto npts            = task.npts;
//    const auto nbf             = task.bfn_screening.nbe;
//
//    const auto* zmat = task.zmat;
//    const auto* bmat = task.bf;
//    const auto* blmat = task.d2bflapl;
//  
//    double znrm = 0.0, bnrm = 0.0, blnrm = 0.0;
//    for(auto j = 0; j < npts*nbf; ++j) {
//      znrm += zmat[j] * zmat[j];
//      bnrm += bmat[j] * bmat[j];
//      blnrm += blmat[j] * blmat[j];
//    }
//
//    const auto* eps = task.eps;
//    const auto* vgamma = task.vgamma;
//    const auto* vtau   = task.vtau;
//    const auto* vlapl   = task.vlapl;
//    const auto* vrho   = task.vrho;
//    const auto* gamma = task.gamma;
//    const auto* tau   = task.tau;
//    const auto* lapl   = task.lapl;
//    const auto* rho   = task.den;
//    double enrm = 0.0, gnrm = 0.0, tnrm = 0.0, rnrm = 0.0, lnrm = 0.0;
//    double vgnrm = 0.0, vtnrm = 0.0, vrnrm = 0.0, vlnrm = 0.0;
//    for(auto j = 0; j < npts; ++j) {
//      enrm += eps[j] * eps[j];
//      vrnrm += vrho[j] * vrho[j];
//      vgnrm += vgamma[j] * vgamma[j];
//      vtnrm += vtau[j] * vtau[j];
//      vlnrm += vlapl[j] * vlapl[j];
//
//      rnrm += rho[j] * rho[j];
//      gnrm += gamma[j] * gamma[j];
//      tnrm += tau[j] * tau[j];
//      lnrm += lapl[j] * lapl[j];
//    }
//
//        printf("ITASK = %lu B = %.6e BL = %.6e R = %.6e G = %.6e T = %.6e L = %.6e E = %.6e VR = %.6e VG = %6e VT = %.6e VL = %.6e Z = %.6e \n", 
//          iT, bnrm, blnrm, rnrm, gnrm, tnrm, lnrm, enrm, vrnrm, vgnrm, vtnrm, vlnrm, znrm);
//  }
//
//}

void mmat_mgga_vxc( size_t            ntasks,
                    int32_t           max_nbf,
                    int32_t           max_npts,
                    XCDeviceTask*     tasks_device,
                    bool              do_lapl,
                    integrator_ks_scheme scheme,
                    density_id sel,
                    device_queue queue ) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>() ;


  dim3 threads(cuda::warp_size,cuda::max_warps_per_thread_block,1);
  dim3 blocks( util::div_ceil( max_npts, threads.x ),
               util::div_ceil( max_nbf,  threads.y ),
               ntasks );

  if(scheme == RKS) {
    if(do_lapl)
      mmat_mgga_vxc_rks_kernel<true><<< blocks, threads, 0, stream >>>( ntasks, tasks_device );
    else
      mmat_mgga_vxc_rks_kernel<false><<< blocks, threads, 0, stream >>>( ntasks, tasks_device );
  } else if(scheme == UKS) {
    switch(sel) {
      case DEN_S:
        if(do_lapl)
          mmat_mgga_vxc_uks_kernel<true, DEN_S><<< blocks, threads, 0, stream >>>( ntasks, tasks_device );
        else
          mmat_mgga_vxc_uks_kernel<false, DEN_S><<< blocks, threads, 0, stream >>>( ntasks, tasks_device );
        break;
      case DEN_Z:
        if(do_lapl)
          mmat_mgga_vxc_uks_kernel<true, DEN_Z><<< blocks, threads, 0, stream >>>( ntasks, tasks_device );
        else
          mmat_mgga_vxc_uks_kernel<false, DEN_Z><<< blocks, threads, 0, stream >>>( ntasks, tasks_device );
        break;
    }
  } else {
    GAUXC_GENERIC_EXCEPTION("MGGA + DEVICE + GKS NYI");
  }
  

  //print_zmat_stats<<<1,1,0,stream>>>(ntasks,tasks_device);
}

}

