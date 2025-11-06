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
#include "device/common/increment_exc_grad.hpp"
#include "cuda_extensions.hpp"
#include "device_specific/cuda_device_constants.hpp"
#include <gauxc/util/div_ceil.hpp>
#include "device_specific/cuda_util.hpp"

namespace GauXC {

template<bool with_weight_derivatives>
__global__ __launch_bounds__(1024,1) void increment_exc_grad_lda_rks_kernel(
  uint32_t                        nshell,
  ShellToTaskDevice* __restrict__ shell_to_task,
  XCDeviceTask*      __restrict__ device_tasks,
  double*            __restrict__ EXC_GRAD
) {

  for( uint32_t ish = blockIdx.z; ish < nshell; ish += gridDim.z ) {
    const uint32_t ntasks      = shell_to_task[ish].ntask;
    const auto shell           = shell_to_task[ish].shell_device;
    const auto task_idx        = shell_to_task[ish].task_idx_device;
    const auto task_shell_offs = shell_to_task[ish].task_shell_offs_device;
    const int iCen = shell_to_task[ish].center_idx;
    const uint32_t shsz   = shell->size();

    const int global_warp_id = 
      (threadIdx.x + blockIdx.x*blockDim.x) / cuda::warp_size;
    const int nwarp_global   = max((blockDim.x*gridDim.x) / cuda::warp_size,1);

    double g_acc_x(0), g_acc_y(0), g_acc_z(0);
    for( uint32_t itask = global_warp_id; itask < ntasks; itask += nwarp_global ) {
      
      const auto*    task   = device_tasks + task_idx[itask];
      const uint32_t npts   = task->npts;
      const size_t   shoff  = task_shell_offs[itask] * npts;
      const int iParent     = task->iParent;
      if constexpr( with_weight_derivatives ) {
        if( iCen == iParent ) 
          continue;
      }
      double g_acc_x_task(0), g_acc_y_task(0), g_acc_z_task(0);

      const auto* __restrict__ basis_x_eval = task->dbfx + shoff;
      const auto* __restrict__ basis_y_eval = task->dbfy + shoff;
      const auto* __restrict__ basis_z_eval = task->dbfz + shoff;

      const auto* __restrict__  xmat = task->zmat + shoff;
      const auto* __restrict__  vrho = task->vrho;

      #pragma unroll 1
      for( uint32_t ipt = threadIdx.x % cuda::warp_size; 
           ipt < npts; 
           ipt += cuda::warp_size ) {

        const double vrho_i = vrho[ipt];
        for( uint32_t ibf = 0; ibf < shsz; ++ibf ) {
          const double z_mu_i    = vrho_i * xmat[ipt + ibf*npts];
          const double dbfx_mu_i = basis_x_eval[ipt + ibf*npts];
          const double dbfy_mu_i = basis_y_eval[ipt + ibf*npts];
          const double dbfz_mu_i = basis_z_eval[ipt + ibf*npts];

          g_acc_x_task += z_mu_i * dbfx_mu_i;
          g_acc_y_task += z_mu_i * dbfy_mu_i;
          g_acc_z_task += z_mu_i * dbfz_mu_i;
        } // Loop over bfns within a shell

      } // Loop over points

      g_acc_x += g_acc_x_task;
      g_acc_y += g_acc_y_task;
      g_acc_z += g_acc_z_task;

      //write to Parent atom with translational invariance
      if constexpr( with_weight_derivatives ) {
        atomicAdd( EXC_GRAD + 3*iParent + 0, 2.0 * g_acc_x_task );
        atomicAdd( EXC_GRAD + 3*iParent + 1, 2.0 * g_acc_y_task );
        atomicAdd( EXC_GRAD + 3*iParent + 2, 2.0 * g_acc_z_task );
      }

    } // Loop over tasks assigned to shell

    constexpr auto warp_size = cuda::warp_size;
    g_acc_x = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_x );
    g_acc_y = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_y );
    g_acc_z = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_z );

    if( (threadIdx.x % cuda::warp_size) == 0 ) {
      
      atomicAdd( EXC_GRAD + 3*iCen + 0, g_acc_x );
      atomicAdd( EXC_GRAD + 3*iCen + 1, g_acc_y );
      atomicAdd( EXC_GRAD + 3*iCen + 2, g_acc_z );
    }

  } // Loop over shells

}

template<bool with_weight_derivatives>
__global__ __launch_bounds__(1024,1) void increment_exc_grad_lda_uks_kernel(
  uint32_t                        nshell,
  ShellToTaskDevice* __restrict__ shell_to_task,
  XCDeviceTask*      __restrict__ device_tasks,
  double*            __restrict__ EXC_GRAD
) {

  for( uint32_t ish = blockIdx.z; ish < nshell; ish += gridDim.z ) {
    const uint32_t ntasks      = shell_to_task[ish].ntask;
    const auto shell           = shell_to_task[ish].shell_device;
    const auto task_idx        = shell_to_task[ish].task_idx_device;
    const auto task_shell_offs = shell_to_task[ish].task_shell_offs_device;
    const int  iCen            = shell_to_task[ish].center_idx;
    const uint32_t shsz   = shell->size();

    const int global_warp_id = 
      (threadIdx.x + blockIdx.x*blockDim.x) / cuda::warp_size;
    const int nwarp_global   = max((blockDim.x*gridDim.x) / cuda::warp_size,1);

    double g_acc_x(0), g_acc_y(0), g_acc_z(0);
    for( uint32_t itask = global_warp_id; itask < ntasks; itask += nwarp_global ) {
      
      const auto*    task   = device_tasks + task_idx[itask];
      const uint32_t npts   = task->npts;
      const size_t   shoff  = task_shell_offs[itask] * npts;
      const int    iParent  = task->iParent;
      if constexpr( with_weight_derivatives ) {
        if( iCen == iParent )
          continue;
      }
      double g_acc_x_task(0), g_acc_y_task(0), g_acc_z_task(0);

      const auto* __restrict__ basis_x_eval = task->dbfx + shoff;
      const auto* __restrict__ basis_y_eval = task->dbfy + shoff;
      const auto* __restrict__ basis_z_eval = task->dbfz + shoff;

      const auto* __restrict__  xmatS = task->xmatS + shoff;
      const auto* __restrict__  xmatZ = task->xmatZ + shoff;
      const auto* __restrict__  vrhop = task->vrho_pos;
      const auto* __restrict__  vrhom = task->vrho_neg;

      #pragma unroll 1
      for( uint32_t ipt = threadIdx.x % cuda::warp_size; 
           ipt < npts; 
           ipt += cuda::warp_size ) {

        const double vrhop_i = vrhop[ipt];
        const double vrhom_i = vrhom[ipt];

        const auto vrhoS_i = 0.5 * (vrhop_i + vrhom_i);
        const auto vrhoZ_i = 0.5 * (vrhop_i - vrhom_i);
        for( uint32_t ibf = 0; ibf < shsz; ++ibf ) {
          const double zS_mu_i    = vrhoS_i * xmatS[ipt + ibf*npts];
          const double zZ_mu_i    = vrhoZ_i * xmatZ[ipt + ibf*npts];
          const double dbfx_mu_i = basis_x_eval[ipt + ibf*npts];
          const double dbfy_mu_i = basis_y_eval[ipt + ibf*npts];
          const double dbfz_mu_i = basis_z_eval[ipt + ibf*npts];

          g_acc_x_task += zS_mu_i * dbfx_mu_i;
          g_acc_y_task += zS_mu_i * dbfy_mu_i;
          g_acc_z_task += zS_mu_i * dbfz_mu_i;
          g_acc_x_task += zZ_mu_i * dbfx_mu_i;
          g_acc_y_task += zZ_mu_i * dbfy_mu_i;
          g_acc_z_task += zZ_mu_i * dbfz_mu_i;
        } // Loop over bfns within a shell

      } // Loop over points

      g_acc_x += g_acc_x_task;
      g_acc_y += g_acc_y_task;
      g_acc_z += g_acc_z_task;

      //write to Parent atom with translational invariance
      if constexpr( with_weight_derivatives ) {
        atomicAdd( EXC_GRAD + 3*iParent + 0, 2.0 * g_acc_x_task );
        atomicAdd( EXC_GRAD + 3*iParent + 1, 2.0 * g_acc_y_task );
        atomicAdd( EXC_GRAD + 3*iParent + 2, 2.0 * g_acc_z_task );
      }

    } // Loop over tasks assigned to shell

    constexpr auto warp_size = cuda::warp_size;
    g_acc_x = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_x );
    g_acc_y = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_y );
    g_acc_z = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_z );

    if( (threadIdx.x % cuda::warp_size) == 0 ) {
      atomicAdd( EXC_GRAD + 3*iCen + 0, g_acc_x );
      atomicAdd( EXC_GRAD + 3*iCen + 1, g_acc_y );
      atomicAdd( EXC_GRAD + 3*iCen + 2, g_acc_z );
    }

  } // Loop over shells

}

void increment_exc_grad_lda( integrator_ks_scheme ks_scheme, size_t nshell, ShellToTaskDevice* shell_to_task,
  XCDeviceTask* device_tasks, double* EXC_GRAD, bool with_weight_derivatives, device_queue queue ) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>();
  #if 0
  int nthreads_per_block = 1024;
  int nwarp_per_block    = nthreads_per_block / cuda::warp_size;
  int nblocks            = util::div_ceil( nshell, nwarp_per_block );

  dim3 threads( nthreads_per_block );
  dim3 blocks( nblocks );
  #else
  dim3 threads(1024), blocks(1,1,nshell);
  #endif

  switch(ks_scheme) {
    case RKS:
      if (with_weight_derivatives) {
        increment_exc_grad_lda_rks_kernel<true><<<blocks, threads, 0 , stream>>>(
          nshell, shell_to_task, device_tasks, EXC_GRAD 
        );
      } else {
        increment_exc_grad_lda_rks_kernel<false><<<blocks, threads, 0 , stream>>>(
          nshell, shell_to_task, device_tasks, EXC_GRAD 
        );
      }
      break;
    case UKS:
      if (with_weight_derivatives) {
        increment_exc_grad_lda_uks_kernel<true><<<blocks, threads, 0 , stream>>>(
          nshell, shell_to_task, device_tasks, EXC_GRAD 
        );
      } else {
        increment_exc_grad_lda_uks_kernel<false><<<blocks, threads, 0 , stream>>>(
          nshell, shell_to_task, device_tasks, EXC_GRAD 
        );
      }
      break;
    default: GAUXC_GENERIC_EXCEPTION("LDA EXC GRAD + GKS NYI");
  }
}















template<bool with_weight_derivatives>
__global__ __launch_bounds__(512,1) void increment_exc_grad_gga_rks_kernel(
  uint32_t                        nshell,
  ShellToTaskDevice* __restrict__ shell_to_task,
  XCDeviceTask*      __restrict__ device_tasks,
  double*            __restrict__ EXC_GRAD
) {

  for( uint32_t ish = blockIdx.z; ish < nshell; ish += gridDim.z ) {
    const uint32_t ntasks      = shell_to_task[ish].ntask;
    const auto shell           = shell_to_task[ish].shell_device;
    const auto task_idx        = shell_to_task[ish].task_idx_device;
    const auto task_shell_offs = shell_to_task[ish].task_shell_offs_device;
    const int iCen = shell_to_task[ish].center_idx;
    const uint32_t shsz   = shell->size();

    const int global_warp_id = 
      (threadIdx.x + blockIdx.x*blockDim.x) / cuda::warp_size;
    const int nwarp_global   = max((blockDim.x*gridDim.x) / cuda::warp_size,1);

    double g_acc_x(0), g_acc_y(0), g_acc_z(0);
    for( uint32_t itask = global_warp_id; itask < ntasks; itask += nwarp_global ) {
      
      const auto*    task   = device_tasks + task_idx[itask];
      const uint32_t npts   = task->npts;
      const size_t   shoff  = task_shell_offs[itask] * npts;
      const int iParent     = task->iParent;
      if constexpr( with_weight_derivatives ) {
        if( iCen == iParent ) 
          continue;
      }
      double g_acc_x_task(0), g_acc_y_task(0), g_acc_z_task(0);

      const auto* __restrict__ basis_x_eval = task->dbfx + shoff;
      const auto* __restrict__ basis_y_eval = task->dbfy + shoff;
      const auto* __restrict__ basis_z_eval = task->dbfz + shoff;

      const auto* __restrict__ basis_xx_eval = task->d2bfxx + shoff;
      const auto* __restrict__ basis_xy_eval = task->d2bfxy + shoff;
      const auto* __restrict__ basis_xz_eval = task->d2bfxz + shoff;
      const auto* __restrict__ basis_yy_eval = task->d2bfyy + shoff;
      const auto* __restrict__ basis_yz_eval = task->d2bfyz + shoff;
      const auto* __restrict__ basis_zz_eval = task->d2bfzz + shoff;

      const auto* __restrict__  xmat = task->zmat + shoff;
      const auto* __restrict__  xmat_x = task->xmat_x + shoff;
      const auto* __restrict__  xmat_y = task->xmat_y + shoff;
      const auto* __restrict__  xmat_z = task->xmat_z + shoff;

      const auto* __restrict__  vrho = task->vrho;
      const auto* __restrict__  vgamma = task->vgamma;

      const auto* __restrict__ den_x = task->dden_sx;
      const auto* __restrict__ den_y = task->dden_sy;
      const auto* __restrict__ den_z = task->dden_sz;

      #pragma unroll 1
      for( uint32_t ipt = threadIdx.x % cuda::warp_size; 
           ipt < npts; 
           ipt += cuda::warp_size ) {

        const double vrho_i   = vrho[ipt];
        const double vgamma_i = vgamma[ipt];

        const double denx_i = den_x[ipt];
        const double deny_i = den_y[ipt];
        const double denz_i = den_z[ipt];
        for( uint32_t ibf = 0; ibf < shsz; ++ibf ) {
          const double z_mu_i    = xmat[ipt + ibf*npts];          
          const double dbfx_mu_i = basis_x_eval[ipt + ibf*npts];
          const double dbfy_mu_i = basis_y_eval[ipt + ibf*npts];
          const double dbfz_mu_i = basis_z_eval[ipt + ibf*npts];

          g_acc_x_task += vrho_i * z_mu_i * dbfx_mu_i;
          g_acc_y_task += vrho_i * z_mu_i * dbfy_mu_i;
          g_acc_z_task += vrho_i * z_mu_i * dbfz_mu_i;

          const double zx = xmat_x[ipt + ibf*npts];
          const double zy = xmat_y[ipt + ibf*npts];
          const double zz = xmat_z[ipt + ibf*npts];

          const double d11_xmat_term = denx_i * zx + deny_i * zy + denz_i * zz;

          const double d2bfxx = basis_xx_eval[ipt + ibf*npts];
          const double d2bfxy = basis_xy_eval[ipt + ibf*npts];
          const double d2bfxz = basis_xz_eval[ipt + ibf*npts];
          const double d2bfyy = basis_yy_eval[ipt + ibf*npts];
          const double d2bfyz = basis_yz_eval[ipt + ibf*npts];
          const double d2bfzz = basis_zz_eval[ipt + ibf*npts];

          const double d2_term_x = d2bfxx*denx_i + d2bfxy*deny_i + d2bfxz*denz_i;
          const double d2_term_y = d2bfxy*denx_i + d2bfyy*deny_i + d2bfyz*denz_i;
          const double d2_term_z = d2bfxz*denx_i + d2bfyz*deny_i + d2bfzz*denz_i;

          g_acc_x_task += 2 * vgamma_i * ( z_mu_i * d2_term_x + dbfx_mu_i * d11_xmat_term );
          g_acc_y_task += 2 * vgamma_i * ( z_mu_i * d2_term_y + dbfy_mu_i * d11_xmat_term );
          g_acc_z_task += 2 * vgamma_i * ( z_mu_i * d2_term_z + dbfz_mu_i * d11_xmat_term );

        } // Loop over bfns within a shell

      } // Loop over points

      g_acc_x += g_acc_x_task;
      g_acc_y += g_acc_y_task;
      g_acc_z += g_acc_z_task;

      //write to Parent atom with translational invariance
      if constexpr( with_weight_derivatives ) {
        atomicAdd( EXC_GRAD + 3*iParent + 0, 2.0 * g_acc_x_task );
        atomicAdd( EXC_GRAD + 3*iParent + 1, 2.0 * g_acc_y_task );
        atomicAdd( EXC_GRAD + 3*iParent + 2, 2.0 * g_acc_z_task );
      }

    } // Loop over tasks assigned to shell

    constexpr auto warp_size = cuda::warp_size;
    g_acc_x = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_x );
    g_acc_y = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_y );
    g_acc_z = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_z );

    if( (threadIdx.x % cuda::warp_size) == 0 ) {
      atomicAdd( EXC_GRAD + 3*iCen + 0, g_acc_x );
      atomicAdd( EXC_GRAD + 3*iCen + 1, g_acc_y );
      atomicAdd( EXC_GRAD + 3*iCen + 2, g_acc_z );
    }

  } // Loop over shells

}

template<bool with_weight_derivatives>
__global__ __launch_bounds__(512,1) void increment_exc_grad_gga_uks_kernel(
  uint32_t                        nshell,
  ShellToTaskDevice* __restrict__ shell_to_task,
  XCDeviceTask*      __restrict__ device_tasks,
  double*            __restrict__ EXC_GRAD
) {

  for( uint32_t ish = blockIdx.z; ish < nshell; ish += gridDim.z ) {
    const uint32_t ntasks      = shell_to_task[ish].ntask;
    const auto shell           = shell_to_task[ish].shell_device;
    const auto task_idx        = shell_to_task[ish].task_idx_device;
    const auto task_shell_offs = shell_to_task[ish].task_shell_offs_device;
    const int iCen = shell_to_task[ish].center_idx;
    const uint32_t shsz   = shell->size();

    const int global_warp_id = 
      (threadIdx.x + blockIdx.x*blockDim.x) / cuda::warp_size;
    const int nwarp_global   = max((blockDim.x*gridDim.x) / cuda::warp_size,1);

    double g_acc_x(0), g_acc_y(0), g_acc_z(0);
    for( uint32_t itask = global_warp_id; itask < ntasks; itask += nwarp_global ) {
      
      const auto*    task   = device_tasks + task_idx[itask];
      const uint32_t npts   = task->npts;
      const size_t   shoff  = task_shell_offs[itask] * npts;
      const int iParent     = task->iParent;
      if constexpr( with_weight_derivatives ) {
        if( iCen == iParent ) 
          continue;
      }
      double g_acc_x_task(0), g_acc_y_task(0), g_acc_z_task(0);

      const auto* __restrict__ basis_x_eval = task->dbfx + shoff;
      const auto* __restrict__ basis_y_eval = task->dbfy + shoff;
      const auto* __restrict__ basis_z_eval = task->dbfz + shoff;

      const auto* __restrict__ basis_xx_eval = task->d2bfxx + shoff;
      const auto* __restrict__ basis_xy_eval = task->d2bfxy + shoff;
      const auto* __restrict__ basis_xz_eval = task->d2bfxz + shoff;
      const auto* __restrict__ basis_yy_eval = task->d2bfyy + shoff;
      const auto* __restrict__ basis_yz_eval = task->d2bfyz + shoff;
      const auto* __restrict__ basis_zz_eval = task->d2bfzz + shoff;

      const auto* __restrict__  xmatS   = task->xmatS   + shoff;
      const auto* __restrict__  xmatS_x = task->xmatS_x + shoff;
      const auto* __restrict__  xmatS_y = task->xmatS_y + shoff;
      const auto* __restrict__  xmatS_z = task->xmatS_z + shoff;

      const auto* __restrict__  xmatZ   = task->xmatZ   + shoff;
      const auto* __restrict__  xmatZ_x = task->xmatZ_x + shoff;
      const auto* __restrict__  xmatZ_y = task->xmatZ_y + shoff;
      const auto* __restrict__  xmatZ_z = task->xmatZ_z + shoff;

      const auto* __restrict__  vrhop = task->vrho_pos;
      const auto* __restrict__  vrhom = task->vrho_neg;

      const auto* __restrict__  vgamma_pp = task->vgamma_pp;
      const auto* __restrict__  vgamma_pm = task->vgamma_pm;
      const auto* __restrict__  vgamma_mm = task->vgamma_mm;

      const auto* __restrict__ dens_x = task->dden_sx;
      const auto* __restrict__ dens_y = task->dden_sy;
      const auto* __restrict__ dens_z = task->dden_sz;

      const auto* __restrict__ denz_x = task->dden_zx;
      const auto* __restrict__ denz_y = task->dden_zy;
      const auto* __restrict__ denz_z = task->dden_zz;

      #pragma unroll 1
      for( uint32_t ipt = threadIdx.x % cuda::warp_size; 
           ipt < npts; 
           ipt += cuda::warp_size ) {

        const double vrhop_i = vrhop[ipt];
        const double vrhom_i = vrhom[ipt];
        const double vrhoS_i = 0.5 * (vrhop_i + vrhom_i);
        const double vrhoZ_i = 0.5 * (vrhop_i - vrhom_i);

        const double vgammapp_i = vgamma_pp[ipt];
        const double vgammapm_i = vgamma_pm[ipt];
        const double vgammamm_i = vgamma_mm[ipt];

        const double denSx_i = dens_x[ipt];
        const double denSy_i = dens_y[ipt];
        const double denSz_i = dens_z[ipt];
        const double denZx_i = denz_x[ipt];
        const double denZy_i = denz_y[ipt];
        const double denZz_i = denz_z[ipt];

        for( uint32_t ibf = 0; ibf < shsz; ++ibf ) {
          const double xN    = xmatS[ipt + ibf*npts];
          const double xZ    = xmatZ[ipt + ibf*npts];
          const double dbfx_mu_i = basis_x_eval[ipt + ibf*npts];
          const double dbfy_mu_i = basis_y_eval[ipt + ibf*npts];
          const double dbfz_mu_i = basis_z_eval[ipt + ibf*npts];

          g_acc_x_task += vrhoS_i * xN * dbfx_mu_i;
          g_acc_y_task += vrhoS_i * xN * dbfy_mu_i;
          g_acc_z_task += vrhoS_i * xN * dbfz_mu_i;
          g_acc_x_task += vrhoZ_i * xZ * dbfx_mu_i;
          g_acc_y_task += vrhoZ_i * xZ * dbfy_mu_i;
          g_acc_z_task += vrhoZ_i * xZ * dbfz_mu_i;

          const double xNx = xmatS_x[ipt + ibf*npts];
          const double xNy = xmatS_y[ipt + ibf*npts];
          const double xNz = xmatS_z[ipt + ibf*npts];
          const double xZx = xmatZ_x[ipt + ibf*npts];
          const double xZy = xmatZ_y[ipt + ibf*npts];
          const double xZz = xmatZ_z[ipt + ibf*npts];

          const double d11nn_xmat_term = denSx_i * xNx + denSy_i * xNy + denSz_i * xNz;
          const double d11nz_xmat_term = denSx_i * xZx + denSy_i * xZy + denSz_i * xZz;
          const double d11zn_xmat_term = denZx_i * xNx + denZy_i * xNy + denZz_i * xNz;
          const double d11zz_xmat_term = denZx_i * xZx + denZy_i * xZy + denZz_i * xZz;

          const double d2bfxx = basis_xx_eval[ipt + ibf*npts];
          const double d2bfxy = basis_xy_eval[ipt + ibf*npts];
          const double d2bfxz = basis_xz_eval[ipt + ibf*npts];
          const double d2bfyy = basis_yy_eval[ipt + ibf*npts];
          const double d2bfyz = basis_yz_eval[ipt + ibf*npts];
          const double d2bfzz = basis_zz_eval[ipt + ibf*npts];

          const double d2n_term_x = d2bfxx*denSx_i + d2bfxy*denSy_i + d2bfxz*denSz_i;
          const double d2n_term_y = d2bfxy*denSx_i + d2bfyy*denSy_i + d2bfyz*denSz_i;
          const double d2n_term_z = d2bfxz*denSx_i + d2bfyz*denSy_i + d2bfzz*denSz_i;
          const double d2z_term_x = d2bfxx*denZx_i + d2bfxy*denZy_i + d2bfxz*denZz_i;
          const double d2z_term_y = d2bfxy*denZx_i + d2bfyy*denZy_i + d2bfyz*denZz_i;
          const double d2z_term_z = d2bfxz*denZx_i + d2bfyz*denZy_i + d2bfzz*denZz_i;

          g_acc_x_task += 0.5 * (vgammapp_i + vgammapm_i + vgammamm_i) * (d2n_term_x * xN + d11nn_xmat_term * dbfx_mu_i);
          g_acc_x_task += 0.5 * (vgammapp_i              - vgammamm_i) * (d2z_term_x * xN + d11zn_xmat_term * dbfx_mu_i);
          g_acc_x_task += 0.5 * (vgammapp_i              - vgammamm_i) * (d2n_term_x * xZ + d11nz_xmat_term * dbfx_mu_i);
          g_acc_x_task += 0.5 * (vgammapp_i - vgammapm_i + vgammamm_i) * (d2z_term_x * xZ + d11zz_xmat_term * dbfx_mu_i);

          g_acc_y_task += 0.5 * (vgammapp_i + vgammapm_i + vgammamm_i) * (d2n_term_y * xN + d11nn_xmat_term * dbfy_mu_i);
          g_acc_y_task += 0.5 * (vgammapp_i              - vgammamm_i) * (d2z_term_y * xN + d11zn_xmat_term * dbfy_mu_i);
          g_acc_y_task += 0.5 * (vgammapp_i              - vgammamm_i) * (d2n_term_y * xZ + d11nz_xmat_term * dbfy_mu_i);
          g_acc_y_task += 0.5 * (vgammapp_i - vgammapm_i + vgammamm_i) * (d2z_term_y * xZ + d11zz_xmat_term * dbfy_mu_i);

          g_acc_z_task += 0.5 * (vgammapp_i + vgammapm_i + vgammamm_i) * (d2n_term_z * xN + d11nn_xmat_term * dbfz_mu_i);
          g_acc_z_task += 0.5 * (vgammapp_i              - vgammamm_i) * (d2z_term_z * xN + d11zn_xmat_term * dbfz_mu_i);
          g_acc_z_task += 0.5 * (vgammapp_i              - vgammamm_i) * (d2n_term_z * xZ + d11nz_xmat_term * dbfz_mu_i);
          g_acc_z_task += 0.5 * (vgammapp_i - vgammapm_i + vgammamm_i) * (d2z_term_z * xZ + d11zz_xmat_term * dbfz_mu_i);

        }// Loop over bfns within a shell

      } // Loop over points

      g_acc_x += g_acc_x_task;
      g_acc_y += g_acc_y_task;
      g_acc_z += g_acc_z_task;

      //write to Parent atom with translational invariance
      if constexpr( with_weight_derivatives ) {
        atomicAdd( EXC_GRAD + 3*iParent + 0, 2.0 * g_acc_x_task );
        atomicAdd( EXC_GRAD + 3*iParent + 1, 2.0 * g_acc_y_task );
        atomicAdd( EXC_GRAD + 3*iParent + 2, 2.0 * g_acc_z_task );
      }

    } // Loop over tasks assigned to shell

    constexpr auto warp_size = cuda::warp_size;
    g_acc_x = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_x );
    g_acc_y = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_y );
    g_acc_z = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_z );

    if( (threadIdx.x % cuda::warp_size) == 0 ) {
      atomicAdd( EXC_GRAD + 3*iCen + 0, g_acc_x );
      atomicAdd( EXC_GRAD + 3*iCen + 1, g_acc_y );
      atomicAdd( EXC_GRAD + 3*iCen + 2, g_acc_z );
    }

  } // Loop over shells

}

void increment_exc_grad_gga( integrator_ks_scheme ks_scheme, size_t nshell, ShellToTaskDevice* shell_to_task,
  XCDeviceTask* device_tasks, double* EXC_GRAD, bool with_weight_derivatives, device_queue queue ) {

  cudaStream_t stream = queue.queue_as<util::cuda_stream>();
  dim3 threads(512), blocks(1,1,nshell);

  switch(ks_scheme) {
    case RKS:
      if (with_weight_derivatives) {
        increment_exc_grad_gga_rks_kernel<true><<<blocks, threads, 0 , stream>>>(
          nshell, shell_to_task, device_tasks, EXC_GRAD 
        );
      } else {
        increment_exc_grad_gga_rks_kernel<false><<<blocks, threads, 0 , stream>>>(
          nshell, shell_to_task, device_tasks, EXC_GRAD 
        );
      }
      break;
    case UKS:
      if (with_weight_derivatives) {
        increment_exc_grad_gga_uks_kernel<true><<<blocks, threads, 0 , stream>>>(
          nshell, shell_to_task, device_tasks, EXC_GRAD 
        );
      } else {
        increment_exc_grad_gga_uks_kernel<false><<<blocks, threads, 0 , stream>>>(
          nshell, shell_to_task, device_tasks, EXC_GRAD 
        );
      }
      break;
    default: GAUXC_GENERIC_EXCEPTION("GGA EXC GRAD + GKS NYI");
  }
}






template<bool with_weight_derivatives>
__global__ __launch_bounds__(512,1) void increment_exc_grad_mgga_rks_kernel(
  uint32_t                        nshell,
  ShellToTaskDevice* __restrict__ shell_to_task,
  XCDeviceTask*      __restrict__ device_tasks,
  double*            __restrict__ EXC_GRAD
) {

  for( uint32_t ish = blockIdx.z; ish < nshell; ish += gridDim.z ) {
    const uint32_t ntasks      = shell_to_task[ish].ntask;
    const auto shell           = shell_to_task[ish].shell_device;
    const auto task_idx        = shell_to_task[ish].task_idx_device;
    const auto task_shell_offs = shell_to_task[ish].task_shell_offs_device;
    const int iCen = shell_to_task[ish].center_idx;
    const uint32_t shsz   = shell->size();

    const int global_warp_id = 
      (threadIdx.x + blockIdx.x*blockDim.x) / cuda::warp_size;
    const int nwarp_global   = max((blockDim.x*gridDim.x) / cuda::warp_size,1);

    double g_acc_x(0), g_acc_y(0), g_acc_z(0);
    for( uint32_t itask = global_warp_id; itask < ntasks; itask += nwarp_global ) {
      
      const auto*    task   = device_tasks + task_idx[itask];
      const uint32_t npts   = task->npts;
      const size_t   shoff  = task_shell_offs[itask] * npts;
      const int iParent     = task->iParent;
      if constexpr( with_weight_derivatives ) {
        if( iCen == iParent )
          continue;
      }
      double g_acc_x_task(0), g_acc_y_task(0), g_acc_z_task(0);

      const auto* __restrict__ basis_x_eval = task->dbfx + shoff;
      const auto* __restrict__ basis_y_eval = task->dbfy + shoff;
      const auto* __restrict__ basis_z_eval = task->dbfz + shoff;

      const auto* __restrict__ basis_xx_eval = task->d2bfxx + shoff;
      const auto* __restrict__ basis_xy_eval = task->d2bfxy + shoff;
      const auto* __restrict__ basis_xz_eval = task->d2bfxz + shoff;
      const auto* __restrict__ basis_yy_eval = task->d2bfyy + shoff;
      const auto* __restrict__ basis_yz_eval = task->d2bfyz + shoff;
      const auto* __restrict__ basis_zz_eval = task->d2bfzz + shoff;

      const auto* __restrict__  xmat   = task->zmat + shoff;
      const auto* __restrict__  xmat_x = task->xmat_x + shoff;
      const auto* __restrict__  xmat_y = task->xmat_y + shoff;
      const auto* __restrict__  xmat_z = task->xmat_z + shoff;

      const auto* __restrict__  vrho   = task->vrho;
      const auto* __restrict__  vgamma = task->vgamma;
      const auto* __restrict__  vtau   = task->vtau;

      const auto* __restrict__ den_x = task->dden_sx;
      const auto* __restrict__ den_y = task->dden_sy;
      const auto* __restrict__ den_z = task->dden_sz;

      #pragma unroll 1
      for( uint32_t ipt = threadIdx.x % cuda::warp_size; 
           ipt < npts; 
           ipt += cuda::warp_size ) {

        const double vrho_i   = vrho[ipt];
        const double vgamma_i = vgamma[ipt];
        const double vtau_i   = 0.5 * vtau[ipt];

        const double denx_i = den_x[ipt];
        const double deny_i = den_y[ipt];
        const double denz_i = den_z[ipt];
        for( uint32_t ibf = 0; ibf < shsz; ++ibf ) {
          const double z_mu_i    = xmat[ipt + ibf*npts];
          const double dbfx_mu_i = basis_x_eval[ipt + ibf*npts];
          const double dbfy_mu_i = basis_y_eval[ipt + ibf*npts];
          const double dbfz_mu_i = basis_z_eval[ipt + ibf*npts];

          g_acc_x_task += vrho_i * z_mu_i * dbfx_mu_i;
          g_acc_y_task += vrho_i * z_mu_i * dbfy_mu_i;
          g_acc_z_task += vrho_i * z_mu_i * dbfz_mu_i;

          const double zx = xmat_x[ipt + ibf*npts];
          const double zy = xmat_y[ipt + ibf*npts];
          const double zz = xmat_z[ipt + ibf*npts];

          const double d11_xmat_term = denx_i * zx + deny_i * zy + denz_i * zz;

          const double d2bfxx = basis_xx_eval[ipt + ibf*npts];
          const double d2bfxy = basis_xy_eval[ipt + ibf*npts];
          const double d2bfxz = basis_xz_eval[ipt + ibf*npts];
          const double d2bfyy = basis_yy_eval[ipt + ibf*npts];
          const double d2bfyz = basis_yz_eval[ipt + ibf*npts];
          const double d2bfzz = basis_zz_eval[ipt + ibf*npts];

          {
          const double d2_term_x = d2bfxx*denx_i + d2bfxy*deny_i + d2bfxz*denz_i;
          const double d2_term_y = d2bfxy*denx_i + d2bfyy*deny_i + d2bfyz*denz_i;
          const double d2_term_z = d2bfxz*denx_i + d2bfyz*deny_i + d2bfzz*denz_i;

          g_acc_x_task += 2 * vgamma_i * ( z_mu_i * d2_term_x + dbfx_mu_i * d11_xmat_term );
          g_acc_y_task += 2 * vgamma_i * ( z_mu_i * d2_term_y + dbfy_mu_i * d11_xmat_term );
          g_acc_z_task += 2 * vgamma_i * ( z_mu_i * d2_term_z + dbfz_mu_i * d11_xmat_term );
          }

          {
          const double d2_term_x = d2bfxx*zx + d2bfxy*zy + d2bfxz*zz;
          const double d2_term_y = d2bfxy*zx + d2bfyy*zy + d2bfyz*zz;
          const double d2_term_z = d2bfxz*zx + d2bfyz*zy + d2bfzz*zz;

          g_acc_x_task += vtau_i * d2_term_x;
          g_acc_y_task += vtau_i * d2_term_y;
          g_acc_z_task += vtau_i * d2_term_z;
          }

        } // Loop over bfns within a shell

      } // Loop over points

      g_acc_x += g_acc_x_task;
      g_acc_y += g_acc_y_task;
      g_acc_z += g_acc_z_task;

      //write to Parent atom with translational invariance
      if constexpr( with_weight_derivatives ) {
        atomicAdd( EXC_GRAD + 3*iParent + 0, 2.0 * g_acc_x_task );
        atomicAdd( EXC_GRAD + 3*iParent + 1, 2.0 * g_acc_y_task );
        atomicAdd( EXC_GRAD + 3*iParent + 2, 2.0 * g_acc_z_task );
      }

    } // Loop over tasks assigned to shell

    constexpr auto warp_size = cuda::warp_size;
    g_acc_x = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_x );
    g_acc_y = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_y );
    g_acc_z = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_z );

    if( (threadIdx.x % cuda::warp_size) == 0 ) {
      atomicAdd( EXC_GRAD + 3*iCen + 0, g_acc_x );
      atomicAdd( EXC_GRAD + 3*iCen + 1, g_acc_y );
      atomicAdd( EXC_GRAD + 3*iCen + 2, g_acc_z );
    }

  } // Loop over shells

}

template<bool with_weight_derivatives>
__global__ __launch_bounds__(512,1) void increment_exc_grad_mgga_uks_kernel(
  uint32_t                        nshell,
  ShellToTaskDevice* __restrict__ shell_to_task,
  XCDeviceTask*      __restrict__ device_tasks,
  double*            __restrict__ EXC_GRAD
) {
  for( uint32_t ish = blockIdx.z; ish < nshell; ish += gridDim.z ) {
    const uint32_t ntasks      = shell_to_task[ish].ntask;
    const auto shell           = shell_to_task[ish].shell_device;
    const auto task_idx        = shell_to_task[ish].task_idx_device;
    const auto task_shell_offs = shell_to_task[ish].task_shell_offs_device;
    const int iCen = shell_to_task[ish].center_idx;
    const uint32_t shsz   = shell->size();

    const int global_warp_id = 
      (threadIdx.x + blockIdx.x*blockDim.x) / cuda::warp_size;
    const int nwarp_global   = max((blockDim.x*gridDim.x) / cuda::warp_size,1);

    double g_acc_x(0), g_acc_y(0), g_acc_z(0);
    for( uint32_t itask = global_warp_id; itask < ntasks; itask += nwarp_global ) {
      
      const auto*    task   = device_tasks + task_idx[itask];
      const uint32_t npts   = task->npts;
      const size_t   shoff  = task_shell_offs[itask] * npts;
      const int iParent     = task->iParent;
      if constexpr( with_weight_derivatives ) {
        if( iCen == iParent ) 
          continue;
      }
      double g_acc_x_task(0), g_acc_y_task(0), g_acc_z_task(0);

      const auto* __restrict__ basis_x_eval = task->dbfx + shoff;
      const auto* __restrict__ basis_y_eval = task->dbfy + shoff;
      const auto* __restrict__ basis_z_eval = task->dbfz + shoff;

      const auto* __restrict__ basis_xx_eval = task->d2bfxx + shoff;
      const auto* __restrict__ basis_xy_eval = task->d2bfxy + shoff;
      const auto* __restrict__ basis_xz_eval = task->d2bfxz + shoff;
      const auto* __restrict__ basis_yy_eval = task->d2bfyy + shoff;
      const auto* __restrict__ basis_yz_eval = task->d2bfyz + shoff;
      const auto* __restrict__ basis_zz_eval = task->d2bfzz + shoff;

      const auto* __restrict__  xmatS   = task->xmatS   + shoff;
      const auto* __restrict__  xmatS_x = task->xmatS_x + shoff;
      const auto* __restrict__  xmatS_y = task->xmatS_y + shoff;
      const auto* __restrict__  xmatS_z = task->xmatS_z + shoff;

      const auto* __restrict__  xmatZ   = task->xmatZ   + shoff;
      const auto* __restrict__  xmatZ_x = task->xmatZ_x + shoff;
      const auto* __restrict__  xmatZ_y = task->xmatZ_y + shoff;
      const auto* __restrict__  xmatZ_z = task->xmatZ_z + shoff;

      const auto* __restrict__  vrhop = task->vrho_pos;
      const auto* __restrict__  vrhom = task->vrho_neg;
      const auto* __restrict__  vtaup = task->vtau_pos;
      const auto* __restrict__  vtaum = task->vtau_neg;

      const auto* __restrict__  vgamma_pp = task->vgamma_pp;
      const auto* __restrict__  vgamma_pm = task->vgamma_pm;
      const auto* __restrict__  vgamma_mm = task->vgamma_mm;

      const auto* __restrict__ dens_x = task->dden_sx;
      const auto* __restrict__ dens_y = task->dden_sy;
      const auto* __restrict__ dens_z = task->dden_sz;

      const auto* __restrict__ denz_x = task->dden_zx;
      const auto* __restrict__ denz_y = task->dden_zy;
      const auto* __restrict__ denz_z = task->dden_zz;

      #pragma unroll 1
      for( uint32_t ipt = threadIdx.x % cuda::warp_size; 
           ipt < npts; 
           ipt += cuda::warp_size ) {

        const double vrhop_i = vrhop[ipt];
        const double vrhom_i = vrhom[ipt];
        const double vrhoS_i = 0.5 * (vrhop_i + vrhom_i);
        const double vrhoZ_i = 0.5 * (vrhop_i - vrhom_i);        

        const double vtaup_i = 0.5 * vtaup[ipt];
        const double vtaum_i = 0.5 * vtaum[ipt];
        const double vtauS_i = 0.5 * (vtaup_i + vtaum_i);
        const double vtauZ_i = 0.5 * (vtaup_i - vtaum_i);

        const double vgammapp_i = vgamma_pp[ipt];
        const double vgammapm_i = vgamma_pm[ipt];
        const double vgammamm_i = vgamma_mm[ipt];

        const double denSx_i = dens_x[ipt];
        const double denSy_i = dens_y[ipt];
        const double denSz_i = dens_z[ipt];
        const double denZx_i = denz_x[ipt];
        const double denZy_i = denz_y[ipt];
        const double denZz_i = denz_z[ipt];

        for( uint32_t ibf = 0; ibf < shsz; ++ibf ) {
          const double xN    = xmatS[ipt + ibf*npts];
          const double xZ    = xmatZ[ipt + ibf*npts];
          const double dbfx_mu_i = basis_x_eval[ipt + ibf*npts];
          const double dbfy_mu_i = basis_y_eval[ipt + ibf*npts];
          const double dbfz_mu_i = basis_z_eval[ipt + ibf*npts];

          g_acc_x_task += vrhoS_i * xN * dbfx_mu_i;
          g_acc_y_task += vrhoS_i * xN * dbfy_mu_i;
          g_acc_z_task += vrhoS_i * xN * dbfz_mu_i;
          g_acc_x_task += vrhoZ_i * xZ * dbfx_mu_i;
          g_acc_y_task += vrhoZ_i * xZ * dbfy_mu_i;
          g_acc_z_task += vrhoZ_i * xZ * dbfz_mu_i;

          const double xNx = xmatS_x[ipt + ibf*npts];
          const double xNy = xmatS_y[ipt + ibf*npts];
          const double xNz = xmatS_z[ipt + ibf*npts];
          const double xZx = xmatZ_x[ipt + ibf*npts];
          const double xZy = xmatZ_y[ipt + ibf*npts];
          const double xZz = xmatZ_z[ipt + ibf*npts];

          const double d11nn_xmat_term = denSx_i * xNx + denSy_i * xNy + denSz_i * xNz;
          const double d11nz_xmat_term = denSx_i * xZx + denSy_i * xZy + denSz_i * xZz;
          const double d11zn_xmat_term = denZx_i * xNx + denZy_i * xNy + denZz_i * xNz;
          const double d11zz_xmat_term = denZx_i * xZx + denZy_i * xZy + denZz_i * xZz;

          const double d2bfxx = basis_xx_eval[ipt + ibf*npts];
          const double d2bfxy = basis_xy_eval[ipt + ibf*npts];
          const double d2bfxz = basis_xz_eval[ipt + ibf*npts];
          const double d2bfyy = basis_yy_eval[ipt + ibf*npts];
          const double d2bfyz = basis_yz_eval[ipt + ibf*npts];
          const double d2bfzz = basis_zz_eval[ipt + ibf*npts];

          {
          const double d2n_term_x = d2bfxx*denSx_i + d2bfxy*denSy_i + d2bfxz*denSz_i;
          const double d2n_term_y = d2bfxy*denSx_i + d2bfyy*denSy_i + d2bfyz*denSz_i;
          const double d2n_term_z = d2bfxz*denSx_i + d2bfyz*denSy_i + d2bfzz*denSz_i;
          const double d2z_term_x = d2bfxx*denZx_i + d2bfxy*denZy_i + d2bfxz*denZz_i;
          const double d2z_term_y = d2bfxy*denZx_i + d2bfyy*denZy_i + d2bfyz*denZz_i;
          const double d2z_term_z = d2bfxz*denZx_i + d2bfyz*denZy_i + d2bfzz*denZz_i;          

          g_acc_x_task += 0.5 * (vgammapp_i + vgammapm_i + vgammamm_i) * (d2n_term_x * xN + d11nn_xmat_term * dbfx_mu_i);
          g_acc_x_task += 0.5 * (vgammapp_i              - vgammamm_i) * (d2z_term_x * xN + d11zn_xmat_term * dbfx_mu_i);
          g_acc_x_task += 0.5 * (vgammapp_i              - vgammamm_i) * (d2n_term_x * xZ + d11nz_xmat_term * dbfx_mu_i);
          g_acc_x_task += 0.5 * (vgammapp_i - vgammapm_i + vgammamm_i) * (d2z_term_x * xZ + d11zz_xmat_term * dbfx_mu_i);

          g_acc_y_task += 0.5 * (vgammapp_i + vgammapm_i + vgammamm_i) * (d2n_term_y * xN + d11nn_xmat_term * dbfy_mu_i);
          g_acc_y_task += 0.5 * (vgammapp_i              - vgammamm_i) * (d2z_term_y * xN + d11zn_xmat_term * dbfy_mu_i);
          g_acc_y_task += 0.5 * (vgammapp_i              - vgammamm_i) * (d2n_term_y * xZ + d11nz_xmat_term * dbfy_mu_i);
          g_acc_y_task += 0.5 * (vgammapp_i - vgammapm_i + vgammamm_i) * (d2z_term_y * xZ + d11zz_xmat_term * dbfy_mu_i);

          g_acc_z_task += 0.5 * (vgammapp_i + vgammapm_i + vgammamm_i) * (d2n_term_z * xN + d11nn_xmat_term * dbfz_mu_i);
          g_acc_z_task += 0.5 * (vgammapp_i              - vgammamm_i) * (d2z_term_z * xN + d11zn_xmat_term * dbfz_mu_i);
          g_acc_z_task += 0.5 * (vgammapp_i              - vgammamm_i) * (d2n_term_z * xZ + d11nz_xmat_term * dbfz_mu_i);
          g_acc_z_task += 0.5 * (vgammapp_i - vgammapm_i + vgammamm_i) * (d2z_term_z * xZ + d11zz_xmat_term * dbfz_mu_i);
          }

          {
          const double d2n_term_x = d2bfxx*xNx + d2bfxy*xNy + d2bfxz*xNz;
          const double d2n_term_y = d2bfxy*xNx + d2bfyy*xNy + d2bfyz*xNz;
          const double d2n_term_z = d2bfxz*xNx + d2bfyz*xNy + d2bfzz*xNz;
          const double d2z_term_x = d2bfxx*xZx + d2bfxy*xZy + d2bfxz*xZz;
          const double d2z_term_y = d2bfxy*xZx + d2bfyy*xZy + d2bfyz*xZz;
          const double d2z_term_z = d2bfxz*xZx + d2bfyz*xZy + d2bfzz*xZz;
          g_acc_x_task += vtauS_i * d2n_term_x;
          g_acc_y_task += vtauS_i * d2n_term_y;
          g_acc_z_task += vtauS_i * d2n_term_z;

          g_acc_x_task += vtauZ_i * d2z_term_x;
          g_acc_y_task += vtauZ_i * d2z_term_y;
          g_acc_z_task += vtauZ_i * d2z_term_z;
          }
        }// Loop over bfns within a shell

      } // Loop over points

      g_acc_x += g_acc_x_task;
      g_acc_y += g_acc_y_task;
      g_acc_z += g_acc_z_task;

      //write to Parent atom with translational invariance
      if constexpr( with_weight_derivatives ) {
        atomicAdd( EXC_GRAD + 3*iParent + 0, 2.0 * g_acc_x_task );
        atomicAdd( EXC_GRAD + 3*iParent + 1, 2.0 * g_acc_y_task );
        atomicAdd( EXC_GRAD + 3*iParent + 2, 2.0 * g_acc_z_task );
      }

    } // Loop over tasks assigned to shell

    constexpr auto warp_size = cuda::warp_size;
    g_acc_x = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_x );
    g_acc_y = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_y );
    g_acc_z = -2. * cuda::warp_reduce_sum<warp_size>( g_acc_z );

    if( (threadIdx.x % cuda::warp_size) == 0 ) {
      atomicAdd( EXC_GRAD + 3*iCen + 0, g_acc_x );
      atomicAdd( EXC_GRAD + 3*iCen + 1, g_acc_y );
      atomicAdd( EXC_GRAD + 3*iCen + 2, g_acc_z );
    }

  } // Loop over shells

}

void increment_exc_grad_mgga( integrator_ks_scheme ks_scheme, size_t nshell, bool need_lapl, 
  ShellToTaskDevice* shell_to_task, XCDeviceTask* device_tasks, 
  double* EXC_GRAD, bool with_weight_derivatives, device_queue queue ) {
 
  if(need_lapl) GAUXC_GENERIC_EXCEPTION("CUDA + MGGA/LAPL EXC GRAD NYI");

  cudaStream_t stream = queue.queue_as<util::cuda_stream>();
  dim3 threads(512), blocks(1,1,nshell);

  switch(ks_scheme) {
    case RKS:
      if (with_weight_derivatives) {
        increment_exc_grad_mgga_rks_kernel<true><<<blocks, threads, 0 , stream>>>(
          nshell, shell_to_task, device_tasks, EXC_GRAD 
        );
      } else {
        increment_exc_grad_mgga_rks_kernel<false><<<blocks, threads, 0 , stream>>>(
          nshell, shell_to_task, device_tasks, EXC_GRAD 
        );
      }
      break;
    case UKS:
      if (with_weight_derivatives) {
        increment_exc_grad_mgga_uks_kernel<true><<<blocks, threads, 0 , stream>>>(
          nshell, shell_to_task, device_tasks, EXC_GRAD 
        );
      } else {
        increment_exc_grad_mgga_uks_kernel<false><<<blocks, threads, 0 , stream>>>(
          nshell, shell_to_task, device_tasks, EXC_GRAD 
        );
      }
      break;
    default: GAUXC_GENERIC_EXCEPTION("GGA EXC GRAD + GKS NYI");
  }
}

}
