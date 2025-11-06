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
#include "scheme1_magma_base.hpp"
#include "device/common/pack_submat.hpp"
#include "device/common/inc_potential.hpp"
#include "device/common/device_blas.hpp"

#ifdef GAUXC_HAS_CUDA
#define GAUXC_ENABLE_EXX
#endif

namespace GauXC {

void AoSScheme1MAGMABase::eval_xmat( double fac, XCDeviceData* _data, bool do_grad, density_id den ){

  if( do_grad ) GAUXC_GENERIC_EXCEPTION("MAGMA + X Gradient NYI");

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();

  // Pack density matrix 
  const auto nbf = data->global_dims.nbf;
  const auto submat_block_size = data->get_submat_chunk_size( nbf, 0 );
  auto static_stack  = data->static_stack;
  auto aos_stack     = data->aos_stack;
  auto master_queue = data->device_backend_->master_magma_queue();
  auto magma_stack = data->magma_stack;
  double* dmat_ptr   = nullptr;
  switch( den ) {
    case DEN_S:
      dmat_ptr = static_stack.dmat_s_device;
      break;
    case DEN_Z:
      dmat_ptr = static_stack.dmat_z_device;
      break;
    case DEN_Y:
      dmat_ptr = static_stack.dmat_y_device;
      break;
    case DEN_X:
      dmat_ptr = static_stack.dmat_x_device;
      break;
    default:
      GAUXC_GENERIC_EXCEPTION( "eval_xmat called with invalid density specifier" );
  }
  sym_pack_submat( ntasks, aos_stack.device_tasks, dmat_ptr,
    nbf, submat_block_size, data->device_backend_->queue() );
  
  // Update dmat on magma_stack 
  std::vector<double*> dmat_host( ntasks );

  for( auto i = 0; i < ntasks; i++ ) {
    auto& task = tasks[i];
    if( task.bfn_screening.ncut > 1 ) {
      dmat_host[i]  = task.nbe_scr;
    } else {
      dmat_host[i]  = dmat_ptr + task.bfn_screening.ibf_begin*(nbf+1);
    }
  }
  data->device_backend_->copy_async( ntasks, dmat_host.data(), 
      magma_stack.xdmat_array_device, "send xdmat array");

  magmablas_dgemm_vbatched( MagmaNoTrans, MagmaNoTrans,
    magma_stack.xmat_m_array_device, magma_stack.xmat_n_array_device, 
    magma_stack.xmat_k_array_device, 
    fac, magma_stack.bf_array_device,    magma_stack.ld_bf_array_device,
        magma_stack.xdmat_array_device, magma_stack.ld_xdmat_array_device,
    0., magma_stack.zmat_array_device,  magma_stack.ld_zmat_array_device,
    ntasks, *master_queue );

}

void AoSScheme1MAGMABase::eval_exx_fmat( XCDeviceData* _data ) {
#ifndef GAUXC_ENABLE_EXX
  GAUXC_GENERIC_EXCEPTION("EXX + non-CUDA NYI");
#else
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();

  // Pack the density matrix into (bfn, cou) shape
  const auto nbf = data->global_dims.nbf;
  const auto submat_block_size = data->get_submat_chunk_size( nbf, 0 );
  auto static_stack  = data->static_stack;
  auto aos_stack     = data->aos_stack;
  asym_pack_submat( ntasks, aos_stack.device_tasks, static_stack.dmat_s_device,
    nbf, submat_block_size, data->device_backend_->queue() );


  auto master_queue = data->device_backend_->master_magma_queue();
  auto magma_stack = data->magma_stack;
  magmablas_dgemm_vbatched( MagmaNoTrans, MagmaNoTrans,
    magma_stack.fmat_m_array_device, magma_stack.fmat_n_array_device, 
    magma_stack.fmat_k_array_device, 
    1., magma_stack.bf_array_device,    magma_stack.ld_bf_array_device,
        magma_stack.fdmat_array_device, magma_stack.ld_fdmat_array_device,
    0., magma_stack.fmat_array_device,  magma_stack.ld_fmat_array_device,
    ntasks, *master_queue );
#endif
}

void AoSScheme1MAGMABase::inc_vxc( XCDeviceData* _data, density_id den, bool do_m){

  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  if(do_m) GAUXC_GENERIC_EXCEPTION("MAGMA + MGGA NYI");

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();

  auto master_queue = data->device_backend_->master_magma_queue();
  auto magma_stack = data->magma_stack;
  magmablas_dsyr2k_vbatched( MagmaLower, MagmaTrans, 
    magma_stack.xmat_n_array_device, magma_stack.xmat_m_array_device,
    1., magma_stack.bf_array_device,   magma_stack.ld_bf_array_device, 
        magma_stack.zmat_array_device, magma_stack.ld_zmat_array_device,
    0., magma_stack.vmat_array_device, magma_stack.ld_vmat_array_device, 
    ntasks, *master_queue );

  // Increment global VXC
  const auto nbf = data->global_dims.nbf;
  const auto submat_block_size = data->get_submat_chunk_size( nbf, 0 );
  auto static_stack  = data->static_stack;
  auto aos_stack     = data->aos_stack;
  double* vxc_ptr    = nullptr;
  switch (den) {
    case DEN_S:
      vxc_ptr = static_stack.vxc_s_device;
      break;
    case DEN_Z:
      vxc_ptr = static_stack.vxc_z_device;
      break;
    case DEN_Y:
      vxc_ptr = static_stack.vxc_y_device;
      break;
    case DEN_X:
      vxc_ptr = static_stack.vxc_x_device;
      break;
    default:
      GAUXC_GENERIC_EXCEPTION( "Inc_vxc called with invalid density specifier" );
  }
  sym_task_inc_potential( ntasks, aos_stack.device_tasks, 
    vxc_ptr,  nbf, submat_block_size, 
    data->device_backend_->queue() );
}

void AoSScheme1MAGMABase::inc_exx_k( XCDeviceData* _data){
#ifndef GAUXC_ENABLE_EXX
  GAUXC_GENERIC_EXCEPTION("EXX + non-CUDA NYI");
#else
  auto* data = dynamic_cast<Data*>(_data);
  if( !data ) GAUXC_BAD_LWD_DATA_CAST();

  if( not data->device_backend_ ) GAUXC_UNINITIALIZED_DEVICE_BACKEND();

  auto& tasks = data->host_device_tasks;
  const auto ntasks = tasks.size();

  auto master_queue = data->device_backend_->master_magma_queue();
  auto magma_stack = data->magma_stack;
  magmablas_dgemm_vbatched( MagmaTrans, MagmaNoTrans,
    magma_stack.fmat_k_array_device, magma_stack.fmat_n_array_device, 
    magma_stack.fmat_m_array_device, 
    1., magma_stack.bf_array_device,   magma_stack.ld_bf_array_device,
        magma_stack.gmat_array_device, magma_stack.ld_fmat_array_device,
    0., magma_stack.kmat_array_device, magma_stack.ld_fdmat_array_device,
    ntasks, *master_queue );

  // Increment EXX_K
  const auto nbf = data->global_dims.nbf;
  const auto submat_block_size = data->get_submat_chunk_size( nbf, 0 );
  auto static_stack  = data->static_stack;
  auto aos_stack     = data->aos_stack;
  asym_task_inc_potential( ntasks, aos_stack.device_tasks, 
    static_stack.exx_k_device, nbf, submat_block_size, 
    data->device_backend_->queue() );
#endif
}

}
