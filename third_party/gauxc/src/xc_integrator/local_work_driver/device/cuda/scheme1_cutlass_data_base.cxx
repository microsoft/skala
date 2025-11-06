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
#include "scheme1_cutlass_base.hpp"
#include "buffer_adaptor.hpp"

namespace GauXC {


void AoSScheme1CUTLASSBase::Data::reset_allocations() {
  base_type::reset_allocations();
  cutlass_stack.reset();
  syr2k_sizes_host.clear();
  problem_sizes_host.clear();
}

size_t AoSScheme1CUTLASSBase::Data::get_static_mem_requirement() {
  return base_type::get_static_mem_requirement() + 
         4 * sizeof(int32_t) +
         2 * sizeof(cutlass::gemm::GemmCoord); 
    // Extra elements in CUTLASS dimension arrays
}

size_t AoSScheme1CUTLASSBase::Data::get_mem_req( integrator_term_tracker terms, 
  const host_task_type& task ) {

  auto is_uks = terms.ks_scheme == UKS;
  auto is_gks = terms.ks_scheme == GKS;
  
  size_t base_size = base_type::get_mem_req(terms, task);

  // TODO: There is probably a better way to check this
  required_term_storage reqt(terms);
  if( reqt.task_nbe_scr ) {
    base_size += 
      4*sizeof(double*) + // batch device pointers (containg trial ones)
      4*sizeof(int64_t) +
      2*sizeof(cutlass::gemm::GemmCoord);  // Dimensions + leading dimensions 
                                           // (extra handled by get_static_mem_requirement)
    if(reqt.task_xmat_grad) {
      base_size += 6 * sizeof(double*);
    }

    if(is_uks or is_gks) {
      base_size += sizeof(double*); // z dmat 
    }
    if(is_gks) {
      base_size += 2*sizeof(double*); // x/y dmat 
    }
    
    if(terms.fxc_contraction) {
      base_size += sizeof(double*); // s tdmat
      if(is_uks or is_gks)
        base_size += sizeof(double*); // z tdmat
      if(is_gks) {
        base_size += 2*sizeof(double*); // x/y tdmat
      }
    }

  }
  return base_size;


}
AoSScheme1CUTLASSBase::Data::device_buffer_t 
  AoSScheme1CUTLASSBase::Data::allocate_dynamic_stack( 
  integrator_term_tracker terms,
  host_task_iterator task_begin, host_task_iterator task_end, 
  device_buffer_t buf ){

  // Allocate base info on the stack
  buf = base_type::allocate_dynamic_stack( terms, task_begin, task_end,
    buf );

  required_term_storage reqt(terms);
  if( not reqt.task_nbe_scr ) return buf;

  auto is_uks = terms.ks_scheme == UKS;
  auto is_gks = terms.ks_scheme == GKS;

  // Allocate additional device memory 
  auto [ ptr, sz ] = buf;
  buffer_adaptor mem( ptr, sz );

  const auto ntask = std::distance( task_begin, task_end );
  cutlass_stack.dmat_s_array_device = mem.aligned_alloc<double*>( ntask, csl );
  cutlass_stack.vmat_array_device   = mem.aligned_alloc<double*>( ntask, csl );
  cutlass_stack.zmat_array_device   = mem.aligned_alloc<double*>( ntask, csl );
  cutlass_stack.bf_array_device     = mem.aligned_alloc<double*>( ntask, csl );
  if(reqt.task_xmat_grad) {
    cutlass_stack.bfx_array_device    = mem.aligned_alloc<double*>( ntask, csl );
    cutlass_stack.bfy_array_device    = mem.aligned_alloc<double*>( ntask, csl );
    cutlass_stack.bfz_array_device    = mem.aligned_alloc<double*>( ntask, csl );
    cutlass_stack.xmat_x_array_device = mem.aligned_alloc<double*>( ntask, csl );
    cutlass_stack.xmat_y_array_device = mem.aligned_alloc<double*>( ntask, csl );
    cutlass_stack.xmat_z_array_device = mem.aligned_alloc<double*>( ntask, csl );
  }

  if(is_uks or is_gks) {
    cutlass_stack.dmat_z_array_device = mem.aligned_alloc<double*>( ntask, csl );
  }

  if(is_gks) {
    cutlass_stack.dmat_y_array_device = mem.aligned_alloc<double*>( ntask, csl );
    cutlass_stack.dmat_x_array_device = mem.aligned_alloc<double*>( ntask, csl );
  }

  if(terms.fxc_contraction) {
    cutlass_stack.tdmat_s_array_device = mem.aligned_alloc<double*>( ntask, csl );
    if(is_uks or is_gks)
      cutlass_stack.tdmat_z_array_device = mem.aligned_alloc<double*>( ntask, csl );
    if(is_gks){
      cutlass_stack.tdmat_y_array_device = mem.aligned_alloc<double*>( ntask, csl );
      cutlass_stack.tdmat_x_array_device = mem.aligned_alloc<double*>( ntask, csl );
    }
  }

  cutlass_stack.ld64_dmat_array_device = mem.aligned_alloc<int64_t>( ntask + 1, csl );
  cutlass_stack.ld64_zmat_array_device = mem.aligned_alloc<int64_t>( ntask + 1, csl );
  cutlass_stack.ld64_vmat_array_device = mem.aligned_alloc<int64_t>( ntask + 1, csl );
  cutlass_stack.ld64_bf_array_device   = mem.aligned_alloc<int64_t>( ntask + 1, csl );
  
  cutlass_stack.problem_sizes_device = mem.aligned_alloc<cutlass::gemm::GemmCoord>( ntask + 1, csl );
  cutlass_stack.syr2k_sizes_device   = mem.aligned_alloc<cutlass::gemm::GemmCoord>( ntask + 1, csl );

  // Update dynmem data for derived impls
  return device_buffer_t{ mem.stack(), mem.nleft() };
}

void AoSScheme1CUTLASSBase::Data::pack_and_send( 
  integrator_term_tracker terms,
  host_task_iterator task_begin, host_task_iterator task_end,
  const BasisSetMap& basis_map ) {

  base_type::pack_and_send( terms, task_begin, task_end, basis_map );
  required_term_storage reqt(terms);
  if( not reqt.task_nbe_scr ) return;

  auto is_uks = terms.ks_scheme == UKS;
  auto is_gks = terms.ks_scheme == GKS;

  const auto ntask = std::distance( task_begin, task_end );
  std::vector<double*> dmat_host( ntask ), zmat_host( ntask ), bf_host( ntask ),
                       vmat_host( ntask ), tdmat_host( ntask );
  problem_sizes_host.resize(ntask);
  syr2k_sizes_host.resize(ntask);
  std::vector<int64_t> ld64_dmat_host( ntask ), ld64_zmat_host( ntask ), 
                       ld64_vmat_host( ntask ), ld64_bf_host( ntask );

  const auto nbf = global_dims.nbf;

  // host_device_tasks should be populated by parent impl called at top
  for( auto i = 0; i < ntask; ++i ) {
    auto& task = host_device_tasks[i];
    zmat_host[i] = task.zmat;    ld64_zmat_host[i] = task.npts;
    bf_host[i]   = task.bf;      ld64_bf_host[i]   = task.npts;
    vmat_host[i] = task.nbe_scr; ld64_vmat_host[i] = task.bfn_screening.nbe;
    if( task.bfn_screening.ncut > 1 ) {
      dmat_host[i]    = task.nbe_scr;
      ld64_dmat_host[i] = task.bfn_screening.nbe;
    } else {
      dmat_host[i]    = static_stack.dmat_s_device + task.bfn_screening.ibf_begin*(nbf+1);
      ld64_dmat_host[i] = nbf;
    }

    cutlass::gemm::GemmCoord problem(task.npts, task.bfn_screening.nbe, task.bfn_screening.nbe);
    problem_sizes_host[i] = problem;

    cutlass::gemm::GemmCoord problem2(task.bfn_screening.nbe, task.bfn_screening.nbe, task.npts);
    syr2k_sizes_host[i] = problem2;
  }

  // Send to device
  device_backend_->copy_async( ntask, dmat_host.data(), 
    cutlass_stack.dmat_s_array_device, "send dmat_s array" );
  device_backend_->copy_async( ntask, zmat_host.data(), 
    cutlass_stack.zmat_array_device, "send zmat array" );
  device_backend_->copy_async( ntask, vmat_host.data(), 
    cutlass_stack.vmat_array_device, "send vmat array" );
  device_backend_->copy_async( ntask, bf_host.data(), 
    cutlass_stack.bf_array_device, "send bf array" );

  device_backend_->copy_async( ntask, problem_sizes_host.data(), 
    cutlass_stack.problem_sizes_device, "send problemsize array" );
  device_backend_->copy_async( ntask, syr2k_sizes_host.data(), 
    cutlass_stack.syr2k_sizes_device, "send problemsize array" );
  device_backend_->copy_async( ntask, ld64_dmat_host.data(), 
    cutlass_stack.ld64_dmat_array_device, "send ld dmat array" );
  device_backend_->copy_async( ntask, ld64_zmat_host.data(), 
    cutlass_stack.ld64_zmat_array_device, "send ld zmat array" );
  device_backend_->copy_async( ntask, ld64_vmat_host.data(), 
    cutlass_stack.ld64_vmat_array_device, "send ld vmat array" );
  device_backend_->copy_async( ntask, ld64_bf_host.data(), 
    cutlass_stack.ld64_bf_array_device, "send ld bf array" );

  if(is_uks or is_gks) {
    std::vector<double*> dmat_z_host( ntask );
    for( auto i = 0; i < ntask; ++i ) {
      auto& task = host_device_tasks[i];
      if( task.bfn_screening.ncut > 1 ) {
        dmat_z_host[i] = task.nbe_scr;
      } else {
        dmat_z_host[i] = static_stack.dmat_z_device + task.bfn_screening.ibf_begin*(nbf+1);
      }
    }
    device_backend_->copy_async( ntask, dmat_z_host.data(), 
      cutlass_stack.dmat_z_array_device, "send dmat_z array" );
  }

  if(is_gks) {
    std::vector<double*> dmat_y_host( ntask );
    std::vector<double*> dmat_x_host( ntask );
    for( auto i = 0; i < ntask; ++i ) {
      auto& task = host_device_tasks[i];
      if( task.bfn_screening.ncut > 1 ) {
        dmat_y_host[i] = task.nbe_scr;
        dmat_x_host[i] = task.nbe_scr;
      } else {
        dmat_y_host[i] = static_stack.dmat_y_device + task.bfn_screening.ibf_begin*(nbf+1);
        dmat_x_host[i] = static_stack.dmat_x_device + task.bfn_screening.ibf_begin*(nbf+1);
      }
    }
    device_backend_->copy_async( ntask, dmat_x_host.data(), 
      cutlass_stack.dmat_x_array_device, "send dmat_x array" );
    device_backend_->copy_async( ntask, dmat_y_host.data(), 
      cutlass_stack.dmat_y_array_device, "send dmat_y array" );
  }

  if(reqt.task_xmat_grad) {
    std::vector<double*> xmat_x_host( ntask ), bfx_host( ntask );
    std::vector<double*> xmat_y_host( ntask ), bfy_host( ntask );
    std::vector<double*> xmat_z_host( ntask ), bfz_host( ntask );
    for( auto i = 0; i < ntask; ++i ) {
      auto& task = host_device_tasks[i];
      xmat_x_host[i] = task.xmat_x;
      xmat_y_host[i] = task.xmat_y;
      xmat_z_host[i] = task.xmat_z;
      bfx_host[i]    = task.dbfx;
      bfy_host[i]    = task.dbfy;
      bfz_host[i]    = task.dbfz;
    }
    device_backend_->copy_async( ntask, xmat_x_host.data(), 
      cutlass_stack.xmat_x_array_device, "send xmat_x array" );
    device_backend_->copy_async( ntask, xmat_y_host.data(), 
      cutlass_stack.xmat_y_array_device, "send xmat_y array" );
    device_backend_->copy_async( ntask, xmat_z_host.data(), 
      cutlass_stack.xmat_z_array_device, "send xmat_z array" );
    device_backend_->copy_async( ntask, bfx_host.data(), 
      cutlass_stack.bfx_array_device, "send bfx array" );
    device_backend_->copy_async( ntask, bfy_host.data(), 
      cutlass_stack.bfy_array_device, "send bfy array" );
    device_backend_->copy_async( ntask, bfz_host.data(), 
      cutlass_stack.bfz_array_device, "send bfz array" );
  }

  if(terms.fxc_contraction) {
    std::vector<double*> tdmat_host( ntask );
    for( auto i = 0; i < ntask; ++i ) {
      auto& task = host_device_tasks[i];
      if( task.bfn_screening.ncut > 1 )
        tdmat_host[i] = task.nbe_scr;
      else 
        tdmat_host[i] = static_stack.tdmat_s_device + task.bfn_screening.ibf_begin*(nbf+1);
    }
    device_backend_->copy_async( ntask, tdmat_host.data(), 
      cutlass_stack.tdmat_s_array_device, "send tdmat_s array" );
    if(is_uks or is_gks) {
      std::vector<double*> tdmat_z_host( ntask );
      for( auto i = 0; i < ntask; ++i ) {
        auto& task = host_device_tasks[i];
        if( task.bfn_screening.ncut > 1 )
          tdmat_z_host[i] = task.nbe_scr;
        else 
          tdmat_z_host[i] = static_stack.tdmat_z_device + task.bfn_screening.ibf_begin*(nbf+1);
      }
      device_backend_->copy_async( ntask, tdmat_z_host.data(), 
        cutlass_stack.tdmat_z_array_device, "send tdmat_z array" );
    }
    if(is_gks) {
      std::vector<double*> tdmat_y_host( ntask );
      std::vector<double*> tdmat_x_host( ntask );
      for( auto i = 0; i < ntask; ++i ) {
        auto& task = host_device_tasks[i];
        if( task.bfn_screening.ncut > 1 ) {
          tdmat_y_host[i] = task.nbe_scr;
          tdmat_x_host[i] = task.nbe_scr;
        } else {
          tdmat_y_host[i] = static_stack.tdmat_y_device + task.bfn_screening.ibf_begin*(nbf+1);
          tdmat_x_host[i] = static_stack.tdmat_x_device + task.bfn_screening.ibf_begin*(nbf+1);
        }
      }
      device_backend_->copy_async( ntask, tdmat_x_host.data(), 
        cutlass_stack.tdmat_x_array_device, "send tdmat_x array" );
      device_backend_->copy_async( ntask, tdmat_y_host.data(), 
        cutlass_stack.tdmat_y_array_device, "send tdmat_y array" );
    }
  }

  device_backend_->master_queue_synchronize(); 

}
}
