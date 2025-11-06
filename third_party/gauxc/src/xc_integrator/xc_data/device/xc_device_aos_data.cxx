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
#include "xc_device_aos_data.hpp"
#include "buffer_adaptor.hpp"
#include "integrator_util/integrator_common.hpp"
#include <gauxc/exceptions.hpp>

namespace GauXC {

void XCDeviceAoSData::reset_allocations() {
  XCDeviceStackData::reset_allocations(); // Base implementation
  aos_stack.reset();
}

size_t XCDeviceAoSData::get_mem_req( integrator_term_tracker terms,
  const host_task_type& task ) {

  size_t base_size = XCDeviceStackData::get_mem_req(terms, task);

  required_term_storage reqt(terms);

  const auto& points           = task.points;
  const auto& submat_cut_bfn   = task.bfn_screening.submat_map;
  const auto& submat_block_bfn = task.bfn_screening.submat_block;
  if( reqt.task_submat_cut_bfn and 
    (!submat_cut_bfn.size() or !submat_block_bfn.size()) 
  )
    GAUXC_GENERIC_EXCEPTION("Must Populate Bfn Submat Maps");

  const auto& submat_cut_cou   = task.cou_screening.submat_map;
  const auto& submat_block_cou = task.cou_screening.submat_block;
  if( reqt.task_submat_cut_cou and  
    (!submat_cut_cou.size() or !submat_block_cou.size()) 
  )
    GAUXC_GENERIC_EXCEPTION("Must Populate Cou Submat Maps");

  // Dimensions
  const size_t npts         = points.size();
  const size_t nbe_bfn      = task.bfn_screening.nbe;
  const size_t ncut_bfn     = submat_cut_bfn.size();
  const size_t nblock_bfn   = submat_block_bfn.size();

  const size_t nbe_cou      = task.cou_screening.nbe;
  const size_t ncut_cou     = submat_cut_cou.size();
  const size_t nblock_cou   = submat_block_cou.size();

  return base_size + 
    // Collocation + Derivatives
    reqt.task_bfn_size     ( nbe_bfn, npts )    * sizeof(double) +
    reqt.task_bfn_grad_size( nbe_bfn, npts )    * sizeof(double) +
    reqt.task_bfn_hess_size( nbe_bfn, npts )    * sizeof(double) +
    reqt.task_bfn_lapl_size( nbe_bfn, npts )    * sizeof(double) +
    reqt.task_bfn_lapgrad_size( nbe_bfn, npts ) * sizeof(double) +

    // LDA/GGA Z Matrix
    reqt.task_zmat_size( nbe_bfn, npts ) * sizeof(double) +

    // X Matrix Gradient
    reqt.task_xmat_grad_size( nbe_bfn, npts ) * sizeof(double) +

    // Persistent X Mat
    reqt.task_xmat_persist_size( nbe_bfn, npts ) * sizeof(double) +

    // EXX Intermediates
    reqt.task_fmat_size( nbe_cou, npts ) * sizeof(double) +
    reqt.task_gmat_size( nbe_cou, npts ) * sizeof(double) +

    // NBE Scratch
    reqt.task_nbe_scr_size(nbe_bfn, nbe_cou) * sizeof(double) +

    // Index Packing (bfn)
    reqt.task_submat_cut_bfn_size( ncut_bfn )     * sizeof(int32_t) +
    reqt.task_submat_block_bfn_size( nblock_bfn ) * sizeof(int32_t) +

    // Index Packing (cou)
    reqt.task_submat_cut_cou_size( ncut_cou )     * sizeof(int32_t) +
    reqt.task_submat_block_cou_size( nblock_cou ) * sizeof(int32_t) +

    // Map from packed to unpacked indices
    reqt.task_bfn_shell_indirection_size( nbe_bfn ) * sizeof(int32_t) +
  
    // Memory associated with task indirection: valid for both AoS and SoA
    reqt.task_indirection_size() * sizeof(XCDeviceTask);
}





XCDeviceAoSData::device_buffer_t XCDeviceAoSData::allocate_dynamic_stack( 
  integrator_term_tracker terms,
  host_task_iterator task_begin, host_task_iterator task_end, 
  device_buffer_t buf ) {

  // Allocate base info in the stack
  buf = XCDeviceStackData::allocate_dynamic_stack( terms, task_begin, task_end, 
    buf );


  required_term_storage reqt(terms);

  // Current Stack
  auto [ ptr, sz ] = buf;
  buffer_adaptor mem( ptr, sz );

  // Get dimensions
  total_nbe_scr_task_batch   = 0;
  total_nbe_bfn_task_batch   = 0;

  total_nbe_bfn_npts_task_batch = 0; 
  total_ncut_bfn_task_batch     = 0; 
  total_nblock_bfn_task_batch   = 0; 

  total_nbe_cou_npts_task_batch = 0; 
  total_ncut_cou_task_batch     = 0; 
  total_nblock_cou_task_batch   = 0; 
  for( auto it = task_begin; it != task_end; ++it ) {

    const auto& points           = it->points;
    const auto& submat_cut_bfn   = it->bfn_screening.submat_map;
    const auto& submat_block_bfn = it->bfn_screening.submat_block;
    if( reqt.task_submat_cut_bfn and 
      (!submat_cut_bfn.size() or !submat_block_bfn.size()) 
    )
      GAUXC_GENERIC_EXCEPTION("Must Populate Bfn Submat Maps");

    const auto& submat_cut_cou   = it->cou_screening.submat_map;
    const auto& submat_block_cou = it->cou_screening.submat_block;
    if( reqt.task_submat_cut_cou and  
      (!submat_cut_cou.size() or !submat_block_cou.size()) 
    )
      GAUXC_GENERIC_EXCEPTION("Must Populate Cou Submat Maps");

    const size_t npts        = points.size();

    const size_t ncut_bfn    = submat_cut_bfn.size();
    const size_t nblock_bfn  = submat_block_bfn.size();
    const auto nbe_bfn       = it->bfn_screening.nbe;

    const size_t ncut_cou    = submat_cut_cou.size();
    const size_t nblock_cou  = submat_block_cou.size();
    const auto nbe_cou       = it->cou_screening.nbe;

    total_nbe_bfn_task_batch += nbe_bfn;

    total_nbe_scr_task_batch += reqt.task_nbe_scr_size(nbe_bfn, nbe_cou);

    total_nbe_bfn_npts_task_batch += reqt.task_bfn_size(nbe_bfn, npts);
    total_ncut_bfn_task_batch   += reqt.task_submat_cut_bfn_size(ncut_bfn);
    total_nblock_bfn_task_batch += reqt.task_submat_block_bfn_size(nblock_bfn);

    total_nbe_cou_npts_task_batch += reqt.task_fmat_size(nbe_cou, npts);
    total_ncut_cou_task_batch   += reqt.task_submat_cut_cou_size(ncut_cou);
    total_nblock_cou_task_batch += reqt.task_submat_block_cou_size(nblock_cou);

  }
  
  // Device task indirection
  if(reqt.task_indirection) {
    const size_t ntask = std::distance( task_begin, task_end );
    aos_stack.device_tasks = mem.aligned_alloc<XCDeviceTask>( ntask, csl );
  }
  // Map packed to unpacked indices
  if(reqt.task_bfn_shell_indirection) {
    aos_stack.bfn_shell_indirection_device =
      mem.aligned_alloc<int32_t>( total_nbe_bfn_task_batch, csl );
  }
  // Collocation + derivatives 
  const size_t bfn_msz = total_nbe_bfn_npts_task_batch;
  if(reqt.task_bfn) {
    aos_stack.bf_eval_device = mem.aligned_alloc<double>( bfn_msz, csl );
  }

  if(reqt.task_bfn_grad) {
    aos_stack.dbf_x_eval_device = mem.aligned_alloc<double>( bfn_msz, csl );
    aos_stack.dbf_y_eval_device = mem.aligned_alloc<double>( bfn_msz, csl );
    aos_stack.dbf_z_eval_device = mem.aligned_alloc<double>( bfn_msz, csl );
  }

  if(reqt.task_bfn_hess) {
    aos_stack.d2bf_xx_eval_device = mem.aligned_alloc<double>( bfn_msz, csl );
    aos_stack.d2bf_xy_eval_device = mem.aligned_alloc<double>( bfn_msz, csl );
    aos_stack.d2bf_xz_eval_device = mem.aligned_alloc<double>( bfn_msz, csl );
    aos_stack.d2bf_yy_eval_device = mem.aligned_alloc<double>( bfn_msz, csl );
    aos_stack.d2bf_yz_eval_device = mem.aligned_alloc<double>( bfn_msz, csl );
    aos_stack.d2bf_zz_eval_device = mem.aligned_alloc<double>( bfn_msz, csl );
  }

  if(reqt.task_bfn_lapl) {
    aos_stack.d2bf_lapl_eval_device = mem.aligned_alloc<double>( bfn_msz, csl );
  }

  if(reqt.task_bfn_lapgrad) {
    aos_stack.d3bf_lapgrad_x_eval_device = mem.aligned_alloc<double>( bfn_msz, csl );
    aos_stack.d3bf_lapgrad_y_eval_device = mem.aligned_alloc<double>( bfn_msz, csl );
    aos_stack.d3bf_lapgrad_z_eval_device = mem.aligned_alloc<double>( bfn_msz, csl );
  }

  // VXC Z Matrix
  if(reqt.task_zmat) {
    aos_stack.zmat_vxc_device = 
      mem.aligned_alloc<double>( bfn_msz, csl);
  }
  // X Matrix Gradient (for GGA EXC Gradient)
  if(reqt.task_xmat_grad) {
    aos_stack.xmat_dx_device = mem.aligned_alloc<double>( bfn_msz, csl);
    aos_stack.xmat_dy_device = mem.aligned_alloc<double>( bfn_msz, csl);
    aos_stack.xmat_dz_device = mem.aligned_alloc<double>( bfn_msz, csl);
  }

  // Persistent X Matrix Gradient
  if(reqt.task_xmat_persist) {
    aos_stack.xmatS_device    = mem.aligned_alloc<double>( bfn_msz, csl);
    aos_stack.xmatZ_device    = mem.aligned_alloc<double>( bfn_msz, csl);
    if(reqt.task_xmat_grad) { 
      aos_stack.xmatS_dx_device = mem.aligned_alloc<double>( bfn_msz, csl);
      aos_stack.xmatS_dy_device = mem.aligned_alloc<double>( bfn_msz, csl);
      aos_stack.xmatS_dz_device = mem.aligned_alloc<double>( bfn_msz, csl);
      aos_stack.xmatZ_dx_device = mem.aligned_alloc<double>( bfn_msz, csl);
      aos_stack.xmatZ_dy_device = mem.aligned_alloc<double>( bfn_msz, csl);
      aos_stack.xmatZ_dz_device = mem.aligned_alloc<double>( bfn_msz, csl);
    }
  }

  // EXX Intermediates
  if(reqt.task_fmat) {
    aos_stack.fmat_exx_device = 
      mem.aligned_alloc<double>(total_nbe_cou_npts_task_batch, csl);
  }
  if(reqt.task_gmat) {
    aos_stack.gmat_exx_device = 
      mem.aligned_alloc<double>(total_nbe_cou_npts_task_batch, csl);
  }

  // Scratch buffer
  if(reqt.task_nbe_scr) {
    aos_stack.nbe_scr_device = 
      mem.aligned_alloc<double>( total_nbe_scr_task_batch, csl);
  }

  // Shell index buffers (bfn)
  if(reqt.task_submat_cut_bfn) {
    aos_stack.submat_cut_bfn_device = 
      mem.aligned_alloc<int32_t>(total_ncut_bfn_task_batch, csl);
  }
  if(reqt.task_submat_block_bfn) {
    aos_stack.submat_block_bfn_device = 
      mem.aligned_alloc<int32_t>(total_nblock_bfn_task_batch, csl);
  }

  // Shell index buffers (cou)
  if(reqt.task_submat_cut_cou) {
    aos_stack.submat_cut_cou_device = 
      mem.aligned_alloc<int32_t>(total_ncut_cou_task_batch, csl);
  }
  if(reqt.task_submat_block_cou) {
    aos_stack.submat_block_cou_device = 
      mem.aligned_alloc<int32_t>(total_nblock_cou_task_batch, csl);
  }

  // Update dynmem data for derived impls
  return device_buffer_t{ mem.stack(), mem.nleft() };
}


void XCDeviceAoSData::pack_and_send( 
  integrator_term_tracker terms,
  host_task_iterator task_begin, host_task_iterator task_end,
  const BasisSetMap& basis_map ) {


  // Pack and send base data
  XCDeviceStackData::pack_and_send( terms, task_begin, task_end, basis_map );

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  required_term_storage reqt(terms);

  // Reset AoS
  host_device_tasks.clear();

  // Host Packing Arrays
  std::vector< std::array<int32_t, 3> > submat_cut_bfn_pack;
  std::vector< int32_t > submat_block_bfn_pack;

  std::vector< std::array<int32_t, 3> > submat_cut_cou_pack;
  std::vector< int32_t > submat_block_cou_pack;

  std::vector<int32_t> bfn_shell_indirection_pack;
  bfn_shell_indirection_pack.reserve(total_nbe_bfn_task_batch);


  // Contatenation utility
  auto concat_iterable = []( auto& a, const auto& b ) {
    a.insert( a.end(), b.begin(), b.end() );
  };

  // Pack AoS data and construct indirections
  for( auto it = task_begin; it != task_end; ++it ) {

    const auto  iAtom            = it->iParent;
    const auto& points           = it->points;
    const auto dist_nearest      = it->dist_nearest;

    const auto& submat_cut_bfn   = it->bfn_screening.submat_map;
    const auto& submat_block_bfn = it->bfn_screening.submat_block;
    if( reqt.task_submat_cut_bfn and 
      (!submat_cut_bfn.size() or !submat_block_bfn.size()) 
    )
      GAUXC_GENERIC_EXCEPTION("Must Populate Bfn Submat Maps");

    const auto& submat_cut_cou   = it->cou_screening.submat_map;
    const auto& submat_block_cou = it->cou_screening.submat_block;
    if( reqt.task_submat_cut_cou and  
      (!submat_cut_cou.size() or !submat_block_cou.size()) 
    )
      GAUXC_GENERIC_EXCEPTION("Must Populate Cou Submat Maps");

    // Dimensions
    const size_t npts         = points.size();

    const size_t ncut_bfn     = submat_cut_bfn.size();
    const size_t nblock_bfn   = submat_block_bfn.size();
    const size_t nshells_bfn  = it->bfn_screening.shell_list.size();
    const auto nbe_bfn        = it->bfn_screening.nbe;

    const size_t ncut_cou     = submat_cut_cou.size();
    const size_t nblock_cou   = submat_block_cou.size();
    const size_t nshells_cou  = it->cou_screening.shell_list.size();
    const auto nbe_cou        = it->cou_screening.nbe;


    // Pack Shell indexing
    if(reqt.task_submat_cut_bfn) {
      concat_iterable( submat_cut_bfn_pack, submat_cut_bfn );
    }
    if(reqt.task_submat_block_bfn) {
      concat_iterable( submat_block_bfn_pack, submat_block_bfn );
    }
    if(reqt.task_submat_cut_cou) {
      concat_iterable( submat_cut_cou_pack, submat_cut_cou );
    }
    if(reqt.task_submat_block_cou) {
      concat_iterable( submat_block_cou_pack, submat_block_cou );
    }

    // Map packed to unpacked indices
    if(reqt.task_bfn_shell_indirection) {
      std::vector<int32_t> bfn_indirection(nbe_bfn);
      auto bit = bfn_indirection.begin();
      for( auto& sh : it->bfn_screening.shell_list ) {
        auto sh_range = basis_map.shell_to_ao_range()[sh];
        for( auto j = sh_range.first; j < sh_range.second; ++j ) {
          *bit = j; ++bit;
        }
      }
      concat_iterable(bfn_shell_indirection_pack, bfn_indirection);
    }

    // Add task to device indirection
    if(reqt.task_indirection) {
      auto& ht = host_device_tasks.emplace_back();

      // Populate indirection with dimensions
      ht.npts         = npts;
      ht.iParent      = iAtom;
      ht.dist_nearest = dist_nearest;

      ht.bfn_screening.nbe     = nbe_bfn;
      ht.bfn_screening.ncut    = ncut_bfn;
      ht.bfn_screening.nblock  = nblock_bfn;
      ht.bfn_screening.nshells = nshells_bfn;

      ht.cou_screening.nbe     = nbe_cou;
      ht.cou_screening.ncut    = ncut_cou;
      ht.cou_screening.nblock  = nblock_cou;
      ht.cou_screening.nshells = nshells_cou;

      auto& shell_list_bfn = it->bfn_screening.shell_list;
      ht.bfn_screening.ibf_begin = 
        shell_list_bfn.size() ?
        basis_map.shell_to_first_ao(shell_list_bfn[0]) : 0;

      auto& shell_list_cou = it->cou_screening.shell_list;
      ht.cou_screening.ibf_begin = 
        shell_list_cou.size() ?
        basis_map.shell_to_first_ao(shell_list_cou[0]) : 0;
    }

  }

  // Send shell index information early to overlap with 
  // indirection construction
  if(reqt.task_submat_cut_bfn) {
    device_backend_->copy_async( 3 * submat_cut_bfn_pack.size(), 
      submat_cut_bfn_pack.data()->data(), aos_stack.submat_cut_bfn_device, 
      "send_submat_cut_bfn"  ); 
  }
  if(reqt.task_submat_block_bfn) {
    device_backend_->copy_async( submat_block_bfn_pack.size(), 
      submat_block_bfn_pack.data(), aos_stack.submat_block_bfn_device, 
      "send_submat_block_bfn"  ); 
  }
  if(reqt.task_submat_cut_cou) {
    device_backend_->copy_async( 3 * submat_cut_cou_pack.size(), 
      submat_cut_cou_pack.data()->data(), aos_stack.submat_cut_cou_device, 
      "send_submat_cut_cou"  ); 
  }
  if(reqt.task_submat_block_cou) {
    device_backend_->copy_async( submat_block_cou_pack.size(), 
      submat_block_cou_pack.data(), aos_stack.submat_block_cou_device, 
      "send_submat_block_cou"  ); 
  }


  if(reqt.task_bfn_shell_indirection) {
    device_backend_->copy_async( bfn_shell_indirection_pack.size(), 
      bfn_shell_indirection_pack.data(), aos_stack.bfn_shell_indirection_device, 
      "send_bfn_shell_indirection"  ); 
  }

  // Construct full indirection
  if(reqt.task_indirection) {

    const size_t total_npts    = total_npts_task_batch * sizeof(double);
    buffer_adaptor points_x_mem( base_stack.points_x_device,  total_npts );
    buffer_adaptor points_y_mem( base_stack.points_y_device,  total_npts );
    buffer_adaptor points_z_mem( base_stack.points_z_device,  total_npts );
    buffer_adaptor weights_mem ( base_stack.weights_device,   total_npts );


    const size_t total_nbe_bfn = total_nbe_bfn_task_batch * sizeof(int32_t);
    buffer_adaptor bfn_shell_indirection_mem( 
      aos_stack.bfn_shell_indirection_device, total_nbe_bfn );

    const size_t total_ncut_bfn   = 
      total_ncut_bfn_task_batch   * sizeof(int32_t);
    const size_t total_nblock_bfn = 
      total_nblock_bfn_task_batch * sizeof(int32_t);
    buffer_adaptor submat_cut_bfn_mem( aos_stack.submat_cut_bfn_device, 
      total_ncut_bfn  );
    buffer_adaptor submat_block_bfn_mem( aos_stack.submat_block_bfn_device, 
      total_nblock_bfn);

    const size_t total_ncut_cou   = 
      total_ncut_cou_task_batch   * sizeof(int32_t);
    const size_t total_nblock_cou = 
      total_nblock_cou_task_batch * sizeof(int32_t);
    buffer_adaptor submat_cut_cou_mem( aos_stack.submat_cut_cou_device, 
      total_ncut_cou  );
    buffer_adaptor submat_block_cou_mem( aos_stack.submat_block_cou_device, 
      total_nblock_cou);

    const size_t total_nbe_scr      = 
      total_nbe_scr_task_batch      * sizeof(double);
    const size_t total_nbe_bfn_npts = 
      total_nbe_bfn_npts_task_batch * sizeof(double);
    const size_t total_nbe_cou_npts = 
      total_nbe_cou_npts_task_batch * sizeof(double);
    buffer_adaptor nbe_mem( aos_stack.nbe_scr_device, total_nbe_scr );
    buffer_adaptor zmat_mem( aos_stack.zmat_vxc_device, 
      total_nbe_bfn_npts );

    buffer_adaptor fmat_mem( aos_stack.fmat_exx_device, total_nbe_cou_npts );
    buffer_adaptor gmat_mem( aos_stack.gmat_exx_device, total_nbe_cou_npts );

    buffer_adaptor bf_mem   ( aos_stack.bf_eval_device,    total_nbe_bfn_npts );
    buffer_adaptor dbf_x_mem( aos_stack.dbf_x_eval_device, total_nbe_bfn_npts );
    buffer_adaptor dbf_y_mem( aos_stack.dbf_y_eval_device, total_nbe_bfn_npts );
    buffer_adaptor dbf_z_mem( aos_stack.dbf_z_eval_device, total_nbe_bfn_npts );

    buffer_adaptor d2bf_xx_mem( aos_stack.d2bf_xx_eval_device, 
      total_nbe_bfn_npts );
    buffer_adaptor d2bf_xy_mem( aos_stack.d2bf_xy_eval_device, 
      total_nbe_bfn_npts );
    buffer_adaptor d2bf_xz_mem( aos_stack.d2bf_xz_eval_device, 
      total_nbe_bfn_npts );
    buffer_adaptor d2bf_yy_mem( aos_stack.d2bf_yy_eval_device, 
      total_nbe_bfn_npts );
    buffer_adaptor d2bf_yz_mem( aos_stack.d2bf_yz_eval_device, 
      total_nbe_bfn_npts );
    buffer_adaptor d2bf_zz_mem( aos_stack.d2bf_zz_eval_device, 
      total_nbe_bfn_npts );

    buffer_adaptor d2bf_lapl_mem( aos_stack.d2bf_lapl_eval_device, 
      total_nbe_bfn_npts );

    buffer_adaptor d3bf_lapgrad_x_mem( aos_stack.d3bf_lapgrad_x_eval_device, 
      total_nbe_bfn_npts );
    buffer_adaptor d3bf_lapgrad_y_mem( aos_stack.d3bf_lapgrad_y_eval_device, 
      total_nbe_bfn_npts );
    buffer_adaptor d3bf_lapgrad_z_mem( aos_stack.d3bf_lapgrad_z_eval_device, 
      total_nbe_bfn_npts );

    buffer_adaptor xmat_dx_mem( aos_stack.xmat_dx_device, total_nbe_bfn_npts );
    buffer_adaptor xmat_dy_mem( aos_stack.xmat_dy_device, total_nbe_bfn_npts );
    buffer_adaptor xmat_dz_mem( aos_stack.xmat_dz_device, total_nbe_bfn_npts );

    buffer_adaptor xmatS_mem( aos_stack.xmatS_device, total_nbe_bfn_npts );
    buffer_adaptor xmatS_dx_mem( aos_stack.xmatS_dx_device, total_nbe_bfn_npts );
    buffer_adaptor xmatS_dy_mem( aos_stack.xmatS_dy_device, total_nbe_bfn_npts );
    buffer_adaptor xmatS_dz_mem( aos_stack.xmatS_dz_device, total_nbe_bfn_npts );

    buffer_adaptor xmatZ_mem( aos_stack.xmatZ_device, total_nbe_bfn_npts );
    buffer_adaptor xmatZ_dx_mem( aos_stack.xmatZ_dx_device, total_nbe_bfn_npts );
    buffer_adaptor xmatZ_dy_mem( aos_stack.xmatZ_dy_device, total_nbe_bfn_npts );
    buffer_adaptor xmatZ_dz_mem( aos_stack.xmatZ_dz_device, total_nbe_bfn_npts );
    
    const bool is_rks = terms.ks_scheme == RKS;
    const bool is_uks = terms.ks_scheme == UKS;
    const bool is_gks = terms.ks_scheme == GKS;
    const bool is_pol  = is_uks or is_gks;
    const bool is_gga = terms.xc_approx == GGA;
    const int den_fac   = is_pol ? 2 : 1;
    const int gamma_fac = is_pol ? 3 : 1;
    // second derivative
    const int rhorho_fac   = is_pol ? 3 : 1;
    const int rhogamma_fac = is_pol ? 6 : 1;
    const int rhotau_fac   = is_pol ? 4 : 1;


    buffer_adaptor eps_mem    ( base_stack.eps_eval_device,     total_npts             );

    // RKS
    buffer_adaptor den_s_mem  ( base_stack.den_s_eval_device,  total_npts  );
    buffer_adaptor tau_s_mem  ( base_stack.tau_s_eval_device,  total_npts  );
    buffer_adaptor lapl_s_mem ( base_stack.lapl_s_eval_device, total_npts  );
    buffer_adaptor gamma_mem  ( base_stack.gamma_eval_device,  total_npts * gamma_fac );
    buffer_adaptor vrho_mem   ( base_stack.vrho_eval_device,   total_npts * den_fac   );
    buffer_adaptor vgamma_mem ( base_stack.vgamma_eval_device, total_npts * gamma_fac );
    buffer_adaptor vtau_mem   ( base_stack.vtau_eval_device,   total_npts * den_fac   );
    buffer_adaptor vlapl_mem  ( base_stack.vlapl_eval_device,  total_npts * den_fac   );

    // Polarized KS
    buffer_adaptor den_interleaved_mem  ( base_stack.den_interleaved_device,  total_npts * den_fac   );
    buffer_adaptor tau_interleaved_mem  ( base_stack.tau_interleaved_device,  total_npts * den_fac   );
    buffer_adaptor lapl_interleaved_mem ( base_stack.lapl_interleaved_device, total_npts * den_fac   );
    buffer_adaptor den_z_mem  ( base_stack.den_z_eval_device,  total_npts  );
    buffer_adaptor den_y_mem  ( base_stack.den_y_eval_device,  total_npts  );
    buffer_adaptor den_x_mem  ( base_stack.den_x_eval_device,  total_npts  );
    buffer_adaptor tau_z_mem  ( base_stack.tau_z_eval_device,  total_npts  );
    buffer_adaptor lapl_z_mem ( base_stack.lapl_z_eval_device, total_npts  );

    buffer_adaptor vrho_pos_mem( base_stack.vrho_pos_eval_device, total_npts );
    buffer_adaptor vrho_neg_mem( base_stack.vrho_neg_eval_device, total_npts );
    buffer_adaptor vtau_pos_mem( base_stack.vtau_pos_eval_device, total_npts );
    buffer_adaptor vtau_neg_mem( base_stack.vtau_neg_eval_device, total_npts );
    buffer_adaptor vlapl_pos_mem( base_stack.vlapl_pos_eval_device, total_npts );
    buffer_adaptor vlapl_neg_mem( base_stack.vlapl_neg_eval_device, total_npts );
    buffer_adaptor gamma_pp_mem( base_stack.gamma_pp_eval_device, total_npts );
    buffer_adaptor gamma_pm_mem( base_stack.gamma_pm_eval_device, total_npts );
    buffer_adaptor gamma_mm_mem( base_stack.gamma_mm_eval_device, total_npts );
    buffer_adaptor vgamma_pp_mem( base_stack.vgamma_pp_eval_device, total_npts );
    buffer_adaptor vgamma_pm_mem( base_stack.vgamma_pm_eval_device, total_npts );
    buffer_adaptor vgamma_mm_mem( base_stack.vgamma_mm_eval_device, total_npts );
    buffer_adaptor K_z_mem    ( base_stack.K_z_eval_device,       total_npts );
    buffer_adaptor K_y_mem    ( base_stack.K_y_eval_device,       total_npts );
    buffer_adaptor K_x_mem    ( base_stack.K_x_eval_device,       total_npts );
    buffer_adaptor H_z_mem    ( base_stack.H_z_eval_device,       total_npts );
    buffer_adaptor H_y_mem    ( base_stack.H_y_eval_device,       total_npts );
    buffer_adaptor H_x_mem    ( base_stack.H_x_eval_device,       total_npts );

    // Gradients
    buffer_adaptor dden_sx_mem( base_stack.dden_sx_eval_device,     total_npts );
    buffer_adaptor dden_sy_mem( base_stack.dden_sy_eval_device,     total_npts );
    buffer_adaptor dden_sz_mem( base_stack.dden_sz_eval_device,     total_npts );
    buffer_adaptor dden_zx_mem( base_stack.dden_zx_eval_device,     total_npts );
    buffer_adaptor dden_zy_mem( base_stack.dden_zy_eval_device,     total_npts );
    buffer_adaptor dden_zz_mem( base_stack.dden_zz_eval_device,     total_npts );
    buffer_adaptor dden_yx_mem( base_stack.dden_yx_eval_device,     total_npts );
    buffer_adaptor dden_yy_mem( base_stack.dden_yy_eval_device,     total_npts );
    buffer_adaptor dden_yz_mem( base_stack.dden_yz_eval_device,     total_npts );
    buffer_adaptor dden_xx_mem( base_stack.dden_xx_eval_device,     total_npts );
    buffer_adaptor dden_xy_mem( base_stack.dden_xy_eval_device,     total_npts );
    buffer_adaptor dden_xz_mem( base_stack.dden_xz_eval_device,     total_npts );

    // second derivative
    // RKS
    buffer_adaptor tden_s_mem( base_stack.tden_s_eval_device, total_npts );
    buffer_adaptor ttau_s_mem( base_stack.ttau_s_eval_device, total_npts );
    buffer_adaptor tlapl_s_mem( base_stack.tlapl_s_eval_device, total_npts );
    buffer_adaptor v2rho2_mem( base_stack.v2rho2_eval_device, total_npts * rhorho_fac );
    buffer_adaptor v2rhogamma_mem( base_stack.v2rhogamma_eval_device, total_npts * rhogamma_fac );
    buffer_adaptor v2rholapl_mem( base_stack.v2rholapl_eval_device, total_npts * rhotau_fac );
    buffer_adaptor v2rhotau_mem( base_stack.v2rhotau_eval_device, total_npts * rhotau_fac );
    buffer_adaptor v2gamma2_mem( base_stack.v2gamma2_eval_device, total_npts * rhogamma_fac );
    buffer_adaptor v2gammalapl_mem( base_stack.v2gammalapl_eval_device, total_npts * rhogamma_fac );
    buffer_adaptor v2gammatau_mem( base_stack.v2gammatau_eval_device, total_npts * rhogamma_fac );
    buffer_adaptor v2lapl2_mem( base_stack.v2lapl2_eval_device, total_npts * rhorho_fac );
    buffer_adaptor v2lapltau_mem( base_stack.v2lapltau_eval_device, total_npts * rhotau_fac );
    buffer_adaptor v2tau2_mem( base_stack.v2tau2_eval_device, total_npts * rhorho_fac );

    // Polarized KS
    buffer_adaptor tden_z_mem( base_stack.tden_z_eval_device, total_npts );
    buffer_adaptor tden_y_mem( base_stack.tden_y_eval_device, total_npts );
    buffer_adaptor tden_x_mem( base_stack.tden_x_eval_device, total_npts );
    buffer_adaptor ttau_z_mem( base_stack.ttau_z_eval_device, total_npts );
    buffer_adaptor tlapl_z_mem( base_stack.tlapl_z_eval_device, total_npts );

    buffer_adaptor v2rho2_a_a_mem( base_stack.v2rho2_a_a_eval_device, total_npts );
    buffer_adaptor v2rho2_a_b_mem( base_stack.v2rho2_a_b_eval_device, total_npts );
    buffer_adaptor v2rho2_b_b_mem( base_stack.v2rho2_b_b_eval_device, total_npts );
    buffer_adaptor v2rhogamma_a_aa_mem( base_stack.v2rhogamma_a_aa_eval_device, total_npts );
    buffer_adaptor v2rhogamma_a_ab_mem( base_stack.v2rhogamma_a_ab_eval_device, total_npts );
    buffer_adaptor v2rhogamma_a_bb_mem( base_stack.v2rhogamma_a_bb_eval_device, total_npts );
    buffer_adaptor v2rhogamma_b_aa_mem( base_stack.v2rhogamma_b_aa_eval_device, total_npts );
    buffer_adaptor v2rhogamma_b_ab_mem( base_stack.v2rhogamma_b_ab_eval_device, total_npts );
    buffer_adaptor v2rhogamma_b_bb_mem( base_stack.v2rhogamma_b_bb_eval_device, total_npts );
    buffer_adaptor v2rholapl_a_a_mem( base_stack.v2rholapl_a_a_eval_device, total_npts );
    buffer_adaptor v2rholapl_a_b_mem( base_stack.v2rholapl_a_b_eval_device, total_npts );
    buffer_adaptor v2rholapl_b_a_mem( base_stack.v2rholapl_b_a_eval_device, total_npts );
    buffer_adaptor v2rholapl_b_b_mem( base_stack.v2rholapl_b_b_eval_device, total_npts );
    buffer_adaptor v2rhotau_a_a_mem( base_stack.v2rhotau_a_a_eval_device, total_npts );
    buffer_adaptor v2rhotau_a_b_mem( base_stack.v2rhotau_a_b_eval_device, total_npts );
    buffer_adaptor v2rhotau_b_a_mem( base_stack.v2rhotau_b_a_eval_device, total_npts );
    buffer_adaptor v2rhotau_b_b_mem( base_stack.v2rhotau_b_b_eval_device, total_npts );
    buffer_adaptor v2gamma2_aa_aa_mem( base_stack.v2gamma2_aa_aa_eval_device, total_npts );
    buffer_adaptor v2gamma2_aa_ab_mem( base_stack.v2gamma2_aa_ab_eval_device, total_npts );
    buffer_adaptor v2gamma2_aa_bb_mem( base_stack.v2gamma2_aa_bb_eval_device, total_npts );
    buffer_adaptor v2gamma2_ab_ab_mem( base_stack.v2gamma2_ab_ab_eval_device, total_npts );
    buffer_adaptor v2gamma2_ab_bb_mem( base_stack.v2gamma2_ab_bb_eval_device, total_npts );
    buffer_adaptor v2gamma2_bb_bb_mem( base_stack.v2gamma2_bb_bb_eval_device, total_npts );
    buffer_adaptor v2gammalapl_aa_a_mem( base_stack.v2gammalapl_aa_a_eval_device, total_npts );
    buffer_adaptor v2gammalapl_aa_b_mem( base_stack.v2gammalapl_aa_b_eval_device, total_npts );
    buffer_adaptor v2gammalapl_ab_a_mem( base_stack.v2gammalapl_ab_a_eval_device, total_npts );
    buffer_adaptor v2gammalapl_ab_b_mem( base_stack.v2gammalapl_ab_b_eval_device, total_npts );
    buffer_adaptor v2gammalapl_bb_a_mem( base_stack.v2gammalapl_bb_a_eval_device, total_npts );
    buffer_adaptor v2gammalapl_bb_b_mem( base_stack.v2gammalapl_bb_b_eval_device, total_npts );
    buffer_adaptor v2gammatau_aa_a_mem( base_stack.v2gammatau_aa_a_eval_device, total_npts );
    buffer_adaptor v2gammatau_aa_b_mem( base_stack.v2gammatau_aa_b_eval_device, total_npts );
    buffer_adaptor v2gammatau_ab_a_mem( base_stack.v2gammatau_ab_a_eval_device, total_npts );
    buffer_adaptor v2gammatau_ab_b_mem( base_stack.v2gammatau_ab_b_eval_device, total_npts );
    buffer_adaptor v2gammatau_bb_a_mem( base_stack.v2gammatau_bb_a_eval_device, total_npts );
    buffer_adaptor v2gammatau_bb_b_mem( base_stack.v2gammatau_bb_b_eval_device, total_npts );
    buffer_adaptor v2lapl2_a_a_mem( base_stack.v2lapl2_a_a_eval_device, total_npts );
    buffer_adaptor v2lapl2_a_b_mem( base_stack.v2lapl2_a_b_eval_device, total_npts );
    buffer_adaptor v2lapl2_b_b_mem( base_stack.v2lapl2_b_b_eval_device, total_npts );
    buffer_adaptor v2lapltau_a_a_mem( base_stack.v2lapltau_a_a_eval_device, total_npts );
    buffer_adaptor v2lapltau_a_b_mem( base_stack.v2lapltau_a_b_eval_device, total_npts );
    buffer_adaptor v2lapltau_b_a_mem( base_stack.v2lapltau_b_a_eval_device, total_npts );
    buffer_adaptor v2lapltau_b_b_mem( base_stack.v2lapltau_b_b_eval_device, total_npts );
    buffer_adaptor v2tau2_a_a_mem( base_stack.v2tau2_a_a_eval_device, total_npts );
    buffer_adaptor v2tau2_a_b_mem( base_stack.v2tau2_a_b_eval_device, total_npts );
    buffer_adaptor v2tau2_b_b_mem( base_stack.v2tau2_b_b_eval_device, total_npts );

    // Trial density gradient 
    buffer_adaptor tdden_sx_mem( base_stack.tdden_sx_eval_device, total_npts );
    buffer_adaptor tdden_sy_mem( base_stack.tdden_sy_eval_device, total_npts );
    buffer_adaptor tdden_sz_mem( base_stack.tdden_sz_eval_device, total_npts );
    buffer_adaptor tdden_zx_mem( base_stack.tdden_zx_eval_device, total_npts );
    buffer_adaptor tdden_zy_mem( base_stack.tdden_zy_eval_device, total_npts );
    buffer_adaptor tdden_zz_mem( base_stack.tdden_zz_eval_device, total_npts );
    buffer_adaptor tdden_yx_mem( base_stack.tdden_yx_eval_device, total_npts );
    buffer_adaptor tdden_yy_mem( base_stack.tdden_yy_eval_device, total_npts );
    buffer_adaptor tdden_yz_mem( base_stack.tdden_yz_eval_device, total_npts );
    buffer_adaptor tdden_xx_mem( base_stack.tdden_xx_eval_device, total_npts );
    buffer_adaptor tdden_xy_mem( base_stack.tdden_xy_eval_device, total_npts );
    buffer_adaptor tdden_xz_mem( base_stack.tdden_xz_eval_device, total_npts );

    // Intermediate matrices for contraction
    buffer_adaptor FXC_A_s_mem(  base_stack.FXC_A_s_eval_device,  total_npts);
    buffer_adaptor FXC_Bx_s_mem( base_stack.FXC_Bx_s_eval_device, total_npts);
    buffer_adaptor FXC_By_s_mem( base_stack.FXC_By_s_eval_device, total_npts);
    buffer_adaptor FXC_Bz_s_mem( base_stack.FXC_Bz_s_eval_device, total_npts);
    buffer_adaptor FXC_C_s_mem(  base_stack.FXC_C_s_eval_device,  total_npts);
    buffer_adaptor FXC_A_z_mem(  base_stack.FXC_A_z_eval_device,  total_npts);
    buffer_adaptor FXC_Bx_z_mem( base_stack.FXC_Bx_z_eval_device, total_npts);
    buffer_adaptor FXC_By_z_mem( base_stack.FXC_By_z_eval_device, total_npts);
    buffer_adaptor FXC_Bz_z_mem( base_stack.FXC_Bz_z_eval_device, total_npts);
    buffer_adaptor FXC_C_z_mem(  base_stack.FXC_C_z_eval_device,  total_npts);

    for( auto& task : host_device_tasks ) {
      const auto npts    = task.npts;
      const auto nbe_bfn     = task.bfn_screening.nbe;
      const auto ncut_bfn    = task.bfn_screening.ncut;
      const auto nblock_bfn  = task.bfn_screening.nblock;

      const auto nbe_cou     = task.cou_screening.nbe;
      const auto ncut_cou    = task.cou_screening.ncut;
      const auto nblock_cou  = task.cou_screening.nblock;

      // Grid points
      if(reqt.grid_points) {
        task.points_x = points_x_mem.aligned_alloc<double>(npts, csl);
        task.points_y = points_y_mem.aligned_alloc<double>(npts, csl);
        task.points_z = points_z_mem.aligned_alloc<double>(npts, csl);
      }

      // Grid weights
      task.weights = weights_mem.aligned_alloc<double>(
        reqt.grid_weights_size(npts), csl); 

      // Shell indexing (bfn)
      task.bfn_screening.submat_cut = 
        submat_cut_bfn_mem.aligned_alloc<int32_t>(
          reqt.task_submat_cut_bfn_size( ncut_bfn ), csl);
      task.bfn_screening.submat_block = 
        submat_block_bfn_mem.aligned_alloc<int32_t>(
          reqt.task_submat_block_bfn_size( nblock_bfn ), csl);

      // Shell indexing (cou)
      task.cou_screening.submat_cut = 
        submat_cut_cou_mem.aligned_alloc<int32_t>(
          reqt.task_submat_cut_cou_size( ncut_cou ), csl);
      task.cou_screening.submat_block = 
        submat_block_cou_mem.aligned_alloc<int32_t>(
          reqt.task_submat_block_cou_size( nblock_cou ), csl);

      // NBE scr
      task.nbe_scr = nbe_mem.aligned_alloc<double>( 
        reqt.task_nbe_scr_size(nbe_bfn, nbe_cou), csl);

      // ZMatrix LDA/GGA
      task.zmat = zmat_mem.aligned_alloc<double>( 
        reqt.task_zmat_size(nbe_bfn, npts), csl);

      // Collocation + derivatives
      task.bf = bf_mem.aligned_alloc<double>( 
        reqt.task_bfn_size(nbe_bfn, npts), csl);
      if( reqt.task_bfn_grad ) {
        task.dbfx = dbf_x_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
        task.dbfy = dbf_y_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
        task.dbfz = dbf_z_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
      }
      if( reqt.task_bfn_hess ) {
        task.d2bfxx = d2bf_xx_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
        task.d2bfxy = d2bf_xy_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
        task.d2bfxz = d2bf_xz_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
        task.d2bfyy = d2bf_yy_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
        task.d2bfyz = d2bf_yz_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
        task.d2bfzz = d2bf_zz_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
      }
      if( reqt.task_bfn_lapl ) {
        task.d2bflapl = d2bf_lapl_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
      }
      if( reqt.task_bfn_lapgrad ) {
        task.d3bflapl_x = d3bf_lapgrad_x_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
        task.d3bflapl_y = d3bf_lapgrad_y_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
        task.d3bflapl_z = d3bf_lapgrad_z_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
      }

      // X Matrix gradient
      if( reqt.task_xmat_grad ) {
        task.xmat_x = xmat_dx_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
        task.xmat_y = xmat_dy_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
        task.xmat_z = xmat_dz_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
      }

      // Persistent X matrix
      if( reqt.task_xmat_persist ) {
        task.xmatS   = xmatS_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
        task.xmatZ   = xmatZ_mem.aligned_alloc<double>( nbe_bfn * npts, csl);

        if( reqt.task_xmat_grad ) {
          task.xmatS_x = xmatS_dx_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
          task.xmatS_y = xmatS_dy_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
          task.xmatS_z = xmatS_dz_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
          task.xmatZ_x = xmatZ_dx_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
          task.xmatZ_y = xmatZ_dy_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
          task.xmatZ_z = xmatZ_dz_mem.aligned_alloc<double>( nbe_bfn * npts, csl);
        }
      }


      // Grid function evaluations
      if (reqt.grid_den) {
        task.den_s        = den_s_mem.aligned_alloc<double>( npts, csl );
        if(is_pol) {
          task.den          = den_interleaved_mem.aligned_alloc<double>(npts*2, csl); //Interleaved memory
          task.den_z        = den_z_mem.aligned_alloc<double>( npts, csl);
          if ( is_gks ) {
            task.den_y        = den_y_mem.aligned_alloc<double>( npts, csl);
            task.den_x        = den_x_mem.aligned_alloc<double>( npts, csl);
          }
        }
      }

      if(reqt.grid_den_grad) {
        task.dden_sx = dden_sx_mem.aligned_alloc<double>(npts, csl);
        task.dden_sy = dden_sy_mem.aligned_alloc<double>(npts, csl);
        task.dden_sz = dden_sz_mem.aligned_alloc<double>(npts, csl);
        if( is_pol ) {
          task.dden_zx    = dden_zx_mem.aligned_alloc<double>( npts, csl );
          task.dden_zy    = dden_zy_mem.aligned_alloc<double>( npts, csl );
          task.dden_zz    = dden_zz_mem.aligned_alloc<double>( npts, csl );
          if( is_gks ) {
            task.dden_yx    = dden_yx_mem.aligned_alloc<double>( npts, csl );
            task.dden_yy    = dden_yy_mem.aligned_alloc<double>( npts, csl );
            task.dden_yz    = dden_yz_mem.aligned_alloc<double>( npts, csl );
            task.dden_xx    = dden_xx_mem.aligned_alloc<double>( npts, csl );
            task.dden_xy    = dden_xy_mem.aligned_alloc<double>( npts, csl );
            task.dden_xz    = dden_xz_mem.aligned_alloc<double>( npts, csl );
          }
        }
      }

      if( reqt.grid_gamma ) {
        task.gamma = gamma_mem.aligned_alloc<double>( npts*gamma_fac, csl);
        if( is_pol ) {
            task.gamma_pp    = gamma_pp_mem.aligned_alloc<double>( npts, csl);
            task.gamma_pm    = gamma_pm_mem.aligned_alloc<double>( npts, csl);
            task.gamma_mm    = gamma_mm_mem.aligned_alloc<double>( npts, csl);
        }
      }

      if (reqt.grid_tau) {
        task.tau_s        = tau_s_mem.aligned_alloc<double>( npts, csl );
        if(is_pol) {
          task.tau          = tau_interleaved_mem.aligned_alloc<double>(npts*2, csl); //Interleaved memory
          task.tau_z        = tau_z_mem.aligned_alloc<double>( npts, csl);
        }
      }

      if (reqt.grid_lapl) {
        task.lapl_s        = lapl_s_mem.aligned_alloc<double>( npts, csl );
        if(is_pol) {
          task.lapl          = lapl_interleaved_mem.aligned_alloc<double>(npts*2, csl); //Interleaved memory
          task.lapl_z        = lapl_z_mem.aligned_alloc<double>( npts, csl);
        }
      }


      
      if(reqt.grid_eps)
        task.eps  =   eps_mem.aligned_alloc<double>( reqt.grid_eps_size(npts), csl);

      if( reqt.grid_vrho ) {
        task.vrho =   vrho_mem.aligned_alloc<double>( npts*den_fac, csl);
        if( is_pol ) {
          task.vrho_pos     = vrho_pos_mem.aligned_alloc<double>( npts, csl);
          task.vrho_neg     = vrho_neg_mem.aligned_alloc<double>( npts, csl); 
        }
      }

      if( reqt.grid_vgamma ) {
        task.vgamma = vgamma_mem.aligned_alloc<double>( npts*gamma_fac, csl);
        if( is_pol ) {
            task.vgamma_pp    = vgamma_pp_mem.aligned_alloc<double>( npts, csl);
            task.vgamma_pm    = vgamma_pm_mem.aligned_alloc<double>( npts, csl);
            task.vgamma_mm    = vgamma_mm_mem.aligned_alloc<double>( npts, csl);
        }
      }

      if( reqt.grid_vtau ) {
        task.vtau =   vtau_mem.aligned_alloc<double>( npts*den_fac, csl);
        if( is_pol ) {
          task.vtau_pos     = vtau_pos_mem.aligned_alloc<double>( npts, csl);
          task.vtau_neg     = vtau_neg_mem.aligned_alloc<double>( npts, csl); 
        }
      }

      if( reqt.grid_vlapl ) {
        task.vlapl =   vlapl_mem.aligned_alloc<double>( npts*den_fac, csl);
        if( is_pol ) {
          task.vlapl_pos     = vlapl_pos_mem.aligned_alloc<double>( npts, csl);
          task.vlapl_neg     = vlapl_neg_mem.aligned_alloc<double>( npts, csl); 
        }
      }

      
      // H, K terms (GKS)
      if( is_gks ) {
        task.K_x    = K_x_mem.aligned_alloc<double>( npts, csl );
        task.K_y    = K_y_mem.aligned_alloc<double>( npts, csl );
        task.K_z    = K_z_mem.aligned_alloc<double>( npts, csl );
        if( is_gga ) {
          task.H_x    = H_x_mem.aligned_alloc<double>( npts, csl );
          task.H_y    = H_y_mem.aligned_alloc<double>( npts, csl );
          task.H_z    = H_z_mem.aligned_alloc<double>( npts, csl );
        }
      }

      // EXX Specific
      task.fmat = fmat_mem.aligned_alloc<double>(
        reqt.task_fmat_size(nbe_cou,npts), csl);
      task.gmat = gmat_mem.aligned_alloc<double>(
        reqt.task_gmat_size(nbe_cou,npts), csl);


      task.bfn_shell_indirection =
        bfn_shell_indirection_mem.aligned_alloc<int32_t>( 
          reqt.task_bfn_shell_indirection_size(nbe_bfn), csl
        );

      // Second derivative
      if( terms.fxc_contraction ) {
        // Trial density
        if(reqt.grid_tden) {
          task.tden_s = tden_s_mem.aligned_alloc<double>( npts, csl );
          if(is_pol) {
            task.tden_z = tden_z_mem.aligned_alloc<double>( npts, csl );
            if(is_gks) {
              task.tden_y = tden_y_mem.aligned_alloc<double>( npts, csl );
              task.tden_x = tden_x_mem.aligned_alloc<double>( npts, csl );
            }
          }
        }

        if(reqt.grid_tden_grad) {
          task.tdden_sx = tdden_sx_mem.aligned_alloc<double>( npts, csl );
          task.tdden_sy = tdden_sy_mem.aligned_alloc<double>( npts, csl );
          task.tdden_sz = tdden_sz_mem.aligned_alloc<double>( npts, csl );
          if(is_pol) {
            task.tdden_zx = tdden_zx_mem.aligned_alloc<double>( npts, csl );
            task.tdden_zy = tdden_zy_mem.aligned_alloc<double>( npts, csl );
            task.tdden_zz = tdden_zz_mem.aligned_alloc<double>( npts, csl );
            if(is_gks) {
              task.tdden_yx = tdden_yx_mem.aligned_alloc<double>( npts, csl );
              task.tdden_yy = tdden_yy_mem.aligned_alloc<double>( npts, csl );
              task.tdden_yz = tdden_yz_mem.aligned_alloc<double>( npts, csl );
              task.tdden_xx = tdden_xx_mem.aligned_alloc<double>( npts, csl );
              task.tdden_xy = tdden_xy_mem.aligned_alloc<double>( npts, csl );
              task.tdden_xz = tdden_xz_mem.aligned_alloc<double>( npts, csl );
            }
          }
        }


        if(reqt.grid_ttau) {
          task.ttau_s = ttau_s_mem.aligned_alloc<double>( npts, csl );
          if(is_pol) {
            task.ttau_z = ttau_z_mem.aligned_alloc<double>( npts, csl );
          }
        }

        if(reqt.grid_tlapl) {
          task.tlapl_s = tlapl_s_mem.aligned_alloc<double>( npts, csl );
          if(is_pol) {
            task.tlapl_z = tlapl_z_mem.aligned_alloc<double>( npts, csl );
          }
        }

        // Second derivatives of XC functional
        if(reqt.grid_v2rho2) {
          task.v2rho2 = v2rho2_mem.aligned_alloc<double>( npts*rhorho_fac, csl );
          if(is_pol) {
            task.v2rho2_a_a = v2rho2_a_a_mem.aligned_alloc<double>( npts, csl );
            task.v2rho2_a_b = v2rho2_a_b_mem.aligned_alloc<double>( npts, csl );
            task.v2rho2_b_b = v2rho2_b_b_mem.aligned_alloc<double>( npts, csl );
          }
        }

        if(reqt.grid_v2rhogamma) {
          task.v2rhogamma = v2rhogamma_mem.aligned_alloc<double>( npts*rhogamma_fac, csl );
          if(is_pol) {
            task.v2rhogamma_a_aa = v2rhogamma_a_aa_mem.aligned_alloc<double>( npts, csl );
            task.v2rhogamma_a_ab = v2rhogamma_a_ab_mem.aligned_alloc<double>( npts, csl );
            task.v2rhogamma_a_bb = v2rhogamma_a_bb_mem.aligned_alloc<double>( npts, csl );
            task.v2rhogamma_b_aa = v2rhogamma_b_aa_mem.aligned_alloc<double>( npts, csl );
            task.v2rhogamma_b_ab = v2rhogamma_b_ab_mem.aligned_alloc<double>( npts, csl );
            task.v2rhogamma_b_bb = v2rhogamma_b_bb_mem.aligned_alloc<double>( npts, csl );
          }
        }

        if(reqt.grid_v2rholapl) {
          task.v2rholapl = v2rholapl_mem.aligned_alloc<double>( npts*rhotau_fac, csl );
          if(is_pol) {
            task.v2rholapl_a_a = v2rholapl_a_a_mem.aligned_alloc<double>( npts, csl );
            task.v2rholapl_a_b = v2rholapl_a_b_mem.aligned_alloc<double>( npts, csl );
            task.v2rholapl_b_a = v2rholapl_b_a_mem.aligned_alloc<double>( npts, csl );
            task.v2rholapl_b_b = v2rholapl_b_b_mem.aligned_alloc<double>( npts, csl );
          }
        }

        if(reqt.grid_v2rhotau) {
          task.v2rhotau = v2rhotau_mem.aligned_alloc<double>( npts*rhotau_fac, csl );
          if(is_pol) {
            task.v2rhotau_a_a = v2rhotau_a_a_mem.aligned_alloc<double>( npts, csl );
            task.v2rhotau_a_b = v2rhotau_a_b_mem.aligned_alloc<double>( npts, csl );
            task.v2rhotau_b_a = v2rhotau_b_a_mem.aligned_alloc<double>( npts, csl );
            task.v2rhotau_b_b = v2rhotau_b_b_mem.aligned_alloc<double>( npts, csl );
          }
        }

        if(reqt.grid_v2gamma2) {
          task.v2gamma2 = v2gamma2_mem.aligned_alloc<double>( npts*rhogamma_fac, csl );
          if(is_pol) {
            task.v2gamma2_aa_aa = v2gamma2_aa_aa_mem.aligned_alloc<double>( npts, csl );
            task.v2gamma2_aa_ab = v2gamma2_aa_ab_mem.aligned_alloc<double>( npts, csl );
            task.v2gamma2_aa_bb = v2gamma2_aa_bb_mem.aligned_alloc<double>( npts, csl );
            task.v2gamma2_ab_ab = v2gamma2_ab_ab_mem.aligned_alloc<double>( npts, csl );
            task.v2gamma2_ab_bb = v2gamma2_ab_bb_mem.aligned_alloc<double>( npts, csl );
            task.v2gamma2_bb_bb = v2gamma2_bb_bb_mem.aligned_alloc<double>( npts, csl );
          }
        }

        if(reqt.grid_v2gammalapl) {
          task.v2gammalapl = v2gammalapl_mem.aligned_alloc<double>( npts*rhogamma_fac, csl );
          if(is_pol) {
            task.v2gammalapl_aa_a = v2gammalapl_aa_a_mem.aligned_alloc<double>( npts, csl );
            task.v2gammalapl_aa_b = v2gammalapl_aa_b_mem.aligned_alloc<double>( npts, csl );
            task.v2gammalapl_ab_a = v2gammalapl_ab_a_mem.aligned_alloc<double>( npts, csl );
            task.v2gammalapl_ab_b = v2gammalapl_ab_b_mem.aligned_alloc<double>( npts, csl );
            task.v2gammalapl_bb_a = v2gammalapl_bb_a_mem.aligned_alloc<double>( npts, csl );
            task.v2gammalapl_bb_b = v2gammalapl_bb_b_mem.aligned_alloc<double>( npts, csl );
          }
        }

        if(reqt.grid_v2gammatau) {
          task.v2gammatau = v2gammatau_mem.aligned_alloc<double>( npts*rhogamma_fac, csl );
          if(is_pol) {
            task.v2gammatau_aa_a = v2gammatau_aa_a_mem.aligned_alloc<double>( npts, csl );
            task.v2gammatau_aa_b = v2gammatau_aa_b_mem.aligned_alloc<double>( npts, csl );
            task.v2gammatau_ab_a = v2gammatau_ab_a_mem.aligned_alloc<double>( npts, csl );
            task.v2gammatau_ab_b = v2gammatau_ab_b_mem.aligned_alloc<double>( npts, csl );
            task.v2gammatau_bb_a = v2gammatau_bb_a_mem.aligned_alloc<double>( npts, csl );
            task.v2gammatau_bb_b = v2gammatau_bb_b_mem.aligned_alloc<double>( npts, csl );
          }
        }

        if(reqt.grid_v2lapl2) {
          task.v2lapl2 = v2lapl2_mem.aligned_alloc<double>( npts*rhorho_fac, csl );
          if(is_pol) {
            task.v2lapl2_a_a = v2lapl2_a_a_mem.aligned_alloc<double>( npts, csl );
            task.v2lapl2_a_b = v2lapl2_a_b_mem.aligned_alloc<double>( npts, csl );
            task.v2lapl2_b_b = v2lapl2_b_b_mem.aligned_alloc<double>( npts, csl );
          }
        }

        if(reqt.grid_v2lapltau) {
          task.v2lapltau = v2lapltau_mem.aligned_alloc<double>( npts*rhotau_fac, csl );
          if(is_pol) {
            task.v2lapltau_a_a = v2lapltau_a_a_mem.aligned_alloc<double>( npts, csl );
            task.v2lapltau_a_b = v2lapltau_a_b_mem.aligned_alloc<double>( npts, csl );
            task.v2lapltau_b_a = v2lapltau_b_a_mem.aligned_alloc<double>( npts, csl );
            task.v2lapltau_b_b = v2lapltau_b_b_mem.aligned_alloc<double>( npts, csl );
          }
        }

        if(reqt.grid_v2tau2) {
          task.v2tau2 = v2tau2_mem.aligned_alloc<double>( npts*rhorho_fac, csl );
          if(is_pol) {
            task.v2tau2_a_a = v2tau2_a_a_mem.aligned_alloc<double>( npts, csl );
            task.v2tau2_a_b = v2tau2_a_b_mem.aligned_alloc<double>( npts, csl );
            task.v2tau2_b_b = v2tau2_b_b_mem.aligned_alloc<double>( npts, csl );
          }
        }

        // Intermediate matrices for contraction
        if(reqt.grid_FXC_A) {
          task.FXC_A_s = FXC_A_s_mem.aligned_alloc<double>( npts, csl );
          if (is_pol)
            task.FXC_A_z = FXC_A_z_mem.aligned_alloc<double>( npts, csl );
        }

        if(reqt.grid_FXC_B) {
          task.FXC_Bx_s = FXC_Bx_s_mem.aligned_alloc<double>( npts, csl );
          task.FXC_By_s = FXC_By_s_mem.aligned_alloc<double>( npts, csl );
          task.FXC_Bz_s = FXC_Bz_s_mem.aligned_alloc<double>( npts, csl );
          if (is_pol) {
            task.FXC_Bx_z = FXC_Bx_z_mem.aligned_alloc<double>( npts, csl );
            task.FXC_By_z = FXC_By_z_mem.aligned_alloc<double>( npts, csl );
            task.FXC_Bz_z = FXC_Bz_z_mem.aligned_alloc<double>( npts, csl );
          }
        }

        if(reqt.grid_FXC_C) {
          task.FXC_C_s = FXC_C_s_mem.aligned_alloc<double>( npts, csl );
          if (is_pol)
            task.FXC_C_z = FXC_C_z_mem.aligned_alloc<double>( npts, csl );
        }
      }

    } // Loop over device tasks

  } // Setup indirection



  // Setup extra pieces to indirection which are algorithm specific
  add_extra_to_indirection(terms, host_device_tasks);

  // Send indirection 
  device_backend_->copy_async( host_device_tasks.size(), host_device_tasks.data(), 
    aos_stack.device_tasks, "send_tasks_device" );


  // Synchronize on the copy stream to keep host vecs in scope
  device_backend_->master_queue_synchronize(); 


}



void XCDeviceAoSData::populate_submat_maps( 
  size_t N,
  host_task_iterator task_begin, host_task_iterator task_end, 
  const BasisSetMap& basis_map ) {


  // Get packing size 
  const size_t submat_chunk_size = this->get_submat_chunk_size(N,0);

  for( auto it = task_begin; it != task_end; ++it ) {

    const auto& shell_list_bfn = it->bfn_screening.shell_list;
    if( shell_list_bfn.size() ) {
      std::tie( it->bfn_screening.submat_map, it->bfn_screening.submat_block ) = 
        gen_compressed_submat_map( basis_map, shell_list_bfn, N, submat_chunk_size );
    }

    const auto& shell_list_cou = it->cou_screening.shell_list;
    if( shell_list_cou.size() ) {
      std::tie( it->cou_screening.submat_map, it->cou_screening.submat_block ) = 
        gen_compressed_submat_map( basis_map, shell_list_cou, N, submat_chunk_size );
    }

  }

}

}
