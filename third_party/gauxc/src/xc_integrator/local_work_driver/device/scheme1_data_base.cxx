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
#include "scheme1_data_base.hpp"
#include "buffer_adaptor.hpp"

namespace GauXC {

Scheme1DataBase::~Scheme1DataBase() noexcept = default;

Scheme1DataBase::Scheme1DataBase(const DeviceRuntimeEnvironment& rt) : 
  base_type(rt) {

  if( device_backend_ ) 
    device_backend_->create_blas_queue_pool(4);

}

void Scheme1DataBase::reset_allocations() {
  base_type::reset_allocations();
  scheme1_stack.reset();
  collocation_stack.reset();
  //coulomb_stack.reset();
  shell_to_task_stack.reset();
  shell_pair_to_task_stack.reset();
  
  l_batched_shell_to_task.clear();

  task_to_shell_pair_stack.reset();
  subtask.clear();
  nprim_pairs_host.clear();
  pp_ptr_host.clear();

  sp_X_AB_host.clear();
  sp_Y_AB_host.clear();
  sp_Z_AB_host.clear();

  task_to_shell_pair.clear();
}

size_t Scheme1DataBase::get_static_mem_requirement() {
  size_t size = 0;

  const size_t nsp = global_dims.nshell_pairs;
  const size_t total_npts = global_dims.total_npts;
  
  size += 
    // Shell Pair map
    global_dims.nshells * sizeof(ShellToTaskDevice) +
    nsp * sizeof(ShellPairToTaskDevice) +
    // Task Map
    nsp * sizeof(int32_t) +      // nprim_pairs
    nsp * sizeof(shell_pair*) +  // shell_pair pointer
    nsp * 3 * sizeof(double) +    // X_AB, Y_AB, Z_AB
    total_npts * 24 * sizeof(double) +   // space for onedft features and results
    1024 * 1024;                 // additional memory for alignment padding

  return size;
}









size_t Scheme1DataBase::get_mem_req( integrator_term_tracker terms, 
  const host_task_type& task ) {

  // All local memory is weights related
  size_t base_size = base_type::get_mem_req(terms, task);

  required_term_storage reqt(terms);
  const auto ldatoms = get_ldatoms();
  const auto npts = task.npts;
  const auto& shell_list_bfn = task.bfn_screening.shell_list;
  const auto& shell_list_cou = task.cou_screening.shell_list;
  const size_t nshells_bfn  = shell_list_bfn.size();
  const size_t nshells_cou  = shell_list_cou.size();

  const int max_l = global_dims.max_l;
  const size_t n_sp_types = (max_l+1) * (max_l+1); 
  const size_t n_sp_types_with_diag = n_sp_types + (max_l+1); 

  base_size += 
    // Weights specific memory
    reqt.grid_to_center_dist_scr_size(ldatoms, npts) * sizeof(double)  +
    reqt.grid_to_center_dist_nearest_size(npts)      * sizeof(double)  +
    reqt.grid_to_parent_center_size(npts)            * sizeof(int32_t) +

    // Shell / Shell Pair lists + indirection
    reqt.task_shell_list_bfn_size(nshells_bfn)            * sizeof(size_t)  +
    reqt.task_shell_offs_bfn_size(nshells_bfn)            * sizeof(size_t)  +
    reqt.shell_to_task_idx_bfn_size(nshells_bfn)          * sizeof(int32_t) +
    reqt.shell_to_task_off_bfn_size(nshells_bfn)          * sizeof(int32_t) +
    reqt.shell_pair_to_task_idx_cou_size(nshells_cou)     * sizeof(int32_t) +
    reqt.shell_pair_to_task_row_off_cou_size(nshells_cou) * sizeof(int32_t) +
    reqt.shell_pair_to_task_col_off_cou_size(nshells_cou) * sizeof(int32_t) +

    // Task to shell pair map
    reqt.task_to_shell_pair_cou_size() * n_sp_types_with_diag * sizeof(TaskToShellPairDevice) +
    reqt.task_to_shell_pair_col_off_cou_size(nshells_cou) * sizeof(int32_t) +
    reqt.task_to_shell_pair_row_off_cou_size(nshells_cou) * sizeof(int32_t) +
    reqt.task_to_shell_pair_idx_cou_size(nshells_cou) * sizeof(int32_t) + 
    reqt.task_to_shell_pair_cou_subtask_size(npts, 256) * sizeof(std::array<int32_t, 4>);


  //std::cout << "MEM REQ: " << base_size << std::endl;
  return base_size;
}










Scheme1DataBase::device_buffer_t Scheme1DataBase::allocate_dynamic_stack( 
  integrator_term_tracker terms,
  host_task_iterator task_begin, host_task_iterator task_end, 
  device_buffer_t buf ){

  // Allocate base info on the stack
  buf = base_type::allocate_dynamic_stack( terms, task_begin, task_end,
    buf );

  // Allocate additional device memory 
  auto [ ptr, sz ] = buf;
  buffer_adaptor mem( ptr, sz );


  required_term_storage reqt(terms);

  // Weights related memory
  if(reqt.grid_to_center_dist_scr) {
    const auto ldatoms = get_ldatoms();
    scheme1_stack.dist_scratch_device = mem.aligned_alloc<double>( 
      ldatoms * total_npts_task_batch, alignof(double2), csl );
  }
  if(reqt.grid_to_center_dist_nearest) {
    scheme1_stack.dist_nearest_device = 
      mem.aligned_alloc<double>( total_npts_task_batch, csl );
  }
  if(reqt.grid_to_parent_center) {
    scheme1_stack.iparent_device = 
      mem.aligned_alloc<int32_t>( total_npts_task_batch, csl );
  }

  // Compute total dimensions for shell(pair) lists
  total_nshells_bfn_task_batch       = 0; 
  total_nshells_cou_sqlt_task_batch  = 0; 
  size_t num_subtasks = 0;
  const int points_per_subtask = get_points_per_subtask();
  for( auto it = task_begin; it != task_end; ++it ) {
    const auto& shell_list_bfn  = it->bfn_screening.shell_list;
    const size_t nshells_bfn  = shell_list_bfn.size();
    total_nshells_bfn_task_batch  += nshells_bfn;

    const auto& shell_list_cou  = it->cou_screening.shell_list;
    const size_t nshells_cou  = shell_list_cou.size();
    const size_t nshells_cou_sqlt = (nshells_cou*(nshells_cou+1))/2;
    total_nshells_cou_sqlt_task_batch  += nshells_cou_sqlt;

    num_subtasks += util::div_ceil(it->npts, points_per_subtask);
  }

  // Shell lists and offs (bfn)
  if(reqt.task_shell_list_bfn) {
    collocation_stack.shell_list_device = 
      mem.aligned_alloc<size_t>( total_nshells_bfn_task_batch , csl);
  }
  if(reqt.task_shell_offs_bfn) {
    collocation_stack.shell_offs_device = 
      mem.aligned_alloc<size_t>( total_nshells_bfn_task_batch , csl);
  }

  // Shell -> Task buffers
  if(reqt.shell_to_task_bfn) {
    shell_to_task_stack.shell_to_task_idx_device = 
      mem.aligned_alloc<int32_t>( total_nshells_bfn_task_batch, csl );
    shell_to_task_stack.shell_to_task_off_device = 
      mem.aligned_alloc<int32_t>( total_nshells_bfn_task_batch, csl );
    shell_to_task_stack.shell_to_task_device =
      mem.aligned_alloc<ShellToTaskDevice>( global_dims.nshells, csl );
  }

  const size_t nsp = global_dims.nshell_pairs;

  // ShellPair -> Task buffer (cou)
  if(reqt.shell_pair_to_task_cou) {
    throw std::runtime_error("SPARSE + SP2TASK NYI");
    shell_pair_to_task_stack.shell_pair_to_task_idx_device = 
      mem.aligned_alloc<int32_t>( total_nshells_cou_sqlt_task_batch, csl );
    shell_pair_to_task_stack.shell_pair_to_task_row_off_device = 
      mem.aligned_alloc<int32_t>( total_nshells_cou_sqlt_task_batch, csl );
    shell_pair_to_task_stack.shell_pair_to_task_col_off_device = 
      mem.aligned_alloc<int32_t>( total_nshells_cou_sqlt_task_batch, csl );

    shell_pair_to_task_stack.shell_pair_to_task_device =
      mem.aligned_alloc<ShellPairToTaskDevice>( nsp, csl );
  }

  // Task -> ShellPair (cou)
  if(reqt.task_to_shell_pair_cou) { 
    const size_t ntasks = std::distance(task_begin, task_end);
    const int max_l = global_dims.max_l;
    const size_t n_sp_types = (max_l+1) * (max_l+1); 
    const size_t n_sp_types_with_diag = n_sp_types + (max_l+1); 
    task_to_shell_pair_stack.task_to_shell_pair_device = 
      mem.aligned_alloc<TaskToShellPairDevice>( ntasks * n_sp_types_with_diag, csl );

    task_to_shell_pair_stack.task_shell_linear_idx_device = 
      mem.aligned_alloc<int32_t>( total_nshells_cou_sqlt_task_batch, csl);
    task_to_shell_pair_stack.task_shell_off_row_device = 
      mem.aligned_alloc<int32_t>( total_nshells_cou_sqlt_task_batch, csl);
    task_to_shell_pair_stack.task_shell_off_col_device = 
      mem.aligned_alloc<int32_t>( total_nshells_cou_sqlt_task_batch, csl);

    task_to_shell_pair_stack.subtask_device = 
      mem.aligned_alloc<std::array<int32_t, 4>>( num_subtasks, 16, csl );

    task_to_shell_pair_stack.nprim_pairs_device = 
      mem.aligned_alloc<int32_t>( nsp, 16, csl );
    task_to_shell_pair_stack.pp_ptr_device = 
      mem.aligned_alloc<GauXC::PrimitivePair<double>*>( nsp, 16, csl );
    task_to_shell_pair_stack.sp_X_AB_device = 
      mem.aligned_alloc<double>( nsp, 16, csl );
    task_to_shell_pair_stack.sp_Y_AB_device = 
      mem.aligned_alloc<double>( nsp, 16, csl );
    task_to_shell_pair_stack.sp_Z_AB_device = 
      mem.aligned_alloc<double>( nsp, 16, csl );

  }



  // Update dynmem data for derived impls
  return device_buffer_t{ mem.stack(), mem.nleft() };
}









void Scheme1DataBase::pack_and_send( 
  integrator_term_tracker terms,
  host_task_iterator task_begin, host_task_iterator task_end,
  const BasisSetMap& basis_map ) {

  // Pack and send base data
  base_type::pack_and_send( terms, task_begin, task_end, basis_map );


  required_term_storage reqt(terms);

  // Host Packing Arrays
  std::vector<int32_t> iparent_pack;
  std::vector<double>  dist_nearest_pack;
  std::vector<size_t> shell_list_bfn_pack, shell_offs_bfn_pack;
  std::vector< std::vector<int32_t> > 
    shell_to_task_idx_bfn, shell_to_task_off_bfn;
  std::vector<int32_t>
    concat_shell_to_task_idx_bfn, concat_shell_to_task_off_bfn,
    concat_shell_pair_to_task_idx_cou, concat_shell_pair_to_task_off_row_cou, 
    concat_shell_pair_to_task_off_col_cou;
  std::vector<ShellToTaskDevice> host_shell_to_task_bfn;
  std::vector<ShellPairToTaskDevice> host_shell_pair_to_task_cou;

  // Contatenation utility
  auto concat_iterable = []( auto& a, const auto& b ) {
    a.insert( a.end(), b.begin(), b.end() );
  };

  using hrt_t = std::chrono::high_resolution_clock;
  using dur_t = std::chrono::duration<double,std::milli>;

  /*******************************************
   *         WEIGHTS RELATED MEMORY          *
   *******************************************/

  auto w_mem_st = hrt_t::now();
  // Nearest Distance Array
  if(reqt.grid_to_center_dist_nearest) {

    // Pack on host
    dist_nearest_pack.reserve( total_npts_task_batch );
    for( auto it = task_begin; it != task_end; ++it ) {
      dist_nearest_pack.insert( dist_nearest_pack.end(), it->points.size(), 
        it->dist_nearest );
    }

    // Send to device
    device_backend_->copy_async( dist_nearest_pack.size(), 
      dist_nearest_pack.data(), scheme1_stack.dist_nearest_device, 
      "send dist_nearest" );

  }

  // IParent Array
  if(reqt.grid_to_parent_center) {

    // Pack on host
    iparent_pack.reserve( total_npts_task_batch );
    for( auto it = task_begin; it != task_end; ++it ) {
      iparent_pack.insert( iparent_pack.end(), it->points.size(), it->iParent );
    }

    // Send to device
    device_backend_->copy_async( iparent_pack.size(), iparent_pack.data(), 
      scheme1_stack.iparent_device, "send iparent"  );

  }
  auto w_mem_en = hrt_t::now();

  /************************************************
   * SHELL LIST, OFFSET and TASK MAP MEMORY (bfn) *
   ************************************************/

  auto sl_mem_st = hrt_t::now();
  // Resize host arrays for Shell -> Task
  if(reqt.shell_to_task_bfn) {
    shell_to_task_idx_bfn.resize( global_dims.nshells );
    shell_to_task_off_bfn.resize( global_dims.nshells );
  }

  // Shell list, offsets + task map (bfn)
  for( auto it = task_begin; it != task_end; ++it ) {
    const auto& shell_list_bfn  = it->bfn_screening.shell_list;
    const size_t nshells_bfn  = shell_list_bfn.size();

    // Pack shell list (bfn)
    if(reqt.task_shell_list_bfn) {
      concat_iterable( shell_list_bfn_pack, shell_list_bfn );
    }
    
    // Generate and pack shell offsets (bfn)
    std::vector<size_t> shell_offs_bfn;
    if(reqt.task_shell_offs_bfn) {
      shell_offs_bfn = basis_map.shell_offs<size_t>( 
        shell_list_bfn.begin(), shell_list_bfn.end() );
      concat_iterable( shell_offs_bfn_pack, shell_offs_bfn );
    }

    // Setup meta data for Shell -> Task (bfn)
    if(reqt.shell_to_task_bfn) {
      const auto itask = std::distance( task_begin, it );
      for( auto i = 0ul; i < nshells_bfn; ++i ) {
        const auto sh_idx = shell_list_bfn.at(i);
        shell_to_task_idx_bfn[sh_idx].emplace_back(itask);
        shell_to_task_off_bfn[sh_idx].emplace_back(shell_offs_bfn.at(i));
      }
    }
  }

  // Send Shell list and offsets (bfn) to device
  if(reqt.task_shell_list_bfn) {
    device_backend_->copy_async( shell_list_bfn_pack.size(), 
      shell_list_bfn_pack.data(), collocation_stack.shell_list_device, 
      "send_shell_list_bfn" );
  } 
  if(reqt.task_shell_offs_bfn) {
    device_backend_->copy_async( shell_offs_bfn_pack.size(), 
      shell_offs_bfn_pack.data(), collocation_stack.shell_offs_device, 
      "send_shell_offs_bfn" );
  }

  auto sl_mem_en = hrt_t::now();
  /*****************************************
   *     GENERATE SHELL -> TASK (bfn)      *
   *****************************************/
  auto s2t_mem_st = hrt_t::now();
  if(reqt.shell_to_task_bfn) {
      
    // Set up buffer allocations from preallocated device segments
    const size_t total_nshells_bfn = 
      total_nshells_bfn_task_batch * sizeof(int32_t);
    buffer_adaptor shell_idx_mem( shell_to_task_stack.shell_to_task_idx_device, 
      total_nshells_bfn );
    buffer_adaptor shell_off_mem( shell_to_task_stack.shell_to_task_off_device, 
      total_nshells_bfn );

    // Reserve memory 
    host_shell_to_task_bfn.resize(global_dims.nshells);

    for( auto ish = 0ul; ish < global_dims.nshells; ++ish ) {
      const auto ntask = shell_to_task_idx_bfn[ish].size();
      auto& bck = host_shell_to_task_bfn[ish];

      // Unpack meta data
      bck.ntask = ntask;
      bck.center_idx = basis_map.shell_to_center( ish );
      bck.true_idx   = ish;
      bck.shell_device = static_stack.shells_device + ish;

      // Allocate device memory 
      bck.task_idx_device = 
        shell_idx_mem.aligned_alloc<int32_t>( ntask, csl );
      bck.task_shell_offs_device = 
        shell_off_mem.aligned_alloc<int32_t>( ntask, csl );

      // Pack host data
      concat_iterable(concat_shell_to_task_idx_bfn, shell_to_task_idx_bfn[ish]);
      concat_iterable(concat_shell_to_task_off_bfn, shell_to_task_off_bfn[ish]);
    }

    // Send data to device
    device_backend_->copy_async( concat_shell_to_task_idx_bfn.size(),
      concat_shell_to_task_idx_bfn.data(), 
      shell_to_task_stack.shell_to_task_idx_device, "shell_to_task_idx_device" );
    device_backend_->copy_async( concat_shell_to_task_off_bfn.size(),
      concat_shell_to_task_off_bfn.data(), 
      shell_to_task_stack.shell_to_task_off_device, "shell_to_task_off_device" );


    // Sort shell indices by L
    std::vector<uint32_t> shell_idx( global_dims.nshells );
    std::iota( shell_idx.begin(), shell_idx.end(), 0 );
    std::stable_sort( shell_idx.begin(), shell_idx.end(),
      [&]( auto i, auto j ){ 
        return basis_map.shell_l(i) < basis_map.shell_l(j); 
      } );

    {
    std::vector<ShellToTaskDevice> shell_to_task_sorted( global_dims.nshells );
    for( auto i = 0ul; i < global_dims.nshells; ++i ) 
      shell_to_task_sorted[i] = host_shell_to_task_bfn[shell_idx[i]];
    host_shell_to_task_bfn = std::move(shell_to_task_sorted);
    }

    // Send Shell -> Task (bfn) map to device
    device_backend_->copy_async( global_dims.nshells, 
      host_shell_to_task_bfn.data(), shell_to_task_stack.shell_to_task_device,
      "shell_to_task_device" );


    // Form angular momenta batches
    auto max_l = basis_map.max_l();
    l_batched_shell_to_task.resize(max_l + 1);
    auto* p = shell_to_task_stack.shell_to_task_device;
    auto* h = host_shell_to_task_bfn.data();
    for( auto l = 0ul; l <= max_l; ++l ) {
      auto nsh  = basis_map.nshells_with_l(l);
      auto pure = basis_map.l_purity(l);
      l_batched_shell_to_task[l].nshells_in_batch     = nsh;
      l_batched_shell_to_task[l].pure                 = pure;
      l_batched_shell_to_task[l].shell_to_task_device = p;
                          
      size_t max_ntask = std::max_element( h, h+nsh,
        [](auto& a, auto& b){ return a.ntask < b.ntask; } )->ntask;

      l_batched_shell_to_task[l].ntask_average = max_ntask;
      l_batched_shell_to_task[l].npts_average  = 0;

      p += nsh;
      h += nsh;
    }
  
  } // Generate Shell -> Task (bfn)
  auto s2t_mem_en = hrt_t::now();

  /*****************************************
   *   GENERATE SHELLPAIR TO TASK (cou)    *
   *****************************************/
  auto sp2t_mem_st = hrt_t::now();
  if(reqt.shell_pair_to_task_cou or reqt.task_to_shell_pair_cou) {

    const size_t nsp = global_dims.nshell_pairs;

  /*****************************************
   *   GENERATE TASK TO SHELLPAIR (cou)    *
   *****************************************/

    auto t2sp_start = hrt_t::now();

    hrt_t::time_point t2sp_1, t2sp_2, t2sp_3, t2sp_4, t2sp_5;

    subtask.clear();
    nprim_pairs_host.clear();
    pp_ptr_host.clear();
    sp_X_AB_host.clear();
    sp_Y_AB_host.clear();
    sp_Z_AB_host.clear();
    task_to_shell_pair.clear();
    l_batch_task_to_shell_pair.clear();
    l_batch_diag_task_to_shell_pair.clear();

    {
      //using point = detail::cartesian_point;
      const int max_l = basis_map.max_l();
      const size_t ntasks = std::distance(task_begin, task_end);

      // Set up task maps for the AM
      for( auto l_i = 0, l_ij = 0; l_i <= max_l; ++l_i )
      for( auto l_j = 0; l_j <= max_l; ++l_j, ++l_ij ) {
        l_batch_task_to_shell_pair.emplace_back();
        auto& batch = l_batch_task_to_shell_pair[l_ij];
        batch.task_to_shell_pair.resize(ntasks);
        batch.lA = l_i;
        batch.lB = l_j;
        batch.max_prim_pairs = 0;
      }

      // Diag terms
      for( auto l_i = 0; l_i <= max_l; ++l_i ) {
        l_batch_diag_task_to_shell_pair.emplace_back();
        auto& batch = l_batch_diag_task_to_shell_pair[l_i];
        batch.task_to_shell_pair.resize(ntasks);
        batch.lA = l_i;
        batch.lB = l_i;
        batch.max_prim_pairs = 0;
      }

      // Generate shell pair device buffer
      nprim_pairs_host = this->shell_pair_soa.shell_pair_nprim_pairs;
      pp_ptr_host = this->shell_pair_soa.prim_pair_dev_ptr;
      for( auto i = 0ul; i < nsp; ++i ) {
        //nprim_pairs_host.push_back(
        //  this->shell_pair_soa.shell_pair_nprim_pairs[i]
        //);
        //sp_ptr_host.push_back(
        //  this->shell_pair_soa.shell_pair_dev_ptr[i]
        //);
        //point rA, rB;
        const auto& [rA, rB] = this->shell_pair_soa.shell_pair_centers[i];

        sp_X_AB_host.push_back(rA.x - rB.x);
        sp_Y_AB_host.push_back(rA.y - rB.y);
        sp_Z_AB_host.push_back(rA.z - rB.z);
      }
    }

    t2sp_1 = hrt_t::now();

    // Total length of the concatenated task map buffers
    size_t task_map_aggregate_length = 0;

    {
    const int max_l = basis_map.max_l();

    std::vector<int> sh_off_flat(nsp);
    const size_t ntask = std::distance(task_begin,task_end);
    for( size_t itask = 0; itask < ntask; ++itask ) {
      auto it = task_begin + itask;

      // Construct the subtasks
      const int points_per_subtask = get_points_per_subtask();
      for (int subtask_i = 0; subtask_i < it->npts; subtask_i += points_per_subtask) {
        subtask.push_back({int(itask), subtask_i, std::min(it->npts, subtask_i+points_per_subtask), 0});
      }

      // Setup ShellPair offset data
      const auto& shell_list_cou  = it->cou_screening.shell_list;
      const size_t nshells_cou  = shell_list_cou.size();

      // Compute shell offsets (cou)
      auto shell_offs_cou = basis_map.shell_offs<size_t>( 
        shell_list_cou.begin(), shell_list_cou.end() );

      for( auto i = 0ul; i < nshells_cou; ++i )
        sh_off_flat[shell_list_cou[i]] = shell_offs_cou[i];

      // Count the number of shell pairs per task
      const size_t task_nsp = it->cou_screening.shell_pair_list.size();
      for(auto i = 0ul; i < task_nsp; ++i) {
        auto [ish, jsh] = it->cou_screening.shell_pair_list[i];
        const auto idx = it->cou_screening.shell_pair_idx_list[i];;

        int32_t lA, lB;
        std::tie(lA, lB) = this->shell_pair_soa.shell_pair_ls[idx];

        // Filter out diag shell pairs
        if (ish != jsh) {
          const int type_index = lA * (max_l+1) + lB;
          auto& ttsp = l_batch_task_to_shell_pair[type_index].task_to_shell_pair[itask];
          ttsp.nsp++;
          task_map_aggregate_length++;

          l_batch_task_to_shell_pair[type_index].max_prim_pairs = std::max(
            l_batch_task_to_shell_pair[type_index].max_prim_pairs,
            nprim_pairs_host[idx]);

        } else {
          const int type_index = lA;
          auto& ttsp = l_batch_diag_task_to_shell_pair[type_index].task_to_shell_pair[itask];
          ttsp.nsp++;
          task_map_aggregate_length++;

          l_batch_diag_task_to_shell_pair[type_index].max_prim_pairs = std::max(
            l_batch_diag_task_to_shell_pair[type_index].max_prim_pairs,
            nprim_pairs_host[idx]);

        }
      }

      // Allocate space for the shell pair data
      for (auto& batch : l_batch_task_to_shell_pair) {
        for (auto& ttsp : batch.task_to_shell_pair) {
          ttsp.shell_pair_linear_idx.resize(ttsp.nsp);
          ttsp.task_shell_off_row.resize(ttsp.nsp);
          ttsp.task_shell_off_col.resize(ttsp.nsp);
          ttsp.nsp_filled = 0;
        }
      }
      for (auto& batch : l_batch_diag_task_to_shell_pair) {
        for (auto& ttsp : batch.task_to_shell_pair) {
          ttsp.shell_pair_linear_idx.resize(ttsp.nsp);
          ttsp.task_shell_off_row.resize(ttsp.nsp);
          ttsp.task_shell_off_col.resize(ttsp.nsp);
          ttsp.nsp_filled = 0;
        }
      }

      // Iterate over shell pairs adding to tasks
      for(auto i = 0ul; i < task_nsp; ++i) {
        auto [ish, jsh] = it->cou_screening.shell_pair_list[i];
        const auto idx = it->cou_screening.shell_pair_idx_list[i];;

        int32_t lA, lB;
        std::tie(lA, lB) = this->shell_pair_soa.shell_pair_ls[idx];

        // Filter out diag shell pairs
        if (ish != jsh) {
          const int type_index = lA * (max_l+1) + lB;
          auto& ttsp = l_batch_task_to_shell_pair[type_index].task_to_shell_pair[itask];

          const int index = ttsp.nsp_filled++;
          ttsp.shell_pair_linear_idx[index] = idx;
          ttsp.task_shell_off_row[index] = (sh_off_flat[ish] * it->npts);
          ttsp.task_shell_off_col[index] = (sh_off_flat[jsh] * it->npts);
        } else {
          const int type_index = lA;
          auto& ttsp = l_batch_diag_task_to_shell_pair[type_index].task_to_shell_pair[itask];

          const int index = ttsp.nsp_filled++;
          ttsp.shell_pair_linear_idx[index] = idx;
          ttsp.task_shell_off_row[index] = (sh_off_flat[ish] * it->npts);
        }
      }
    }

    }

    t2sp_2 = hrt_t::now();

    // Concat host buffers and copy to device
    buffer_adaptor task_sp_mem( 
      task_to_shell_pair_stack.task_shell_linear_idx_device, 
      total_nshells_cou_sqlt_task_batch * sizeof(int32_t) );
    buffer_adaptor task_row_off_mem( 
      task_to_shell_pair_stack.task_shell_off_row_device, 
      total_nshells_cou_sqlt_task_batch * sizeof(int32_t) );
    buffer_adaptor task_col_off_mem( 
      task_to_shell_pair_stack.task_shell_off_col_device, 
      total_nshells_cou_sqlt_task_batch  * sizeof(int32_t) );

    {
    const size_t ntasks = std::distance(task_begin, task_end);
    const int max_l = basis_map.max_l();
    const int num_sp_types = (max_l+1)*(max_l+1);
    const int num_sp_types_with_diag = num_sp_types + (max_l+1);

    std::vector<TaskToShellPairDevice> host_task_to_shell_pair_task(ntasks * num_sp_types_with_diag);

    std::vector<int32_t> concat_task_to_shell_pair_idx;
    std::vector<int32_t> concat_task_to_shell_pair_off_row;
    std::vector<int32_t> concat_task_to_shell_pair_off_col;

    concat_task_to_shell_pair_idx.reserve(task_map_aggregate_length);
    concat_task_to_shell_pair_off_row.reserve(task_map_aggregate_length);
    concat_task_to_shell_pair_off_col.reserve(task_map_aggregate_length);

    t2sp_3 = hrt_t::now();

    for( auto l_i = 0, l_ij = 0; l_i <= max_l; ++l_i )
    for( auto l_j = 0; l_j <= max_l; ++l_j, ++l_ij ) {
      for( auto itask = 0ul; itask < ntasks; ++itask ) {
        auto& ttsp = l_batch_task_to_shell_pair[l_ij].task_to_shell_pair[itask];

        auto& bck = host_task_to_shell_pair_task[l_ij * ntasks + itask];
        bck.nsp = ttsp.nsp;

        bck.shell_pair_linear_idx_device = task_sp_mem.aligned_alloc<int32_t>(ttsp.nsp, csl);
        bck.task_shell_off_row_device = task_row_off_mem.aligned_alloc<int32_t>(ttsp.nsp, csl);
        bck.task_shell_off_col_device = task_col_off_mem.aligned_alloc<int32_t>(ttsp.nsp, csl);

        concat_iterable( concat_task_to_shell_pair_idx, ttsp.shell_pair_linear_idx );
        concat_iterable( concat_task_to_shell_pair_off_row, ttsp.task_shell_off_row );
        concat_iterable( concat_task_to_shell_pair_off_col, ttsp.task_shell_off_col );
      }
    }
    for( auto l_i = 0; l_i <= max_l; ++l_i ) {
      for( auto itask = 0ul; itask < ntasks; ++itask ) {
        auto& ttsp = l_batch_diag_task_to_shell_pair[l_i].task_to_shell_pair[itask];

        auto& bck = host_task_to_shell_pair_task[(num_sp_types + l_i) * ntasks + itask];
        bck.nsp = ttsp.nsp;

        bck.shell_pair_linear_idx_device = task_sp_mem.aligned_alloc<int32_t>(ttsp.nsp, csl);
        bck.task_shell_off_row_device = task_row_off_mem.aligned_alloc<int32_t>(ttsp.nsp, csl);
        bck.task_shell_off_col_device = task_col_off_mem.aligned_alloc<int32_t>(ttsp.nsp, csl);

        concat_iterable( concat_task_to_shell_pair_idx, ttsp.shell_pair_linear_idx );
        concat_iterable( concat_task_to_shell_pair_off_row, ttsp.task_shell_off_row );
        concat_iterable( concat_task_to_shell_pair_off_col, ttsp.task_shell_off_col );
      }
    }
    t2sp_4 = hrt_t::now();

    device_backend_->copy_async( concat_task_to_shell_pair_idx.size(),
      concat_task_to_shell_pair_idx.data(),
      task_to_shell_pair_stack.task_shell_linear_idx_device,
      "task_shell_linear_idx_device");
    device_backend_->copy_async( concat_task_to_shell_pair_off_row.size(),
      concat_task_to_shell_pair_off_row.data(),
      task_to_shell_pair_stack.task_shell_off_row_device,
      "task_shell_off_row_device");
    device_backend_->copy_async( concat_task_to_shell_pair_off_col.size(),
      concat_task_to_shell_pair_off_col.data(),
      task_to_shell_pair_stack.task_shell_off_col_device,
      "task_shell_off_col_device");

    device_backend_->copy_async(host_task_to_shell_pair_task.size(),
      host_task_to_shell_pair_task.data(),
      task_to_shell_pair_stack.task_to_shell_pair_device,
      "task_to_shell_pair_device");

    device_backend_->copy_async(subtask.size(),
      subtask.data(),
      task_to_shell_pair_stack.subtask_device,
      "subtask_device");

    device_backend_->copy_async(nprim_pairs_host.size(),
      nprim_pairs_host.data(),
      task_to_shell_pair_stack.nprim_pairs_device,
      "nprim_pairs_device");

    device_backend_->copy_async(pp_ptr_host.size(),
      pp_ptr_host.data(),
      task_to_shell_pair_stack.pp_ptr_device,
      "pp_ptr_device");
      
    device_backend_->copy_async(sp_X_AB_host.size(),
      sp_X_AB_host.data(),
      task_to_shell_pair_stack.sp_X_AB_device,
      "sp_X_AB_device");

    device_backend_->copy_async(sp_Y_AB_host.size(),
      sp_Y_AB_host.data(),
      task_to_shell_pair_stack.sp_Y_AB_device,
      "sp_Y_AB_device");

    device_backend_->copy_async(sp_Z_AB_host.size(),
      sp_Z_AB_host.data(),
      task_to_shell_pair_stack.sp_Z_AB_device,
      "sp_Z_AB_device");

    t2sp_5 = hrt_t::now();

    l_batch_task_to_shell_pair_device.clear();
    l_batch_task_to_shell_pair_device.resize(num_sp_types);
    for( auto l_i = 0, l_ij = 0; l_i <= max_l; ++l_i )
    for( auto l_j = 0; l_j <= max_l; ++l_j, ++l_ij ) {
      auto& map = l_batch_task_to_shell_pair_device[l_ij];
      map.task_to_shell_pair_device = task_to_shell_pair_stack.task_to_shell_pair_device + l_ij * ntasks;
      map.lA = l_i;
      map.lB = l_j;
      map.max_prim_pairs = l_batch_task_to_shell_pair[l_ij].max_prim_pairs;
    }

    l_batch_diag_task_to_shell_pair_device.clear();
    l_batch_diag_task_to_shell_pair_device.resize(max_l+1);
    for( auto l_i = 0; l_i <= max_l; ++l_i ) {
      auto& map = l_batch_diag_task_to_shell_pair_device[l_i];
      const int offset = (l_i + num_sp_types) * ntasks;
      map.task_to_shell_pair_device = task_to_shell_pair_stack.task_to_shell_pair_device + offset;
      map.lA = l_i;
      map.lB = l_i;
      map.max_prim_pairs = l_batch_diag_task_to_shell_pair[l_i].max_prim_pairs;
    }
    }

    auto t2sp_end = hrt_t::now();

  dur_t t2sp_dur_total = t2sp_end - t2sp_start;
  dur_t t2sp_dur_1 = t2sp_1 - t2sp_start;
  dur_t t2sp_dur_2 = t2sp_2 - t2sp_1;
  dur_t t2sp_dur_3 = t2sp_3 - t2sp_2;
  dur_t t2sp_dur_4 = t2sp_4 - t2sp_3;
  dur_t t2sp_dur_5 = t2sp_5 - t2sp_4;
  dur_t t2sp_dur_6 = t2sp_end - t2sp_5;
  //std::cout << "T2SP TOTAL  = " << t2sp_dur_total.count() << std::endl;
  //std::cout << "T2SP 1 = " << t2sp_dur_1.count() << std::endl;
  //std::cout << "T2SP 2 = " << t2sp_dur_2.count() << std::endl;
  //std::cout << "T2SP 3 = " << t2sp_dur_3.count() << std::endl;
  //std::cout << "T2SP 4 = " << t2sp_dur_4.count() << std::endl;
  //std::cout << "T2SP 5 = " << t2sp_dur_5.count() << std::endl;
  //std::cout << "T2SP 6 = " << t2sp_dur_6.count() << std::endl;
  //std::cout << "INTERIM = " << interim_dur << std::endl;


  } // Generate ShellPair -> Task (cou)
  auto sp2t_mem_en = hrt_t::now();

  dur_t w_mem_dur = w_mem_en - w_mem_st;
  dur_t sl_mem_dur = sl_mem_en - sl_mem_st;
  dur_t s2t_mem_dur = s2t_mem_en - s2t_mem_st;
  dur_t sp2t_mem_dur = sp2t_mem_en - sp2t_mem_st;

  //std::cout << "W DUR    = " << w_mem_dur.count() << std::endl;
  //std::cout << "SL DUR   = " << sl_mem_dur.count() << std::endl;
  //std::cout << "S2T DUR  = " << s2t_mem_dur.count() << std::endl;
  //std::cout << "SP2T DUR = " << sp2t_mem_dur.count() << std::endl;

  //auto snd_st = hrt_t::now();
  device_backend_->master_queue_synchronize(); 
  //auto snd_en = hrt_t::now();
  //std::cout << "SND_WAIT = " << dur_t(snd_en-snd_st).count() << std::endl;
}








void Scheme1DataBase::add_extra_to_indirection( 
  integrator_term_tracker terms, std::vector<XCDeviceTask>& tasks  ) {

  // Weights Specific
  if( terms.weights ) {
    const auto ldatoms = get_ldatoms();
    buffer_adaptor dist_scratch_mem( scheme1_stack.dist_scratch_device, 
      ldatoms * total_npts_task_batch * sizeof(double) );

    // Extra indirection for dist scratch
    for( auto& task : tasks ) {
      task.dist_scratch  = dist_scratch_mem.aligned_alloc<double>( 
        ldatoms * task.npts, sizeof(double2), csl );
    }
  }

  if( terms.exx or terms.exc_vxc or terms.exc_grad or terms.den or terms.exx_ek_screening or terms.fxc_contraction ) {
    const size_t total_nshells_bfn = total_nshells_bfn_task_batch * sizeof(size_t);
    buffer_adaptor 
      shell_list_bfn_mem( collocation_stack.shell_list_device, total_nshells_bfn );
    buffer_adaptor 
      shell_offs_bfn_mem( collocation_stack.shell_offs_device, total_nshells_bfn );

    for( auto& task : tasks ) {
      const auto nshells_bfn = task.bfn_screening.nshells;
      task.bfn_screening.shell_list = 
        shell_list_bfn_mem.aligned_alloc<size_t>( nshells_bfn , csl); 
      task.bfn_screening.shell_offs = 
        shell_offs_bfn_mem.aligned_alloc<size_t>( nshells_bfn , csl); 
    }
  }

}

}
