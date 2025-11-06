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
#ifdef GAUXC_HAS_HIP
#include "collocation_common.hpp"
#include "device/common/collocation_device.hpp"
#include "device_specific/hip_util.hpp"
#include <gauxc/basisset_map.hpp>


auto populate_device_hip( const BasisSet<double>& basis,
                           const std::vector<ref_collocation_data>& ref_data,
                           bool pop_grad, bool pop_hess ) {

  std::vector< XCDeviceTask > tasks;

  auto shells_device  = util::hip_malloc<Shell<double>>( basis.size() );
  std::vector<Shell<double>> shells( basis );
  util::hip_copy( basis.size(), shells_device, shells.data() );

  for( auto& d : ref_data ) {
    const auto npts = d.pts.size();
    const auto nbf  = d.eval.size() / npts;

    const auto& mask = d.mask;
    const auto& pts  = d.pts;

    /// XXX: THIS DOES NOT POPULATE A VALID TASK, ONLY WHAT's REQUIRED FOR THIS
    //  TEST
    auto& task = tasks.emplace_back();
    task.bfn_screening.nbe     = nbf;
    task.npts    = npts;
    task.bfn_screening.nshells = mask.size();

    //task.points     = util::hip_malloc<double>( 3 * npts );
    task.points_x     = util::hip_malloc<double>( npts );
    task.points_y     = util::hip_malloc<double>( npts );
    task.points_z     = util::hip_malloc<double>( npts );
    task.bfn_screening.shell_offs = util::hip_malloc<size_t>( mask.size() );
    task.bfn_screening.shell_list = util::hip_malloc<size_t>( mask.size() );
    task.bf         = util::hip_malloc<double>( nbf * npts );
    if(pop_grad) {
      task.dbfx = util::hip_malloc<double>( nbf * npts );
      task.dbfy = util::hip_malloc<double>( nbf * npts );
      task.dbfz = util::hip_malloc<double>( nbf * npts );
    }

    if(pop_hess) {
      task.d2bfxx = util::hip_malloc<double>( nbf * npts );
      task.d2bfxy = util::hip_malloc<double>( nbf * npts );
      task.d2bfxz = util::hip_malloc<double>( nbf * npts );
      task.d2bfyy = util::hip_malloc<double>( nbf * npts );
      task.d2bfyz = util::hip_malloc<double>( nbf * npts );
      task.d2bfzz = util::hip_malloc<double>( nbf * npts );
    }

    //auto* pts_device = task.points;
    auto* pts_x_device = task.points_x;
    auto* pts_y_device = task.points_y;
    auto* pts_z_device = task.points_z;
    auto* offs_device = task.bfn_screening.shell_offs;
    auto* mask_device = task.bfn_screening.shell_list;


    //util::hip_copy( 3*npts, pts_device, pts.data()->data() );
    std::vector<double> pts_x, pts_y, pts_z;
    for( auto pt : pts ) {
      pts_x.emplace_back(pt[0]);
      pts_y.emplace_back(pt[1]);
      pts_z.emplace_back(pt[2]);
    }
    util::hip_copy( npts, pts_x_device, pts_x.data() );
    util::hip_copy( npts, pts_y_device, pts_y.data() );
    util::hip_copy( npts, pts_z_device, pts_z.data() );

    std::vector<size_t> mask_ul( mask.size() );
    std::copy( mask.begin(), mask.end(), mask_ul.begin() );
    util::hip_copy( mask.size(), mask_device, mask_ul.data() );

    std::vector<size_t> offs( mask.size() );
    offs[0] = 0;
    for( int i = 1; i < mask.size(); ++i )
      offs[i] = offs[i-1] + basis[mask[i-1]].size();
    util::hip_copy( offs.size(), offs_device, offs.data()  );

  }

  return std::pair(shells_device,tasks);
}


void hip_check_collocation( const std::vector<XCDeviceTask>& tasks,
                             const std::vector<ref_collocation_data>& ref_data,
                             bool check_grad, bool check_hess) {

  for( int i = 0; i < tasks.size(); i++ ) {

    auto* ref_eval = ref_data[i].eval.data();
    std::vector<double> eval (tasks[i].bfn_screening.nbe * tasks[i].npts);
    util::hip_copy( eval.size(), eval.data(), tasks[i].bf );

    check_collocation_transpose( tasks[i].npts, tasks[i].bfn_screening.nbe, ref_eval, 
      eval.data(), "IT = " + std::to_string(i) + " BF EVAL" );

    if( check_grad ) {
      auto* ref_deval_x = ref_data[i].deval_x.data();
      auto* ref_deval_y = ref_data[i].deval_y.data();
      auto* ref_deval_z = ref_data[i].deval_z.data();

      std::vector<double> deval_x (tasks[i].bfn_screening.nbe * tasks[i].npts);
      std::vector<double> deval_y (tasks[i].bfn_screening.nbe * tasks[i].npts);
      std::vector<double> deval_z (tasks[i].bfn_screening.nbe * tasks[i].npts);

      util::hip_copy( eval.size(), deval_x.data(), tasks[i].dbfx );
      util::hip_copy( eval.size(), deval_y.data(), tasks[i].dbfy );
      util::hip_copy( eval.size(), deval_z.data(), tasks[i].dbfz );

      auto npts = tasks[i].npts;
      auto nbe  = tasks[i].bfn_screening.nbe;
      check_collocation_transpose( npts, nbe, ref_deval_x, deval_x.data(), "IT = " + std::to_string(i) + " BFX EVAL" );
      check_collocation_transpose( npts, nbe, ref_deval_y, deval_y.data(), "IT = " + std::to_string(i) + " BFY EVAL" );
      check_collocation_transpose( npts, nbe, ref_deval_z, deval_z.data(), "IT = " + std::to_string(i) + " BFZ EVAL" );
    }

    if( check_hess ) {
      auto* ref_d2eval_xx = ref_data[i].d2eval_xx.data();
      auto* ref_d2eval_xy = ref_data[i].d2eval_xy.data();
      auto* ref_d2eval_xz = ref_data[i].d2eval_xz.data();
      auto* ref_d2eval_yy = ref_data[i].d2eval_yy.data();
      auto* ref_d2eval_yz = ref_data[i].d2eval_yz.data();
      auto* ref_d2eval_zz = ref_data[i].d2eval_zz.data();

      std::vector<double> d2eval_xx (tasks[i].bfn_screening.nbe * tasks[i].npts);
      std::vector<double> d2eval_xy (tasks[i].bfn_screening.nbe * tasks[i].npts);
      std::vector<double> d2eval_xz (tasks[i].bfn_screening.nbe * tasks[i].npts);
      std::vector<double> d2eval_yy (tasks[i].bfn_screening.nbe * tasks[i].npts);
      std::vector<double> d2eval_yz (tasks[i].bfn_screening.nbe * tasks[i].npts);
      std::vector<double> d2eval_zz (tasks[i].bfn_screening.nbe * tasks[i].npts);

      util::hip_copy( eval.size(), d2eval_xx.data(), tasks[i].d2bfxx );
      util::hip_copy( eval.size(), d2eval_xy.data(), tasks[i].d2bfxy );
      util::hip_copy( eval.size(), d2eval_xz.data(), tasks[i].d2bfxz );
      util::hip_copy( eval.size(), d2eval_yy.data(), tasks[i].d2bfyy );
      util::hip_copy( eval.size(), d2eval_yz.data(), tasks[i].d2bfyz );
      util::hip_copy( eval.size(), d2eval_zz.data(), tasks[i].d2bfzz );

      auto npts = tasks[i].npts;
      auto nbe  = tasks[i].bfn_screening.nbe;
      check_collocation_transpose( npts, nbe, ref_d2eval_xx, d2eval_xx.data(), "IT = " + std::to_string(i) + " BFXX EVAL" );
      check_collocation_transpose( npts, nbe, ref_d2eval_xy, d2eval_xy.data(), "IT = " + std::to_string(i) + " BFXY EVAL" );
      check_collocation_transpose( npts, nbe, ref_d2eval_xz, d2eval_xz.data(), "IT = " + std::to_string(i) + " BFXZ EVAL" );
      check_collocation_transpose( npts, nbe, ref_d2eval_yy, d2eval_yy.data(), "IT = " + std::to_string(i) + " BFYY EVAL" );
      check_collocation_transpose( npts, nbe, ref_d2eval_yz, d2eval_yz.data(), "IT = " + std::to_string(i) + " BFYZ EVAL" );
      check_collocation_transpose( npts, nbe, ref_d2eval_zz, d2eval_zz.data(), "IT = " + std::to_string(i) + " BFZZ EVAL" );
    }

  }

}


    









void test_hip_collocation_masked_combined( const BasisSet<double>& basis, std::ifstream& in_file, bool grad ) {



  std::vector<ref_collocation_data> ref_data;

  {
    cereal::BinaryInputArchive ar( in_file );
    ar( ref_data );
  }


  device_queue stream( std::make_shared<util::hip_stream>() );
  auto [shells_device,tasks] = populate_device_hip( basis, ref_data, grad, false );


  const auto nshells_max = std::max_element( tasks.begin(), tasks.end(),
    []( const auto& a, const auto& b ) {
      return a.bfn_screening.nshells < b.bfn_screening.nshells;
    })->bfn_screening.nshells;

  const auto npts_max = std::max_element( tasks.begin(), tasks.end(),
    []( const auto& a, const auto& b ) {
      return a.npts < b.npts;
    })->npts;

  auto* tasks_device = util::hip_malloc<XCDeviceTask>( tasks.size() );
  util::hip_copy( tasks.size(), tasks_device, tasks.data() );

  if(grad)
    eval_collocation_masked_combined_deriv1( tasks.size(), npts_max,
      nshells_max, shells_device, tasks_device, stream );
  else
    eval_collocation_masked_combined( tasks.size(), npts_max,
      nshells_max, shells_device, tasks_device, stream );

  util::hip_device_sync();

  hip_check_collocation( tasks, ref_data, grad, false );


  for( auto& t : tasks ) {
    util::hip_free( t.points_x, t.points_y, t.points_z, t.bfn_screening.shell_offs, t.bfn_screening.shell_list, t.bf );
    if(grad) util::hip_free( t.dbfx, t.dbfy, t.dbfz );
  }
  util::hip_free( tasks_device, shells_device );
}

void test_hip_collocation( const BasisSet<double>& basis, 
  std::ifstream& in_file ) {

  test_hip_collocation_masked_combined( basis, in_file, false );

}
void test_hip_collocation_deriv1( const BasisSet<double>& basis,
  std::ifstream& in_file ) {

  test_hip_collocation_masked_combined( basis, in_file, true );

}
  













#if 0
void test_hip_collocation_shell_to_task( const BasisSet<double>& basis,  const BasisSetMap& basis_map,
  std::ifstream& in_file, bool grad, bool hess) {

  // Load reference data
  std::vector<ref_collocation_data> ref_data;
  {
    cereal::BinaryInputArchive ar( in_file );
    ar( ref_data );
  }

  // Populate base task information
  device_queue stream( std::make_shared<util::hip_stream>() );
  auto [shells_device,tasks] = populate_device_hip( basis, ref_data, grad, hess );

  // Send tasks to device
  auto* tasks_device = util::hip_malloc<XCDeviceTask>( tasks.size() );
  util::hip_copy( tasks.size(), tasks_device, tasks.data() );


  // Form Shell -> Task data structures 
  std::vector< std::vector<int32_t> >
    shell_to_task_idx( basis.size() ),
    shell_to_task_off( basis.size() );

  int itask = 0;
  for( auto& d : ref_data ) {
    const auto& mask = d.mask;

    // Reform offsets 
    std::vector<size_t> offs( mask.size() );
    offs[0] = 0;
    for( int i = 1; i < mask.size(); ++i )
      offs[i] = offs[i-1] + basis[mask[i-1]].size();

    // Form shell -> task
    for( auto i = 0; i < mask.size(); ++i ) {
      auto ish = mask[i];
      shell_to_task_idx[ish].emplace_back(itask);
      shell_to_task_off[ish].emplace_back( offs[i] );
    }
    itask++;

  }

  std::vector<ShellToTaskDevice> shell_to_task;
  for( auto ish = 0; ish < basis.size(); ++ish ) {
    shell_to_task.emplace_back();

    const auto ntask = shell_to_task_idx[ish].size();
    shell_to_task.back().ntask = ntask;
    shell_to_task.back().shell_device = shells_device + ish;
    shell_to_task.back().task_idx_device = util::hip_malloc<int32_t>( ntask );
    shell_to_task.back().task_shell_offs_device =
      util::hip_malloc<int32_t>( ntask );

    util::hip_copy( ntask, shell_to_task.back().task_idx_device, 
      shell_to_task_idx[ish].data() );
    util::hip_copy( ntask, shell_to_task.back().task_shell_offs_device, 
      shell_to_task_off[ish].data() );

  }


  // Sort shells by L
  std::vector<uint32_t> shell_idx( basis.size() );
  std::iota( shell_idx.begin(), shell_idx.end(), 0 );

  std::sort( shell_idx.begin(), shell_idx.end(),
    [&]( auto i, auto j ){ return basis.at(i).l() < basis.at(j).l(); } );

  {
  std::vector<ShellToTaskDevice> shell_to_task_sorted( basis.size() );
  for( auto i = 0; i < basis.size(); ++i ) 
    shell_to_task_sorted[i] = shell_to_task[shell_idx[i]];
  shell_to_task = std::move(shell_to_task_sorted);
  }


  // Send Shell -> Task to device
  auto* shell_to_task_device = util::hip_malloc<ShellToTaskDevice>(basis.size());
  util::hip_copy( basis.size(), shell_to_task_device, shell_to_task.data() );
  util::hip_device_sync();

  // Form angular momentum batches for collocation eval
  auto max_l = std::max_element(basis.begin(),basis.end(),
    [](const auto&a, const auto& b){ return a.l() < b.l(); } )->l();
  std::vector<AngularMomentumShellToTaskBatch> l_batched_shell_to_task(max_l+1);
  {
  auto* p = shell_to_task_device;
  auto* h = shell_to_task.data();
  for( auto l = 0; l <= max_l; ++l ) {
    auto nsh = basis_map.nshells_with_l(l);
    auto pure = basis_map.l_purity(l);
    l_batched_shell_to_task[l].nshells_in_batch     = nsh;
    l_batched_shell_to_task[l].pure                 = pure;
    l_batched_shell_to_task[l].shell_to_task_device = p;
    
    size_t total_ntask = std::accumulate( h, h + nsh, 0ul,
      [](auto& a, auto& b){ return a + b.ntask; } );
    l_batched_shell_to_task[l].ntask_average = total_ntask / nsh;

    p += nsh;
    h += nsh;
  }
  }


  if( hess )
    eval_collocation_shell_to_task_hessian( max_l, l_batched_shell_to_task.data(), 
      tasks_device, stream );
  else if( grad ) 
    eval_collocation_shell_to_task_gradient( max_l, l_batched_shell_to_task.data(), 
      tasks_device, stream );
  else       
    eval_collocation_shell_to_task( max_l, l_batched_shell_to_task.data(), 
      tasks_device, stream );



  util::hip_device_sync();
  hip_check_collocation( tasks, ref_data, grad, hess );

      
  for( auto& t : tasks ) {
    util::hip_free( t.points_x, t.points_y, t.points_z, t.bfn_screening.shell_offs, t.bfn_screening.shell_list, t.bf );
    if(grad) util::hip_free( t.dbfx, t.dbfy, t.dbfz );
    if(hess) util::hip_free( t.d2bfxx, t.d2bfxy, t.d2bfxz, t.d2bfyy, t.d2bfyz, t.d2bfzz );
  }
  util::hip_free( tasks_device, shells_device, shell_to_task_device );
  for( auto& s : shell_to_task ) {
    util::hip_free( s.task_idx_device, s.task_shell_offs_device );
  }
}



void test_hip_collocation_shell_to_task( const BasisSet<double>& basis,  
  const BasisSetMap& basis_map, std::ifstream& in_file) {

  test_hip_collocation_shell_to_task(basis,basis_map,in_file,false, false);

}
void test_hip_collocation_shell_to_task_gradient( const BasisSet<double>& basis,  
  const BasisSetMap& basis_map, std::ifstream& in_file) {

  test_hip_collocation_shell_to_task(basis,basis_map,in_file,true, false);

}
void test_hip_collocation_shell_to_task_hessian( const BasisSet<double>& basis,  
  const BasisSetMap& basis_map, std::ifstream& in_file) {

  test_hip_collocation_shell_to_task(basis,basis_map,in_file,true, true);

}
#endif





















#endif // GAUXC_HAS_HIP

