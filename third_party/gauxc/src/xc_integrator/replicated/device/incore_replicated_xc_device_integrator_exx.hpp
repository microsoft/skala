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

#include "incore_replicated_xc_device_integrator.hpp"
#include "device/local_device_work_driver.hpp"
#include "host/reference_local_host_work_driver.hpp"
#include <stdexcept>
#include "device/xc_device_aos_data.hpp"
#include <fstream>
#include <gauxc/util/unused.hpp>

#include "integrator_util/exx_screening.hpp"
#include "integrator_util/integral_bounds.hpp"

namespace GauXC  {
namespace detail {

template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  eval_exx_( int64_t m, int64_t n, const value_type* P,
             int64_t ldp, value_type* K, int64_t ldk, 
             const IntegratorSettingsEXX& settings ) { 


  const auto& basis = this->load_balancer_->basis();

  // Check that P / K are sane
  const int64_t nbf = basis.nbf();
  if( m != n ) 
    GAUXC_GENERIC_EXCEPTION("P/K Must Be Square");
  if( m != nbf ) 
    GAUXC_GENERIC_EXCEPTION("P/K Must Have Same Dimension as Basis");
  if( ldp < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDP");
  if( ldk < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDK");

  // Allocate Device memory
  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get() );
  auto rt  = detail::as_device_runtime(this->load_balancer_->runtime());
  auto device_data_ptr = lwd->create_device_data(rt);

  GAUXC_MPI_CODE(MPI_Barrier(rt.comm());)

  this->timer_.time_op("XCIntegrator.EXX_Screening", [&]() { 
    exx_ek_screening_local_work_( basis, P, ldp, *device_data_ptr, settings);
  });


  // Get Tasks
  auto& tasks = this->load_balancer_->get_tasks();
  if( this->reduction_driver_->takes_device_memory() ) {
    //GAUXC_GENERIC_EXCEPTION("EXX + NCCL NYI");

    // Compute local contributions to K and keep on device
    this->timer_.time_op("XCIntegrator.LocalWork_EXX", [&](){
      exx_local_work_( basis, P, ldp, 
        tasks.begin(), tasks.end(), *device_data_ptr, settings);
      rt.device_backend()->master_queue_synchronize();
    });

    GAUXC_MPI_CODE(
    this->timer_.time_op("XCIntegrator.ImbalanceWait_EXX",[&](){
      MPI_Barrier(rt.comm());
    });  
    )

    // Reduce results in device memory
    this->timer_.time_op("XCIntegrator.Allreduce_EXX", [&](){
      this->reduction_driver_->allreduce_inplace(
        device_data_ptr->exx_k_device_data(), nbf*nbf, ReductionOp::Sum, 
        device_data_ptr->queue());
    });

    // Receive K from host
    this->timer_.time_op("XCIntegrator.DeviceToHostCopy_EXX",[&](){
      device_data_ptr->retrieve_exx_integrands( K, ldk );
    });

  } else {

    // Compute local contributions to K and retrieve
    // data from device 
    this->timer_.time_op("XCIntegrator.LocalWork_EXX", [&](){
      exx_local_work_( basis, P, ldp, K, ldk, 
        tasks.begin(), tasks.end(), *device_data_ptr, settings);
    });

    GAUXC_MPI_CODE(
    this->timer_.time_op("XCIntegrator.ImbalanceWait_EXX",[&](){
      MPI_Barrier(rt.comm());
    });  
    )

    // Reduce Results in host mem
    this->timer_.time_op("XCIntegrator.Allreduce_EXX", [&](){
      this->reduction_driver_->allreduce_inplace( K, nbf*nbf, ReductionOp::Sum );
    });

  }
}

template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  exx_ek_screening_local_work_( const basis_type& basis, const value_type* P, int64_t ldp, 
                       XCDeviceData& device_data,
                       const IntegratorSettingsEXX& settings ) {

  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get());
  IntegratorSettingsSNLinK sn_link_settings;
  if( auto* tmp = dynamic_cast<const IntegratorSettingsSNLinK*>(&settings) ) {
    sn_link_settings = *tmp;
  }

  // Get Tasks
  auto& tasks = this->load_balancer_->get_tasks();
  auto task_begin = tasks.begin();
  auto task_end = tasks.end();

  // Setup Aliases
  const auto& mol   = this->load_balancer_->molecule();

  const auto nbf     = basis.nbf();
  const auto nshells = basis.nshells();


  // Get basis map and shell pairs
  auto& basis_map   = this->load_balancer_->basis_map();
  auto& shell_pairs = this->load_balancer_->shell_pairs();

  // Populate submat maps
  device_data.populate_submat_maps( basis.nbf(), task_begin, task_end, basis_map );


  // Check that Partition Weights have been calculated
  auto& lb_state = this->load_balancer_->state();
  if( not lb_state.modified_weights_are_stored ) {
    GAUXC_GENERIC_EXCEPTION("Weights Have Not Been Modified"); 
  }

  // Reset the coulomb screening data
  for( auto it = task_begin; it != task_end; ++it) {
    it->cou_screening = XCTask::screening_data();
  }

  // Compute base screening quantities
  const size_t nb2 = basis.nbf() * basis.nbf();
  std::vector<double> P_abs(nb2);
  for( auto i = 0ul; i < nb2; ++i ) P_abs[i] = std::abs(P[i]);

  // Loop over sparse shell pairs
  const size_t ns2 = nshells * nshells;
  std::vector<double> V_max(ns2, 0.0);
  this->timer_.time_op("XCIntegrator.VM_EXX", [&](){
  const auto sp_row_ptr = shell_pairs.row_ptr();
  const auto sp_col_ind = shell_pairs.col_ind();
  for( auto i = 0; i < nshells; ++i ) {
    const auto j_st = sp_row_ptr[i];
    const auto j_en = sp_row_ptr[i+1];
    for( auto _j = j_st; _j < j_en; ++_j ) {
      const auto j = sp_col_ind[_j];
      const auto mv = util::max_coulomb( basis.at(i), basis.at(j) );
      V_max[i + j*nshells] = mv;
      if( i != j ) V_max[j + i*nshells] = mv;
    }
  }
  });

#if 1
  exx_ek_screening( basis, basis_map, shell_pairs, P_abs.data(), basis.nbf(),
    V_max.data(), nshells, sn_link_settings.energy_tol, 
    sn_link_settings.k_tol, device_data, lwd, task_begin, task_end );
#else
  for( auto it = task_begin; it != task_end; ++it) {
    it->cou_screening = XCTask::screening_data();
  }
  // Create LocalHostWorkDriver
  LocalHostWorkDriver host_lwd(
    std::make_unique<ReferenceLocalHostWorkDriver>()
  );
  exx_ek_screening( basis, basis_map, P_abs.data(), basis.nbf(),
    V_max.data(), nshells, sn_link_settings.energy_tol, 
    sn_link_settings.k_tol, &host_lwd, task_begin, task_end );
#endif

  //this->load_balancer_->rebalance_exx();

}

template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  exx_local_work_( const basis_type& basis, const value_type* P, int64_t ldp, 
                       value_type* K, int64_t ldk,
                       host_task_iterator task_begin, host_task_iterator task_end,
                       XCDeviceData& device_data,
                       const IntegratorSettingsEXX& settings ) {


  exx_local_work_(basis, P, ldp, task_begin, task_end, device_data, settings);
  auto rt  = detail::as_device_runtime(this->load_balancer_->runtime());
  rt.device_backend()->master_queue_synchronize();

  // Receive K from host
  this->timer_.time_op("XCIntegrator.DeviceToHostCopy_EXX",[&](){
    device_data.retrieve_exx_integrands( K, ldk );
  });

}

template <typename ValueType>
void IncoreReplicatedXCDeviceIntegrator<ValueType>::
  exx_local_work_( const basis_type& basis, const value_type* P, int64_t ldp, 
                       host_task_iterator task_begin, host_task_iterator task_end,
                       XCDeviceData& device_data,
                       const IntegratorSettingsEXX& settings ) {

  auto* lwd = dynamic_cast<LocalDeviceWorkDriver*>(this->local_work_driver_.get() );
  IntegratorSettingsSNLinK sn_link_settings;
  if( auto* tmp = dynamic_cast<const IntegratorSettingsSNLinK*>(&settings) ) {
    sn_link_settings = *tmp;
  }

  // Setup Aliases
  const auto& mol   = this->load_balancer_->molecule();

  const auto nbf     = basis.nbf();
  const auto nshells = basis.nshells();


  // Get basis map and shell pairs
  auto& basis_map   = this->load_balancer_->basis_map();
  auto& shell_pairs = this->load_balancer_->shell_pairs();

  // Populate submat maps
  device_data.populate_submat_maps( basis.nbf(), task_begin, task_end, basis_map );



  // Sort tasks 
  auto task_comparator = []( const XCTask& a, const XCTask& b ) {
    return (a.points.size() * a.bfn_screening.nbe) > (b.points.size() * b.bfn_screening.nbe);
  };
  std::sort( task_begin, task_end, task_comparator );



  // Check that Partition Weights have been calculated
  auto& lb_state = this->load_balancer_->state();
  if( not lb_state.modified_weights_are_stored ) {
    GAUXC_GENERIC_EXCEPTION("Weights Have Not Been Modified"); 
  }

  task_end = std::stable_partition( task_begin, task_end,
    []( const auto& t ) { return t.cou_screening.shell_list.size() > 0; } );

#if 0
  // Lexicographic ordering of tasks
  auto task_order = []( const auto& a, const auto& b ) {

    // Sort by iParent first
    if( a.iParent < b.iParent )      return true;
    else if( a.iParent > b.iParent ) return false;

    // Equal iParent: lex sort on bfn shell list
    else if(a.bfn_screening.shell_list < b.bfn_screening.shell_list) return true;
    else if(a.bfn_screening.shell_list > b.bfn_screening.shell_list) return false;
    
    // Equal iParent and bfn shell list: lex sort on cou shell list
    else return a.cou_screening.shell_list < b.cou_screening.shell_list;

  };

  std::sort( task_begin, task_end, task_order ); 
  auto task_equiv = []( const auto& a, const auto& b ) {
    return a.equiv_with(b) and 
      a.cou_screening.equiv_with(b.cou_screening);
  };
  std::vector<XCTask> local_work_unique(task_begin, task_end);
  auto last_unique =
    std::unique( local_work_unique.begin(),
                 local_work_unique.end(),
                 task_equiv );
  local_work_unique.erase( last_unique, local_work_unique.end() );

  // Merge tasks
  for( auto&& t : local_work_unique ) {
    t.points.clear();
    t.weights.clear();
    t.npts = 0;
  }

  auto cur_lw_begin = task_begin;
  auto cur_uniq_it  = local_work_unique.begin();

  for( auto lw_it = task_begin; lw_it != task_end; ++lw_it ) 
  if( not task_equiv( *lw_it, *cur_uniq_it ) ) {

    if( cur_uniq_it == local_work_unique.end() )
      GAUXC_GENERIC_EXCEPTION("Messed up in unique");

    cur_uniq_it->merge_with( cur_lw_begin, lw_it );

    cur_lw_begin = lw_it;
    cur_uniq_it++;

  }

  // Merge the last set of batches
  for( ; cur_lw_begin != task_end; ++cur_lw_begin )
    cur_uniq_it->merge_with( *cur_lw_begin );
  cur_uniq_it++;

  std::copy(local_work_unique.begin(), local_work_unique.end(),
    task_begin);
  task_end = task_begin + local_work_unique.size();
#endif
  
  std::sort(task_begin,task_end,
    [](auto& a, auto& b){ return a.cou_screening.shell_pair_list.size() >
      b.cou_screening.shell_pair_list.size(); });


  size_t total_npts = std::accumulate( task_begin, task_end, 0ul,
    [](const auto& a, const auto& b) { return a + b.npts; } );
  //std::cout << "TOTAL NPTS " << total_npts << std::endl;

  size_t total_nbe_bfn = std::accumulate( task_begin, task_end, 0ul,
    [](const auto& a, const auto& b) { return a + b.bfn_screening.nbe; } );
  size_t total_nbe_cou = std::accumulate( task_begin, task_end, 0ul,
    [](const auto& a, const auto& b) { return a + b.cou_screening.nbe; } );

  size_t ntasks = std::distance(task_begin,task_end);

  int world_rank = 0;
  GAUXC_MPI_CODE(
  MPI_Comm_rank(this->load_balancer_->runtime().comm(), &world_rank);
  )
  //printf("RANK %d, LC_EXX = %lu\n",
  //  world_rank,
  //  std::accumulate(task_begin, task_end, 0ul, [](auto c, const auto& t){ return c + t.cost_exx(); })
  //);

  // Populate submat maps
  device_data.populate_submat_maps( basis.nbf(), task_begin, task_end, basis_map );



  // Do EXX integration in task batches
  device_data.reset_allocations();
  device_data.allocate_static_data_exx( nbf, nshells, shell_pairs.npairs(), shell_pairs.nprim_pair_total(), basis_map.max_l() );
  device_data.send_static_data_density_basis( P, ldp, nullptr, 0, nullptr, 0, nullptr, 0, basis );
  device_data.send_static_data_shell_pairs( basis, shell_pairs );

  // Zero integrands
  device_data.zero_exx_integrands();

  // Processes batches in groups that saturadate available device memory
  integrator_term_tracker enabled_terms;
  enabled_terms.exx = true;

  //GAUXC_GENERIC_EXCEPTION("DIE DIE DIE");
  auto task_it = task_begin;
  while( task_it != task_end ) {

    // Determine next task batch, send relevant data to device (EXX only)
    task_it = 
      device_data.generate_buffers( enabled_terms, basis_map, task_it, task_end );

#if 1
    /*** Process the batches ***/

    // Evaluate collocation
    lwd->eval_collocation( &device_data );

    // Evaluate F(mu,i) = P(mu,nu) * B(nu,i)
    // mu runs over significant ek shells
    // nu runs over the bfn shell list
    // i runs over all points
    lwd->eval_exx_fmat( &device_data );

    // Compute G(mu,i) = w(i) * A(mu,nu,i) * F(nu,i)
    // mu/nu run over significant ek shells
    // i runs over all points
    lwd->eval_exx_gmat( &device_data, basis_map );

    // Increment K(mu,nu) += B(mu,i) * G(nu,i)
    // mu runs over bfn shell list
    // nu runs over ek shells
    // i runs over all points
    lwd->inc_exx_k( &device_data );
#endif

  } // Loop over batches of batches 

#if 1
  // Symmetrize K in device memory
  lwd->symmetrize_exx_k( &device_data);
#endif

}

}
}
