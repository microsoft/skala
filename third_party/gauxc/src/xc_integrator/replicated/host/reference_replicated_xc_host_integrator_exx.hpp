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
#pragma once

#include "reference_replicated_xc_host_integrator.hpp"
#include "integrator_util/integrator_common.hpp"
#include "integrator_util/integral_bounds.hpp"
#include "integrator_util/exx_screening.hpp"
#include "host/local_host_work_driver.hpp"
#include "host/blas.hpp"
#include <stdexcept>
#include <set>

#include <gauxc/util/geometry.hpp>


namespace std {
template <typename T>
ostream& operator<<( ostream& out, const vector<T>& v ) {
  for( auto _v : v ) out << _v << " ";
  return out;
}
}

namespace GauXC::detail {

template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exx_( int64_t m, int64_t n, const value_type* P,
             int64_t ldp, value_type* K, int64_t ldk,
             const IntegratorSettingsEXX& settings ) {

  const auto& basis = this->load_balancer_->basis();

  // Check that P / VXC are sane
  const int64_t nbf = basis.nbf();
  if( m != n ) 
    GAUXC_GENERIC_EXCEPTION("P/VXC Must Be Square");
  if( m != nbf ) 
    GAUXC_GENERIC_EXCEPTION("P/VXC Must Have Same Dimension as Basis");
  if( ldp < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDP");
  if( ldk < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDVXC");


  // Get Tasks
  this->load_balancer_->get_tasks();

  // Compute Local contributions to EXC / VXC
  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
    exx_local_work_( P, ldp, K, ldk, settings );
  });

  #ifdef GAUXC_HAS_MPI
  this->timer_.time_op("XCIntegrator.LocalWait", [&](){
    MPI_Barrier( this->load_balancer_->runtime().comm() );
  });
  #endif

  // Reduce Results
  this->timer_.time_op("XCIntegrator.Allreduce", [&](){

    if( not this->reduction_driver_->takes_host_memory() )
      GAUXC_GENERIC_EXCEPTION("This Module Only Works With Host Reductions");

    this->reduction_driver_->allreduce_inplace( K, nbf*nbf, ReductionOp::Sum );

  });

}





#if 0

// MBFS(i) = sqrt(W[i]) * sum_mu B(mu,i)
// return max_i MBFS(i)
double compute_max_bf_sum( size_t npts, size_t nbe_bfn, const double* weights,
  const double* basis_eval, size_t ldb ) {

  std::vector<double> bf_sums( npts );
  for( auto ipt = 0ul; ipt < npts; ++ipt ) {
    double tmp = 0.;
    for( auto ibf = 0ul; ibf < nbe_bfn; ++ibf ) 
      tmp += std::abs( basis_eval[ibf + ipt * ldb] );
    bf_sums[ipt] = std::sqrt(weights[ipt]) * tmp;
  }

  return *std::max_element( bf_sums.begin(), bf_sums.end() );

}


auto compute_approx_f_max( size_t npts, size_t nshells_bf, size_t nbf, 
  size_t nbe_bfn, const BasisSetMap& basis_map, const double* weights, 
  const std::vector<std::array<int32_t,3>>& submat_bfn, const double* basis_eval,
  size_t ldb, const double* P_abs, size_t ldp, LocalHostWorkDriver* lwd,
  double* nbe_scr) {

  // Get max value for each bfn over grid 
  std::vector<double> max_bf_grid( nbe_bfn );
  for( auto ibf = 0ul; ibf < nbe_bfn; ++ibf ) {
    double tmp = 0.;
    for( auto ipt = 0ul; ipt < npts; ++ipt )
      tmp = std::max( tmp,
        std::sqrt(weights[ipt]) *
        std::abs(basis_eval[ibf + ipt*ldb])
      );
    max_bf_grid[ibf] = tmp;
  }

  // Compute approximate F max over basis functions
  std::vector<double> max_F_approx_bfn( nbf );
  std::vector<std::array<int32_t,3>> submat_full = {
    std::array<int32_t,3>{0, (int32_t)nbf, 0}
  };

  lwd->eval_exx_fmat( 1, nbf, nbf, nbe_bfn, submat_full, submat_bfn,
    P_abs, ldp, max_bf_grid.data(), nbe_bfn, max_F_approx_bfn.data(),
    nbf, nbe_scr );

  // Collapse approx F max over shells 
  std::vector<double> max_F_approx( nshells_bf );
  for( auto ish = 0ul; ish < nshells_bf; ++ish ) {
    const auto sh_st = basis_map.shell_to_first_ao(ish);
    const auto sh_sz = basis_map.shell_size(ish);
    double tmp = 0.;
    for( auto i = sh_st; i < sh_st + sh_sz; ++i )
      tmp = std::max( tmp, std::abs(max_F_approx_bfn[i]) );
    max_F_approx[ish] = tmp;
  }

  return max_F_approx;
}



auto compute_true_f_max( size_t npts, size_t nshells_bra, size_t nbe_bra,
  const BasisSetMap& basis_map, const std::vector<int32_t>& shell_list_bra,
  const double* weights, const double* F, size_t ldf ) {

  std::vector<double> max_F( nshells_bra );
  size_t sh_st = 0;
  for( auto i = 0ul; i < nshells_bra; ++i ) {
    const auto ish = shell_list_bra[i];
    const auto sh_sz = basis_map.shell_size(ish);

    double tmp_max = 0.;
    for( auto ipt = 0ul; ipt < npts; ++ipt ) 
    for( auto ii = 0ul;   ii < sh_sz;  ++ii  )
      tmp_max = std::max( tmp_max,
        std::sqrt(weights[ipt]) *
        std::abs( F[ sh_st + ii + ipt*ldf] )
      );
    max_F[i] = tmp_max;

    sh_st += sh_sz;
  }

  return max_F;

}



auto compute_sn_LinK_E_set( size_t nshells, const std::vector<int32_t>& shell_list,
  const double* V_max, size_t ldv, const double* max_F, double E_tol ) {

  std::set<int32_t> E_shells;
  for( auto j = 0ul; j < nshells; ++j ) 
  for( auto i = j;   i < nshells; ++i ) {

    const auto ish = shell_list[i];
    const auto jsh = shell_list[j];

    const auto V_ij = V_max[ish + jsh*ldv];
    const auto F_i  = max_F[i];
    const auto F_j  = max_F[j];

    const double eps_E_compare = F_i * F_j * V_ij;
    if( eps_E_compare > E_tol )  {
      E_shells.insert(ish); 
      E_shells.insert(jsh); 
    }

  }

  return E_shells;

}

auto compute_sn_LinK_K_set( size_t nshells, const std::vector<int32_t>& shell_list,
  const double* V_max, size_t ldv, const double* max_F, double max_bf_sum,
  double K_tol ) {

  std::set<int32_t> K_shells;
  for( auto j = 0ul; j < nshells; ++j ) 
  for( auto i = j;   i < nshells; ++i ) {

    const auto ish = shell_list[i];
    const auto jsh = shell_list[j];

    const auto V_ij = V_max[ish + jsh*ldv];
    const auto F_i  = max_F[i];
    const auto F_j  = max_F[j];

    const double eps_K_compare = std::max(F_i, F_j) * V_ij * max_bf_sum;
    if( eps_K_compare > K_tol )  {
      K_shells.insert(ish); 
      K_shells.insert(jsh); 
    }

  }

  return K_shells;

}

auto compute_sn_LinK_ek_set( size_t nshells, const std::vector<int32_t>& shell_list,
  const double* V_max, size_t ldv, const double* max_F, double max_bf_sum,
  double E_tol, double K_tol ) {

  std::set<int32_t> ek_shells;
  for( auto j = 0ul; j < nshells; ++j ) 
  for( auto i = j;   i < nshells; ++i ) {

    const auto ish = shell_list[i];
    const auto jsh = shell_list[j];

    const auto V_ij = V_max[ish + jsh*ldv];
    const auto F_i  = max_F[i];
    const auto F_j  = max_F[j];

    const double eps_E_compare = F_i * F_j * V_ij;
    const double eps_K_compare = std::max(F_i, F_j) * V_ij * max_bf_sum;
    if( eps_K_compare > K_tol or eps_E_compare > E_tol)  {
      ek_shells.insert(ish); 
      ek_shells.insert(jsh); 
    }

  }

  return ek_shells;

}


#endif











template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  exx_local_work_( const value_type* P, int64_t ldp, 
    value_type* K, int64_t ldk, const IntegratorSettingsEXX& settings ) {

  // Cast LWD to LocalHostWorkDriver
  auto* lwd = dynamic_cast<LocalHostWorkDriver*>(this->local_work_driver_.get());

  // Setup Aliases
  const auto& basis   = this->load_balancer_->basis();
  const auto& mol     = this->load_balancer_->molecule();
  const auto& shpairs = this->load_balancer_->shell_pairs();


  // Get basis map
  BasisSetMap basis_map(basis,mol);

  const int32_t nbf = basis.nbf();

  // Sort tasks on size (XXX: maybe doesnt matter?)
  auto task_comparator = []( const XCTask& a, const XCTask& b ) {
    return (a.points.size() * a.bfn_screening.nbe) > (b.points.size() * b.bfn_screening.nbe);
  };

  auto& tasks = this->load_balancer_->get_tasks();
  std::sort( tasks.begin(), tasks.end(), task_comparator );


  // Check that Partition Weights have been calculated
  auto& lb_state = this->load_balancer_->state();
  if( not lb_state.modified_weights_are_stored ) {
    GAUXC_GENERIC_EXCEPTION("Weights Have Not Been Modified"); 
  }

  // Zero out integrands
  for( auto j = 0; j < nbf; ++j )
  for( auto i = 0; i < nbf; ++i ) 
    K[i + j*ldk] = 0.;

   
  // Compute V upper bounds per shell pair
  const size_t nshells_bf = basis.size();
  std::vector<double> V_max( nshells_bf * nshells_bf );
  // Loop over sparse shell pairs
  const auto sp_row_ptr = shpairs.row_ptr();
  const auto sp_col_ind = shpairs.col_ind();
  for( auto i = 0; i < nshells_bf; ++i ) {
    const auto j_st = sp_row_ptr[i];
    const auto j_en = sp_row_ptr[i+1];
    for( auto _j = j_st; _j < j_en; ++_j ) {
      const auto j = sp_col_ind[_j];
      const auto mv = util::max_coulomb( basis.at(i), basis.at(j) );
      V_max[i + j*nshells_bf] = mv;
      if( i != j ) V_max[j + i*nshells_bf] = mv;
    }
  }

  // Absolute value of P
  std::vector<double> P_abs(nbf*nbf);
  for( auto i = 0; i < nbf*nbf; ++i ) P_abs[i] = std::abs(P[i]);

  // Full shell list
  std::vector<int32_t> full_shell_list_( basis.nshells() );
  std::iota( full_shell_list_.begin(), full_shell_list_.end(), 0 );
  std::vector< std::array<int32_t,3> > full_submat_map = { {0, nbf, 0} };

  // Screening settings
  IntegratorSettingsSNLinK sn_link_settings;
  if( auto* tmp = dynamic_cast<const IntegratorSettingsSNLinK*>(&settings) ) {
    sn_link_settings = *tmp;
  }

  const bool screen_ek = sn_link_settings.screen_ek;
  const double eps_K   = sn_link_settings.k_tol;
  const double eps_E   = sn_link_settings.energy_tol;

  int world_rank = 0;
  #ifdef GAUXC_HAS_MPI
  auto comm = this->load_balancer_->runtime().comm();
  MPI_Comm_rank( comm, &world_rank );
  #endif
  //if( !world_rank ) {
  //  std::cout << "sn-LinK Settings:" << std::endl
  //            << "  SCREEN_EK     = " << std::boolalpha << screen_ek << std::endl
  //            << "  EPS_E         = " << eps_E << std::endl
  //            << "  EPS_K         = " << eps_K << std::endl
  //            << std::endl;
  //}

  // Reset the coulomb screening data
  for(auto& task : tasks) task.cou_screening = XCTask::screening_data();

  // Precompute EK shell screening
  exx_ek_screening( basis, basis_map, shpairs, P_abs.data(), nbf, V_max.data(), 
    nshells_bf, eps_E, eps_K, lwd, tasks.begin(), tasks.end() );

  // Allow for merging of tasks with different iParent
  for(auto& task : tasks) task.iParent = 0;

#if 1
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

  std::sort( tasks.begin(), tasks.end(), task_order ); 
  auto task_equiv = []( const auto& a, const auto& b ) {
    return a.equiv_with(b) and 
      a.cou_screening.equiv_with(b.cou_screening);
  };
  std::vector<XCTask> local_work_unique(tasks.begin(), tasks.end());
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

  auto cur_lw_begin = tasks.begin();
  auto cur_uniq_it  = local_work_unique.begin();

  for( auto lw_it = tasks.begin(); lw_it != tasks.end(); ++lw_it ) 
  if( not task_equiv( *lw_it, *cur_uniq_it ) ) {

    if( cur_uniq_it == local_work_unique.end() )
      GAUXC_GENERIC_EXCEPTION("Messed up in unique");

    cur_uniq_it->merge_with( cur_lw_begin, lw_it );

    cur_lw_begin = lw_it;
    cur_uniq_it++;

  }

  // Merge the last set of batches
  for( ; cur_lw_begin != tasks.end(); ++cur_lw_begin )
    cur_uniq_it->merge_with( *cur_lw_begin );
  cur_uniq_it++;

  tasks = std::move(local_work_unique);
#endif

  std::sort(tasks.begin(),tasks.end(),
    [](auto& a, auto& b){ return a.cou_screening.shell_pair_list.size() >
      b.cou_screening.shell_pair_list.size(); });


  // Loop over tasks
  const size_t ntasks = tasks.size();
  //std::cout << "NTASKS = " << ntasks << std::endl;
  //std::cout << "NTASKS NNZ = " << std::count_if(tasks.begin(),tasks.end(),[](const auto& t){ return t.cou_screening.shell_pair_list.size(); }) << std::endl;
  #pragma omp parallel
  {

  XCHostData<value_type> host_data; // Thread local host data
  std::vector<double> K_local(nbf*nbf,0.0);

  #pragma omp for schedule(dynamic)
  for( size_t iT = 0; iT < ntasks; ++iT ) {

    //std::cout << iT << "/" << ntasks << std::endl;
    // Alias current task
    const auto& task = tasks[iT];

    // Early exit
    auto ek_shell_list = task.cou_screening.shell_list;
    if( ek_shell_list.size() == 0 ) {
      continue;
    }
    std::vector< std::array<int32_t,3> > ek_submat_map;
    std::tie( ek_submat_map, std::ignore ) =
      gen_compressed_submat_map( basis_map, ek_shell_list, nbf, nbf );

    // Get tasks constants
    const int32_t  npts    = task.points.size();

    const auto* points      = task.points.data()->data();
    const auto* weights     = task.weights.data();

    // Basis function shell list
    auto shell_list_bfn_ = task.bfn_screening.shell_list;
    int32_t* shell_list_bfn = shell_list_bfn_.data();
    size_t nshells_bfn = shell_list_bfn_.size();
    size_t nbe_bfn     = 
      basis.nbf_subset( shell_list_bfn_.begin(), shell_list_bfn_.end() );

    std::vector< std::array<int32_t, 3> > submat_map_bfn;
    std::tie(submat_map_bfn, std::ignore) =
      gen_compressed_submat_map( basis_map, shell_list_bfn_, nbf, nbf );
    


    // Allocate data screening independent data
    host_data.basis_eval.resize( npts * nbe_bfn );
    host_data.nbe_scr   .resize( nbe_bfn * nbf );
    auto* basis_eval = host_data.basis_eval.data();
    auto* nbe_scr    = host_data.nbe_scr.data();



    // Evaluate collocation B(mu,i)
    // mu ranges over the bfn shell list and i runs over all points
    lwd->eval_collocation( npts, nshells_bfn, nbe_bfn, points, basis, 
      shell_list_bfn, basis_eval );

    const auto nbe_ek = basis.nbf_subset( ek_shell_list.begin(), ek_shell_list.end() );
    const auto nshells_ek = ek_shell_list.size();


    // Allocate Screening Dependent Data
    host_data.zmat.resize( npts * nbe_ek );
    host_data.gmat.resize( npts * nbe_ek );
    auto* zmat = host_data.zmat.data();
    auto* gmat = host_data.gmat.data();

    // Evaluate F(mu,i) = P(mu,nu) * B(nu,i)
    // mu runs over significant ek shells
    // nu runs over the bfn shell list
    // i runs over all points
    lwd->eval_exx_fmat( npts, nbf, nbe_ek, nbe_bfn, ek_submat_map,
      submat_map_bfn, P, ldp, basis_eval, nbe_bfn, zmat, nbe_ek, nbe_scr );

    // Get True Max F for shell pairs
    //auto max_F = compute_true_f_max( npts, nshells_ek, nbe_ek, basis_map,
    //  ek_shell_list, weights, zmat, nbe_ek );


    // Compute G(mu,i) = w(i) * A(mu,nu,i) * F(nu,i)
    // mu/nu run over significant ek shells
    // i runs over all points
    const size_t nshell_pairs = task.cou_screening.shell_pair_list.size();
    const auto*  shell_pair_list = task.cou_screening.shell_pair_list.data();
    lwd->eval_exx_gmat( npts, nshells_ek, nshell_pairs, nbe_ek, points, weights, 
      basis, shpairs,basis_map, ek_shell_list.data(), shell_pair_list, zmat, 
      nbe_ek, gmat, nbe_ek );

    // Increment K(mu,nu) += B(mu,i) * G(nu,i)
    // mu runs over bfn shell list
    // nu runs over ek shells
    // i runs over all points
    lwd->inc_exx_k( npts, nbf, nbe_bfn, nbe_ek, basis_eval, submat_map_bfn,
      ek_submat_map, gmat, nbe_ek, K, ldk, nbe_scr );

  } // Loop over tasks 


  } // End OpenMP region

  // Symmetrize K
  for( auto j = 0; j < nbf; ++j ) 
  for( auto i = 0; i < j;   ++i ) {
    const auto K_ij = K[i + j*ldk];
    const auto K_ji = K[j + i*ldk];
    const auto K_symm = 0.5 * (K_ij + K_ji);
    K[i + j*ldk] = K_symm;
    K[j + i*ldk] = K_symm;
  }

}

} // namespace GauXC::detail
