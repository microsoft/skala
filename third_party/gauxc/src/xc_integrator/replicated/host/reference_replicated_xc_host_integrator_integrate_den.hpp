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
#include "host/local_host_work_driver.hpp"
#include <stdexcept>

namespace GauXC::detail {

template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  integrate_den_( int64_t m, int64_t n, const value_type* P,
                  int64_t ldp, value_type* N_EL ) {

  const auto& basis = this->load_balancer_->basis();

  // Check that P / VXC are sane
  const int64_t nbf = basis.nbf();
  if( m != n ) 
    GAUXC_GENERIC_EXCEPTION("P Must Be Square");
  if( m != nbf ) 
    GAUXC_GENERIC_EXCEPTION("P Must Have Same Dimension as Basis");
  if( ldp < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDP");


  // Get Tasks
  this->load_balancer_->get_tasks();

  *N_EL = 0.;
  // Compute Local contributions to EXC / VXC
  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
    integrate_den_local_work_( P, ldp, N_EL );
  });


  // Reduce Results
  this->timer_.time_op("XCIntegrator.Allreduce", [&](){

    if( not this->reduction_driver_->takes_host_memory() )
      GAUXC_GENERIC_EXCEPTION("This Module Only Works With Host Reductions");

    this->reduction_driver_->allreduce_inplace( N_EL, 1, ReductionOp::Sum );

  });

}

template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  integrate_den_local_work_( const value_type* P, int64_t ldp, 
    value_type* N_EL ) {

  // Cast LWD to LocalHostWorkDriver
  auto* lwd = dynamic_cast<LocalHostWorkDriver*>(this->local_work_driver_.get());

  // Setup Aliases
  const auto& basis = this->load_balancer_->basis();
  const auto& mol   = this->load_balancer_->molecule();

  // Get basis map
  BasisSetMap basis_map(basis,mol);

  const int32_t nbf = basis.nbf();

  // Sort tasks on size (XXX: maybe doesnt matter?)
  auto task_comparator = []( const XCTask& a, const XCTask& b ) {
    return (a.points.size() * a.bfn_screening.nbe) > (b.points.size() * b.bfn_screening.nbe);
  };

  auto& tasks = this->load_balancer_->get_tasks();
  std::sort( tasks.begin(), tasks.end(), task_comparator );


  // Compute Partition Weights
  auto& lb_state = this->load_balancer_->state();
  if( not lb_state.modified_weights_are_stored ) {
    GAUXC_GENERIC_EXCEPTION("Weights Have Not Been Modified"); 
  }


  // Loop over tasks
  const size_t ntasks = tasks.size();
  double N_EL_WORK = 0.0;

  #pragma omp parallel
  {

  XCHostData<value_type> host_data; // Thread local host data
  double N_EL_LOCAL = 0.;

  #pragma omp for schedule(dynamic)
  for( size_t iT = 0; iT < ntasks; ++iT ) {

    //std::cout << iT << "/" << ntasks << std::endl;
    // Alias current task
    const auto& task = tasks[iT];

    // Get tasks constants
    const int32_t  npts    = task.points.size();
    const int32_t  nbe     = task.bfn_screening.nbe;
    const int32_t  nshells = task.bfn_screening.shell_list.size();

    const auto* points      = task.points.data()->data();
    const auto* weights     = task.weights.data();
    const int32_t* shell_list = task.bfn_screening.shell_list.data();

    // Allocate enough memory for batch

    host_data.nbe_scr .resize( nbe * nbe  );
    host_data.zmat    .resize( npts * nbe );

    host_data.basis_eval .resize( npts * nbe );
    host_data.den_scr    .resize( npts );


    // Alias/Partition out scratch memory
    auto* basis_eval = host_data.basis_eval.data();
    auto* den_eval   = host_data.den_scr.data();
    auto* nbe_scr    = host_data.nbe_scr.data();
    auto* zmat       = host_data.zmat.data();


    // Get the submatrix map for batch
    std::vector< std::array<int32_t, 3> > submat_map;
    std::tie(submat_map, std::ignore) =
          gen_compressed_submat_map(basis_map, task.bfn_screening.shell_list, nbf, nbf);

    // Evaluate Collocation (+ Grad)
    lwd->eval_collocation( npts, nshells, nbe, points, basis, shell_list, 
      basis_eval );


    // Evaluate X matrix (P * B) -> store in Z
    lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, P, ldp, basis_eval, nbe,
      zmat, nbe, nbe_scr );


    // Evaluate density on grid
    lwd->eval_uvvar_lda_rks( npts, nbe, basis_eval, zmat, nbe, den_eval );

    // Scalar integrations
    for( int32_t i = 0; i < npts; ++i ) {
      N_EL_LOCAL += weights[i] * den_eval[i];
    }

  } // Loop over tasks 

  #pragma omp atomic 
  N_EL_WORK += N_EL_LOCAL;

  } // End OpenMP region

  // Commit return value
  *N_EL = N_EL_WORK;

}

} // namespace GauXC::detail
