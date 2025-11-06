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
#include "integrator_util/spherical_harmonics.hpp"
#include "host/local_host_work_driver.hpp"
#include <gauxc/molgrid/defaults.hpp>
#include <stdexcept>
#ifdef GAUXC_ENABLE_OPENMP
#include <omp.h>
#endif

namespace GauXC::detail {
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_dd_psi_( int64_t m, int64_t n, const value_type* P,
                int64_t ldp, unsigned max_Ylm, value_type* ddPsi, 
                int64_t ldPsi ) {

  const auto& basis = this->load_balancer_->basis();
  const auto& mol   = this->load_balancer_->molecule();
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
  // Compute Local contributions to ddPsi
  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
   dd_psi_local_work_( P, ldp, max_Ylm, ddPsi, ldPsi );
  });


  // Reduce Results
  this->timer_.time_op("XCIntegrator.Allreduce", [&](){

    if( not this->reduction_driver_->takes_host_memory() )
      GAUXC_GENERIC_EXCEPTION("This Module Only Works With Host Reductions");

    this->reduction_driver_->allreduce_inplace( ddPsi, ldPsi * mol.size(), ReductionOp::Sum );

  });
}

template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  dd_psi_local_work_( const value_type* P, int64_t ldp, unsigned max_Ylm,
    value_type* dd_Psi, int64_t ldPsi) {

  // Cast LWD to LocalHostWorkDriver
  auto* lwd = dynamic_cast<LocalHostWorkDriver*>(this->local_work_driver_.get());

  // Setup Aliases
  const auto& basis = this->load_balancer_->basis();
  const auto& mol   = this->load_balancer_->molecule();

  // Atom-specific data
  int natom = mol.size();
  std::vector<double> radii(natom);
  for (int i = 0; i < natom; ++i) {
    radii[i] = uff_radius_103(mol[i].Z);
  }
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
  #ifdef GAUXC_ENABLE_OPENMP
  #pragma omp parallel
  #endif
  {

  XCHostData<value_type> host_data; // Thread local host data

  #ifdef GAUXC_ENABLE_OPENMP
  #pragma omp for schedule(dynamic) reduction(+:dd_Psi[:natom * ldPsi])
  #endif
  for( size_t iT = 0; iT < ntasks; ++iT ) {

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

    int nharmonics = (max_Ylm + 1) * (max_Ylm + 1);

    // Get the submatrix map for batch
    std::vector< std::array<int32_t, 3> > submat_map;
    std::tie(submat_map, std::ignore) =
          gen_compressed_submat_map(basis_map, task.bfn_screening.shell_list, nbf, nbf);

    // Evaluate Collocation
    lwd->eval_collocation( npts, nshells, nbe, points, basis, shell_list, 
      basis_eval );

    // Evaluate X matrix (P * B) -> store in Z
    lwd->eval_xmat( npts, nbf, nbe, submat_map, 1.0, P, ldp, basis_eval, nbe,
      zmat, nbe, nbe_scr );

    // Evaluate density on grid
    lwd->eval_uvvar_lda_rks( npts, nbe, basis_eval, zmat, nbe, den_eval );

    // Populate dd_Psi
    const size_t atom_offset = task.iParent * ldPsi;
    const double radius = radii[task.iParent];
    const std::array<double, 3> center = {mol[task.iParent].x, mol[task.iParent].y, mol[task.iParent].z};

    std::vector<double> ylm_matrix(npts * nharmonics);
    scaled_ylm_matrix(max_Ylm, points, npts, center, radius, ylm_matrix.data());

    for (int i = 0; i < npts; ++i) {
      den_eval[i] *= -weights[i];
    }
    std::vector<double> offset_local_dd_psi(ldPsi, 0.0);
    blas::gemm('N', 'N', ldPsi, 1, npts,  
            1.0, ylm_matrix.data(), ldPsi,   
            den_eval, npts,     
            0.0, offset_local_dd_psi.data(), ldPsi); 
    for (int j = 0; j < ldPsi; ++j) {
      dd_Psi[atom_offset + j] += offset_local_dd_psi[j];
    }

  } // Loop over tasks 
  } // End OpenMP region
}
} // namespace GauXC::detail

