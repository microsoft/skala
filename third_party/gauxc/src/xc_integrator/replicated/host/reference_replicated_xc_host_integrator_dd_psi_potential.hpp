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
#include <stdexcept>
#include "host/blas.hpp"
#include "host/util.hpp"

#ifdef GAUXC_ENABLE_OPENMP
#include <omp.h>
#endif

namespace GauXC::detail {
template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_dd_psi_potential_( int64_t m, int64_t n, const value_type* X, unsigned max_Ylm, 
    value_type* Vddx ) {

  const auto& basis = this->load_balancer_->basis();
  const int32_t nbf = basis.nbf();

  // Check that m is natom, n is nharmonics
  const auto& mol = this->load_balancer_->molecule();
  const size_t natom = mol.size();
  const size_t nharmonics = (max_Ylm + 1) * (max_Ylm + 1);
  if (m != nharmonics || n != natom) {
    GAUXC_GENERIC_EXCEPTION("m must be nharmonics and n must be natom");
  }
  // Get Tasks
  this->load_balancer_->get_tasks();
  // Compute Local contributions to EXC / VXC
  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
   dd_psi_potential_local_work_( X, Vddx, max_Ylm );
  });

  // Reduce Results
  this->timer_.time_op("XCIntegrator.Allreduce", [&](){

    if( not this->reduction_driver_->takes_host_memory() )
      GAUXC_GENERIC_EXCEPTION("This Module Only Works With Host Reductions");

    this->reduction_driver_->allreduce_inplace( Vddx, nbf * nbf, ReductionOp::Sum );

  });
}

template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  dd_psi_potential_local_work_( const value_type* X, value_type* Vddx, unsigned max_Ylm ) {

  // Cast LWD to LocalHostWorkDriver
  auto* lwd = dynamic_cast<LocalHostWorkDriver*>(this->local_work_driver_.get());

  // Setup Aliases
  const auto& basis = this->load_balancer_->basis();
  const auto& mol   = this->load_balancer_->molecule();

  // Atom-specific data
  std::vector<double> radii(mol.size());
  for (int i = 0; i < mol.size(); ++i) {
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
  #pragma omp for schedule(dynamic)
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
    host_data.basis_eval .resize( npts * nbe );
    auto* basis_eval = host_data.basis_eval.data();

    host_data.nbe_scr .resize( nbe * nbe  );
    auto* vddx_scr = host_data.nbe_scr.data();

    host_data.den_scr    .resize( npts );
    auto etas = host_data.den_scr.data();

    host_data.zmat    .resize( npts * nbe );
    auto* zmat = host_data.zmat.data();
    
    int nharmonics = (max_Ylm + 1) * (max_Ylm + 1);

    // Get the submatrix map for batch
    std::vector< std::array<int32_t, 3> > submat_map;
    std::tie(submat_map, std::ignore) =
          gen_compressed_submat_map(basis_map, task.bfn_screening.shell_list, nbf, nbf);
    
    // Evaluate Collocation
    lwd->eval_collocation( npts, nshells, nbe, points, basis, shell_list, 
      basis_eval );
    
    // Project X onto the spherical harmonics basis
    const size_t atom_offset = task.iParent * nharmonics;
    const double radius = radii[task.iParent];
    std::array<double, 3> center = {mol[task.iParent].x, mol[task.iParent].y, mol[task.iParent].z};
    const value_type* X_i = X + atom_offset;

    std::vector<double> ylm_matrix(npts * nharmonics);
    scaled_ylm_matrix(max_Ylm, points, npts, center, radius, ylm_matrix.data());

    blas::gemm('T', 'N', npts, 1, nharmonics, 
              1.0, ylm_matrix.data(), nharmonics, 
              X_i, nharmonics,                
              0.0, etas, npts);

    // zmat = phi * etas
    for (int ipt = 0; ipt < npts; ipt++) {
      etas[ipt] *= weights[ipt];
      for (int ibe = 0; ibe < nbe; ibe++) {
        zmat[ipt * nbe + ibe] = basis_eval[ipt * nbe + ibe] * etas[ipt]; // nbe is fastest, col in column-major
      }
    }

    // vddx_scr = phi^T * etas * weights * phi
    blas::gemm('N', 'T', nbe, nbe, npts, 1.0, basis_eval, nbe, zmat, nbe, 0.0, vddx_scr, nbe);

    detail::inc_by_submat_atomic( nbf, nbf, nbe, nbe, Vddx, nbf, vddx_scr, nbe,
                        submat_map );
  } // Loop over tasks 
  } // End OpenMP region
}

} // namespace GauXC::detail

