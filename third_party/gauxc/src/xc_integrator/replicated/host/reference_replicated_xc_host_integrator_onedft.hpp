/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */
#pragma once

#include "reference_replicated_xc_host_integrator.hpp"
#include "integrator_util/integrator_common.hpp"
#include "integrator_util/onedft_util.hpp"
#include "host/local_host_work_driver.hpp"
#include "host/blas.hpp"

#include <stdexcept>
#include <string>

namespace GauXC  {
namespace detail {
  
FeatureDict prepare_onedft_features(const int ndm, std::vector<XCTask>& tasks, const Molecule& mol, 
  const std::vector<std::string> feature_keys, const RuntimeEnvironment& rt, std::vector<int>& sendcounts, 
  std::vector<int>& displs);

void send_buffer_onedft_outputs(const int ndm, const FeatureDict features_dict, std::vector<XCTask>& tasks, 
  const RuntimeEnvironment& rt, std::vector<int> sendcounts, std::vector<int> displs);

void interleave_data(const double* a, const double* b, const size_t n, double* out);

void eval_zmat_gga_vxc_uks(size_t npts, size_t nbf, 
  const double* vdden_eval_a, const double* vdden_eval_b, 
  const double* vdden_x_eval_a, const double* vdden_x_eval_b, const double* vdden_y_eval_a, 
  const double* vdden_y_eval_b, const double* vdden_z_eval_a, const double* vdden_z_eval_b,
  const double* basis_eval, const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval,
  double* Zs, size_t ldzs, double* Zz, size_t ldzz);

void eval_zmat_mgga_vxc_uks(size_t npts, size_t nbf, 
  const double* vdden_eval_a, const double* vdden_eval_b, 
  const double* vlapl_a, const double* vlapl_b,
  const double* vdden_x_eval_a, const double* vdden_x_eval_b, const double* vdden_y_eval_a, 
  const double* vdden_y_eval_b, const double* vdden_z_eval_a, const double* vdden_z_eval_b,
  const double* basis_eval, const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval,
  const double* lbasis_eval, double* Zs, size_t ldzs, double* Zz, size_t ldzz);

template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  eval_exc_vxc_onedft_( int64_t m, int64_t n, 
    const value_type* Ps, int64_t ldps,
    const value_type* Pz, int64_t ldpz,
    value_type* VXCs, int64_t ldvxcs,
    value_type* VXCz, int64_t ldvxcz,
    value_type* EXC, const IntegratorSettingsXC& settings ) {

  const auto& basis = this->load_balancer_->basis();

  // Check that P / VXC are sane
  const int64_t nbf = basis.nbf();
  if( m != n ) 
    GAUXC_GENERIC_EXCEPTION("P Must Be Square");
  if( m != nbf ) 
    GAUXC_GENERIC_EXCEPTION("P Must Have Same Dimension as Basis");

  if( ldps and ldps < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDP");
  if( ldpz and ldpz < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDPZ");

  if( ldvxcs < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDVXCS");
  if( ldvxcz and ldvxcz < nbf )
    GAUXC_GENERIC_EXCEPTION("Invalid LDVXCZ");

  // const bool is_exc_only = (!VXCs) and (!VXCz) and (!VXCy) and (!VXCx);

  const bool is_uks = (Pz != nullptr);
  if (not is_uks) {
    // TODO: duplicate the density matrix ot duplicate the feature results?
    // Pz = Ps;
    GAUXC_GENERIC_EXCEPTION("RKS Not Yet Implemented");
  }
  // Get Tasks
  auto& tasks = this->load_balancer_->get_tasks();  
#ifdef GAUXC_HAS_DEVICE
  auto rt  = detail::as_device_runtime(this->load_balancer_->runtime());
#else
  auto rt = this->load_balancer_->runtime();
#endif
  int32_t world_rank = rt.comm_rank();
  // Temporary electron count to judge integrator accuracy
  value_type N_EL; 
  
  // load model from parameter/interator settings
  OneDFTSettings onedft_settings;
  if( auto* tmp = dynamic_cast<const OneDFTSettings*>(&settings) ) {
    onedft_settings = *tmp;
  }
  const auto model_path = onedft_settings.model;
  torch::DeviceType device = torch::kCPU;
  auto [exc_func, feature_keys] = load_model(model_path, device);
  
  // determine what feature we need based on the keys
  if (feature_keys.size() == 0) {
    GAUXC_GENERIC_EXCEPTION("No feature keys found in model");
  }
  bool is_gga = false;
  bool is_mgga = false;
  for (const auto& key : feature_keys) {
    if ( not valueExists(key) ) GAUXC_GENERIC_EXCEPTION("Feature Key Required Not Implemented: " + key);
    if (key == feat_map.at(ONEDFT_FEATURE::TAU)) is_mgga = true;
    if (key == feat_map.at(ONEDFT_FEATURE::DDEN)) is_gga = true;
  }
  if (is_mgga) is_gga = false;

  // Compute Local contributions to EXC / VXC
  this->timer_.time_op("XCIntegrator.LocalWork", [&](){
    pre_onedft_local_work_( basis, Ps, ldps, Pz, ldpz, &N_EL, is_gga, is_mgga, false /*needs_laplacian*/);
  });
  std::vector<int> sendcounts(rt.comm_size(), 0);
  std::vector<int> displs(rt.comm_size(), 0);
  FeatureDict features_dict = prepare_onedft_features(2/*ndm*/, tasks, this->load_balancer_->molecule(), feature_keys, rt, 
    sendcounts, displs);
  if (world_rank == 0) {
    auto exc_on_grid = get_exc(exc_func, features_dict);
    // check is_nan
    if (exc_on_grid.isnan().any().item<bool>()) {
      GAUXC_GENERIC_EXCEPTION("exc_on_grid has NaN");
    }
    auto exc = (exc_on_grid * features_dict.at(feat_map.at(ONEDFT_FEATURE::WEIGHTS))).sum();
    exc.backward();
    EXC[0] = exc.item().to<double>();
    std::cout << "EXC: " << EXC[0] << std::endl;
  }
  // MPI_Bcast(EXC, 1, MPI_DOUBLE, 0, rt.comm());
  // TODO: stop here if only exc

  send_buffer_onedft_outputs(2/*ndm*/, features_dict, tasks, rt, sendcounts, displs);

  this->timer_.time_op("XCIntegrator.LocalWork2", [&](){
    post_onedft_local_work_( basis, Ps, ldps, Pz, ldpz, VXCs, n, VXCz, n, is_gga, is_mgga, false /*needs_laplacian*/);
  });

  this->timer_.time_op("XCIntegrator.Allreduce", [&](){

    if( not this->reduction_driver_->takes_host_memory() )
      GAUXC_GENERIC_EXCEPTION("This Module Only Works With Host Reductions");

    this->reduction_driver_->allreduce_inplace( VXCs, nbf*nbf, ReductionOp::Sum );
    if(VXCz) this->reduction_driver_->allreduce_inplace( VXCz, nbf*nbf, ReductionOp::Sum );

    this->reduction_driver_->allreduce_inplace( EXC,   1    , ReductionOp::Sum );
    this->reduction_driver_->allreduce_inplace( &N_EL, 1    , ReductionOp::Sum );

  });

}

template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  pre_onedft_local_work_( const basis_type& basis, const value_type* Ps, int64_t ldps,
    const value_type* Pz, int64_t ldpz, value_type *N_EL, 
    const bool is_gga, const bool is_mgga, const bool needs_laplacian) {

  const bool is_uks = (Pz != nullptr);
  const bool is_rks = not is_uks;
  const bool is_lda = not is_gga and not is_mgga;
  // Cast LWD to LocalHostWorkDriver
  auto* lwd = dynamic_cast<LocalHostWorkDriver*>(this->local_work_driver_.get());
  const auto& mol   = this->load_balancer_->molecule();
  BasisSetMap basis_map(basis,mol);
  const int32_t nbf = basis.nbf();

  auto& tasks = this->load_balancer_->get_tasks();
  const size_t ntasks = tasks.size();

  // Check that Partition Weights have been calculated
  auto& lb_state = this->load_balancer_->state();
  if( not lb_state.modified_weights_are_stored ) {
    GAUXC_GENERIC_EXCEPTION("Weights Have Not Been Modified");
  }

  double NEL_WORK = 0.0;

  #pragma omp parallel
  {

  XCHostData<value_type> host_data; // Thread local host data
  
  #pragma omp for schedule(dynamic)
  for( size_t iT = 0; iT < ntasks; ++iT ) {
    auto& task = tasks[iT];

    // Get tasks constants
    const int32_t  npts    = task.points.size();
    const int32_t  nbe     = task.bfn_screening.nbe;
    const int32_t  nshells = task.bfn_screening.shell_list.size();

    const auto* points      = task.points.data()->data();
    const auto* weights     = task.weights.data();
    const int32_t* shell_list = task.bfn_screening.shell_list.data();
    const size_t spin_dim_scal = is_rks ? 1 : 2; 
    const size_t mgga_dim_scal = is_mgga ? 4 : 1; // basis + d1basis
 
    // Things that every calc needs
    host_data.nbe_scr .resize(nbe  * nbe);
    host_data.zmat    .resize(npts * nbe * spin_dim_scal * mgga_dim_scal); 

    // LDA data requirements
    if( is_lda ){
      host_data.basis_eval .resize( npts * nbe );
      task.feat.den_eval.resize(npts * spin_dim_scal);
    }
     
    // GGA data requirements
    const size_t gga_dim_scal = is_rks ? 1 : 3;
    if( is_gga ){
      host_data.basis_eval .resize( 4 * npts * nbe );
      host_data.gamma      .resize( gga_dim_scal * npts ); // TODO: delete gamma
      task.feat.den_eval.resize(npts * spin_dim_scal);
      task.feat.dden_x_eval.resize(npts * spin_dim_scal);
      task.feat.dden_y_eval.resize(npts * spin_dim_scal);
      task.feat.dden_z_eval.resize(npts * spin_dim_scal);
    }
    if( is_mgga ){
      host_data.basis_eval .resize( 4 * npts * nbe ); // basis + grad (3)
      host_data.gamma      .resize( gga_dim_scal * npts );
      task.feat.den_eval.resize(npts * spin_dim_scal);
      task.feat.dden_x_eval.resize(npts * spin_dim_scal);
      task.feat.dden_y_eval.resize(npts * spin_dim_scal);
      task.feat.dden_z_eval.resize(npts * spin_dim_scal);
      task.feat.tau.resize(npts * spin_dim_scal);
    }

    // Alias/Partition out scratch memory
    auto* basis_eval = host_data.basis_eval.data();
    auto* nbe_scr    = host_data.nbe_scr.data();
    auto* zmat       = host_data.zmat.data();
    auto* gamma      = host_data.gamma.data();
    auto* lapl       = host_data.lapl.data();

    decltype(zmat) zmat_z = nullptr;
    decltype(zmat) zmat_x = nullptr;
    decltype(zmat) zmat_y = nullptr;
    if(!is_rks) {
      zmat_z = zmat + mgga_dim_scal * nbe * npts;
    }
    
    auto* den_eval   = task.feat.den_eval.data();
    auto* tau        = task.feat.tau.data();
    auto* dden_x_eval = task.feat.dden_x_eval.data();
    auto* dden_y_eval = task.feat.dden_y_eval.data();
    auto* dden_z_eval = task.feat.dden_z_eval.data();

    value_type* dbasis_x_eval = nullptr;
    value_type* dbasis_y_eval = nullptr;
    value_type* dbasis_z_eval = nullptr;
    value_type* lbasis_eval = nullptr;

    value_type* mmat_x      = nullptr;
    value_type* mmat_y      = nullptr;
    value_type* mmat_z      = nullptr;
    value_type* mmat_x_z    = nullptr;
    value_type* mmat_y_z    = nullptr;
    value_type* mmat_z_z    = nullptr;

    if( is_gga ) {
      dbasis_x_eval = basis_eval    + npts * nbe;
      dbasis_y_eval = dbasis_x_eval + npts * nbe;
      dbasis_z_eval = dbasis_y_eval + npts * nbe;
    }

    if ( is_mgga ) {
      dbasis_x_eval = basis_eval    + npts * nbe;
      dbasis_y_eval = dbasis_x_eval + npts * nbe;
      dbasis_z_eval = dbasis_y_eval + npts * nbe;
      mmat_x        = zmat + npts * nbe;
      mmat_y        = mmat_x + npts * nbe;
      mmat_z        = mmat_y + npts * nbe;
      if(is_uks) {
        mmat_x_z = zmat_z + npts * nbe;
        mmat_y_z = mmat_x_z + npts * nbe;
        mmat_z_z = mmat_y_z + npts * nbe;
      }
    }


    // Get the submatrix map for batch
    std::vector< std::array<int32_t, 3> > submat_map;
    std::tie(submat_map, std::ignore) =
          gen_compressed_submat_map(basis_map, task.bfn_screening.shell_list, nbf, nbf);

    // Evaluate Collocation
    if( is_mgga ) {
        lwd->eval_collocation_gradient( npts, nshells, nbe, points, basis, shell_list,
          basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval );
    } else if( is_gga )
      lwd->eval_collocation_gradient( npts, nshells, nbe, points, basis, shell_list,
        basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval );
    else
      lwd->eval_collocation( npts, nshells, nbe, points, basis, shell_list,
        basis_eval );

     
    // Evaluate X matrix (fac * P * B) -> store in Z
    const auto xmat_fac = is_rks ? 2.0 : 1.0; // TODO Fix for spinor RKS input
    lwd->eval_xmat( mgga_dim_scal * npts, nbf, nbe, submat_map, xmat_fac, Ps, ldps, basis_eval, nbe,
      zmat, nbe, nbe_scr );

    // X matrix for Pz
    if(not is_rks) {
      lwd->eval_xmat( mgga_dim_scal * npts, nbf, nbe, submat_map, 1.0, Pz, ldpz, basis_eval, nbe,
        zmat_z, nbe, nbe_scr);
    }

    // Evaluate U and V variables
    if( is_mgga ) {
      if (is_rks) {
        lwd->eval_uvvar_mgga_rks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, lbasis_eval, zmat, nbe, mmat_x, mmat_y, mmat_z, 
          nbe, den_eval, dden_x_eval, dden_y_eval, dden_z_eval, gamma, tau, lapl);
      } else if (is_uks) {
        lwd->eval_uvvar_mgga_uks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, lbasis_eval, zmat, nbe, zmat_z, nbe, 
          mmat_x, mmat_y, mmat_z, nbe, mmat_x_z, mmat_y_z, mmat_z_z, nbe, 
          den_eval, dden_x_eval, dden_y_eval, dden_z_eval, gamma, tau, lapl);
      }
    } else if ( is_gga ) {
      if(is_rks) {
        lwd->eval_uvvar_gga_rks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, zmat, nbe, den_eval, dden_x_eval, dden_y_eval, dden_z_eval,
          gamma );
      } else if(is_uks) {
        lwd->eval_uvvar_gga_uks( npts, nbe, basis_eval, dbasis_x_eval, dbasis_y_eval,
          dbasis_z_eval, zmat, nbe, zmat_z, nbe, den_eval, dden_x_eval, 
          dden_y_eval, dden_z_eval, gamma );
      }
     } else {
      if(is_rks) {
        lwd->eval_uvvar_lda_rks( npts, nbe, basis_eval, zmat, nbe, den_eval );
      } else if(is_uks) {
        lwd->eval_uvvar_lda_uks( npts, nbe, basis_eval, zmat, nbe, zmat_z, nbe,
          den_eval );
      }
    }
    
    // Scalar integrations
    double NEL_local = 0.0;
    for( int32_t i = 0; i < npts; ++i ) {
      const auto den = is_rks ? den_eval[i] : (den_eval[2*i] + den_eval[2*i+1]);
      NEL_local += weights[i] * den;
    }

    // Atomic updates
    #pragma omp atomic
    NEL_WORK += NEL_local;
  } // Loop over tasks
}  // End OpenMP region
*N_EL = NEL_WORK;
// std::cout << "N_EL: " << *N_EL << std::endl;
}

template <typename ValueType>
void ReferenceReplicatedXCHostIntegrator<ValueType>::
  post_onedft_local_work_( const basis_type& basis, const value_type* Ps, int64_t ldps,
    const value_type* Pz, int64_t ldpz,
    value_type* VXCs, int64_t ldvxcs,
    value_type* VXCz, int64_t ldvxcz,
    const bool is_gga, const bool is_mgga, const bool needs_laplacian) {


    const bool is_uks = (Pz != nullptr);
    const bool is_rks = not is_uks;
    const bool is_lda = not is_gga and not is_mgga;
    auto* lwd = dynamic_cast<LocalHostWorkDriver*>(this->local_work_driver_.get());

    const auto& mol   = this->load_balancer_->molecule();
    BasisSetMap basis_map(basis,mol);
    const int32_t nbf = basis.nbf();

    // Zero out integrands
    
    if(VXCs)
    for( auto j = 0; j < nbf; ++j ) {
      for( auto i = 0; i < nbf; ++i ) {
        VXCs[i + j*ldvxcs] = 0.;
      }
    }

    if(VXCz) {
      for( auto j = 0; j < nbf; ++j ) {
        for( auto i = 0; i < nbf; ++i ) {
          VXCz[i + j*ldvxcz] = 0.;
        }
      }
    }

  // Loop over tasks
  auto& tasks = this->load_balancer_->get_tasks();
  const size_t ntasks = tasks.size();

  #pragma omp parallel
  {

  XCHostData<value_type> host_data; // Thread local host data

  #pragma omp for schedule(dynamic)
  for( size_t iT = 0; iT < ntasks; ++iT ) {
    const auto& task = tasks[iT];

    // Get tasks constants
    const int32_t  npts    = task.points.size();
    const int32_t  nbe     = task.bfn_screening.nbe;
    const int32_t  nshells = task.bfn_screening.shell_list.size();

    const auto* points      = task.points.data()->data();
    const auto* weights     = task.weights.data();
    const int32_t* shell_list = task.bfn_screening.shell_list.data();
    const size_t spin_dim_scal = is_rks ? 1 : is_uks ? 2 : 4; // last case is_gks
    const size_t mgga_dim_scal = is_mgga ? 4 : 1; // basis + d1basis
 
    // Things that every calc needs
    host_data.nbe_scr .resize(nbe  * nbe);
    host_data.zmat    .resize(npts * nbe * spin_dim_scal * mgga_dim_scal); 

    // LDA data requirements
    if( is_lda ){
      host_data.basis_eval .resize( npts * nbe );
    }
    // GGA data requirements
    const size_t gga_dim_scal = is_rks ? 1 : 3;
    if( is_gga ){
      host_data.basis_eval .resize( 4 * npts * nbe );
    }
    if( is_mgga ){
      host_data.basis_eval .resize( 4 * npts * nbe ); // basis + grad (3)
    }

    // Alias/Partition out scratch memory
    auto* basis_eval = host_data.basis_eval.data();
    auto* nbe_scr    = host_data.nbe_scr.data();
    auto* zmat       = host_data.zmat.data();
    auto* lapl       = host_data.lapl.data();

    decltype(zmat) zmat_z = nullptr;
    decltype(zmat) zmat_x = nullptr;
    decltype(zmat) zmat_y = nullptr;
    if(!is_rks) {
      zmat_z = zmat + mgga_dim_scal * nbe * npts;
    }

    value_type* dbasis_x_eval = nullptr;
    value_type* dbasis_y_eval = nullptr;
    value_type* dbasis_z_eval = nullptr;
    value_type* lbasis_eval = nullptr;
    value_type* mmat_x      = nullptr;
    value_type* mmat_y      = nullptr;
    value_type* mmat_z      = nullptr;
    value_type* mmat_x_z    = nullptr;
    value_type* mmat_y_z    = nullptr;
    value_type* mmat_z_z    = nullptr;

    if( is_gga ) {
      dbasis_x_eval = basis_eval    + npts * nbe;
      dbasis_y_eval = dbasis_x_eval + npts * nbe;
      dbasis_z_eval = dbasis_y_eval + npts * nbe;
    }

    if ( is_mgga ) {
      dbasis_x_eval = basis_eval    + npts * nbe;
      dbasis_y_eval = dbasis_x_eval + npts * nbe;
      dbasis_z_eval = dbasis_y_eval + npts * nbe;
      mmat_x        = zmat + npts * nbe;
      mmat_y        = mmat_x + npts * nbe;
      mmat_z        = mmat_y + npts * nbe;
      if(is_uks) {
        mmat_x_z = zmat_z + npts * nbe;
        mmat_y_z = mmat_x_z + npts * nbe;
        mmat_z_z = mmat_y_z + npts * nbe;
      }
    }

    // assume always uks
    const value_type* vdden_eval_a, *vdden_eval_b;
    const value_type* vdden_x_eval_a, *vdden_y_eval_a, *vdden_z_eval_a;
    const value_type* vdden_x_eval_b, *vdden_y_eval_b, *vdden_z_eval_b;
    const value_type* vtau;
    std::vector<value_type> vrho;

    vdden_eval_a = task.feat.vdden_eval_a.data();
    vdden_eval_b = task.feat.vdden_eval_b.data();
    if (is_gga || is_mgga) {
      vdden_x_eval_a = task.feat.vdden_x_eval_a.data();
      vdden_y_eval_a = task.feat.vdden_y_eval_a.data();
      vdden_z_eval_a = task.feat.vdden_z_eval_a.data();
      vdden_x_eval_b = task.feat.vdden_x_eval_b.data();
      vdden_y_eval_b = task.feat.vdden_y_eval_b.data();
      vdden_z_eval_b = task.feat.vdden_z_eval_b.data();
    } else { // lda
      vrho.resize(npts * spin_dim_scal);
      interleave_data(task.feat.vdden_eval_a.data(), task.feat.vdden_eval_b.data(), npts, vrho.data());
    }
    if (is_mgga) {
      vtau = task.feat.vtau.data();
    }
    // Get the submatrix map for batch
    std::vector< std::array<int32_t, 3> > submat_map;
    std::tie(submat_map, std::ignore) =
          gen_compressed_submat_map(basis_map, task.bfn_screening.shell_list, nbf, nbf);

    // Evaluate Collocation (+ Grad and Hessian)
    if( is_mgga ) {
      lwd->eval_collocation_gradient( npts, nshells, nbe, points, basis, shell_list,
        basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval );
    } else if( is_gga )
      lwd->eval_collocation_gradient( npts, nshells, nbe, points, basis, shell_list,
        basis_eval, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval );
    else
      lwd->eval_collocation( npts, nshells, nbe, points, basis, shell_list,
        basis_eval );

    // Evaluate Z matrix for VXC
    if (is_gga){
        eval_zmat_gga_vxc_uks( npts, nbe, vdden_eval_a, vdden_eval_b, vdden_x_eval_a, vdden_x_eval_b, vdden_y_eval_a, 
                                vdden_y_eval_b, vdden_z_eval_a, vdden_z_eval_b, basis_eval,
                                dbasis_x_eval, dbasis_y_eval, dbasis_z_eval, zmat, nbe, zmat_z, nbe);
    } else if (is_mgga) {
        eval_zmat_mgga_vxc_uks( npts, nbe, vdden_eval_a, vdden_eval_b, 
                              nullptr, nullptr, /* vlapl_a, vlapl_b */
                              vdden_x_eval_a, vdden_x_eval_b, vdden_y_eval_a, 
                              vdden_y_eval_b, vdden_z_eval_a, vdden_z_eval_b, basis_eval,
                              dbasis_x_eval, dbasis_y_eval, dbasis_z_eval, 
                              lbasis_eval, zmat, nbe, zmat_z, nbe);
        lwd->eval_mmat_mgga_vxc_uks( npts, nbe, vtau, nullptr /*vlapl*/, dbasis_x_eval, dbasis_y_eval, dbasis_z_eval,
                                mmat_x, mmat_y, mmat_z, nbe, mmat_x_z, mmat_y_z, mmat_z_z, nbe);
    } else {
      lwd->eval_zmat_lda_vxc_uks( npts, nbe, vrho.data(), basis_eval, zmat, nbe, zmat_z, nbe );
    }

    // Increment VXC
    lwd->inc_vxc( mgga_dim_scal * npts, nbf, nbe, basis_eval, submat_map, zmat, nbe, VXCs, ldvxcs, nbe_scr );
    if(is_uks) {
      lwd->inc_vxc( mgga_dim_scal * npts, nbf, nbe, basis_eval, submat_map, zmat_z, nbe,VXCz, ldvxcz, nbe_scr);
    }
  } // loop over tasks
  } // end OpenMP region

  for( int32_t j = 0;   j < nbf; ++j ) {
    for( int32_t i = j+1; i < nbf; ++i ) {
      VXCs[ j + i*ldvxcs ] = VXCs[ i + j*ldvxcs ];
    }
  }
  if(not is_rks) {
    for( int32_t j = 0;   j < nbf; ++j ) {
      for( int32_t i = j+1; i < nbf; ++i ) {
        VXCz[ j + i*ldvxcz ] = VXCz[ i + j*ldvxcz ];
      }
    }
  }
}

void eval_zmat_gga_vxc_uks(size_t npts, size_t nbf, 
  const double* vdden_eval_a, const double* vdden_eval_b, const double* vdden_x_eval_a, const double* vdden_x_eval_b,
  const double* vdden_y_eval_a, const double* vdden_y_eval_b, const double* vdden_z_eval_a, const double* vdden_z_eval_b,
  const double* basis_eval, const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval,
  double* Zs, size_t ldzs, double* Zz, size_t ldzz) {
  if( ldzs != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
  if( ldzz != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
  blas::lacpy( 'A', nbf, npts, basis_eval, nbf, Zs, ldzs);
  blas::lacpy( 'A', nbf, npts, basis_eval, nbf, Zz, ldzz);

  for( int32_t i = 0; i < (int32_t)npts; ++i ) {

    const int32_t ioff = i * nbf;

    auto* zs_col = Zs + ioff;
    auto* zz_col = Zz + ioff;
    auto* bf_x_col = dbasis_x_eval + ioff;
    auto* bf_y_col = dbasis_y_eval + ioff;
    auto* bf_z_col = dbasis_z_eval + ioff;

    const double factp = 0.5 * vdden_eval_a[i];
    const double factm = 0.5 * vdden_eval_b[i];

    GauXC::blas::scal( nbf, 0.5*(factp + factm), zs_col, 1 );
    GauXC::blas::scal( nbf, 0.5*(factp - factm), zz_col, 1 );

    const double x_factp = 0.5*(vdden_x_eval_a[i] + vdden_x_eval_b[i]);
    const double y_factp = 0.5*(vdden_y_eval_a[i] + vdden_y_eval_b[i]);
    const double z_factp = 0.5*(vdden_z_eval_a[i] + vdden_z_eval_b[i]);
    const double x_factm = 0.5*(vdden_x_eval_a[i] - vdden_x_eval_b[i]);
    const double y_factm = 0.5*(vdden_y_eval_a[i] - vdden_y_eval_b[i]);
    const double z_factm = 0.5*(vdden_z_eval_a[i] - vdden_z_eval_b[i]);
    GauXC::blas::axpy( nbf, x_factp, bf_x_col, 1, zs_col, 1 );
    GauXC::blas::axpy( nbf, y_factp, bf_y_col, 1, zs_col, 1 );
    GauXC::blas::axpy( nbf, z_factp, bf_z_col, 1, zs_col, 1 );

    GauXC::blas::axpy( nbf, x_factm, bf_x_col, 1, zz_col, 1 );
    GauXC::blas::axpy( nbf, y_factm, bf_y_col, 1, zz_col, 1 );
    GauXC::blas::axpy( nbf, z_factm, bf_z_col, 1, zz_col, 1 );
  }
}

void eval_zmat_mgga_vxc_uks(size_t npts, size_t nbf, 
  const double* vdden_eval_a, const double* vdden_eval_b, 
  const double* vlapl_a, const double* vlapl_b,
  const double* vdden_x_eval_a, const double* vdden_x_eval_b, const double* vdden_y_eval_a, 
  const double* vdden_y_eval_b, const double* vdden_z_eval_a, const double* vdden_z_eval_b,
  const double* basis_eval, const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval,
  const double* lbasis_eval, double* Zs, size_t ldzs, double* Zz, size_t ldzz){

  if( ldzs != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
  if( ldzz != nbf ) GAUXC_GENERIC_EXCEPTION(std::string("Invalid Dims"));
  blas::lacpy( 'A', nbf, npts, basis_eval, nbf, Zs, ldzs);
  blas::lacpy( 'A', nbf, npts, basis_eval, nbf, Zz, ldzz);

  for( int32_t i = 0; i < (int32_t)npts; ++i ) {

    const int32_t ioff = i * nbf;

    auto* zs_col = Zs + ioff;
    auto* zz_col = Zz + ioff;
    auto* bf_x_col = dbasis_x_eval + ioff;
    auto* bf_y_col = dbasis_y_eval + ioff;
    auto* bf_z_col = dbasis_z_eval + ioff;
    auto* lbf_col = lbasis_eval + ioff;

    const double factp = 0.5 * vdden_eval_a[i];
    const double factm = 0.5 * vdden_eval_b[i];

    GauXC::blas::scal( nbf, 0.5*(factp + factm), zs_col, 1 ); 
    GauXC::blas::scal( nbf, 0.5*(factp - factm), zz_col, 1 );

    const double x_factp = 0.5*(vdden_x_eval_a[i] + vdden_x_eval_b[i]);
    const double y_factp = 0.5*(vdden_y_eval_a[i] + vdden_y_eval_b[i]);
    const double z_factp = 0.5*(vdden_z_eval_a[i] + vdden_z_eval_b[i]);
    const double x_factm = 0.5*(vdden_x_eval_a[i] - vdden_x_eval_b[i]);
    const double y_factm = 0.5*(vdden_y_eval_a[i] - vdden_y_eval_b[i]);
    const double z_factm = 0.5*(vdden_z_eval_a[i] - vdden_z_eval_b[i]);
    
    GauXC::blas::axpy( nbf, x_factp, bf_x_col, 1, zs_col, 1 );
    GauXC::blas::axpy( nbf, y_factp, bf_y_col, 1, zs_col, 1 );
    GauXC::blas::axpy( nbf, z_factp, bf_z_col, 1, zs_col, 1 );

    GauXC::blas::axpy( nbf, x_factm, bf_x_col, 1, zz_col, 1 );
    GauXC::blas::axpy( nbf, y_factm, bf_y_col, 1, zz_col, 1 );
    GauXC::blas::axpy( nbf, z_factm, bf_z_col, 1, zz_col, 1 );

    if (vlapl_a != nullptr) {
      const auto lfactp = vlapl_a[i];
      const auto lfactm = vlapl_b[i];
      blas::axpy( nbf, 0.5*(lfactp + lfactm), lbf_col, 1, zs_col, 1);
      blas::axpy( nbf, 0.5*(lfactp - lfactm), lbf_col, 1, zz_col, 1);
    }
  }
}

void sz_to_ab(std::vector<double>& dden){
  for (size_t i = 0; i < dden.size()/2; i++) {
    double s = dden[2*i];
    double z = dden[2*i+1];
    dden[2*i] = 0.5 * (s + z);
    dden[2*i+1] = 0.5 * (s - z);
  }
}

void interleave_data(const double* a, const double* b, const size_t n, double* result) {
  for (size_t i = 0; i < n; ++i) {
    result[2*i] = a[i];
    result[2*i+1] = b[i];
  }
}

FeatureDict prepare_onedft_features(const int ndm, std::vector<XCTask>& tasks, const Molecule& mol, 
                  const std::vector<std::string> feature_keys, const RuntimeEnvironment& rt,
                  std::vector<int>& sendcounts, std::vector<int>& displs) {
  std::vector<double> den_eval, dden_eval, tau, grid_coords, grid_weights;
  size_t total_npts = std::accumulate( tasks.begin(), tasks.end(), 0ul,
    [](const auto& a, const auto& b) { return a + b.npts; } );
  grid_coords.reserve(total_npts * 3);
  grid_weights.reserve(total_npts);
  den_eval.reserve(total_npts * ndm);
  dden_eval.resize(total_npts * 6);  // 2 values per point, 3 components
  tau.reserve(total_npts * ndm);

  int offset = 0;
  for (auto& task : tasks) {
    for (const auto& point : task.points) {
      grid_coords.push_back(point[0]);
      grid_coords.push_back(point[1]);
      grid_coords.push_back(point[2]);
    }
    std::copy(task.weights.begin(), task.weights.end(), std::back_inserter(grid_weights));
    std::copy(task.feat.den_eval.begin(), task.feat.den_eval.end(), std::back_inserter(den_eval));

    if (task.feat.dden_x_eval.size() != 0){
      sz_to_ab(task.feat.dden_x_eval);
      sz_to_ab(task.feat.dden_y_eval);
      sz_to_ab(task.feat.dden_z_eval);
      for (size_t i = 0; i < task.points.size(); i++) {
        dden_eval[6 * i + 0 + 6 * offset] = task.feat.dden_x_eval[2 * i];
        dden_eval[6 * i + 1 + 6 * offset] = task.feat.dden_y_eval[2 * i];
        dden_eval[6 * i + 2 + 6 * offset] = task.feat.dden_z_eval[2 * i];
        dden_eval[6 * i + 3 + 6 * offset] = task.feat.dden_x_eval[2 * i + 1];  
        dden_eval[6 * i + 4 + 6 * offset] = task.feat.dden_y_eval[2 * i + 1];
        dden_eval[6 * i + 5 + 6 * offset] = task.feat.dden_z_eval[2 * i + 1];
      }
    }
    offset += task.points.size();
    std::copy(task.feat.tau.begin(), task.feat.tau.end(), std::back_inserter(tau));
  }
  
  int world_rank = rt.comm_rank();  
  GAUXC_MPI_CODE(
    total_npts = mpi_gather_onedft_inputs(den_eval, dden_eval, tau, grid_coords, grid_weights, total_npts, 
      world_rank, rt.comm_size(), sendcounts, displs);
  );
  FeatureDict featmap;
  if (world_rank == 0) {
    size_t natoms = mol.size();
    std::vector<double> coarse_0_atomic_coords (natoms*3); 
    for (int i = 0; i < natoms; i++) {
      coarse_0_atomic_coords[3*i] = mol[i].x;
      coarse_0_atomic_coords[3*i+1] = mol[i].y;
      coarse_0_atomic_coords[3*i+2] = mol[i].z;
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    for (const auto& key : feature_keys) {
      auto enum_key = reverse_feat_map.at(key);
      at::Tensor tensor;
      switch (enum_key) {
      case ONEDFT_FEATURE::DEN: {
        auto flat_tensor = torch::from_blob(den_eval.data(), {ndm, total_npts}, {1, ndm}, options);
        tensor = flat_tensor.clone().requires_grad_(true);
        break;
      }
      case ONEDFT_FEATURE::DDEN: {
        auto flat_tensor = torch::from_blob(dden_eval.data(), {ndm, 3, total_npts}, {3, 1, 3*ndm}, options);
        tensor = flat_tensor.clone().requires_grad_(true);
        break;
      }
      case ONEDFT_FEATURE::TAU: {
        auto flat_tensor = torch::from_blob(tau.data(), {ndm, total_npts}, {1, ndm}, options);
        tensor = flat_tensor.clone().requires_grad_(true);
        break;
      }
      case ONEDFT_FEATURE::POINTS: {
        auto flat_tensor = torch::from_blob(grid_coords.data(), {total_npts, 3}, options);
        tensor = flat_tensor.clone();
        break;
      }
      case ONEDFT_FEATURE::WEIGHTS: {
        auto flat_tensor = torch::from_blob(grid_weights.data(), {total_npts}, options);
        tensor = flat_tensor.clone();
        break;
      }
      case ONEDFT_FEATURE::COORDS: {
        auto flat_tensor = torch::from_blob(coarse_0_atomic_coords.data(), {natoms, 3}, options);
        tensor = flat_tensor.clone();
        break;
      }
      default:
        GAUXC_GENERIC_EXCEPTION("Feature Key Not Implemented: " + key);
      }
      if (tensor.isnan().any().item<bool>()) {
        GAUXC_GENERIC_EXCEPTION("NaN detected in feature tensor: " + key);
      }
      featmap.insert(key, tensor);
    }
  }
  return featmap;
}

void send_buffer_onedft_outputs(const int ndm, const FeatureDict features_dict, std::vector<XCTask>& tasks, 
                                const RuntimeEnvironment& rt, std::vector<int> sendcounts, std::vector<int> displs) {

  std::vector<double> den_eval, dden_eval, tau;
  auto total_npts = mpi_scatter_onedft_outputs(features_dict, rt.comm_rank(), rt.comm_size(),
                                                sendcounts, displs, den_eval, dden_eval, tau);

  size_t offset = 0;
  for (auto&task : tasks) {
    int64_t npts = task.points.size();
    task.feat.vdden_eval_a.resize(npts);
    task.feat.vdden_eval_b.resize(npts);
    auto den_a_slice = den_eval.data() + offset;
    auto den_b_slice = den_eval.data() + total_npts + offset;
    std::copy(den_a_slice, den_a_slice + npts, task.feat.vdden_eval_a.begin());
    std::copy(den_b_slice, den_b_slice + npts, task.feat.vdden_eval_b.begin());
    if (task.feat.dden_x_eval.size() != 0){
      task.feat.vdden_x_eval_a.resize(npts);
      task.feat.vdden_y_eval_a.resize(npts);
      task.feat.vdden_z_eval_a.resize(npts);
      task.feat.vdden_x_eval_b.resize(npts);
      task.feat.vdden_y_eval_b.resize(npts);
      task.feat.vdden_z_eval_b.resize(npts);

      auto dden_a_x_slice = dden_eval.data() + offset;
      auto dden_a_y_slice = dden_eval.data() + total_npts + offset;
      auto dden_a_z_slice = dden_eval.data() + total_npts * 2 + offset;
      auto dden_b_x_slice = dden_eval.data() + total_npts * 3 + offset;
      auto dden_b_y_slice = dden_eval.data() + total_npts * 4 + offset;
      auto dden_b_z_slice = dden_eval.data() + total_npts * 5 + offset;

      std::copy(dden_a_x_slice, dden_a_x_slice + npts, task.feat.vdden_x_eval_a.begin());
      std::copy(dden_a_y_slice, dden_a_y_slice + npts, task.feat.vdden_y_eval_a.begin());
      std::copy(dden_a_z_slice, dden_a_z_slice + npts, task.feat.vdden_z_eval_a.begin());
      std::copy(dden_b_x_slice, dden_b_x_slice + npts, task.feat.vdden_x_eval_b.begin());
      std::copy(dden_b_y_slice, dden_b_y_slice + npts, task.feat.vdden_y_eval_b.begin());
      std::copy(dden_b_z_slice, dden_b_z_slice + npts, task.feat.vdden_z_eval_b.begin());
    }

    if (task.feat.tau.size() != 0){
      task.feat.vtau.resize(npts * 2);
      auto vtau_a = tau.data() + offset;
      auto vtau_b = tau.data() + total_npts + offset;
      interleave_data(vtau_a, vtau_b, npts, task.feat.vtau.data());
    }
    offset += npts;
  }
  if (offset != total_npts) {
    GAUXC_GENERIC_EXCEPTION("Mismatch in number of points for onedft features.");
  }
}

// RKS OneDFT driver - delegates to generic GKS impl
// template <typename ValueType>
// void ReferenceReplicatedXCHostIntegrator<ValueType>::
//   eval_exc_vxc_onedft_( int64_t m, int64_t n, 
//                  const value_type* P, int64_t ldp,
//                  value_type* VXC, int64_t ldvxc,
//                  value_type* EXC, const IntegratorSettingsXC& ks_settings) {

//   eval_exc_vxc_onedft_(m, n, P, ldp, nullptr, 0, nullptr, 0, nullptr, 0,
//     VXC, ldvxc, nullptr, 0, nullptr, 0, nullptr, 0, EXC, ks_settings);

// }


} // namespace detail
} // namespace GauXC
