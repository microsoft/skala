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

#include "xc_device_data.hpp"
#include "xc_device_shell_pair_soa.hpp"
#include "device/device_backend.hpp"
#include <cstring>
#include <gauxc/runtime_environment/fwd.hpp>

namespace GauXC {

// Collection of dimensions used in the XC integration
struct allocated_dims {
  size_t nshells      = 0; ///< Number of shells allocated for static data
  size_t nshell_pairs = 0; ///< Number of shell pairs allocated for static data
  size_t nprim_pairs  = 0; ///< Total number of prim pairs allocated 
  size_t nbf          = 0; ///< Number of bfns allocated for static data
  size_t natoms       = 0; ///< Number of atoms allocated for static data
  size_t max_l        = 0; ///< Highest angular momentum value used
  size_t ntask_ek     = 0; ///< Number of total tasks allocated for static data (EK)
  size_t total_npts   = 0; ///< Total number of grid points allocated for onedft data
};

/// Base type for XCDeviceData instances that use stack data allocation.
struct XCDeviceStackData : public XCDeviceData {

  using XCDeviceData::host_task_type;
  using XCDeviceData::host_task_container;
  using XCDeviceData::host_task_iterator;

  allocated_dims global_dims; ///< Global dimensions for allocated data structures
  integrator_term_tracker allocated_terms;
  
  void* device_ptr = nullptr; ///< Device buffer for all device allocations
  void* dynmem_ptr = nullptr; ///< Device buffer for dynamic allocations (mod static)
  size_t devmem_sz = 0;       ///< Length of device_ptr in bytes
  size_t dynmem_sz = 0;       ///< Length of dynmem_ptr in bytes 

  // Stack static data (not dynamically allocated for each task batch)

  struct static_data {
    Shell<double>* shells_device = nullptr; ///< Array of static basis shells (nshells)
    PrimitivePair<double>* prim_pairs_device = nullptr;

    double* rab_device    = nullptr; ///< Static RAB matrix storage (*,natoms)
    double* coords_device = nullptr; ///< Static atomic positions (3 * natoms)

    double* exc_device     = nullptr;  ///< EXC storage (1)
    double* nel_device     = nullptr;  ///< N_EL storage (1)
    double* exx_k_device   = nullptr;  ///< EXX K storage (nbf,nbf)
    double* acc_scr_device = nullptr;  ///< Accumulaion scratch (1)
    double* exc_grad_device = nullptr; ///< EXC Gradient storage (3*natoms)
    double* fxc_device     = nullptr; ///< FXC contraction storage (nbf,nbf)

    double* vshell_max_sparse_device = nullptr;
    size_t* shpair_row_ind_device = nullptr;
    size_t* shpair_col_ind_device = nullptr;
    double* ek_max_bfn_sum_device = nullptr;
    double* ek_bfn_max_device     = nullptr;
    int32_t* shell_to_bf_device = nullptr;
    int32_t* shell_sizes_device = nullptr;

    double* dmat_s_device   = nullptr;  ///< Static density matrix storage (nbf,nbf)
    double* dmat_z_device   = nullptr;  /// Ditto for Z,Y,X densities
    double* dmat_y_device   = nullptr;
    double* dmat_x_device   = nullptr;
    double* vxc_s_device    = nullptr;  ///< VXC storage (nbf, nbf)
    double* vxc_z_device    = nullptr;  /// Ditto for Z,Y,X densities
    double* vxc_y_device    = nullptr;
    double* vxc_x_device    = nullptr;
    
    // Second derivatives
    double* tdmat_s_device  = nullptr;  ///< Static trial density matrix storage (nbf,nbf)
    double* tdmat_z_device  = nullptr;  /// Ditto for Z,Y,X trial densities
    double* tdmat_y_device  = nullptr;
    double* tdmat_x_device  = nullptr;
    double* fxc_s_device    = nullptr;  ///< FXC storage (nbf, nbf)
    double* fxc_z_device    = nullptr;  /// Ditto for Z,Y,X densities
    double* fxc_y_device    = nullptr;
    double* fxc_x_device    = nullptr;

    // onedft input and output
    double* grid_weights_device = nullptr; ///< Grid weights
    double* grid_coords_device  = nullptr; ///< Grid coordinates
    double* den_eval_device     = nullptr; ///< Grid densities
    double* dden_eval_device = nullptr; ///< Grid density gradients
    double* tau_device    = nullptr;

    // onedft output
    double* den_grad_device     = nullptr; ///< Grid density gradients
    double* dden_grad_device    = nullptr; ///< Grid density gradients eps gradients
    double* tau_grad_device     = nullptr;

    inline void reset() { std::memset( this, 0, sizeof(static_data) ); }

    inline double* den_selector(density_id den) {
      switch(den) {
        case DEN_S: return dmat_s_device;
        case DEN_Z: return dmat_z_device;
        case DEN_Y: return dmat_y_device;
        case DEN_X: return dmat_x_device;
        default: GAUXC_GENERIC_EXCEPTION("den_selector: density_id not recognized");
      } 
      return nullptr;
    }

    inline double* vxc_selector(density_id den) {
      switch(den) {
        case DEN_S: return vxc_s_device;
        case DEN_Z: return vxc_z_device;
        case DEN_Y: return vxc_y_device;
        case DEN_X: return vxc_x_device;
        default: GAUXC_GENERIC_EXCEPTION("vxc_selector: density_id not recognized");
      } 
      return nullptr;
    }

    inline double* tden_selector(density_id den) {
      switch(den) {
        case DEN_S: return tdmat_s_device;
        case DEN_Z: return tdmat_z_device;
        case DEN_Y: return tdmat_y_device;
        case DEN_X: return tdmat_x_device;
        default: GAUXC_GENERIC_EXCEPTION("tden_selector: density_id not recognized");
      } 
      return nullptr;
    }

    inline double* fxc_selector(density_id den) {
      switch(den) {
        case DEN_S: return fxc_s_device;
        case DEN_Z: return fxc_z_device;
        case DEN_Y: return fxc_y_device;
        case DEN_X: return fxc_x_device;
        default: GAUXC_GENERIC_EXCEPTION("fxc_selector: density_id not recognized");
      } 
      return nullptr;
    }

  };

  XCDeviceShellPairSoA shell_pair_soa;
  static_data static_stack;


  // Stack dynamic data

  size_t total_npts_task_batch = 0; ///< Number of grid points in task batch
  struct base_stack_data {

    //double* points_device  = nullptr; ///< Grid points for task batch
    double* points_x_device = nullptr;
    double* points_y_device = nullptr;
    double* points_z_device = nullptr;
    double* weights_device = nullptr; ///< Grid weights for task batch

    // U variables
    double* den_s_eval_device      = nullptr; ///< scalar density for task batch
    double* dden_sx_eval_device    = nullptr; ///< d/dx scalar density for task batch
    double* dden_sy_eval_device    = nullptr; ///< d/dy scalar density for task batch
    double* dden_sz_eval_device    = nullptr; ///< d/dz scalar density for task batch
    double* tau_s_eval_device      = nullptr; ///< scalar tau for task batch
    double* lapl_s_eval_device     = nullptr; ///< scalar density laplacian for task batch
    
    double* den_z_eval_device      = nullptr; ///< z density for task batch
    double* dden_zx_eval_device    = nullptr; ///< d/dx z density for task batch
    double* dden_zy_eval_device    = nullptr; ///< d/dy z density for task batch
    double* dden_zz_eval_device    = nullptr; ///< d/dz z density for task batch
    double* tau_z_eval_device      = nullptr; ///< z tau for task batch
    double* lapl_z_eval_device     = nullptr; ///< z density laplacian for task batch

    double* den_y_eval_device   = nullptr; ///< y density for task batch
    double* dden_yx_eval_device = nullptr; ///< d/dx y density for task batch
    double* dden_yy_eval_device = nullptr; ///< d/dy y density for task batch
    double* dden_yz_eval_device = nullptr; ///< d/dz y density for task batch

    double* den_x_eval_device   = nullptr; ///< x density for task batch
    double* dden_xx_eval_device = nullptr; ///< d/dx x density for task batch
    double* dden_xy_eval_device = nullptr; ///< d/dy x density for task batch
    double* dden_xz_eval_device = nullptr; ///< d/dz x density for task batch
    
    double* den_interleaved_device  = nullptr; /// Storage for interleaved density (non-RKS only)
    double* tau_interleaved_device  = nullptr; /// Storage for interleaved tau (non-RKS only)
    double* lapl_interleaved_device = nullptr; /// Storage for interleaved lapl (non-RKS only)

    // V variables / XC output
    double* gamma_eval_device  = nullptr; ///< gamma for task batch
    double* eps_eval_device    = nullptr; ///< XC energy density for task batch
    double* vrho_eval_device   = nullptr; ///< Rho XC derivative for task batch
    double* vgamma_eval_device = nullptr; ///< Gamma XC derivative for task batch
    double* vtau_eval_device   = nullptr; ///< Tau XC derivative for task batch
    double* vlapl_eval_device  = nullptr; ///< Lapl XC derivative for task batch

    double* vrho_pos_eval_device   = nullptr;  ///< Polarized Rho+ XC derivative for task batch
    double* vrho_neg_eval_device   = nullptr;  ///< Polarized Rho+ XC derivative for task batch
    double* vtau_pos_eval_device   = nullptr;
    double* vtau_neg_eval_device   = nullptr;
    double* vlapl_pos_eval_device  = nullptr;
    double* vlapl_neg_eval_device  = nullptr;
    

    double* gamma_pp_eval_device  = nullptr;  ///< Polarized Gamma++ for task batch
    double* gamma_pm_eval_device  = nullptr;  ///< Polarized Gamma+- for task batch
    double* gamma_mm_eval_device  = nullptr;  ///< Polarized Gamma-- for task batch
    double* vgamma_pp_eval_device  = nullptr; ///< Polarized Gamma++ XC derivative for task batch
    double* vgamma_pm_eval_device  = nullptr; ///< Polarized Gamma+- XC derivative for task batch
    double* vgamma_mm_eval_device  = nullptr; ///< Polarized Gamma-- XC derivative for task batch

    double* H_x_eval_device     = nullptr;    ///< norm(m) dependent GGA X transformation factor for task batch
    double* H_y_eval_device     = nullptr;    ///< norm(m) dependent GGA Y transformation factor for task batch
    double* H_z_eval_device     = nullptr;    ///< norm(m) dependent GGA Z transformation factor for task batch
    double* K_x_eval_device     = nullptr;    ///< norm(m) dependent LDA X transformation factor for task batch
    double* K_y_eval_device     = nullptr;    ///< norm(m) dependent LDA Y transformation factor for task batch
    double* K_z_eval_device     = nullptr;    ///< norm(m) dependent LDA Z transformation factor for task batch

    // Second derivative intermediates - Trial variables (T)
    double* tden_s_eval_device      = nullptr; ///< scalar trial density for task batch
    double* tdden_sx_eval_device    = nullptr; ///< d/dx scalar trial density for task batch
    double* tdden_sy_eval_device    = nullptr; ///< d/dy scalar trial density for task batch
    double* tdden_sz_eval_device    = nullptr; ///< d/dz scalar trial density for task batch
    double* ttau_s_eval_device      = nullptr; ///< scalar trial tau for task batch
    double* tlapl_s_eval_device     = nullptr; ///< scalar trial density laplacian for task batch
    
    double* tden_z_eval_device      = nullptr; ///< z trial density for task batch
    double* tdden_zx_eval_device    = nullptr; ///< d/dx z trial density for task batch
    double* tdden_zy_eval_device    = nullptr; ///< d/dy z trial density for task batch
    double* tdden_zz_eval_device    = nullptr; ///< d/dz z trial density for task batch
    double* ttau_z_eval_device      = nullptr; ///< z trial tau for task batch
    double* tlapl_z_eval_device     = nullptr; ///< z trial density laplacian for task batch

    double* tden_y_eval_device      = nullptr; ///< y trial density for task batch
    double* tdden_yx_eval_device    = nullptr; ///< d/dx y trial density for task batch
    double* tdden_yy_eval_device    = nullptr; ///< d/dy y trial density for task batch
    double* tdden_yz_eval_device    = nullptr; ///< d/dz y trial density for task batch

    double* tden_x_eval_device      = nullptr; ///< x trial density for task batch
    double* tdden_xx_eval_device    = nullptr; ///< d/dx x trial density for task batch
    double* tdden_xy_eval_device    = nullptr; ///< d/dy x trial density for task batch
    double* tdden_xz_eval_device    = nullptr; ///< d/dz x trial density for task batch

    // Second derivative kernel outputs (V2 variables)
    double* v2rho2_eval_device      = nullptr; ///< 2nd derivative of XC wrt rho^2
    double* v2rhogamma_eval_device  = nullptr; ///< 2nd derivative of XC wrt rho-gamma
    double* v2rholapl_eval_device   = nullptr; ///< 2nd derivative of XC wrt rho-lapl
    double* v2rhotau_eval_device    = nullptr; ///< 2nd derivative of XC wrt rho-tau
    double* v2gamma2_eval_device    = nullptr; ///< 2nd derivative of XC wrt gamma^2
    double* v2gammalapl_eval_device = nullptr; ///< 2nd derivative of XC wrt gamma-lapl
    double* v2gammatau_eval_device  = nullptr; ///< 2nd derivative of XC wrt gamma-tau
    double* v2lapl2_eval_device     = nullptr; ///< 2nd derivative of XC wrt lapl^2
    double* v2lapltau_eval_device   = nullptr; ///< 2nd derivative of XC wrt lapl-tau
    double* v2tau2_eval_device      = nullptr; ///< 2nd derivative of XC wrt tau^2
    // in unrestricted case, these are 2nd derivatives of XC with alpha (+) and beta (-) densities
    double* v2rho2_a_a_eval_device = nullptr;
    double* v2rho2_a_b_eval_device = nullptr;
    double* v2rho2_b_b_eval_device = nullptr;
    double* v2rhogamma_a_aa_eval_device = nullptr;
    double* v2rhogamma_a_ab_eval_device = nullptr;
    double* v2rhogamma_a_bb_eval_device = nullptr;
    double* v2rhogamma_b_aa_eval_device = nullptr;
    double* v2rhogamma_b_ab_eval_device = nullptr;
    double* v2rhogamma_b_bb_eval_device = nullptr;
    double* v2rholapl_a_a_eval_device = nullptr;
    double* v2rholapl_a_b_eval_device = nullptr;
    double* v2rholapl_b_a_eval_device = nullptr;
    double* v2rholapl_b_b_eval_device = nullptr;
    double* v2rhotau_a_a_eval_device = nullptr;
    double* v2rhotau_a_b_eval_device = nullptr;
    double* v2rhotau_b_a_eval_device = nullptr;
    double* v2rhotau_b_b_eval_device = nullptr;
    double* v2gamma2_aa_aa_eval_device = nullptr;
    double* v2gamma2_aa_ab_eval_device = nullptr;
    double* v2gamma2_aa_bb_eval_device = nullptr;
    double* v2gamma2_ab_ab_eval_device = nullptr;
    double* v2gamma2_ab_bb_eval_device = nullptr;
    double* v2gamma2_bb_bb_eval_device = nullptr;
    double* v2gammalapl_aa_a_eval_device = nullptr;
    double* v2gammalapl_aa_b_eval_device = nullptr;
    double* v2gammalapl_ab_a_eval_device = nullptr;
    double* v2gammalapl_ab_b_eval_device = nullptr;
    double* v2gammalapl_bb_a_eval_device = nullptr;
    double* v2gammalapl_bb_b_eval_device = nullptr;
    double* v2gammatau_aa_a_eval_device = nullptr;
    double* v2gammatau_aa_b_eval_device = nullptr;
    double* v2gammatau_ab_a_eval_device = nullptr;
    double* v2gammatau_ab_b_eval_device = nullptr;
    double* v2gammatau_bb_a_eval_device = nullptr;
    double* v2gammatau_bb_b_eval_device = nullptr;
    double* v2lapl2_a_a_eval_device = nullptr;
    double* v2lapl2_a_b_eval_device = nullptr;
    double* v2lapl2_b_b_eval_device = nullptr;
    double* v2lapltau_a_a_eval_device = nullptr;
    double* v2lapltau_a_b_eval_device = nullptr;
    double* v2lapltau_b_a_eval_device = nullptr;
    double* v2lapltau_b_b_eval_device = nullptr;
    double* v2tau2_a_a_eval_device = nullptr;
    double* v2tau2_a_b_eval_device = nullptr;
    double* v2tau2_b_b_eval_device = nullptr;
    
    // Second derivative kernel outputs (A,B,C variables)
    double* FXC_A_s_eval_device           = nullptr;
    double* FXC_Bx_s_eval_device          = nullptr;
    double* FXC_By_s_eval_device          = nullptr;
    double* FXC_Bz_s_eval_device          = nullptr;
    double* FXC_C_s_eval_device           = nullptr;
    double* FXC_A_z_eval_device           = nullptr;
    double* FXC_Bx_z_eval_device          = nullptr;
    double* FXC_By_z_eval_device          = nullptr;
    double* FXC_Bz_z_eval_device          = nullptr;
    double* FXC_C_z_eval_device           = nullptr;

    inline void reset() { std::memset( this, 0, sizeof(base_stack_data) ); }
  };

  base_stack_data base_stack;

  /// Device backend instance to handle device specific execution
  const DeviceRuntimeEnvironment& runtime_;
  DeviceBackend* device_backend_ = nullptr;

  XCDeviceStackData() = delete; // No default ctor, must have device backend
  XCDeviceStackData( const DeviceRuntimeEnvironment& rt );

  virtual ~XCDeviceStackData() noexcept;

  // Final overrides
  host_task_iterator generate_buffers( integrator_term_tracker, const BasisSetMap&,
    host_task_iterator, host_task_iterator) override final;
  void allocate_static_data_weights( int32_t natoms ) override final;
  void allocate_static_data_onedft( int32_t nbf, int32_t nshells, int32_t natoms, int32_t total_npts, integrator_term_tracker enabled_terms ) override final;
  void allocate_static_data_exc_vxc( int32_t nbf, int32_t nshells, integrator_term_tracker enabled_terms, bool do_vxc ) override final;
  void allocate_static_data_fxc_contraction( int32_t nbf, int32_t nshells, integrator_term_tracker enabled_terms ) override final;
  void allocate_static_data_den( int32_t nbf, int32_t nshells ) override final;
  void allocate_static_data_exc_grad( int32_t nbf, int32_t nshells, int32_t natoms, integrator_term_tracker enabled_terms ) override final;
  void allocate_static_data_exx( int32_t nbf, int32_t nshells, size_t nshell_pairs, size_t nprim_pair_total, int32_t max_l ) override final;
  void allocate_static_data_exx_ek_screening( size_t ntasks, int32_t nbf, int32_t nshells, int nshell_pairs, int32_t max_l ) override final;
  void send_static_data_weights( const Molecule& mol, const MolMeta& meta ) override final;
  void send_static_data_density_basis( const double* Ps, int32_t ldps, const double* Pz, int32_t ldpz,
                                        const double* Py, int32_t ldpy, const double* Px, int32_t ldpx,
    const BasisSet<double>& basis ) override final;
  void send_static_data_onedft( const Molecule& mol, const double* Ps, int32_t ldps, const double* Pz, int32_t ldpz, const double* Py, 
      int32_t ldpy, const double* Px, int32_t ldpx, const BasisSet<double>& basis ) override final;
  void send_static_data_onedft_results( int32_t total_npts, int32_t ndm, const double* EXC, const double* DEN, const double* DDEN, const double* TAU) override final;
  void send_static_data_trial_density(
    const double* tPs, int32_t ldtps, const double* tPz, int32_t ldtpz,
    const double* tPy, int32_t ldtpy, const double* tPx, int32_t ldtpx ) override final;
  void send_static_data_shell_pairs( const BasisSet<double>&, const ShellPairCollection<double>& ) 
    override final;
  void send_static_data_exx_ek_screening( const double* V_max, int32_t ldv, const BasisSetMap&, const ShellPairCollection<double>& ) override final;
  void zero_den_integrands() override final;
  void zero_exc_vxc_integrands(integrator_term_tracker t) override final;
  void zero_fxc_contraction_integrands() override final;
  void zero_exc_grad_integrands() override final;
  void zero_exx_integrands() override final;
  void zero_exx_ek_screening_intermediates() override final;
  void retrieve_exc_vxc_integrands( double* EXC, double* N_EL,
    double* VXCscalar, int32_t ldvxcscalar, double* VXCz, int32_t ldvxcz,
    double* VXCy     , int32_t ldvxcy     , double* VXCx, int32_t ldvxcx ) override final;
  void retrieve_onedft_features( int32_t total_npts, int32_t ndm, double* DEN, 
    double* DDEN, double* TAU, double* POINTS, double* WEIGHTS) override final;
  void retrieve_fxc_contraction_integrands( double* N_EL,
    double* FXCs, int32_t ldfxcs, double* FXCz, int32_t ldfxcz,
    double* FXCy, int32_t ldfxcy, double* FXCx, int32_t ldfxcx ) override final;
  void retrieve_exc_grad_integrands( double* EXC_GRAD, double* N_EL ) override final;
  void retrieve_den_integrands( double* N_EL ) override final;
  void retrieve_exx_integrands( double* K, int32_t ldk ) override final;
  void retrieve_exx_ek_max_bfn_sum( double* MBS, int32_t nt) override final;
  void copy_weights_to_tasks( host_task_iterator task_begin, host_task_iterator task_end ) override final;

  double* vxc_s_device_data() override;
  double* vxc_z_device_data() override;
  double* vxc_y_device_data() override;
  double* vxc_x_device_data() override;
  double* exc_device_data() override;
  double* nel_device_data() override;
  double* exx_k_device_data() override;

  double* grid_weights_device_data() override;
  double* grid_coords_device_data() override;
  double* den_eval_device_data() override;
  double* dden_eval_device_data() override;
  double* tau_device_data() override;
  double* coords_device_data() override;

  double* fxc_s_device_data() override;
  double* fxc_z_device_data() override;
  double* fxc_y_device_data() override;
  double* fxc_x_device_data() override;
  device_queue queue() override;


  virtual void reset_allocations() override;

  // New overridable APIs
  using device_buffer_t = std::tuple<void*, size_t>;

  /** Allocate and populate device memory for a given task batch
   *
   *  Overridable in devrived classes - derived classes should call
   *  this function explicitly to ensure that the correct information
   *  is allocated on the stack
   *
   *  @param[in] begin      Start iterator for task batch
   *  @param[in] end        End iterator for task batch
   *  @param[in] buf        Current state of dynamic memory stack
   *  @param[in] basis_map  Basis map instance for pass basis set 
   *                        (TODO: should persist internally)
   *
   *  @returns The state of the dynamic memory stack after allocating
   *           base information.
   */


  virtual device_buffer_t allocate_dynamic_stack( integrator_term_tracker terms,
    host_task_iterator begin, host_task_iterator end, device_buffer_t buf );

  virtual void pack_and_send( integrator_term_tracker terms,
    host_task_iterator begin, host_task_iterator end, 
    const BasisSetMap& basis_map );



  /** Obtain the memory requirement for an XC task
   *
   *  Overridable in devrived classes - derived classes should call
   *  this function explicitly to ensure that the correct information
   *  is allocated on the stack
   *
   *  @param[in] task       Task to obtain the memory requirement
   *
   *  @returns Memory requirement (bytes) for `task` in device memory
   */
  virtual size_t get_mem_req( integrator_term_tracker terms,
    const host_task_type& task );


  // Implementation specific APIs
  virtual size_t get_ldatoms()   = 0; ///< Stride of RAB in device memory
  virtual size_t get_rab_align() = 0; ///< Alignment of RAB in device memory
  virtual int get_points_per_subtask() = 0; ///< Number of points per subtask for OS kernels
  virtual size_t get_static_mem_requirement() = 0;
    ///< Static memory requirment for task batch which is independent of batch size

};

}
