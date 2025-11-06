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
#include <gauxc/xc_task.hpp>
#include <gauxc/util/div_ceil.hpp>
#include <vector>
#include <gauxc/basisset_map.hpp>
#include <gauxc/shell_pair.hpp>
#include <gauxc/molmeta.hpp>
//#include <gauxc/reduction_driver.hpp>
#include <any>
#include <cstring>
#include "device/device_queue.hpp"

namespace GauXC {

enum integrator_xc_approx : uint32_t {
  _UNDEF_APPROX         = 0,
  LDA                   = 1,
  GGA                   = 2,
  MGGA_TAU              = 3,
  MGGA_LAPL             = 4
};

enum integrator_ks_scheme : uint32_t {
  _UNDEF_SCHEME             = 0,
  RKS                       = 1,
  UKS                       = 2,
  GKS                       = 3
};

enum density_id : uint32_t {
  _UNDEF_DEN      = 0,
  DEN_S           = 1,    // RKS, UKS, GKS
  DEN_Z           = 2,    // UKS, GKS
  DEN_Y           = 3,    // GKS
  DEN_X           = 4     // GKS
};

struct integrator_term_tracker {
  bool weights                   = false;
  bool den                       = false;
  bool exc_vxc                   = false;
  bool exc_grad                  = false;
  bool exx                       = false;
  bool exx_ek_screening          = false;
  bool onedft                    = false;
  bool fxc_contraction           = false;
  integrator_xc_approx xc_approx = _UNDEF_APPROX;
  integrator_ks_scheme ks_scheme = _UNDEF_SCHEME;
  inline void reset() {
    std::memset( this, 0, sizeof(integrator_term_tracker) );
  }
};

#define PRDVL(pred,val) (pred) ? (val) : 0ul

struct required_term_storage {
  bool grid_points  = false;
  bool grid_weights = false;

  inline size_t grid_points_size(size_t npts) { 
    return PRDVL(grid_points, 3 * npts); 
  }
  inline size_t grid_weights_size(size_t npts) { 
    return PRDVL(grid_weights, npts); 
  }

  // Evaluation of functions on the grid (linear storage)
  bool grid_den      = false;
  bool grid_den_grad = false;
  bool grid_lapl     = false;
  bool grid_gamma    = false;
  bool grid_tau      = false;
  bool grid_eps      = false;
  bool grid_vrho     = false;
  bool grid_vgamma   = false;
  bool grid_vtau     = false;
  bool grid_vlapl    = false;
  
  // Second derivative variables
  bool grid_tden      = false;
  bool grid_tden_grad = false;
  bool grid_ttau      = false;
  bool grid_tlapl     = false;
  bool grid_v2rho2      = false;
  bool grid_v2rhogamma  = false;
  bool grid_v2rholapl   = false;
  bool grid_v2rhotau    = false;
  bool grid_v2gamma2    = false;
  bool grid_v2gammalapl = false;
  bool grid_v2gammatau  = false;
  bool grid_v2lapl2     = false;
  bool grid_v2lapltau   = false;
  bool grid_v2tau2      = false;
  bool grid_FXC_A           = false;
  bool grid_FXC_B           = false;
  bool grid_FXC_C           = false;


  // Reference flags for memory management use
  integrator_term_tracker ref_tracker;
  
  inline size_t grid_den_size(size_t npts){ 
    // For RKS, only den_s_eval is used
    if( grid_den ) {
      if( ref_tracker.ks_scheme == RKS ) return npts; 
      if( ref_tracker.den )              return npts; 
      // 2*npts for S,Z densities, 2*npts for interleaved density
      if( ref_tracker.ks_scheme == UKS ) return 4*npts;
      // Same as above, but also X,Y densities
      if( ref_tracker.ks_scheme == GKS ) return 6*npts;  
    }
    return 0ul;
  }
  inline size_t grid_den_grad_size(size_t npts){ 
    if( grid_den_grad ) {
      // 3*npts for each density in play
      if( ref_tracker.ks_scheme == RKS ) return 3*npts;
      if( ref_tracker.ks_scheme == UKS ) return 6*npts;
      if( ref_tracker.ks_scheme == GKS ) return 12*npts;
    }
    return 0ul;
  }
  inline size_t grid_gamma_size(size_t npts){
    if( grid_gamma ) {
      if(  ref_tracker.ks_scheme == RKS ) return npts;
      if(  ref_tracker.ks_scheme == UKS 
        or ref_tracker.ks_scheme == GKS ) return 6*npts;
    }
    return 0ul;
  }
  inline size_t grid_lapl_size(size_t npts){ 
    if(grid_lapl) {
      switch(ref_tracker.ks_scheme) {
        case UKS:
        case GKS:
          return 4 * npts;
        default:
          return npts;
      }
    } 
    return 0ul;
  }
  inline size_t grid_tau_size(size_t npts){ 
    if(grid_tau) {
      switch(ref_tracker.ks_scheme) {
        case UKS:
        case GKS:
          return 4 * npts;
        default:
          return npts;
      }
    } 
    return 0ul;
  }
  inline size_t grid_eps_size(size_t npts){ 
    return PRDVL(grid_eps, npts);
  }
  inline size_t grid_vrho_size(size_t npts){ 
    if( grid_vrho ) {
      if(   ref_tracker.ks_scheme == RKS ) return npts;
      if(   ref_tracker.ks_scheme == UKS 
        or  ref_tracker.ks_scheme == GKS ) return 4*npts;
    }
    return 0ul;
  }
  inline size_t grid_vgamma_size(size_t npts){ 
    if( grid_vgamma ) {
      if(   ref_tracker.ks_scheme == RKS ) return npts;
      if(   ref_tracker.ks_scheme == UKS 
        or  ref_tracker.ks_scheme == GKS ) return 6*npts;
    }
    return 0ul;
  }
  inline size_t grid_HK_size(size_t npts){
    if( ref_tracker.ks_scheme == GKS ) {
      if( ref_tracker.xc_approx == GGA ) return 6*npts;
      if( ref_tracker.xc_approx == LDA ) return 3*npts;
    }
    return 0ul;
  }
  inline size_t grid_vtau_size(size_t npts){ 
    if(grid_vtau) {
      switch(ref_tracker.ks_scheme) {
        case UKS:
        case GKS:
          return 4 * npts;
        default:
          return npts;
      }
    } 
    return 0ul;
  }
  inline size_t grid_vlapl_size(size_t npts){ 
    if(grid_vlapl) {
      switch(ref_tracker.ks_scheme) {
        case UKS:
        case GKS:
          return 4 * npts;
        default:
          return npts;
      }
    } 
    return 0ul;
  }
  
  // Size calculators for second derivative variables
  inline size_t grid_tden_size(size_t npts){ 
    if( grid_tden ) {
      if( ref_tracker.ks_scheme == RKS ) return npts; 
      // 2*npts for S,Z densities, 2*npts for interleaved density
      if( ref_tracker.ks_scheme == UKS ) return 2*npts;
      // Same as above, but also X,Y densities
      if( ref_tracker.ks_scheme == GKS ) return 4*npts;  
    }
    return 0ul;
  }
  
  inline size_t grid_tden_grad_size(size_t npts){ 
    if( grid_tden_grad ) {
      // 3*npts for each density in play
      if( ref_tracker.ks_scheme == RKS ) return 3*npts;
      if( ref_tracker.ks_scheme == UKS ) return 6*npts;
      if( ref_tracker.ks_scheme == GKS ) return 12*npts;
    }
    return 0ul;
  }
  
  inline size_t grid_tlapl_size(size_t npts){ 
    if(grid_tlapl) {
      switch(ref_tracker.ks_scheme) {
        case UKS:
        case GKS:
          return 2 * npts;
        default:
          return npts;
      }
    } 
    return 0ul;
  }
  
  inline size_t grid_ttau_size(size_t npts){ 
    if(grid_ttau) {
      switch(ref_tracker.ks_scheme) {
        case UKS:
        case GKS:
          return 2 * npts;
        default:
          return npts;
      }
    } 
    return 0ul;
  }
  
  inline size_t grid_v2rho2_size(size_t npts){
    if(grid_v2rho2) {
      if( ref_tracker.ks_scheme == RKS ) return npts;
      if( ref_tracker.ks_scheme == UKS or ref_tracker.ks_scheme == GKS ) return 6*npts;
    }
    return 0ul;
  }
  
  inline size_t grid_v2rhogamma_size(size_t npts){
    if(grid_v2rhogamma) {
      if( ref_tracker.ks_scheme == RKS ) return npts;
      if( ref_tracker.ks_scheme == UKS or ref_tracker.ks_scheme == GKS ) return 12*npts;
    }
    return 0ul;
  }
  
  inline size_t grid_v2rholapl_size(size_t npts){
    if(grid_v2rholapl) {
      if( ref_tracker.ks_scheme == RKS ) return npts;
      if( ref_tracker.ks_scheme == UKS or ref_tracker.ks_scheme == GKS ) return 8*npts;
    }
    return 0ul;
  }
  
  inline size_t grid_v2rhotau_size(size_t npts){
    if(grid_v2rhotau) {
      if( ref_tracker.ks_scheme == RKS ) return npts;
      if( ref_tracker.ks_scheme == UKS or ref_tracker.ks_scheme == GKS ) return 8*npts;
    }
    return 0ul;
  }
  
  inline size_t grid_v2gamma2_size(size_t npts){
    if(grid_v2gamma2) {
      if( ref_tracker.ks_scheme == RKS ) return npts;
      if( ref_tracker.ks_scheme == UKS or ref_tracker.ks_scheme == GKS ) return 12*npts;
    }
    return 0ul;
  }
  
  inline size_t grid_v2gammalapl_size(size_t npts){
    if(grid_v2gammalapl) {
      if( ref_tracker.ks_scheme == RKS ) return npts;
      if( ref_tracker.ks_scheme == UKS or ref_tracker.ks_scheme == GKS ) return 12*npts;
    }
    return 0ul;
  }
  
  inline size_t grid_v2gammatau_size(size_t npts){
    if(grid_v2gammatau) {
      if( ref_tracker.ks_scheme == RKS ) return npts;
      if( ref_tracker.ks_scheme == UKS or ref_tracker.ks_scheme == GKS ) return 12*npts;
    }
    return 0ul;
  }
  
  inline size_t grid_v2lapl2_size(size_t npts){
    if(grid_v2lapl2) {
      if( ref_tracker.ks_scheme == RKS ) return npts;
      if( ref_tracker.ks_scheme == UKS or ref_tracker.ks_scheme == GKS ) return 6*npts;
    }
    return 0ul;
  }
  
  inline size_t grid_v2lapltau_size(size_t npts){
    if(grid_v2lapltau) {
      if( ref_tracker.ks_scheme == RKS ) return npts;
      if( ref_tracker.ks_scheme == UKS or ref_tracker.ks_scheme == GKS ) return 8*npts;
    }
    return 0ul;
  }
  
  inline size_t grid_v2tau2_size(size_t npts){
    if(grid_v2tau2) {
      if( ref_tracker.ks_scheme == RKS ) return npts;
      if( ref_tracker.ks_scheme == UKS or ref_tracker.ks_scheme == GKS ) return 6*npts;
    }
    return 0ul;
  }

  inline size_t grid_FXC_A_size(size_t npts){
    if( grid_FXC_A ) {
      if( ref_tracker.ks_scheme == RKS ) return npts;
      if( ref_tracker.ks_scheme == UKS or ref_tracker.ks_scheme == GKS ) return 2*npts;
    }
  }
  inline size_t grid_FXC_B_size(size_t npts){
    if( grid_FXC_B ) {
      if( ref_tracker.ks_scheme == RKS ) return 3*npts;
      if( ref_tracker.ks_scheme == UKS or ref_tracker.ks_scheme == GKS ) return 6*npts;
    }
  }
  inline size_t grid_FXC_C_size(size_t npts){
    if( grid_FXC_C ) {
      if( ref_tracker.ks_scheme == RKS ) return npts;
      if( ref_tracker.ks_scheme == UKS or ref_tracker.ks_scheme == GKS ) return 2*npts;
    }
  }



  // Task-local matrices
  bool task_bfn           = false;
  bool task_bfn_grad      = false;
  bool task_bfn_hess      = false;
  bool task_bfn_lapl      = false;
  bool task_bfn_lapgrad   = false;
  bool task_zmat          = false;
  bool task_xmat          = false;
  bool task_xmat_grad     = false;
  bool task_xmat_persist  = false;
  bool task_fmat          = false;
  bool task_gmat          = false;
  bool task_nbe_scr       = false;
  bool task_bfn_shell_indirection = false;


  inline size_t task_bfn_size(size_t nbe, size_t npts) {
    return PRDVL(task_bfn, nbe * npts);
  }
  inline size_t task_bfn_grad_size(size_t nbe, size_t npts) {
    return PRDVL(task_bfn_grad, 3 * nbe * npts);
  }
  inline size_t task_bfn_hess_size(size_t nbe, size_t npts) {
    return PRDVL(task_bfn_hess, 6 * nbe * npts);
  }
  inline size_t task_bfn_lapl_size(size_t nbe, size_t npts) {
    return PRDVL(task_bfn_lapl, nbe * npts);
  }
  inline size_t task_bfn_lapgrad_size(size_t nbe, size_t npts) {
    return PRDVL(task_bfn_lapgrad, 3 * nbe * npts);
  }
  inline size_t task_zmat_size(size_t nbe, size_t npts) {
    return PRDVL(task_zmat, nbe * npts);
  }
  inline size_t task_xmat_grad_size(size_t nbe, size_t npts) {
    return PRDVL(task_xmat_grad, 3 * nbe * npts);
  }
  inline size_t task_xmat_persist_size(size_t nbe, size_t npts) {
    // TODO Make this more robust
    return PRDVL(task_xmat_persist, 2 * (task_xmat_grad ? 4 : 1) * nbe * npts);
  }
  inline size_t task_fmat_size(size_t nbe, size_t npts) {
    return PRDVL(task_fmat, nbe * npts);
  }
  inline size_t task_gmat_size(size_t nbe, size_t npts) {
    return PRDVL(task_gmat, nbe * npts);
  }
  inline size_t task_nbe_scr_size(size_t nbe_bfn, size_t nbe_cou) {
    return PRDVL(task_nbe_scr, std::max(nbe_bfn,nbe_cou) * nbe_bfn);
  }
  inline size_t task_bfn_shell_indirection_size(size_t nbe) {
    return PRDVL(task_bfn_shell_indirection, nbe);
  }

  // Index packing
  bool task_submat_cut_bfn   = false;
  bool task_submat_block_bfn = false;
  bool task_submat_cut_cou   = false;
  bool task_submat_block_cou = false;

  inline size_t task_submat_cut_bfn_size(size_t ncut) {
    return PRDVL(task_submat_cut_bfn, 3*ncut);
  }
  inline size_t task_submat_block_bfn_size(size_t nblock) {
    return PRDVL(task_submat_block_bfn, nblock);
  }
  inline size_t task_submat_cut_cou_size(size_t ncut) {
    return PRDVL(task_submat_cut_cou, 3*ncut);
  }
  inline size_t task_submat_block_cou_size(size_t nblock) {
    return PRDVL(task_submat_block_cou, nblock);
  }

  // Task indirection
  bool task_indirection = false;
  inline size_t task_indirection_size() {
    return PRDVL(task_indirection, 1ul);
  }

  // Weights kernel scratch
  bool grid_to_center_dist_scr     = false;
  bool grid_to_center_dist_nearest = false;
  bool grid_to_parent_center       = false;

  inline size_t grid_to_center_dist_scr_size(size_t ldatom, size_t npts) {
    return PRDVL(grid_to_center_dist_scr, ldatom * npts);
  }
  inline size_t grid_to_center_dist_nearest_size(size_t npts) {
    return PRDVL(grid_to_center_dist_nearest, npts);
  }
  inline size_t grid_to_parent_center_size(size_t npts) {
    return PRDVL(grid_to_parent_center, npts);
  }

  // Shell/Shell pairs lists + indirection
  bool task_shell_list_bfn    = false;
  bool task_shell_offs_bfn    = false;
  bool shell_to_task_bfn      = false;
  bool shell_pair_to_task_cou = false;
  bool task_to_shell_pair_cou = false;
  
  inline size_t task_shell_list_bfn_size(size_t nshells) {
    return PRDVL(task_shell_list_bfn, nshells);
  }
  inline size_t task_shell_offs_bfn_size(size_t nshells) {
    return PRDVL(task_shell_offs_bfn, nshells);
  }
  inline size_t shell_to_task_idx_bfn_size(size_t nshells) {
    return PRDVL(shell_to_task_bfn, nshells);
  }
  inline size_t shell_to_task_off_bfn_size(size_t nshells) {
    return PRDVL(shell_to_task_bfn, nshells);
  }
  inline size_t shell_pair_to_task_idx_cou_size(size_t nshells) {
    const size_t nslt = (nshells * (nshells+1)) / 2;
    return PRDVL(shell_pair_to_task_cou, nslt);
  }
  inline size_t shell_pair_to_task_row_off_cou_size(size_t nshells) {
    const size_t nslt = (nshells * (nshells+1)) / 2;
    return PRDVL(shell_pair_to_task_cou, nslt);
  }
  inline size_t shell_pair_to_task_col_off_cou_size(size_t nshells) {
    const size_t nslt = (nshells * (nshells+1)) / 2;
    return PRDVL(shell_pair_to_task_cou, nslt);
  }
  inline size_t task_to_shell_pair_col_off_cou_size(size_t nshells) {
    const size_t nslt = (nshells * (nshells+1)) / 2;
    return PRDVL(task_to_shell_pair_cou, nslt);
  }
  inline size_t task_to_shell_pair_row_off_cou_size(size_t nshells) {
    const size_t nslt = (nshells * (nshells+1)) / 2;
    return PRDVL(task_to_shell_pair_cou, nslt);
  }
  inline size_t task_to_shell_pair_idx_cou_size(size_t nshells) {
    const size_t nslt = (nshells * (nshells+1)) / 2;
    return PRDVL(task_to_shell_pair_cou, nslt);
  }
  inline size_t task_to_shell_pair_cou_size() {
    return PRDVL(task_to_shell_pair_cou, 1ul);
  }
  inline size_t task_to_shell_pair_cou_subtask_size(size_t npts, size_t subtask_size) {
    const size_t num_subtasks = util::div_ceil(npts, subtask_size);
    return PRDVL(task_to_shell_pair_cou, num_subtasks);
  }



  inline explicit required_term_storage(integrator_term_tracker tracker) {
    // Everything under the sun needs the grid
    grid_points  = true;
    grid_weights = true;

    if(tracker.weights) {
      grid_to_center_dist_scr     = true;
      grid_to_center_dist_nearest = true; 
      grid_to_parent_center       = true;
    }

    // Allocated terms for XC calculations
    const bool is_xc = tracker.exc_vxc or tracker.exc_grad or tracker.fxc_contraction or tracker.onedft;
    const bool is_2nd_deriv = tracker.fxc_contraction;
    
    ref_tracker = tracker;

    if(is_xc) {
      if( tracker.xc_approx == _UNDEF_APPROX )
        GAUXC_GENERIC_EXCEPTION("No XC Approx Set");
      if( tracker.ks_scheme == _UNDEF_SCHEME )
        GAUXC_GENERIC_EXCEPTION("No KS Scheme Set");
      //const bool is_lda  = is_xc and tracker.xc_approx == LDA;
      const bool is_gga  = is_xc and tracker.xc_approx == GGA;
      const bool need_tau  = tracker.xc_approx == MGGA_TAU;
      const bool need_lapl = tracker.xc_approx == MGGA_LAPL;
      const bool is_mgga = is_xc and (need_tau or need_lapl);
      const bool is_grad = tracker.exc_grad;
      const bool is_rks  = tracker.ks_scheme == RKS;

      grid_den      = true;
      grid_den_grad = is_gga or is_mgga or is_grad;
      grid_lapl     = need_lapl;
      grid_gamma    = is_gga or is_mgga;
      grid_tau      = is_mgga;
      grid_eps      = true;
      grid_vrho     = true;
      grid_vgamma   = is_gga or is_mgga;
      grid_vtau     = is_mgga;
      grid_vlapl    = need_lapl;

      task_bfn          = true;
      task_bfn_grad     = is_gga or  is_mgga or is_grad;
      task_bfn_hess     = (is_gga or is_mgga) and is_grad;
      task_bfn_lapl     = need_lapl;
      task_bfn_lapgrad  = need_lapl and is_grad;
      task_zmat         = true;
      task_xmat         = true;
      task_xmat_grad    = is_mgga or (is_gga and is_grad);
      task_xmat_persist = is_grad and not is_rks;
      task_nbe_scr      = true;

      task_submat_cut_bfn   = true;
      task_submat_block_bfn = true;
      task_indirection      = true;

      task_shell_list_bfn = true;
      task_shell_offs_bfn = true;
      shell_to_task_bfn   = true;
    }

    if(is_2nd_deriv) {
      grid_eps      = false;

      grid_tden      = true;
      grid_tden_grad = true;
      grid_tlapl     = true;
      grid_ttau      = true;
      grid_v2rho2    = true;
      grid_v2rhogamma= true;
      grid_v2rholapl = true;
      grid_v2rhotau  = true;
      grid_v2gamma2  = true;
      grid_v2gammalapl= true;
      grid_v2gammatau= true;
      grid_v2lapl2   = true;
      grid_v2lapltau = true;
      grid_v2tau2    = true;
      grid_FXC_A         = true;
      grid_FXC_B         = true;
      grid_FXC_C         = true;

      // task_bfn_hess     = is_gga or is_mgga or is_grad; // TODO: Check this
      // task_bfn_lapgrad  = need_lapl and is_grad; // TODO: Check this
    }

    // Density integration
    if(tracker.den) {
      grid_den              = true;
      task_bfn              = true;
      task_nbe_scr          = true;
      task_xmat             = true;
      task_zmat             = true;
      task_submat_cut_bfn   = true;
      task_submat_block_bfn = true;
      task_indirection      = true;

      task_shell_list_bfn = true;
      task_shell_offs_bfn = true;
      shell_to_task_bfn   = true;
    }

    // EXX integration
    if(tracker.exx) {
      task_bfn              = true;
      task_fmat             = true;
      task_gmat             = true;
      task_nbe_scr          = true;
      task_submat_cut_bfn   = true;
      task_submat_block_bfn = true;
      task_submat_cut_cou   = true;
      task_submat_block_cou = true;
      task_indirection      = true;

      task_shell_list_bfn    = true;
      task_shell_offs_bfn    = true;
      shell_to_task_bfn      = true;
      //shell_pair_to_task_cou = true;
      task_to_shell_pair_cou = true;
    }

    if(tracker.exx_ek_screening) {
      task_bfn              = true;
      task_indirection      = true;

      task_shell_list_bfn        = true;
      task_shell_offs_bfn        = true;
      task_bfn_shell_indirection = true;
      shell_to_task_bfn          = true;
    }

  }
};

#undef PRDVL



inline 
std::ostream& operator<<( std::ostream& out, const integrator_term_tracker& t ) {
  out << std::boolalpha;
  out << "Integrator Terms:" << std::endl;
  out << "  WEIGHTS  " << t.weights << std::endl;
  out << "  DEN      " << t.den << std::endl;
  out << "  EXC_VXC  " << t.exc_vxc << std::endl;
  out << "  FXC_CONTRACTION " << t.fxc_contraction << std::endl;
  out << "  EXC_GRAD " << t.exc_grad << std::endl;
  out << "  EXX      " << t.exx << std::endl;
  return out;
}

/** Base class for all XCDeviceData types
 *
 *  Exposes virtual API to manage device memory and batch XC
 *  integration tasks.
 */
struct XCDeviceData {

  using host_task_type        = XCTask;
  using host_task_container   = std::vector<host_task_type>;
  using host_task_iterator    = host_task_container::iterator;

  virtual ~XCDeviceData() noexcept = default;

  /// Allocate device memory for data that will persist on the device.
  virtual void reset_allocations() = 0;
  virtual void allocate_static_data_weights( int32_t natoms ) = 0;
  virtual void allocate_static_data_onedft( int32_t nbf, int32_t nshells, int32_t natoms, int32_t total_npts, integrator_term_tracker enabled_terms ) = 0;
  virtual void allocate_static_data_exc_vxc( int32_t nbf, int32_t nshells, integrator_term_tracker enabled_terms, bool do_vxc ) = 0;
  virtual void allocate_static_data_den( int32_t nbf, int32_t nshells ) = 0;
  virtual void allocate_static_data_exc_grad( int32_t nbf, int32_t nshells, int32_t natoms, integrator_term_tracker enabled_terms ) = 0;
  virtual void allocate_static_data_exx( int32_t nbf, int32_t nshells, size_t nshell_pairs, size_t nprim_pair_total, int32_t max_l ) = 0;
  virtual void allocate_static_data_exx_ek_screening( size_t ntasks, int32_t nbf, int32_t nshells, int nshell_pairs, int32_t max_l ) = 0;
  virtual void allocate_static_data_fxc_contraction( int32_t nbf, int32_t nshells, integrator_term_tracker enabled_terms) = 0;

  // Send persistent data from host to device
  virtual void send_static_data_weights( const Molecule& mol, const MolMeta& meta ) = 0;
  virtual void send_static_data_onedft( const Molecule& mol, const double* Ps, int32_t ldps, const double* Pz, int32_t ldpz, const double* Py, int32_t ldpy, const double* Px, int32_t ldpx, const BasisSet<double>& basis ) = 0;
  virtual void send_static_data_onedft_results( int32_t total_npts, int32_t ndm, const double* EXC, const double* DEN, const double* DDEN, const double* TAU) = 0;
  virtual void send_static_data_density_basis( const double* Ps, int32_t ldps, 
    const double* Pz, int32_t ldpz, const double* Py, int32_t ldpy, 
    const double* Px, int32_t ldpx, const BasisSet<double>& basis ) = 0;
  virtual void send_static_data_trial_density(
    const double* tPs, int32_t ldtps, const double* tPz, int32_t ldtpz,
    const double* tPy, int32_t ldtpy, const double* tPx, int32_t ldtpx ) = 0;
  virtual void send_static_data_shell_pairs( const BasisSet<double>&, const ShellPairCollection<double>& ) = 0;
  virtual void send_static_data_exx_ek_screening( const double* V_max, int32_t ldv, const BasisSetMap&, const ShellPairCollection<double>& ) = 0;

  /// Zero out the density integrands in device memory
  virtual void zero_den_integrands() = 0;

  /// Zero out the EXC / VXC integrands in device memory
  virtual void zero_exc_vxc_integrands(integrator_term_tracker enabled_terms) = 0;

  /// Zero out the EXC Gradient integrands in device memory
  virtual void zero_exc_grad_integrands() = 0;

  /// Zero out the EXX integrands in device memory
  virtual void zero_exx_integrands() = 0;

  /// Zero out intermediates for EXX EK screening
  virtual void zero_exx_ek_screening_intermediates() = 0;

  /// Zero out the FXC contraction integrands in device memory
  virtual void zero_fxc_contraction_integrands() = 0;

  /** Generate task batch to execute on device
   *
   *  Generate a batch of XC tasks to execute on the device and 
   *  populate device memory for said batch.
   *
   *  TODO: this will depend on the integrand, we should refactor this
   *  to only allocate what is needed
   *
   *  @param[in] basis_map  Basis set map instance for passed basis object
   *                        (TODO, this should probably persist to avoid clashes)
   *  @param[in] task_begin Start iterator for XC task queue
   *  @param[in] task_end   End iterator for XC task queue
   *
   *  @returns iterator to last XC task queue which was not kept in the
   *           allocated batch (!= task_end)
   */
  virtual host_task_iterator generate_buffers( integrator_term_tracker terms,
    const BasisSetMap& basis_map, host_task_iterator task_begin,
    host_task_iterator task_end ) = 0;

  /** Retreive EXC/VXC integrands from device memory
   *
   *  @param[out] EXC  Integrated XC energy (host) for XC task
   *  @param[out] N_EL Integrated # electrons (host) for XC queue (accuracy metric)
   *  @param[out[ VXC  Integrated XC potential (host) for XC queue
   */
  virtual void retrieve_exc_vxc_integrands( double* EXC, double* N_EL,
    double* VXCs, int32_t ldvxcs, double* VXCz, int32_t ldvxcz,
    double* VXCy, int32_t ldvxcy, double* VXCx, int32_t ldvxcx ) = 0;

  /** Retreive OneDFT features from device memory
   */
  virtual void retrieve_onedft_features( int32_t total_npts, int32_t ndm, double* DEN, 
    double* DDEN, double* TAU, double* POINTS, double* WEIGHTS ) = 0;
    
  virtual void retrieve_fxc_contraction_integrands( double* N_EL,
    double* FXCs, int32_t ldfxcs, double* FXCz, int32_t ldfxcz,
    double* FXCy, int32_t ldfxcy, double* FXCx, int32_t ldfxcx ) = 0;

  /** Retreive EXC Gradient integrands from device memory
   *
   *  @param[out] EXC_GRAD  Integrated XC Gradient (host) for XC task
   *  @param[out] N_EL      Integrated # electrons (host) for XC queue 
   */
  virtual void retrieve_exc_grad_integrands( double* EXC_GRAD, double* N_EL ) = 0;

  /** Retreive Density integrands from device memory
   *
   *  @param[out] N_EL      Integrated # electrons (host) for XC queue 
   */
  virtual void retrieve_den_integrands( double* N_EL ) = 0;


  virtual void retrieve_exx_integrands( double* K, int32_t ldk ) = 0;

  virtual void retrieve_exx_ek_max_bfn_sum( double* MBS, int32_t nt) = 0;


  virtual void copy_weights_to_tasks( host_task_iterator task_begin, host_task_iterator task_end ) = 0;
  virtual void populate_submat_maps ( size_t, host_task_iterator begin, host_task_iterator end, const BasisSetMap& ) = 0;

  virtual double* vxc_z_device_data() = 0;
  virtual double* vxc_s_device_data() = 0;
  virtual double* vxc_y_device_data() = 0;
  virtual double* vxc_x_device_data() = 0;
  virtual double* exc_device_data() = 0;
  virtual double* nel_device_data() = 0;
  virtual double* exx_k_device_data() = 0;

  virtual double* grid_weights_device_data() = 0;
  virtual double* grid_coords_device_data() = 0;
  virtual double* den_eval_device_data() = 0;
  virtual double* dden_eval_device_data() = 0;
  virtual double* tau_device_data() = 0;
  virtual double* coords_device_data() = 0;

  virtual double* fxc_z_device_data() = 0;
  virtual double* fxc_s_device_data() = 0;
  virtual double* fxc_y_device_data() = 0;
  virtual double* fxc_x_device_data() = 0;
  virtual device_queue queue() = 0;


};

}
