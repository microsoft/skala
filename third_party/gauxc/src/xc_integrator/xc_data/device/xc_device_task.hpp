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

namespace GauXC {

struct XCDeviceTask {

  size_t npts = 0;

  struct screening_quantities {
    size_t nbe             = 0;
    size_t ncut            = 0;
    size_t nblock          = 0;
    size_t nshells         = 0;
    size_t ibf_begin       = 0;
    size_t* shell_list     = nullptr;
    size_t* shell_offs     = nullptr;
    int32_t* submat_cut    = nullptr;
    int32_t* submat_block  = nullptr;
  };

  screening_quantities bfn_screening;
  screening_quantities cou_screening;

  double* points_x       = nullptr;
  double* points_y       = nullptr;
  double* points_z       = nullptr;
  double* weights        = nullptr;

  double*   nbe_scr = nullptr;
  double*   zmat    = nullptr;
  double*   fmat    = nullptr;
  double*   gmat    = nullptr;
  double*   xmat_x  = nullptr;
  double*   xmat_y  = nullptr;
  double*   xmat_z  = nullptr;
  double*   bf      = nullptr;
  double*   dbfx    = nullptr;
  double*   dbfy    = nullptr;
  double*   dbfz    = nullptr;
  double*   d2bfxx    = nullptr;
  double*   d2bfxy    = nullptr;
  double*   d2bfxz    = nullptr;
  double*   d2bfyy    = nullptr;
  double*   d2bfyz    = nullptr;
  double*   d2bfzz    = nullptr;
  double*   eps     = nullptr;

  double* den    = nullptr;
  double* gamma  = nullptr;
  double* tau    = nullptr;
  double* lapl   = nullptr;
  double* vrho   = nullptr;
  double* vgamma = nullptr;
  double* vtau   = nullptr;
  double* vlapl  = nullptr;
    
  // (S,Z,Y,X) densities
  double* den_s     = nullptr;
  double* den_z     = nullptr;
  double* den_y     = nullptr;
  double* den_x     = nullptr;
  double* tau_s     = nullptr;
  double* tau_z     = nullptr;
  double* tau_y     = nullptr;
  double* tau_x     = nullptr;
  double* lapl_s    = nullptr;
  double* lapl_z    = nullptr;
  double* lapl_y    = nullptr;
  double* lapl_x    = nullptr;

  // Del(S,Z,Y,X) Gradients
  double* dden_sx   = nullptr;
  double* dden_sy   = nullptr;
  double* dden_sz   = nullptr;
  double* dden_zx   = nullptr;
  double* dden_zy   = nullptr;
  double* dden_zz   = nullptr;
  double* dden_yx   = nullptr;
  double* dden_yy   = nullptr;
  double* dden_yz   = nullptr;
  double* dden_xx   = nullptr;
  double* dden_xy   = nullptr;
  double* dden_xz   = nullptr;
  
  // 2C U vars
  double* vrho_pos  = nullptr;
  double* vrho_neg  = nullptr;
  double* gamma_pp  = nullptr;
  double* gamma_pm  = nullptr;
  double* gamma_mm  = nullptr;
  double* vgamma_pp  = nullptr;
  double* vgamma_pm  = nullptr;
  double* vgamma_mm  = nullptr;
  double* vtau_pos  = nullptr;
  double* vtau_neg  = nullptr;
  double* vlapl_pos  = nullptr;
  double* vlapl_neg  = nullptr;

  // GKS K,H matrices
  double* K_z        = nullptr;
  double* K_y        = nullptr;
  double* K_x        = nullptr;
  double* H_z        = nullptr;
  double* H_y        = nullptr;
  double* H_x        = nullptr;

  // MGGA
  double*   d2bflapl    = nullptr;
  double*   d3bflapl_x    = nullptr;
  double*   d3bflapl_y    = nullptr;
  double*   d3bflapl_z    = nullptr;

  // Persistent X matrices for EXC gradients
  double* xmatS   = nullptr;
  double* xmatS_x = nullptr;
  double* xmatS_y = nullptr;
  double* xmatS_z = nullptr;
  double* xmatZ   = nullptr;
  double* xmatZ_x = nullptr;
  double* xmatZ_y = nullptr;
  double* xmatZ_z = nullptr;

  // Second derivatives - Trial density and derivatives
  double* tden    = nullptr;
  double* ttau    = nullptr;
  double* tlapl   = nullptr;
  double* v2rho2      = nullptr;
  double* v2rhogamma  = nullptr;
  double* v2rholapl   = nullptr;
  double* v2rhotau    = nullptr;
  double* v2gamma2    = nullptr;
  double* v2gammalapl = nullptr;
  double* v2gammatau  = nullptr;
  double* v2lapl2     = nullptr;
  double* v2lapltau   = nullptr;
  double* v2tau2      = nullptr;
  
  // (S,Z,Y,X) trial densities
  double* tden_s     = nullptr;
  double* tden_z     = nullptr;
  double* tden_y     = nullptr;
  double* tden_x     = nullptr;
  double* ttau_s     = nullptr;
  double* ttau_z     = nullptr;
  double* ttau_y     = nullptr;
  double* ttau_x     = nullptr;
  double* tlapl_s    = nullptr;
  double* tlapl_z    = nullptr;
  double* tlapl_y    = nullptr;
  double* tlapl_x    = nullptr;

  // Del(S,Z,Y,X) trial density gradients
  double* tdden_sx   = nullptr;
  double* tdden_sy   = nullptr;
  double* tdden_sz   = nullptr;
  double* tdden_zx   = nullptr;
  double* tdden_zy   = nullptr;
  double* tdden_zz   = nullptr;
  double* tdden_yx   = nullptr;
  double* tdden_yy   = nullptr;
  double* tdden_yz   = nullptr;
  double* tdden_xx   = nullptr;
  double* tdden_xy   = nullptr;
  double* tdden_xz   = nullptr;
  
  //2C U variables for second derivatives
  double* v2rho2_a_a = nullptr;
  double* v2rho2_a_b = nullptr;
  double* v2rho2_b_b = nullptr;
  double* v2rhogamma_a_aa = nullptr;
  double* v2rhogamma_a_ab = nullptr;
  double* v2rhogamma_a_bb = nullptr;
  double* v2rhogamma_b_aa = nullptr;
  double* v2rhogamma_b_ab = nullptr;
  double* v2rhogamma_b_bb = nullptr;
  double* v2rholapl_a_a = nullptr;
  double* v2rholapl_a_b = nullptr;
  double* v2rholapl_b_a = nullptr;
  double* v2rholapl_b_b = nullptr;
  double* v2rhotau_a_a = nullptr;
  double* v2rhotau_a_b = nullptr;
  double* v2rhotau_b_a = nullptr;
  double* v2rhotau_b_b = nullptr;
  double* v2gamma2_aa_aa = nullptr;
  double* v2gamma2_aa_ab = nullptr;
  double* v2gamma2_aa_bb = nullptr;
  double* v2gamma2_ab_ab = nullptr;
  double* v2gamma2_ab_bb = nullptr;
  double* v2gamma2_bb_bb = nullptr;
  double* v2gammalapl_aa_a = nullptr;
  double* v2gammalapl_aa_b = nullptr;
  double* v2gammalapl_ab_a = nullptr;
  double* v2gammalapl_ab_b = nullptr;
  double* v2gammalapl_bb_a = nullptr;
  double* v2gammalapl_bb_b = nullptr;
  double* v2gammatau_aa_a = nullptr;
  double* v2gammatau_aa_b = nullptr;
  double* v2gammatau_ab_a = nullptr;
  double* v2gammatau_ab_b = nullptr;
  double* v2gammatau_bb_a = nullptr;
  double* v2gammatau_bb_b = nullptr;
  double* v2lapl2_a_a = nullptr;
  double* v2lapl2_a_b = nullptr;
  double* v2lapl2_b_b = nullptr;
  double* v2lapltau_a_a = nullptr;
  double* v2lapltau_a_b = nullptr;
  double* v2lapltau_b_a = nullptr;
  double* v2lapltau_b_b = nullptr;
  double* v2tau2_a_a = nullptr;
  double* v2tau2_a_b = nullptr;
  double* v2tau2_b_b = nullptr;

  // Second derivatives intermediate output
  double* FXC_A_s = nullptr;
  double* FXC_Bx_s = nullptr;
  double* FXC_By_s = nullptr;
  double* FXC_Bz_s = nullptr;
  double* FXC_C_s = nullptr;
  double* FXC_A_z = nullptr;
  double* FXC_Bx_z = nullptr;
  double* FXC_By_z = nullptr;
  double* FXC_Bz_z = nullptr;
  double* FXC_C_z = nullptr;

  int32_t iParent       = -1;
  double dist_nearest   = 0.;
  double * dist_scratch = nullptr;

  int32_t* bfn_shell_indirection = nullptr;

};

}
