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
#include <gauxc/xc_integrator/local_work_driver.hpp>

#include <memory>
#include <gauxc/molmeta.hpp>
#include <gauxc/basisset.hpp>
#include <gauxc/shell_pair.hpp>
#include <gauxc/basisset_map.hpp>
#include <gauxc/xc_task.hpp>


namespace GauXC {
namespace detail {

struct LocalHostWorkDriverPIMPL;

}

/// Base class for local work drivers in Host execution spaces 
class LocalHostWorkDriver : public LocalWorkDriver {

  using pimpl_type = std::unique_ptr<detail::LocalHostWorkDriverPIMPL>;

public:

  using submat_map_t = std::vector< std::array<int32_t,3> >;
  using task_container = std::vector<XCTask>;
  using task_iterator  = typename task_container::iterator;

  /// Construct LocalHostWorkDriver instance in invalid state
  LocalHostWorkDriver();

  /** Construct LocalHostWorkDriver instance given implementation pointer
   *  @param[in] ptr Pointer to implementation
   */
  LocalHostWorkDriver( pimpl_type&& ptr );

  /// Destructor (default)
  ~LocalHostWorkDriver() noexcept;

  // Remove copy ctor
  LocalHostWorkDriver( const LocalHostWorkDriver& ) = delete;

  /** Construct LocalHostWorkDriver by transferring ownership
   *  @param[in] other LocalHostWorkDriver instance to take ownership
   */
  LocalHostWorkDriver( LocalHostWorkDriver&& other ) noexcept;


  // Public APIs

  /** Evaluate the molecular partition weights
   *
   *  Overwrites the weights of passed XC Tasks to include molecular
   *  partition weights.
   *
   *  @param[in] weight_alg Molecular partitioning scheme
   *  @param[in] mol        Molecule being partitioned
   *  @param[in] molmeta    Metadata associated with mol
   *
   *  @param[in/out] task_begin Start iterator for task container to be modified
   *  @param[in/out] task_end   End iterator for task container to be modified
   */
  void partition_weights( XCWeightAlg weight_alg, const Molecule& mol, 
    const MolMeta& meta, task_iterator task_begin, task_iterator task_end );

  /** Evaluate the weight first derivative contracted with a function
   *
   *  @param[in] weight_alg Molecular partitioning scheme
   *  @param[in] mol        Molecule being partitioned
   *  @param[in] molmeta    Metadata associated with mol
   *  @param[in] task       Task Data
   *  @param[in] w_times_f  Weight times function evaluation
   * 
   *  @param[in/out] exc_grad_w  Weight first derivative times function evaluation (added to this array)
   *                              Assumed to have length 3*natoms. Example: exc_grad
   */
  void eval_weight_1st_deriv_contracted( XCWeightAlg weight_alg, const Molecule& mol, 
    const MolMeta& meta, const XCTask& task, const double* w_times_f, double* exc_grad_w );


  /** Evaluation the collocation matrix
   *
   *  @param[in] npts     Number of points on which to evaluate the basis
   *  @param[in] nshells  Number of shells to evaluate (length of shell_list)
   *  @param[in] nbe      Total number of basis functions to evaluate (sum over shell_list)
   *  @param[in] pts      Grid points (AoS)
   *  @param[in] basis    Full basis set
   *  @param[in] shell_list List of indices (0-based) to evaulate from basis
   *
   *  @param[out] basis_eval Collocation matrix in col major (bfn,pts). 
   *                         Assumed to have leading dimension of nbe.
   */
  void eval_collocation( size_t npts, size_t nshells, size_t nbe, 
    const double* pts, const BasisSet<double>& basis, const int32_t* shell_list, 
    double* basis_eval );


  /** Evaluation the collocation matrix + gradient
   *
   *  @param[in] npts     Same as `eval_collocation`
   *  @param[in] nshells  Same as `eval_collocation`
   *  @param[in] nbe      Same as `eval_collocation`
   *  @param[in] pts      Same as `eval_collocation`
   *  @param[in] basis    Same as `eval_collocation`
   *  @param[in] shell_list Same as `eval_collocation`
   *
   *  @param[out] basis_eval    Same as `eval_collocation`
   *  @param[out] dbasis_x_eval Derivative of `basis_eval` wrt x (same dimensions)
   *  @param[out] dbasis_y_eval Derivative of `basis_eval` wrt y (same dimensions)
   *  @param[out] dbasis_z_eval Derivative of `basis_eval` wrt z (same dimensions)
   */
  void eval_collocation_gradient( size_t npts, size_t nshells, size_t nbe, 
    const double* pts, const BasisSet<double>& basis, const int32_t* shell_list, 
    double* basis_eval, double* dbasis_x_eval, double* dbasis_y_eval, 
    double* dbasis_z_eval);


  /** Evaluation the collocation matrix + gradient + hessian
   *
   *  @param[in] npts     Same as `eval_collocation`
   *  @param[in] nshells  Same as `eval_collocation`
   *  @param[in] nbe      Same as `eval_collocation`
   *  @param[in] pts      Same as `eval_collocation`
   *  @param[in] basis    Same as `eval_collocation`
   *  @param[in] shell_list Same as `eval_collocation`
   *
   *  @param[out] basis_eval    Same as `eval_collocation`
   *  @param[out] dbasis_x_eval Same as `eval_collocation_gradient`
   *  @param[out] dbasis_y_eval Same as `eval_collocation_gradient`
   *  @param[out] dbasis_z_eval Same as `eval_collocation_gradient`
   *  @param[out] d2basis_xx_eval Derivative of `basis_eval` wrt x+x (same dimensions)
   *  @param[out] d2basis_xy_eval Derivative of `basis_eval` wrt x+y (same dimensions)
   *  @param[out] d2basis_xz_eval Derivative of `basis_eval` wrt x+z (same dimensions)
   *  @param[out] d2basis_yy_eval Derivative of `basis_eval` wrt y+y (same dimensions)
   *  @param[out] d2basis_yz_eval Derivative of `basis_eval` wrt y+z (same dimensions)
   *  @param[out] d2basis_zz_eval Derivative of `basis_eval` wrt z+z (same dimensions)
   */
  void eval_collocation_hessian( size_t npts, size_t nshells, size_t nbe, 
    const double* pts, const BasisSet<double>& basis, const int32_t* shell_list, 
    double* basis_eval, double* dbasis_x_eval, double* dbasis_y_eval, 
    double* dbasis_z_eval, double* d2basis_xx_eval, double* d2basis_xy_eval,
    double* d2basis_xz_eval, double* d2basis_yy_eval, double* d2basis_yz_eval,
    double* d2basis_zz_eval );

  /** Evaluation the collocation matrix + gradient + hessian + 3rd derivatives
   *
   *  @param[in] npts     Same as `eval_collocation`
   *  @param[in] nshells  Same as `eval_collocation`
   *  @param[in] nbe      Same as `eval_collocation`
   *  @param[in] pts      Same as `eval_collocation`
   *  @param[in] basis    Same as `eval_collocation`
   *  @param[in] shell_list Same as `eval_collocation`
   *
   *  @param[out] basis_eval    Same as `eval_collocation`
   *  @param[out] dbasis_x_eval Same as `eval_collocation_gradient`
   *  @param[out] dbasis_y_eval Same as `eval_collocation_gradient`
   *  @param[out] dbasis_z_eval Same as `eval_collocation_gradient`
   *  @param[out] d2basis_xx_eval Derivative of `basis_eval` wrt x+x (same dimensions)
   *  @param[out] d2basis_xy_eval Derivative of `basis_eval` wrt x+y (same dimensions)
   *  @param[out] d2basis_xz_eval Derivative of `basis_eval` wrt x+z (same dimensions)
   *  @param[out] d2basis_yy_eval Derivative of `basis_eval` wrt y+y (same dimensions)
   *  @param[out] d2basis_yz_eval Derivative of `basis_eval` wrt y+z (same dimensions)
   *  @param[out] d2basis_zz_eval Derivative of `basis_eval` wrt z+z (same dimensions)
   *  @param[out] d3basis_xxx_eval Derivative of `basis_eval` wrt x+x+x (same dimensions)
   *  @param[out] d3basis_xxy_eval Derivative of `basis_eval` wrt x+x+y (same dimensions)
   *  @param[out] d3basis_xxz_eval Derivative of `basis_eval` wrt x+x+z (same dimensions)
   *  @param[out] d3basis_xyy_eval Derivative of `basis_eval` wrt x+y+y (same dimensions)
   *  @param[out] d3basis_xyz_eval Derivative of `basis_eval` wrt x+y+z (same dimensions)
   *  @param[out] d3basis_xzz_eval Derivative of `basis_eval` wrt x+z+z (same dimensions)
   *  @param[out] d3basis_yyy_eval Derivative of `basis_eval` wrt y+y+y (same dimensions)
   *  @param[out] d3basis_yyz_eval Derivative of `basis_eval` wrt y+y+z (same dimensions)
   *  @param[out] d3basis_yzz_eval Derivative of `basis_eval` wrt y+z+z (same dimensions)
   *  @param[out] d3basis_zzz_eval Derivative of `basis_eval` wrt z+z+z (same dimensions)
   */
  void eval_collocation_der3( size_t npts, size_t nshells, size_t nbe, 
    const double* pts, const BasisSet<double>& basis, const int32_t* shell_list, 
    double* basis_eval, double* dbasis_x_eval, double* dbasis_y_eval, 
    double* dbasis_z_eval, double* d2basis_xx_eval, double* d2basis_xy_eval,
    double* d2basis_xz_eval, double* d2basis_yy_eval, double* d2basis_yz_eval,
    double* d2basis_zz_eval, double* d3basis_xxx_eval, double* d3basis_xxy_eval,
    double* d3basis_xxz_eval, double* d3basis_xyy_eval, double* d3basis_xyz_eval,
    double* d3basis_xzz_eval, double* d3basis_yyy_eval, double* d3basis_yyz_eval,
    double* d3basis_yzz_eval, double* d3basis_zzz_eval);

  /** Evaluate the compressed "X" matrix = fac * P * B
   *
   *  @param[in]  npts        The number of points in the collocation matrix 
   *  @param[in]  nbf         The total number of bfns
   *  @param[in]  nbe         The number of non-negligible bfns
   *  @param[in]  submat_map  Map from the full matrix to non-negligible submatrices
   *  @param[in]  fac         Scaling factor in front of matrix multiplication
   *  @param[in]  P           The alpha density matrix ( (nbf,nbf) col major)
   *  @param[in]  ldp         The leading dimension of P
   *  @param[in]  basis_eval  The collocation matrix ( (nbe,npts) col major)
   *  @param[in]  ldb         The leading dimension of basis_eval
   *  @param[out] X           The X matrix ( (nbe,npts) col major)
   *  @param[in]  ldx         The leading dimension of X
   *  @param[in/out] scr      Scratch space of at least nbe*nbe
   */
  void eval_xmat( size_t npts, size_t nbf, size_t nbe, 
    const submat_map_t& submat_map, double fac, const double* P, size_t ldp,
    const double* basis_eval, size_t ldb, double* X, size_t ldx, 
    double* scr );

  void eval_exx_fmat( size_t npts, size_t nbf, size_t nbe_bra,
    size_t nbe_ket, const submat_map_t& submat_map_bra,
    const submat_map_t& submat_map_ket, const double* P, size_t ldp,
    const double* basis_eval, size_t ldb, double* F, size_t ldf,
    double* scr );

  void eval_exx_gmat( size_t npts, size_t nshells, size_t nshell_pairs,
    size_t nbe, const double* points, const double* weights, 
    const BasisSet<double>& basis, const ShellPairCollection<double>& shpairs, 
    const BasisSetMap& basis_map, const int32_t* shell_list, 
    const std::pair<int32_t,int32_t>* shell_pair_list, 
    const double* X, size_t ldx, double* G, size_t ldg );

  void inc_exx_k( size_t npts, size_t nbf, size_t nbe_bra, size_t nbe_ket, 
    const double* basis_eval, const submat_map_t& submat_map_bra, 
    const submat_map_t& submat_map_ket, const double* G, size_t ldg, double* K, 
    size_t ldk, double* scr );
    
  /** Evaluate the U and V variavles for RKS LDA
   *
   *  U = V = rho (total density)
   *
   *  @param[in] npts       The number of points to evaluate the U/V variables
   *  @param[in] nbe        The number of basis functions in collocation matrix
   *  @param[in] basis_eval The collocation matrix ( (nbe,npts), col major, lb=nbe)
   *  @param[in] X          The X matrix (P*B, (nbe,npts) col major)
   *  @param[in] ldx        The leading dimension of X
   *  @param[out] den_eval  The total density evaluated on the grid (npts)
   *
   */
  void eval_uvvar_lda_rks( size_t npts, size_t nbe, const double* basis_eval,
    const double* X, size_t ldx, double* den_eval);

  /** Evaluate the U and V variavles for RKS LDA
   *
   *  U = rho_+ / rho_- (alpha and beta densities)
   *  V = rho_s / rho_z (scalar and spin densities)
   *
   *  @param[in] npts       The number of points to evaluate the U/V variables
   *  @param[in] nbe        The number of basis functions in collocation matrix
   *  @param[in] basis_eval The collocation matrix ( (nbe,npts), col major, lb=nbe)
   *  @param[in] Xs         The Xs matrix (Ps*B, (nbe,npts) col major)
   *  @param[in] Xz         The Xz matrix (Pz*B, (nbe,npts) col major)
   *  @param[in] ldx        The leading dimension of X
   *  @param[out] den_eval  The total density evaluated on the grid (npts)
   *
   */
  void eval_uvvar_lda_uks( size_t npts, size_t nbe, const double* basis_eval,
    const double* Xs, size_t ldxs, const double* Xz, size_t ldxz,
    double* den_eval);

  void eval_uvvar_lda_gks( size_t npts, size_t nbe, const double* basis_eval,
    const double* Xs, size_t ldxs, const double* Xz, size_t ldxz,
    const double* Xx, size_t ldxx, const double* Xy, size_t ldxy, double* den_eval, double* K, const double dtol);


  /** Evaluate the U and V variavles for RKS GGA
   *
   *  U = rho + gradient
   *  V = rho + gamma
   *
   *  @param[in] npts          Same as `eval_uvvar_lda`
   *  @param[in] nbe           Same as `eval_uvvar_lda`
   *  @param[in] basis_eval    Same as `eval_uvvar_lda`
   *  @param[in] dbasis_x_eval Derivative of `basis_eval` wrt x (same dims)
   *  @param[in] dbasis_y_eval Derivative of `basis_eval` wrt y (same dims)
   *  @param[in] dbasis_z_eval Derivative of `basis_eval` wrt z (same dims)
   *  @param[in] X             Same as `eval_uvvar_lda`
   *  @param[in] ldx           Same as `eval_uvvar_lda`
   *  @param[out] den_eval     Same as `eval_uvvar_lda`
   *  @param[out] dden_x_eval  Derivative of `den_eval` wrt x (npts)
   *  @param[out] dden_y_eval  Derivative of `den_eval` wrt y (npts)
   *  @param[out] dden_z_eval  Derivative of `den_eval` wrt z (npts)
   *  @param[out] gamma        |grad rho|^2 (npts)
   *                        
   */
  void eval_uvvar_gga_rks( size_t npts, size_t nbe, const double* basis_eval,
    const double* dbasis_x_eavl, const double *dbasis_y_eval, 
    const double* dbasis_z_eval, const double* X, size_t ldx, double* den_eval, 
    double* dden_x_eval, double* dden_y_eval, double* dden_z_eval, double* gamma );

  void eval_uvvar_gga_uks( size_t npts, size_t nbe, const double* basis_eval,
    const double* dbasis_x_eavl, const double *dbasis_y_eval,
    const double* dbasis_z_eval, const double* Xs, size_t ldxs, 
    const double* Xz, size_t ldxz, double* den_eval,
    double* dden_x_eval, double* dden_y_eval, double* dden_z_eval, double* gamma );

  void eval_uvvar_gga_gks( size_t npts, size_t nbe, const double* basis_eval,
    const double* dbasis_x_eavl, const double *dbasis_y_eval,
    const double* dbasis_z_eval, const double* Xs, size_t ldxs,
    const double* Xz, size_t ldxz, const double* Xx, size_t ldxx,
    const double* Xy, size_t ldxy, double* den_eval,
    double* dden_x_eval, double* dden_y_eval, double* dden_z_eval, double* gamma, double* K, double* H, const double dtol );
  
  /** Evaluate the U and V variavles for RKS MGGA
   *
   *  U = rho + gradient + tau + lapl
   *  V = rho + gamma + tau + lapl
   *
   *  @param[in] npts          Same as `eval_uvvar_lda`
   *  @param[in] nbe           Same as `eval_uvvar_lda`
   *  @param[in] basis_eval    Same as `eval_uvvar_lda`
   *  @param[in] dbasis_x_eval Derivative of `basis_eval` wrt x (same dims)
   *  @param[in] dbasis_y_eval Derivative of `basis_eval` wrt y (same dims)
   *  @param[in] dbasis_z_eval Derivative of `basis_eval` wrt z (same dims)
   *  @param[in] lbasis_eval   Laplacian of `basis_eval` (same dims)
   *  @param[in] X             Same as `eval_uvvar_lda`
   *  @param[in] ldx           Same as `eval_uvvar_lda`
   *  @param[in] mmat_x
   *  @param[in] mmat_y
   *  @param[in] mmat_z
   *  @param[in] ldm
   *  @param[out] den_eval     Same as `eval_uvvar_lda`
   *  @param[out] dden_x_eval  Derivative of `den_eval` wrt x (npts)
   *  @param[out] dden_y_eval  Derivative of `den_eval` wrt y (npts)
   *  @param[out] dden_z_eval  Derivative of `den_eval` wrt z (npts)
   *  @param[out] gamma        |grad rho|^2 (npts)
   *  @param[out] tau
   *  @param[out] lapl
   *                        
   */
  void eval_uvvar_mgga_rks( size_t npts, size_t nbe, const double* basis_eval,
    const double* dbasis_x_eavl, const double* dbasis_y_eval, 
    const double* dbasis_z_eval, const double* lbasis_eval,
    const double* X, size_t ldx, const double* mmat_x,
    const double* mmat_y, const double* mmat_z, size_t ldm, double* den_eval, 
    double* dden_x_eval, double* dden_y_eval, double* dden_z_eval, double* gamma,
    double* tau, double* lapl);
  void eval_uvvar_mgga_uks( size_t npts, size_t nbe, const double* basis_eval,
    const double* dbasis_x_eavl, const double* dbasis_y_eval, 
    const double* dbasis_z_eval, const double* lbasis_eval,
    const double* Xs, size_t ldxs, const double* Xz, size_t ldxz, 
    const double* mmat_xs, const double* mmat_ys, const double* mmat_zs, size_t ldms, 
    const double* mmat_xz, const double* mmat_yz, const double* mmat_zz, size_t ldmz, 
    double* den_eval, double* dden_x_eval, double* dden_y_eval, double* dden_z_eval, 
    double* gamma, double* tau, double* lapl);

    /** Evaluate the VXC Z Matrix for RKS LDA
   *
   *  Z(mu,i) = 0.5 * vrho(i) * B(mu, i)
   *
   *  TODO: Need to add an API for UKS/GKS
   *
   *  @param[in] npts        Number of grid points
   *  @param[in] nbe         Number of non-negligible bfns
   *  @param[in] vrho        Derivative of XC functional wrt rho scaled by quad weight (npts)
   *  @param[in] basis_eval  Collocation matrix ((nbe,npts), col major, ld=nbe)
   *  @param[out] Z          The Z Matrix ((nbe,npts), col major)
   *  @param[in]  ldz        Leading dimension of Z
   *
   */
  void eval_zmat_lda_vxc_rks( size_t npts, size_t nbe, const double* vrho, 
    const double* basis_eval, double* Z, size_t ldz );

  void eval_zmat_lda_vxc_uks( size_t npts, size_t nbe, const double* vrho,
    const double* basis_eval, double* Zs, size_t ldzs, double* Zz,
    size_t ldzz );

  void eval_zmat_lda_vxc_gks( size_t npts, size_t nbe, const double* vrho,
    const double* basis_eval, double* Zs, size_t ldzs, double* Zz, size_t ldzz,
    double* Zx, size_t ldzx,double* Zy, size_t ldzy, double *K );

  /** Evaluate the VXC Z Matrix for RKS LDA
   *
   *  Z(mu,i) = 0.5 * vrho(i)   * B(mu, i) +
   *            2.0 * vgamma(i) * (grad B(mu,i)) . (grad rho(i))
   *
   *  TODO: Need to add an API for UKS/GKS
   *
   *  @param[in] npts           Same as `eval_zmat_lda_vxc`
   *  @param[in] nbe            Same as `eval_zmat_lda_vxc`
   *  @param[in] vrho           Same as `eval_zmat_lda_vxc`
   *  @param[in] vgamma         Derivative of the XC functional wrt gamma scaled by quad weights (npts)
   *  @param[in] basis_eval     Same as `eval_zmat_lda_vxc`
   *  @param[in] dbasis_x_eval  Derivative of `basis_eval` wrt x (same dims)
   *  @param[in] dbasis_y_eval  Derivative of `basis_eval` wrt y (same dims)
   *  @param[in] dbasis_z_eval  Derivative of `basis_eval` wrt z (same dims)
   *  @param[in] dden_x_eval    Derivative of rho wrt x (npts)
   *  @param[in] dden_y_eval    Derivative of rho wrt y (npts)
   *  @param[in] dden_z_eval    Derivative of rho wrt z (npts)
   *  @param[out] Z             Same as `eval_zmat_lda_vxc`
   *  @param[in]  ldz           Same as `eval_zmat_lda_vxc`
   *
   */
  void eval_zmat_gga_vxc_rks( size_t npts, size_t nbe, const double* vrho, 
    const double* vgamma, const double* basis_eval, const double* dbasis_x_eval,
    const double* dbasis_y_eval, const double* dbasis_z_eval, 
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval,
    double* Z, size_t ldz );

  void eval_zmat_gga_vxc_uks( size_t npts, size_t nbe, const double* vrho,
    const double* vgamma, const double* basis_eval, const double* dbasis_x_eval,
    const double* dbasis_y_eval, const double* dbasis_z_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval,
    double* Zs, size_t ldzs, double* Zz, size_t ldzz );

  void eval_zmat_gga_vxc_gks( size_t npts, size_t nbe, const double* vrho,
    const double* vgamma, const double* basis_eval, const double* dbasis_x_eval,
    const double* dbasis_y_eval, const double* dbasis_z_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval,
    double* Zs, size_t ldzs, double* Zz, size_t ldzz, double* Zx, size_t ldzx, 
    double* Zy, size_t ldzy, double* K, double* H );

  /** Evaluate the VXC Z Matrix for RKS MGGA
   *
   *  Z(mu,i) = 0.5 * vrho(i)   * B(mu, i) +
   *            2.0 * vgamma(i) * (grad B(mu,i)) . (grad rho(i)) +
   *            0.5 * vlapl(i) * lapl B(mu, i)
   *
   *  TODO: Need to add an API for UKS/GKS
   *
   *  @param[in] npts           Same as `eval_zmat_lda_vxc`
   *  @param[in] nbe            Same as `eval_zmat_lda_vxc`
   *  @param[in] vrho           Same as `eval_zmat_lda_vxc`
   *  @param[in] vgamma         Derivative of the XC functional wrt gamma scaled by quad weights (npts)
   *  @param[in] basis_eval     Same as `eval_zmat_lda_vxc`
   *  @param[in] dbasis_x_eval  Derivative of `basis_eval` wrt x (same dims)
   *  @param[in] dbasis_y_eval  Derivative of `basis_eval` wrt y (same dims)
   *  @param[in] dbasis_z_eval  Derivative of `basis_eval` wrt z (same dims)
   *  @param[in] lbasis_eval    Laplacian of `basis_eval` (same dims)
   *  @param[in] dden_x_eval    Derivative of rho wrt x (npts)
   *  @param[in] dden_y_eval    Derivative of rho wrt y (npts)
   *  @param[in] dden_z_eval    Derivative of rho wrt z (npts)
   *  @param[out] Z             Same as `eval_zmat_lda_vxc`
   *  @param[in]  ldz           Same as `eval_zmat_lda_vxc`
   *
   */
  void eval_zmat_mgga_vxc_rks( size_t npts, size_t nbe, const double* vrho, 
    const double* vgamma, const double* vlapl, const double* basis_eval, 
    const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval, 
    const double* lbasis_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval,
    double* Z, size_t ldz );
  void eval_zmat_mgga_vxc_uks( size_t npts, size_t nbe, const double* vrho, 
    const double* vgamma, const double* vlapl, const double* basis_eval, 
    const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval, 
    const double* lbasis_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval,
    double* Zs, size_t ldzs, double* Zz, size_t ldzz );
  void eval_mmat_mgga_vxc_rks( size_t npts, size_t nbe, const double* vtau,
      const double* vlapl, const double* dbasis_x_eval, const double* dbasis_y_eval,
      const double* dbasis_z_eval, double* mmat_x, double* mmat_y, double* mmat_z,
      size_t ldm);
  void eval_mmat_mgga_vxc_uks( size_t npts, size_t nbe, const double* vtau,
      const double* vlapl, const double* dbasis_x_eval, const double* dbasis_y_eval,
      const double* dbasis_z_eval, double* mmat_xs, double* mmat_ys, double* mmat_zs,
      size_t ldms, double* mmat_xz, double* mmat_yz, double* mmat_zz, size_t ldmz);



  /** Increment VXC integrand given Z / Collocation (RKS LDA+GGA)
   *
   *  VXC += Z**H * B + h.c.
   *  VXC += M**H . dB + h.c.
   *
   *  Only updates lower triangle
   *
   *  @param[in] npts        Number of grid points
   *  @param[in] nbf         Number of bfns in full basis
   *  @param[in] nbe         Number of non-negligible bfns
   *  @paran[in] basis_eval  Compressed collocation matrix ((nbe,npts), col major, ld=nbe)
   *  @param[in] submat_map  Map between non-negilgible bfns to full basis
   *  @param[in] Z           Compressed Z Matrix ((nbe,npts), col major)
   *  @param[in] ldz         Leading dimension of Z
   *  @param[in/out] VXC     VXC integrand ((nbf,nbf), col major)
   *  @param[in]  ldvxc      Leading dimension of VXC
   *  @param[out] scr        Scratch space at least nbe*nbe
   *
   */
  void inc_vxc( size_t npts, size_t nbf, size_t nbe, const double* basis_eval,
    const submat_map_t& submat_map, const double* Z, size_t ldz, 
    double* VXC, size_t ldvxc, double* scr );

  /** Evaluate the intermediate vector variables tmat for Fxc contraction of LDA 
   *
   *  See Jiashu's notes for details
   *
   *  @param[in] npts       The number of points to evaluate the U/V variables
   *  @param[in] v2rho2     the second derivative of the XC functional wrt rho
   *  @param[in] trho       The trial density calculated from the trial density matrix
   *  @param[out] A         intermediate output to form zmat (npts, 1) for RKS, (npts, 2) for UKS
   *
   */
  void eval_tmat_lda_vxc_rks( size_t npts, const double* v2rho2, const double* trho, double* A);
  void eval_tmat_lda_vxc_uks( size_t npts, const double* v2rho2, const double* trho, double* A);
  
  /**
   * Evaluate the intermediate vector variables tmat for Fxc contraction of GGA
   * 
   * See Jiashu's notes for details
   * 
   * @param[in] npts       The number of points to evaluate the U/V variables
   * @param[in] vgamma     the derivative of the XC functional wrt gamma
   * @param[in] v2rho2 the second derivative of the XC functional wrt rho twice
   * @param[in] v2rhogamma the second derivative of the XC functional wrt rho and gamma
   * @param[in] v2gamma2 the second derivative of the XC functional wrt gamma twice
   * @param[in] tden_eval  The trial density calculated from the trial density matrix
   * @param[in] tdden_x_eval the gradient of the trial density calculated from the trial density matrix, similar for y and z
   * @param[in] dden_x_eval the gradient of the density (npts) calculated from the density matrix, similar for y and z
   * @param[out] A      intermediate output to form zmat (npts, 1) for RKS, (npts, 2) for UKS
   * @param[out] B      intermediate output to form zmat (npts, 3) for RKS, (npts, 6) for UKS
   */
  void eval_tmat_gga_vxc_rks( size_t npts, const double* vgamma, 
    const double* v2rho2, const double* v2rhogamma, const double* v2gamma2, 
    const double* tden_eval, const double* tdden_x_eval, const double* tdden_y_eval, const double* tdden_z_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* A, double* B );
  void eval_tmat_gga_vxc_uks( size_t npts, const double* vgamma, 
    const double* v2rho2, const double* v2rhogamma, const double* v2gamma2, 
    const double* trho, const double* tdden_x_eval, const double* tdden_y_eval, const double* tdden_z_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* A, double* B );
  
  /**
   *  Evaluate the intermediate vector variables tmat for Fxc contraction of MGGA
   * 
   * See Jiashu's notes for details
   * 
   * @param[in] npts       The number of points to evaluate the U/V variables
   * @param[in] vgamma     the derivative of the XC functional wrt gamma
   * @param[in] v2rho2   the second derivative of the XC functional wrt rho twice
   * @param[in] v2rhogamma the second derivative of the XC functional wrt rho and gamma
   * @param[in] v2rholapl the second derivative of the XC functional wrt rho and laplacian
   * @param[in] v2rhotau  the second derivative of the XC functional wrt rho and tau
   * @param[in] v2gamma2 the second derivative of the XC functional wrt gamma twice
   * @param[in] v2gammalapl the second derivative of the XC functional wrt gamma and laplacian
   * @param[in] v2gammatau the second derivative of the XC functional wrt gamma and tau
   * @param[in] v2lapl2 the second derivative of the XC functional wrt laplacian twice
   * @param[in] v2lapltau the second derivative of the XC functional wrt laplacian and tau
   * @param[in] v2tau2 the second derivative of the XC functional wrt tau twice
   * @param[in] tden_eval  The trial density calculated from the trial density matrix
   * @param[in] tdden_x_eval the gradient of the trial density calculated from the trial density matrix, similar for y and z
   * @param[in] dden_x_eval the gradient of the density (npts) calculated from the density matrix, similar for y and z
   * @param[in] ttau      the kinetic energy density calculated from the trial density matrix
   * @param[out] A     intermediate output to form zmat (npts, 1) for RKS, (npts, 2) for UKS
   * @param[out] B     intermediate output to form zmat (npts, 3) for RKS, (npts, 6) for UKS
   * @param[out] C     intermediate output to form mmat (npts, 1) for RKS, (npts, 2) for UKS
   */
  void eval_tmat_mgga_vxc_rks( size_t npts, const double* vgamma, 
    const double* v2rho2, const double* v2rhogamma, const double* v2rholapl, const double* v2rhotau, 
    const double* v2gamma2, const double* v2gammalapl, const double* v2gammatau,
    const double* v2lapl2, const double* v2lapltau, const double* v2tau2, 
    const double* tden_eval, const double* tdden_x_eval, const double* tdden_y_eval, const double* tdden_z_eval, const double* ttau, 
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* A, double* B, double* C);
  void eval_tmat_mgga_vxc_uks( size_t npts, const double* vgamma, 
    const double* v2rho2, const double* v2rhogamma, const double* v2rholapl, const double* v2rhotau, 
    const double* v2gamma2, const double* v2gammalapl, const double* v2gamma_tau,
    const double* v2lapl2, const double* v2tau_lapl, const double* v2tau2, 
    const double* trho, const double* tdden_x_eval, const double* tdden_y_eval, const double* tdden_z_eval, const double* ttau, 
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* A, double* B, double* C);

  
  void eval_zmat_lda_vxc_uks_ts( size_t npts, size_t nbe, const double* vrho,
    const double* basis_eval, double* Za, size_t ldza, double* Zb,
    size_t ldzb );
  void eval_Bvec_gga_vxc_uks_ts( size_t npts, const double* vgamma, 
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* B );
  void eval_zmat_gga_vxc_uks_ts( size_t npts, size_t nbf, const double* A, const double* B, const double* basis_eval,
    const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval,
    double* Za, size_t ldza, double* Zb, size_t ldzb );
  void eval_Bvec_gga_vxc_rks_ts( size_t npts, const double* vgamma, 
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval, double* B );
  void eval_zmat_gga_vxc_rks_ts( size_t npts, size_t nbf, const double* A, const double* B, const double* basis_eval,
    const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval, 
    double* Z, size_t ldz );

  void eval_zmat_gga_vxc_uks_ts( size_t npts, size_t nbe, const double* vrho,
    const double* vgamma, const double* basis_eval, const double* dbasis_x_eval,
    const double* dbasis_y_eval, const double* dbasis_z_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval,
    double* Za, size_t ldza, double* Zb, size_t ldzb );
  void eval_zmat_mgga_vxc_uks_ts( size_t npts, size_t nbe, const double* vrho, 
    const double* vgamma, const double* vlapl, const double* basis_eval, 
    const double* dbasis_x_eval, const double* dbasis_y_eval, const double* dbasis_z_eval, 
    const double* lbasis_eval,
    const double* dden_x_eval, const double* dden_y_eval, const double* dden_z_eval,
    double* Za, size_t ldza, double* Zb, size_t ldzb );
  void eval_mmat_mgga_vxc_uks_ts( size_t npts, size_t nbe, const double* vtau,
      const double* vlapl, const double* dbasis_x_eval, const double* dbasis_y_eval,
      const double* dbasis_z_eval, double* mmat_xs, double* mmat_ys, double* mmat_zs,
      size_t ldms, double* mmat_xz, double* mmat_yz, double* mmat_zz, size_t ldmz);

private: 

  pimpl_type pimpl_; ///< Implementation

};

}
