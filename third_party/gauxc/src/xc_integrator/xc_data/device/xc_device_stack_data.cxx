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
#include "xc_device_stack_data.hpp"
#include "buffer_adaptor.hpp"
#include <gauxc/runtime_environment.hpp>

namespace GauXC {

namespace detail {
  size_t memory_cap() {
    if( getenv("GAUXC_DEVICE_MEMORY_CAP" ) ) {
      return std::stoull( getenv("GAUXC_DEVICE_MEMORY_CAP") );
    } else { return std::numeric_limits<size_t>::max(); }
  }
}

XCDeviceStackData::XCDeviceStackData(const DeviceRuntimeEnvironment& rt) :
  runtime_(rt) { 
    device_ptr = runtime_.device_memory();
    devmem_sz  = runtime_.device_memory_size();
    device_backend_ = runtime_.device_backend();
    reset_allocations(); 
  }





XCDeviceStackData::~XCDeviceStackData() noexcept = default;


double* XCDeviceStackData::vxc_s_device_data() { return static_stack.vxc_s_device; }
double* XCDeviceStackData::vxc_z_device_data() { return static_stack.vxc_z_device; }
double* XCDeviceStackData::vxc_y_device_data() { return static_stack.vxc_y_device; }
double* XCDeviceStackData::vxc_x_device_data() { return static_stack.vxc_x_device; }
double* XCDeviceStackData::exc_device_data() { return static_stack.exc_device; }
double* XCDeviceStackData::nel_device_data() { return static_stack.nel_device; }
double* XCDeviceStackData::exx_k_device_data() { return static_stack.exx_k_device; }
double* XCDeviceStackData::fxc_s_device_data() { return static_stack.fxc_s_device; }
double* XCDeviceStackData::fxc_z_device_data() { return static_stack.fxc_z_device; }
double* XCDeviceStackData::fxc_y_device_data() { return static_stack.fxc_y_device; }
double* XCDeviceStackData::fxc_x_device_data() { return static_stack.fxc_x_device; }

double* XCDeviceStackData::grid_weights_device_data() { return static_stack.grid_weights_device; }
double* XCDeviceStackData::grid_coords_device_data() { return static_stack.grid_coords_device; }
double* XCDeviceStackData::den_eval_device_data() { return static_stack.den_eval_device; }
double* XCDeviceStackData::dden_eval_device_data() { return static_stack.dden_eval_device; }
double* XCDeviceStackData::tau_device_data() { return static_stack.tau_device; }
double* XCDeviceStackData::coords_device_data() { return static_stack.coords_device; }

device_queue XCDeviceStackData::queue() { 
  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");
  return device_backend_->queue();
}




void XCDeviceStackData::reset_allocations() {
  dynmem_ptr = device_ptr;
  dynmem_sz  = devmem_sz;
  allocated_terms.reset();
  static_stack.reset();
  base_stack.reset();
}

void XCDeviceStackData::allocate_static_data_weights( int32_t natoms ) {

  if( allocated_terms.weights ) 
    GAUXC_GENERIC_EXCEPTION("Attempting to reallocate Stack Weights");

  // Save state
  global_dims.natoms  = natoms;

  // Allocate static memory with proper alignment
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );

  static_stack.coords_device = mem.aligned_alloc<double>( 3 * natoms, csl );

  // Allow for RAB to be strided and properly aligned
  const auto ldatoms   = get_ldatoms();
  const auto rab_align = get_rab_align();
  static_stack.rab_device = mem.aligned_alloc<double>( natoms * ldatoms, rab_align, csl );

  // Get current stack location
  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft(); 

  allocated_terms.weights = true;
}

void XCDeviceStackData::allocate_static_data_exc_vxc( int32_t nbf, int32_t nshells, integrator_term_tracker enabled_terms, bool do_vxc ) {

  if( allocated_terms.exc_vxc ) 
    GAUXC_GENERIC_EXCEPTION("Attempting to reallocate Stack EXC VXC");
  if( enabled_terms.ks_scheme == _UNDEF_SCHEME )
    GAUXC_GENERIC_EXCEPTION("Must have a KS Scheme set to allocate Stack EXC VXC");

  // Save state
  global_dims.nshells = nshells;
  global_dims.nbf     = nbf; 

  // Allocate static memory with proper alignment
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );

  static_stack.shells_device     = mem.aligned_alloc<Shell<double>>( nshells , csl);
  static_stack.exc_device        = mem.aligned_alloc<double>( 1 , csl);
  static_stack.nel_device        = mem.aligned_alloc<double>( 1 , csl);
  static_stack.acc_scr_device    = mem.aligned_alloc<double>( 1 , csl);
  
  allocated_terms.ks_scheme = enabled_terms.ks_scheme;
  static_stack.dmat_s_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
  if( not (allocated_terms.ks_scheme == RKS) ) {
      static_stack.dmat_z_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      if( allocated_terms.ks_scheme == GKS ) {
        static_stack.dmat_y_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        static_stack.dmat_x_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      }
  }

  if( do_vxc ) {
    static_stack.vxc_s_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
    if( not (allocated_terms.ks_scheme == RKS) ) {
        static_stack.vxc_z_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        if( allocated_terms.ks_scheme == GKS ) {
          static_stack.vxc_y_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
          static_stack.vxc_x_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        }
    }
  }

  // Get current stack location
  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft(); 
    

  allocated_terms.exc_vxc = true;
}


void XCDeviceStackData::allocate_static_data_onedft( int32_t nbf, int32_t nshells, int32_t natoms, 
  int32_t total_npts, integrator_term_tracker enabled_terms ) {

  if( allocated_terms.onedft ) 
    GAUXC_GENERIC_EXCEPTION("Attempting to reallocate Stack OneDFT");
  if( enabled_terms.ks_scheme == _UNDEF_SCHEME )
    GAUXC_GENERIC_EXCEPTION("Must have a KS Scheme set to allocate Stack OneDFT");

  // Save state
  global_dims.nshells = nshells;
  global_dims.nbf     = nbf; 
  global_dims.natoms  = natoms; 
  global_dims.total_npts  = total_npts;

  // Allocate static memory with proper alignment
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );

  static_stack.shells_device     = mem.aligned_alloc<Shell<double>>( nshells , csl);
  static_stack.exc_device        = mem.aligned_alloc<double>( 1 , csl);
  static_stack.nel_device        = mem.aligned_alloc<double>( 1 , csl);
  static_stack.acc_scr_device    = mem.aligned_alloc<double>( 1 , csl);
  static_stack.coords_device     = mem.aligned_alloc<double>( 3 * natoms, csl );

  allocated_terms.ks_scheme = enabled_terms.ks_scheme;
  static_stack.dmat_s_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
  if( not (allocated_terms.ks_scheme == RKS) ) {
      static_stack.dmat_z_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      if( allocated_terms.ks_scheme == GKS ) {
        GAUXC_GENERIC_EXCEPTION("GKS NYI for OneDFT Device");
      }
  }

  static_stack.vxc_s_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
  if( not (allocated_terms.ks_scheme == RKS) ) {
      static_stack.vxc_z_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
  }
  if (enabled_terms.ks_scheme == RKS) {
    GAUXC_GENERIC_EXCEPTION("RKS NYI for OneDFT Device");
  } else if (enabled_terms.ks_scheme == UKS) {
    static_stack.grid_weights_device = mem.aligned_alloc<double>( total_npts , csl );
    static_stack.grid_coords_device  = mem.aligned_alloc<double>( 3 * total_npts , csl );
    static_stack.den_eval_device     = mem.aligned_alloc<double>( 2 * total_npts , csl );
    static_stack.dden_eval_device    = mem.aligned_alloc<double>( 2 * 3 * total_npts , csl );
    static_stack.tau_device          = mem.aligned_alloc<double>( 2 * total_npts , csl );
    
    static_stack.den_grad_device     = mem.aligned_alloc<double>( 2 * total_npts , csl );
    static_stack.dden_grad_device    = mem.aligned_alloc<double>( 2 * 3 * total_npts , csl );
    static_stack.tau_grad_device     = mem.aligned_alloc<double>( 2 * total_npts , csl );
  }

  // Get current stack location
  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft(); 

  allocated_terms.onedft = true;
}

void XCDeviceStackData::allocate_static_data_fxc_contraction( int32_t nbf, int32_t nshells, integrator_term_tracker enabled_terms ) {

  if( allocated_terms.fxc_contraction ) 
    GAUXC_GENERIC_EXCEPTION("Attempting to reallocate Stack FXC Contraction");
  if( enabled_terms.ks_scheme == _UNDEF_SCHEME )
    GAUXC_GENERIC_EXCEPTION("Must have a KS Scheme set to allocate Stack EXC VXC");

  // Save state
  global_dims.nshells = nshells;
  global_dims.nbf     = nbf; 

  // Allocate static memory with proper alignment
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );

  static_stack.shells_device     = mem.aligned_alloc<Shell<double>>( nshells , csl);
  static_stack.nel_device        = mem.aligned_alloc<double>( 1 , csl);
  static_stack.acc_scr_device    = mem.aligned_alloc<double>( 1 , csl);
  static_stack.dmat_s_device   = mem.aligned_alloc<double>( nbf * nbf , csl );
  static_stack.tdmat_s_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
  static_stack.fxc_s_device    = mem.aligned_alloc<double>( nbf * nbf , csl );
  
  allocated_terms.ks_scheme = enabled_terms.ks_scheme;
  if( not (allocated_terms.ks_scheme == RKS) ) {
      static_stack.dmat_z_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      static_stack.tdmat_z_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      static_stack.fxc_z_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      if( allocated_terms.ks_scheme == GKS ) {
        static_stack.dmat_y_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        static_stack.dmat_x_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        static_stack.tdmat_y_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        static_stack.tdmat_x_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        static_stack.fxc_y_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        static_stack.fxc_x_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      }
  }

  // Get current stack location
  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft(); 
    

  allocated_terms.fxc_contraction = true;
}

void XCDeviceStackData::allocate_static_data_den( int32_t nbf, int32_t nshells ) {

  if( allocated_terms.den ) 
    GAUXC_GENERIC_EXCEPTION("Attempting to reallocate Stack Density");

  // Save state
  global_dims.nshells = nshells;
  global_dims.nbf     = nbf; 

  // Allocate static memory with proper alignment
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );

  static_stack.shells_device     = mem.aligned_alloc<Shell<double>>( nshells , csl);
  static_stack.acc_scr_device    = mem.aligned_alloc<double>( 1 , csl);
  static_stack.nel_device        = mem.aligned_alloc<double>( 1 , csl);

  static_stack.dmat_s_device = mem.aligned_alloc<double>( nbf * nbf , csl);

  // Get current stack location
  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft(); 

  allocated_terms.den = true;
}

void XCDeviceStackData::allocate_static_data_exc_grad( int32_t nbf, int32_t nshells, int32_t natoms, integrator_term_tracker enabled_terms ) {

  if( allocated_terms.exc_grad ) 
    GAUXC_GENERIC_EXCEPTION("Attempting to reallocate Stack EXC GRAD");

  // Save state
  global_dims.nshells = nshells;
  global_dims.nbf     = nbf; 
  global_dims.natoms  = natoms; 

  // Allocate static memory with proper alignment
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );

  static_stack.shells_device     = mem.aligned_alloc<Shell<double>>( nshells , csl);
  static_stack.exc_grad_device   = mem.aligned_alloc<double>( 3*natoms , csl);
  static_stack.nel_device        = mem.aligned_alloc<double>( 1 , csl);
  static_stack.acc_scr_device    = mem.aligned_alloc<double>( 1 , csl);

  allocated_terms.ks_scheme = enabled_terms.ks_scheme;
  static_stack.dmat_s_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
  if( not (allocated_terms.ks_scheme == RKS) ) {
      static_stack.dmat_z_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      if( allocated_terms.ks_scheme == GKS ) {
        static_stack.dmat_y_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
        static_stack.dmat_x_device  = mem.aligned_alloc<double>( nbf * nbf , csl );
      }
  }

  // Get current stack location
  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft(); 

  allocated_terms.exc_grad = true;
}


void XCDeviceStackData::allocate_static_data_exx( int32_t nbf, int32_t nshells, size_t nshell_pairs, size_t nprim_pair_total, int32_t max_l ) {

  if( allocated_terms.exx ) 
    GAUXC_GENERIC_EXCEPTION("Attempting to reallocate Stack EXX");

  // Save state
  global_dims.nshells      = nshells;
  global_dims.nshell_pairs = nshell_pairs;
  global_dims.nprim_pairs  = nprim_pair_total;
  global_dims.nbf          = nbf; 
  global_dims.max_l        = max_l; 

  // Allocate static memory with proper alignment
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );

  static_stack.shells_device = mem.aligned_alloc<Shell<double>>( nshells , csl);
  static_stack.prim_pairs_device = 
      mem.aligned_alloc<PrimitivePair<double>>(nprim_pair_total, csl);

  static_stack.exx_k_device = mem.aligned_alloc<double>( nbf * nbf , csl);
  static_stack.dmat_s_device  = mem.aligned_alloc<double>( nbf * nbf , csl);

  // Get current stack location
  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft(); 

  allocated_terms.exx = true;
}

void XCDeviceStackData::allocate_static_data_exx_ek_screening( size_t ntasks, int32_t nbf, int32_t nshells, int nshell_pairs, int32_t max_l ) {

  if( allocated_terms.exx_ek_screening ) 
    GAUXC_GENERIC_EXCEPTION("Attempting to reallocate Stack EXX-EK Screening");

  // Save state
  global_dims.nshells      = nshells;
  global_dims.nshell_pairs = nshell_pairs;
  global_dims.nbf          = nbf; 
  global_dims.max_l        = max_l; 
  global_dims.ntask_ek     = ntasks;



  // Allocate static memory with proper alignment
  buffer_adaptor mem( dynmem_ptr, dynmem_sz );

  static_stack.shells_device = mem.aligned_alloc<Shell<double>>( nshells , csl);
  static_stack.dmat_s_device   = mem.aligned_alloc<double>( nbf * nbf , csl);
  static_stack.ek_max_bfn_sum_device =
    mem.aligned_alloc<double>( ntasks , csl);
  static_stack.vshell_max_sparse_device = 
    mem.aligned_alloc<double>( nshell_pairs , csl);
  static_stack.shpair_row_ind_device = 
    mem.aligned_alloc<size_t>( nshell_pairs , csl);
  static_stack.shpair_col_ind_device = 
    mem.aligned_alloc<size_t>( nshell_pairs , csl);
  static_stack.ek_bfn_max_device = 
    mem.aligned_alloc<double>( nbf * ntasks , csl);
  static_stack.shell_to_bf_device =
    mem.aligned_alloc<int32_t>( nshells, csl );
  static_stack.shell_sizes_device =
    mem.aligned_alloc<int32_t>( nshells, csl );

  // Get current stack location
  dynmem_ptr = mem.stack();
  dynmem_sz  = mem.nleft(); 

  allocated_terms.exx_ek_screening = true;
}






void XCDeviceStackData::send_static_data_weights( const Molecule& mol, const MolMeta& meta ) {

  if( not allocated_terms.weights ) 
    GAUXC_GENERIC_EXCEPTION("Weights Not Stack Allocated");

  const auto natoms = global_dims.natoms;
  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  // Copy Atomic Coordinates
  std::vector<double> coords( 3*natoms );
  for( auto i = 0ul; i < natoms; ++i ) {
    coords[ 3*i + 0 ] = mol[i].x;
    coords[ 3*i + 1 ] = mol[i].y;
    coords[ 3*i + 2 ] = mol[i].z;
  }
  device_backend_->copy_async( 3*natoms, coords.data(), static_stack.coords_device, 
    "Coords H2D" );

  // Invert and send RAB
  const auto ldatoms = get_ldatoms();
  std::vector<double> rab_inv(natoms*natoms);
  for( auto i = 0ul; i < (natoms*natoms); ++i) rab_inv[i] = 1./meta.rab().data()[i];
  device_backend_->copy_async_2d( natoms, natoms, rab_inv.data(), natoms,
    static_stack.rab_device, ldatoms, "RAB H2D" );

  device_backend_->master_queue_synchronize(); 
}

void XCDeviceStackData::send_static_data_onedft( const Molecule& mol, const double* Ps, 
  int32_t ldps, const double* Pz, int32_t ldpz, const double* Py, 
  int32_t ldpy, const double* Px, int32_t ldpx, const BasisSet<double>& basis ) {
  
  const bool is_gks = (Pz != nullptr) and (Py != nullptr) and (Px != nullptr);
  const bool is_uks = (Pz != nullptr) and (Py == nullptr) and (Px == nullptr);
  const bool is_rks = (Ps != nullptr) and (not is_uks and not is_gks);
  if( not is_rks and not is_uks and not is_gks )
    GAUXC_GENERIC_EXCEPTION("Densities do not match RKS, UKS, or GKS schemes");

  if( not (allocated_terms.onedft ) )
    GAUXC_GENERIC_EXCEPTION("OneDFT Stack Not Allocated");

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  // Copy Atomic Coordinates
  const auto natoms = global_dims.natoms;
  std::vector<double> coords( 3*natoms );
  for( auto i = 0ul; i < natoms; ++i ) {
    coords[ 3*i + 0 ] = mol[i].x;
    coords[ 3*i + 1 ] = mol[i].y;
    coords[ 3*i + 2 ] = mol[i].z;
  }
  device_backend_->copy_async( 3*natoms, coords.data(), static_stack.coords_device, 
    "Coords H2D" );

  const auto nbf    = global_dims.nbf;
  // Check dimensions and copy density
  if( ldps != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDPs must bf NBF");
  device_backend_->copy_async( nbf*nbf, Ps, static_stack.dmat_s_device, "P_scalar H2D" );
  if( not is_rks ) {
    if( ldpz != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDPz must bf NBF");
    device_backend_->copy_async( nbf*nbf, Pz, static_stack.dmat_z_device, "P_z H2D" );
    if( is_gks ) {
      if( ldpy != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDPy must bf NBF");
      if( ldpx != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDPx must bf NBF");
      device_backend_->copy_async( nbf*nbf, Py, static_stack.dmat_y_device, "P_y H2D" );
      device_backend_->copy_async( nbf*nbf, Px, static_stack.dmat_x_device, "P_x H2D" );
    }
  }

  // Copy Basis Set
  device_backend_->copy_async( basis.nshells(), basis.data(), static_stack.shells_device,
    "Shells H2D" );

  device_backend_->master_queue_synchronize(); 

}

void XCDeviceStackData::send_static_data_onedft_results( int32_t total_npts, int32_t ndm, 
  const double* EXC, const double* DEN, const double* DDEN, const double* TAU) {

  if( not (allocated_terms.onedft ) )
    GAUXC_GENERIC_EXCEPTION("OneDFT Stack Not Allocated");

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  device_backend_->copy_async(1, EXC, static_stack.exc_device, "Copy OneDFT EXC");
  
  device_backend_->copy_async(ndm * total_npts, DEN, static_stack.den_grad_device, 
                              "Copy OneDFT den_grad_device");
  
  if (DDEN != nullptr) {
    device_backend_->copy_async(ndm * 3 * total_npts, DDEN, static_stack.dden_grad_device, 
                                "Copy OneDFT dden_grad_device");
  }
  if (TAU != nullptr) {
    device_backend_->copy_async(ndm * total_npts, TAU, static_stack.tau_grad_device,
                                "Copy OneDFT tau_grad_device");
  }
}

void XCDeviceStackData::send_static_data_density_basis( const double* Ps, int32_t ldps, const double* Pz, int32_t ldpz, const double* Py, int32_t ldpy, const double* Px, int32_t ldpx,
  const BasisSet<double>& basis ) {
  const bool is_gks = (Pz != nullptr) and (Py != nullptr) and (Px != nullptr);
  const bool is_uks = (Pz != nullptr) and (Py == nullptr) and (Px == nullptr);
  const bool is_rks = (Ps != nullptr) and (not is_uks and not is_gks);
  if( not is_rks and not is_uks and not is_gks )
    GAUXC_GENERIC_EXCEPTION("Densities do not match RKS, UKS, or GKS schemes");

  if( not (allocated_terms.exx or allocated_terms.exc_vxc or allocated_terms.exc_grad or allocated_terms.den or allocated_terms.exx_ek_screening or allocated_terms.fxc_contraction ) ) 
    GAUXC_GENERIC_EXCEPTION("Density/Basis Not Stack Allocated");

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");


  const auto nbf    = global_dims.nbf;
  // Check dimensions and copy density
  if( ldps != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDPs must bf NBF");
  device_backend_->copy_async( nbf*nbf, Ps, static_stack.dmat_s_device, "P_scalar H2D" );
  if( not is_rks ) {
    if( ldpz != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDPz must bf NBF");
    device_backend_->copy_async( nbf*nbf, Pz, static_stack.dmat_z_device, "P_z H2D" );
    if( is_gks ) {
      if( ldpy != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDPy must bf NBF");
      if( ldpx != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDPx must bf NBF");
      device_backend_->copy_async( nbf*nbf, Py, static_stack.dmat_y_device, "P_y H2D" );
      device_backend_->copy_async( nbf*nbf, Px, static_stack.dmat_x_device, "P_x H2D" );
    }
  }

  // Copy Basis Set
  device_backend_->copy_async( basis.nshells(), basis.data(), static_stack.shells_device,
    "Shells H2D" );

  device_backend_->master_queue_synchronize(); 
}


void XCDeviceStackData::send_static_data_trial_density(
  const double* tPs, int32_t ldtps, const double* tPz, int32_t ldtpz,
  const double* tPy, int32_t ldtpy, const double* tPx, int32_t ldtpx ) {

  const bool is_gks = (tPz != nullptr) && (tPy != nullptr) && (tPx != nullptr);
  const bool is_uks = (tPz != nullptr) && (tPy == nullptr) && (tPx == nullptr);
  const bool is_rks = (tPs != nullptr) && (not is_uks and not is_gks);
  if( not is_rks and not is_uks and not is_gks )
    GAUXC_GENERIC_EXCEPTION("Trial densities do not match RKS, UKS, or GKS schemes");

  if( not allocated_terms.fxc_contraction )
    GAUXC_GENERIC_EXCEPTION("Trial Density Not Stack Allocated");

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  const auto nbf = global_dims.nbf;
  // Check dimensions and copy density
  if( ldtps != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDTps must bf NBF");
  device_backend_->copy_async( nbf*nbf, tPs, static_stack.tdmat_s_device, "tP_scalar H2D" );
  if( not is_rks ) {
    if( ldtpz != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDTpz must bf NBF");
    device_backend_->copy_async( nbf*nbf, tPz, static_stack.tdmat_z_device, "tP_z H2D" );
    if( is_gks ) {
      if( ldtpy != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDTpy must bf NBF");
      if( ldtpx != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDTpx must bf NBF");
      device_backend_->copy_async( nbf*nbf, tPy, static_stack.tdmat_y_device, "tP_y H2D" );
      device_backend_->copy_async( nbf*nbf, tPx, static_stack.tdmat_x_device, "tP_x H2D" );
    }
  }
  
  device_backend_->master_queue_synchronize();
}


void XCDeviceStackData::send_static_data_shell_pairs( 
  const BasisSet<double>& basis,
  const ShellPairCollection<double>& shell_pairs ) {

  if( not allocated_terms.exx ) 
    GAUXC_GENERIC_EXCEPTION("ShellPairs Not Stack Allocated");

  const auto nshells = global_dims.nshells;
  if( shell_pairs.nshells() != nshells )
    GAUXC_GENERIC_EXCEPTION("Incompatible Basis for Stack Allocation");

  const auto nshell_pairs = global_dims.nshell_pairs;
  if( shell_pairs.npairs() != nshell_pairs )
    GAUXC_GENERIC_EXCEPTION("Incompatible ShellPairs for Stack Allocation");

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  // Copy primitive pairs
  std::vector<GauXC::PrimitivePair<double>> pp_host;
  for(const auto& sp : shell_pairs) {
    pp_host.insert( pp_host.end(), sp.prim_pairs(), sp.prim_pairs() + sp.nprim_pairs());
  }
  device_backend_->copy_async( global_dims.nprim_pairs, pp_host.data(),
    static_stack.prim_pairs_device, "PrimPairs H2D" );

  // Create SoA
  shell_pair_soa.reset();
  using point = XCDeviceShellPairSoA::point;
  const auto sp_row_ptr = shell_pairs.row_ptr();
  const auto sp_col_ind = shell_pairs.col_ind();

  shell_pair_soa.sp_row_ptr = sp_row_ptr;
  shell_pair_soa.sp_col_ind = sp_col_ind;

  GauXC::PrimitivePair<double>* prim_pair_ptr = static_stack.prim_pairs_device;
  for( auto i = 0ul, idx = 0ul; i < nshells; ++i ) {
    const auto j_st = sp_row_ptr[i];
    const auto j_en = sp_row_ptr[i+1];
    for( auto _j = j_st; _j < j_en; ++_j, idx++ ) {
      const auto j = sp_col_ind[_j];

      const auto& sp = shell_pairs.shell_pairs()[idx];
      const auto nprim_pairs = sp.nprim_pairs();
      shell_pair_soa.prim_pair_dev_ptr.emplace_back( prim_pair_ptr );
      prim_pair_ptr += nprim_pairs;

      shell_pair_soa.shell_pair_nprim_pairs.push_back(nprim_pairs);
      auto& bra = basis[i];
      auto& ket = basis[j];
      shell_pair_soa.shell_pair_shidx.emplace_back(i,j);
      shell_pair_soa.shell_pair_ls.emplace_back( bra.l(), ket.l());
      shell_pair_soa.shell_pair_centers.emplace_back(
        point{ bra.O()[0], bra.O()[1], bra.O()[2] },
        point{ ket.O()[0], ket.O()[1], ket.O()[2] }
      );
    }
  }
  
  device_backend_->master_queue_synchronize(); 
}

void XCDeviceStackData::send_static_data_exx_ek_screening( const double* V_max, 
  int32_t ldv, const BasisSetMap& basis_map, 
  const ShellPairCollection<double>& shpairs ) {

  if( not allocated_terms.exx_ek_screening ) 
    GAUXC_GENERIC_EXCEPTION("VMAX Not Stack Allocated");

  const auto nshells      = global_dims.nshells;
  const auto nshell_pairs = global_dims.nshell_pairs;
  if( ldv != (int)nshells ) GAUXC_GENERIC_EXCEPTION("LDV must bf NSHELLS");
  if( shpairs.npairs() != nshell_pairs ) 
    GAUXC_GENERIC_EXCEPTION("Inconsistent ShellPairs"); 
  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");


  // Pack VMAX
  std::vector<double> V_pack(nshell_pairs);
  const auto sp_row_ptr = shpairs.row_ptr();
  const auto sp_col_ind = shpairs.col_ind();
  for( auto i = 0ul; i < nshells; ++i ) {
    const auto j_st = sp_row_ptr[i];
    const auto j_en = sp_row_ptr[i+1];
    for( auto _j = j_st; _j < j_en; ++_j ) {
      const auto j = sp_col_ind[_j];
      V_pack[_j] = V_max[i + j*ldv];
    }
  }

  // Copy VMAX
  device_backend_->copy_async( nshell_pairs, V_pack.data(), 
    static_stack.vshell_max_sparse_device, "VMAX Sparse H2D");

  // Create sparse triplet for device
  std::vector<size_t> rowind(nshell_pairs);
  for( auto i = 0ul; i < nshells; ++i ) {
    const auto j_st = sp_row_ptr[i];
    const auto j_en = sp_row_ptr[i+1];
    for( auto _j = j_st; _j < j_en; ++_j ) {
      rowind[_j] = i;
    }
  }

  

  // Send adjacency
  device_backend_->copy_async( nshell_pairs, rowind.data(),
    static_stack.shpair_row_ind_device, "SP RowInd H2D");
  device_backend_->copy_async( nshell_pairs, sp_col_ind.data(),
    static_stack.shpair_col_ind_device, "SP ColInd H2D");

  std::vector<int32_t> shell2bf(nshells);
  std::vector<int32_t> shell_sizes(nshells);
  for(auto i = 0ul; i < nshells; ++i) {
    shell2bf[i] = basis_map.shell_to_first_ao(i);
    shell_sizes[i] = basis_map.shell_size(i);
  }
  

  device_backend_->copy_async( nshells, shell2bf.data(), static_stack.shell_to_bf_device,
    "Shell2BF H2D");
  device_backend_->copy_async( nshells, shell_sizes.data(), static_stack.shell_sizes_device,
    "ShellSizes H2D");
  
  device_backend_->master_queue_synchronize(); 

}


void XCDeviceStackData::zero_den_integrands() {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  device_backend_->set_zero( 1, static_stack.nel_device, "NEL Zero" );

}


void XCDeviceStackData::zero_exc_vxc_integrands(integrator_term_tracker enabled_terms) {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  const auto nbf = global_dims.nbf;
  if(static_stack.vxc_s_device) device_backend_->set_zero( nbf*nbf, static_stack.vxc_s_device, "VXCs Zero" );
  if(static_stack.vxc_z_device) device_backend_->set_zero( nbf*nbf, static_stack.vxc_z_device, "VXCz Zero" );
  if(static_stack.vxc_y_device) device_backend_->set_zero( nbf*nbf, static_stack.vxc_y_device, "VXCy Zero" );
  if(static_stack.vxc_x_device) device_backend_->set_zero( nbf*nbf, static_stack.vxc_x_device, "VXCx Zero" );
  device_backend_->set_zero( 1,       static_stack.exc_device, "EXC Zero" );
  device_backend_->set_zero( 1,       static_stack.nel_device, "NEL Zero" );

}

void XCDeviceStackData::zero_fxc_contraction_integrands() {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  const auto nbf = global_dims.nbf;
  if(static_stack.fxc_s_device) device_backend_->set_zero( nbf*nbf, static_stack.fxc_s_device, "FXCs Zero" );
  if(static_stack.fxc_z_device) device_backend_->set_zero( nbf*nbf, static_stack.fxc_z_device, "FXCz Zero" );
  if(static_stack.fxc_y_device) device_backend_->set_zero( nbf*nbf, static_stack.fxc_y_device, "FXCy Zero" );
  if(static_stack.fxc_x_device) device_backend_->set_zero( nbf*nbf, static_stack.fxc_x_device, "FXCx Zero" );
  device_backend_->set_zero( 1,       static_stack.nel_device, "NEL Zero" );

}

void XCDeviceStackData::zero_exc_grad_integrands() {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  const auto natoms = global_dims.natoms;
  device_backend_->set_zero( 3*natoms, static_stack.exc_grad_device, "EXC Gradient Zero" );
  device_backend_->set_zero( 1,        static_stack.nel_device, "NEL Zero" );

}


void XCDeviceStackData::zero_exx_integrands() {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  const auto nbf = global_dims.nbf;
  device_backend_->set_zero( nbf*nbf, static_stack.exx_k_device, "K Zero" );

}

void XCDeviceStackData::zero_exx_ek_screening_intermediates() {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  const auto ntask_ek = global_dims.ntask_ek;
  const auto nbf      = global_dims.nbf;
  device_backend_->set_zero( ntask_ek*nbf, static_stack.ek_bfn_max_device, "EK BFNMAX Zero" );

}


void XCDeviceStackData::retrieve_exc_vxc_integrands( double* EXC, double* N_EL,
  double* VXCs, int32_t ldvxcs, double* VXCz, int32_t ldvxcz,
  double* VXCy, int32_t ldvxcy, double* VXCx, int32_t ldvxcx ) {

  const auto nbf = global_dims.nbf;

  device_backend_->copy_async( 1,       static_stack.nel_device, N_EL, "NEL D2H" );
  device_backend_->copy_async( 1,       static_stack.exc_device, EXC,  "EXC D2H" );

  if( ldvxcs and (ldvxcs != (int)nbf) ) GAUXC_GENERIC_EXCEPTION("LDVXCs must be NBF");
  if( VXCs )
    device_backend_->copy_async( nbf*nbf, static_stack.vxc_s_device, VXCs,  "VXCs D2H" );

  if( ldvxcz and (ldvxcz != (int)nbf) ) GAUXC_GENERIC_EXCEPTION("LDVXCz must be NBF");
  if( VXCz )
    device_backend_->copy_async( nbf*nbf, static_stack.vxc_z_device, VXCz,  "VXCz D2H" );

  if( ldvxcy and (ldvxcy != (int)nbf) ) GAUXC_GENERIC_EXCEPTION("LDVXCy must be NBF");
  if( VXCy )
    device_backend_->copy_async( nbf*nbf, static_stack.vxc_y_device, VXCy,  "VXCy D2H" );

  if( ldvxcx and (ldvxcx != (int)nbf) ) GAUXC_GENERIC_EXCEPTION("LDVXCx must be NBF");
  if( VXCx )
    device_backend_->copy_async( nbf*nbf, static_stack.vxc_x_device, VXCx,  "VXCx D2H" );

}

void XCDeviceStackData::retrieve_onedft_features( int32_t total_npts, int32_t ndm, 
  double* DEN, double* DDEN, double* TAU, double* POINTS, double* WEIGHTS) {

  if( DEN )
    device_backend_->copy_async( total_npts*ndm, static_stack.den_eval_device, DEN,  "DEN D2H" );

  if( DDEN )
    device_backend_->copy_async( total_npts*ndm*3, static_stack.dden_eval_device, DDEN,  "DDEN D2H" );

  if( TAU )
    device_backend_->copy_async( total_npts*ndm, static_stack.tau_device, TAU,  "TAU D2H" );

  if( POINTS )
    device_backend_->copy_async( total_npts*3, static_stack.grid_coords_device, POINTS,  "POINTS D2H" );

  if( WEIGHTS )
    device_backend_->copy_async( total_npts, static_stack.grid_weights_device, WEIGHTS,  "WEIGHTS D2H" );

}



void XCDeviceStackData::retrieve_fxc_contraction_integrands( double* N_EL,
  double* FXCs, int32_t ldfxcs, double* FXCz, int32_t ldfxcz,
  double* FXCy, int32_t ldfxcy, double* FXCx, int32_t ldfxcx ) {

  const auto nbf = global_dims.nbf;
  device_backend_->copy_async( 1,       static_stack.nel_device, N_EL, "NEL D2H" );

  if( ldfxcs and (ldfxcs != (int)nbf) ) GAUXC_GENERIC_EXCEPTION("LDFXCs must be NBF");
  if( FXCs )
    device_backend_->copy_async( nbf*nbf, static_stack.fxc_s_device, FXCs,  "FXCs D2H" );

  if( ldfxcz and (ldfxcz != (int)nbf) ) GAUXC_GENERIC_EXCEPTION("LDFXCz must be NBF");
  if( FXCz )
    device_backend_->copy_async( nbf*nbf, static_stack.fxc_z_device, FXCz,  "FXCz D2H" );

  if( ldfxcy and (ldfxcy != (int)nbf) ) GAUXC_GENERIC_EXCEPTION("LDFXCy must be NBF");
  if( FXCy )
    device_backend_->copy_async( nbf*nbf, static_stack.fxc_y_device, FXCy,  "FXCy D2H" );

  if( ldfxcx and (ldfxcx != (int)nbf) ) GAUXC_GENERIC_EXCEPTION("LDFXCx must be NBF");
  if( FXCx )
    device_backend_->copy_async( nbf*nbf, static_stack.fxc_x_device, FXCx,  "FXCx D2H" );

}

void XCDeviceStackData::retrieve_den_integrands( double* N_EL ) {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");
  
  device_backend_->copy_async( 1, static_stack.nel_device, N_EL, "NEL D2H" );

}

void XCDeviceStackData::retrieve_exc_grad_integrands( double* EXC_GRAD, double* N_EL ) {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");
  
  const auto natoms = global_dims.natoms;
  device_backend_->copy_async( 3*natoms, static_stack.exc_grad_device, EXC_GRAD,  "EXC Gradient D2H" );
  device_backend_->copy_async( 1,        static_stack.nel_device,      N_EL,      "NEL D2H" );

}

void XCDeviceStackData::retrieve_exx_integrands( double* K, int32_t ldk ) {

  const auto nbf = global_dims.nbf;
  if( ldk != (int)nbf ) GAUXC_GENERIC_EXCEPTION("LDK must bf NBF");
  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");
  
  device_backend_->copy_async( nbf*nbf, static_stack.exx_k_device, K,  "K D2H" );

}

void XCDeviceStackData::retrieve_exx_ek_max_bfn_sum( double* MBS, int32_t nt ) {

  const auto ntask_ek = global_dims.ntask_ek;
  if( nt != (int)ntask_ek ) GAUXC_GENERIC_EXCEPTION("Inconsistent Task Count");
  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  device_backend_->copy_async( ntask_ek , static_stack.ek_max_bfn_sum_device, MBS, 
    "MBS D2H");

}






XCDeviceStackData::host_task_iterator XCDeviceStackData::generate_buffers(
  integrator_term_tracker terms,
  const BasisSetMap& basis_map,
  host_task_iterator task_begin,
  host_task_iterator task_end
) {

  if( get_static_mem_requirement() > dynmem_sz )
    GAUXC_GENERIC_EXCEPTION("Insufficient memory to even start!");

  size_t mem_left = dynmem_sz - get_static_mem_requirement();

  // Determine the number of batches that will fit into device memory
  host_task_iterator task_it = task_begin;
  while( task_it != task_end ) {

    // Get memory requirement for batch
    size_t mem_req_batch = get_mem_req( terms, *task_it );

    // Break out of loop if we can't allocate for this batch
    if( mem_req_batch > mem_left ) break;

    // Update remaining memory and increment task iterator
    mem_left -= mem_req_batch;
    task_it++;

  }

  // TODO: print this if verbose
  //std::cout << "XCDeviceStackData will allocate for " << std::distance(task_begin, task_it) << " Tasks MEMLEFT = " << mem_left << std::endl;

  // Pack host data and send to device
  allocate_dynamic_stack( terms, task_begin, task_it,
    device_buffer_t{dynmem_ptr, dynmem_sz} );

  pack_and_send( terms, task_begin, task_it, basis_map );

  return task_it;
}





size_t XCDeviceStackData::get_mem_req( 
  integrator_term_tracker terms,
  const host_task_type& task
) {

  const auto& points = task.points;
  const size_t npts  = points.size();

  required_term_storage reqt(terms);
  
  size_t mem_req = 
    // Grid
    reqt.grid_points_size (npts)  * sizeof(double) + 
    reqt.grid_weights_size(npts)  * sizeof(double) +

    // U Variables
    reqt.grid_den_size(npts)      * sizeof(double) + 
    reqt.grid_den_grad_size(npts) * sizeof(double) +
    reqt.grid_lapl_size(npts)     * sizeof(double) +

    // H/K Matrices (GKS)
    reqt.grid_HK_size(npts)       * sizeof(double) +

    // V Variables
    reqt.grid_gamma_size(npts)    * sizeof(double) +
    reqt.grid_tau_size(npts)      * sizeof(double) +

    // XC output
    reqt.grid_eps_size(npts)      * sizeof(double) +
    reqt.grid_vrho_size(npts)     * sizeof(double) +
    reqt.grid_vgamma_size(npts)   * sizeof(double) +
    reqt.grid_vtau_size(npts)     * sizeof(double) +
    reqt.grid_vlapl_size(npts)    * sizeof(double) ;

    // second derivatives
    mem_req += 
    // U variables
    reqt.grid_tden_size(npts)      * sizeof(double) +
    reqt.grid_tden_grad_size(npts) * sizeof(double) +
    reqt.grid_tlapl_size(npts)     * sizeof(double) +
    reqt.grid_ttau_size(npts)      * sizeof(double) +
    // XC output
    reqt.grid_v2rho2_size(npts)    * sizeof(double) +
    reqt.grid_v2rhogamma_size(npts)  * sizeof(double) +
    reqt.grid_v2rholapl_size(npts)   * sizeof(double) +
    reqt.grid_v2rhotau_size(npts)  * sizeof(double) +
    reqt.grid_v2gamma2_size(npts) * sizeof(double) +
    reqt.grid_v2gammalapl_size(npts) * sizeof(double) +
    reqt.grid_v2gammatau_size(npts) * sizeof(double) +
    reqt.grid_v2lapl2_size(npts) * sizeof(double) +
    reqt.grid_v2lapltau_size(npts) * sizeof(double) +
    reqt.grid_v2tau2_size(npts) * sizeof(double) +
    // intermediate output
    reqt.grid_FXC_A_size(npts) * sizeof(double) +
    reqt.grid_FXC_B_size(npts) * sizeof(double) +
    reqt.grid_FXC_C_size(npts) * sizeof(double);

  return mem_req;
}









XCDeviceStackData::device_buffer_t XCDeviceStackData::allocate_dynamic_stack( 
  integrator_term_tracker terms,
  host_task_iterator task_begin, host_task_iterator task_end, 
  device_buffer_t buf ) {


  // Get total npts
  total_npts_task_batch = std::accumulate( task_begin, task_end, 0ul,
    [](const auto& a, const auto& b){ return a + b.points.size(); } );

  // Allocate device memory
  auto [ ptr, sz ] = buf;
  buffer_adaptor mem( ptr, sz );


  required_term_storage reqt(terms);
  const size_t msz = total_npts_task_batch;
  const size_t aln = 256;
  
  const bool is_rks = terms.ks_scheme == RKS;
  const bool is_uks = terms.ks_scheme == UKS;
  const bool is_gks = terms.ks_scheme == GKS;
  const bool is_pol = is_uks or is_gks;
  const bool is_gga = terms.xc_approx == GGA;

  const bool is_den = terms.den;
  
  // Grid Points
  if( reqt.grid_points ) {
    base_stack.points_x_device = mem.aligned_alloc<double>( msz, aln, csl);
    base_stack.points_y_device = mem.aligned_alloc<double>( msz, aln, csl);
    base_stack.points_z_device = mem.aligned_alloc<double>( msz, aln, csl);
  }


  // Grid Weights
  if( reqt.grid_weights ) {
    base_stack.weights_device = mem.aligned_alloc<double>(msz, csl);
  }

  // Grid function evaluations
  if( reqt.grid_den ) { // Density 
    base_stack.den_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);

    if(is_pol) {
      base_stack.den_interleaved_device = mem.aligned_alloc<double>(2*msz, aln, csl);
      base_stack.den_z_eval_device      = mem.aligned_alloc<double>(msz, aln, csl);
    }

    if(is_gks){   
      base_stack.den_y_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
      base_stack.den_x_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
    }
  }

  if( reqt.grid_den_grad ) { // Density gradient
    base_stack.dden_sx_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    base_stack.dden_sy_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    base_stack.dden_sz_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 

    if(is_pol) { 
      base_stack.dden_zx_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.dden_zy_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
      base_stack.dden_zz_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
    }
    if( is_gks ) { 
      base_stack.dden_yx_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.dden_yy_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
      base_stack.dden_yz_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
      base_stack.dden_xx_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.dden_xy_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
      base_stack.dden_xz_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
    }
  }

  if( reqt.grid_tau ) { // Tau 
    base_stack.tau_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    if(is_pol) {
      base_stack.tau_interleaved_device = mem.aligned_alloc<double>(2*msz, aln, csl);
      base_stack.tau_z_eval_device      = mem.aligned_alloc<double>(msz, aln, csl);
    } 
  }

  if( reqt.grid_lapl ) { // Density Laplacian
    base_stack.lapl_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    if(is_pol) {
      base_stack.lapl_interleaved_device = mem.aligned_alloc<double>(2*msz, aln, csl);
      base_stack.lapl_z_eval_device      = mem.aligned_alloc<double>(msz, aln, csl);
    } 
  }

  if( reqt.grid_gamma ) { // Gamma
    if( is_pol  ) {  
      base_stack.gamma_eval_device    = mem.aligned_alloc<double>(3 * msz, aln, csl);
      base_stack.gamma_pp_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.gamma_pm_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.gamma_mm_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
    } else {           
      base_stack.gamma_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    }
  }

  if( reqt.grid_vrho ) { // Vrho
    if( is_pol  ) { 
      base_stack.vrho_eval_device     = mem.aligned_alloc<double>(2 * msz, aln, csl);
      base_stack.vrho_pos_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.vrho_neg_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
    } else {          
      base_stack.vrho_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    }
  }

  if( reqt.grid_vgamma ) { // Vgamma
    if( is_pol  ) {  
      base_stack.vgamma_eval_device    = mem.aligned_alloc<double>(3*msz, aln, csl);
      base_stack.vgamma_pp_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.vgamma_pm_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.vgamma_mm_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
    } else {
      base_stack.vgamma_eval_device    = mem.aligned_alloc<double>(msz, aln, csl);
    }
  }

  if( is_gks ) {       // H, K matrices
    base_stack.K_x_eval_device   = mem.aligned_alloc<double>(msz, aln, csl);
    base_stack.K_y_eval_device   = mem.aligned_alloc<double>(msz, aln, csl);
    base_stack.K_z_eval_device   = mem.aligned_alloc<double>(msz, aln, csl);
    if( is_gga ) {
      base_stack.H_x_eval_device   = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.H_y_eval_device   = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.H_z_eval_device   = mem.aligned_alloc<double>(msz, aln, csl);
    }
  }

  if( reqt.grid_eps ) { // Energy density 
    base_stack.eps_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
  }

  if( reqt.grid_vtau ) { // Vtau
    if( is_pol  ) { 
      base_stack.vtau_eval_device     = mem.aligned_alloc<double>(2 * msz, aln, csl);
      base_stack.vtau_pos_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.vtau_neg_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
    } else {          
      base_stack.vtau_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    }
  }

  if( reqt.grid_vlapl ) { // Vlapl
    if( is_pol  ) { 
      base_stack.vlapl_eval_device     = mem.aligned_alloc<double>(2 * msz, aln, csl);
      base_stack.vlapl_pos_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.vlapl_neg_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
    } else {          
      base_stack.vlapl_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    }
  }

  if( terms.fxc_contraction ) {
    // Trial density evaluation
    if( reqt.grid_tden ) { 
      base_stack.tden_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      if(is_pol) {
        base_stack.tden_z_eval_device      = mem.aligned_alloc<double>(msz, aln, csl);
      }
      if(is_gks){
        base_stack.tden_y_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
        base_stack.tden_x_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
      }
    }

    // Trial density gradient
    if( reqt.grid_tden_grad ) {
      base_stack.tdden_sx_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.tdden_sy_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.tdden_sz_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 

      if(is_pol) { 
        base_stack.tdden_zx_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.tdden_zy_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
        base_stack.tdden_zz_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
      }
      if( is_gks ) { 
        base_stack.tdden_yx_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.tdden_yy_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
        base_stack.tdden_yz_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
        base_stack.tdden_xx_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.tdden_xy_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
        base_stack.tdden_xz_eval_device = mem.aligned_alloc<double>(msz, aln, csl); 
      }
    }

    // Trial tau
    if( reqt.grid_ttau ) {
      base_stack.ttau_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      if(is_pol) {
        base_stack.ttau_z_eval_device      = mem.aligned_alloc<double>(msz, aln, csl);
      } 
    }

    // Trial laplacian
    if( reqt.grid_tlapl ) {
      base_stack.tlapl_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      if(is_pol) {
        base_stack.tlapl_z_eval_device      = mem.aligned_alloc<double>(msz, aln, csl);
      } 
    }

    // Second derivatives of XC functional
    if( reqt.grid_v2rho2 ) {
      if( is_pol  ) { 
        base_stack.v2rho2_eval_device = mem.aligned_alloc<double>(3 * msz, aln, csl);
        base_stack.v2rho2_a_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rho2_a_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rho2_b_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2rho2_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2rhogamma ) {
      if( is_pol  ) { 
        base_stack.v2rhogamma_eval_device = mem.aligned_alloc<double>(6 * msz, aln, csl);
        base_stack.v2rhogamma_a_aa_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rhogamma_a_ab_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rhogamma_a_bb_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rhogamma_b_aa_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rhogamma_b_ab_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rhogamma_b_bb_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2rhogamma_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2rholapl ) {
      if( is_pol  ) { 
        base_stack.v2rholapl_eval_device = mem.aligned_alloc<double>(4 * msz, aln, csl);
        base_stack.v2rholapl_a_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rholapl_a_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rholapl_b_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rholapl_b_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2rholapl_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2rhotau ) {
      if( is_pol  ) { 
        base_stack.v2rhotau_eval_device = mem.aligned_alloc<double>(4 * msz, aln, csl);
        base_stack.v2rhotau_a_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rhotau_a_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rhotau_b_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2rhotau_b_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2rhotau_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2gamma2 ) {
      if( is_pol  ) { 
        base_stack.v2gamma2_eval_device = mem.aligned_alloc<double>(6 * msz, aln, csl);
        base_stack.v2gamma2_aa_aa_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gamma2_aa_ab_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gamma2_aa_bb_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gamma2_ab_ab_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gamma2_ab_bb_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gamma2_bb_bb_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2gamma2_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2gammalapl ) {
      if( is_pol  ) { 
        base_stack.v2gammalapl_eval_device = mem.aligned_alloc<double>(6 * msz, aln, csl);
        base_stack.v2gammalapl_aa_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammalapl_aa_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammalapl_ab_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammalapl_ab_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammalapl_bb_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammalapl_bb_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2gammalapl_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2gammatau ) {
      if( is_pol  ) { 
        base_stack.v2gammatau_eval_device = mem.aligned_alloc<double>(6 * msz, aln, csl);
        base_stack.v2gammatau_aa_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammatau_aa_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammatau_ab_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammatau_ab_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammatau_bb_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2gammatau_bb_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2gammatau_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2lapl2 ) {
      if( is_pol  ) { 
        base_stack.v2lapl2_eval_device = mem.aligned_alloc<double>(3 * msz, aln, csl);
        base_stack.v2lapl2_a_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2lapl2_a_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2lapl2_b_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2lapl2_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2lapltau ) {
      if( is_pol  ) { 
        base_stack.v2lapltau_eval_device = mem.aligned_alloc<double>(4 * msz, aln, csl);
        base_stack.v2lapltau_a_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2lapltau_a_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2lapltau_b_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2lapltau_b_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2lapltau_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_v2tau2 ) {
      if( is_pol  ) { 
        base_stack.v2tau2_eval_device = mem.aligned_alloc<double>(3 * msz, aln, csl);
        base_stack.v2tau2_a_a_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2tau2_a_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.v2tau2_b_b_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      } else {          
        base_stack.v2tau2_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    // Intermediate matrices for contraction
    if( reqt.grid_FXC_A ) {
      base_stack.FXC_A_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      if( is_pol  ) 
        base_stack.FXC_A_z_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    }
    
    if( reqt.grid_FXC_B ) {
      base_stack.FXC_Bx_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.FXC_By_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      base_stack.FXC_Bz_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      if( is_pol  ) { 
        base_stack.FXC_Bx_z_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.FXC_By_z_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
        base_stack.FXC_Bz_z_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      }
    }

    if( reqt.grid_FXC_C ) {
      base_stack.FXC_C_s_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
      if( is_pol  ) 
        base_stack.FXC_C_z_eval_device = mem.aligned_alloc<double>(msz, aln, csl);
    }
  }



  // Update dynmem data for derived impls
  return device_buffer_t{ mem.stack(), mem.nleft() };
}

void XCDeviceStackData::pack_and_send( integrator_term_tracker terms,
  host_task_iterator task_begin, host_task_iterator task_end, const BasisSetMap& ) {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  // Host data packing arrays
  std::vector<double> points_x_pack, points_y_pack, points_z_pack;
  std::vector< double > weights_pack;

  // Contatenation utility
  auto concat_iterable = []( auto& a, const auto& b ) {
    a.insert( a.end(), b.begin(), b.end() );
  };

  // Pack points / weights
  for( auto it = task_begin; it != task_end; ++it ) {

    const auto& points  = it->points;
    const auto& weights = it->weights;

    //concat_iterable( points_pack,  points  );
    std::vector<double> pts_x, pts_y, pts_z;
    for( auto pt : points ) {
      pts_x.emplace_back( pt[0] );
      pts_y.emplace_back( pt[1] );
      pts_z.emplace_back( pt[2] );
    }
    concat_iterable( points_x_pack, pts_x );
    concat_iterable( points_y_pack, pts_y );
    concat_iterable( points_z_pack, pts_z );

    concat_iterable( weights_pack, weights );
    
  } // Loop over tasks

  if( points_x_pack.size() != total_npts_task_batch )
    GAUXC_GENERIC_EXCEPTION("Inconsistent Points-X allocation");
  if( points_y_pack.size() != total_npts_task_batch )
    GAUXC_GENERIC_EXCEPTION("Inconsistent Points-Y allocation");
  if( points_z_pack.size() != total_npts_task_batch )
    GAUXC_GENERIC_EXCEPTION("Inconsistent Points-Z allocation");
  if( weights_pack.size() != total_npts_task_batch )
    GAUXC_GENERIC_EXCEPTION("Inconsistent weights allocation");



  // Send grid data
  device_backend_->copy_async( points_x_pack.size(), points_x_pack.data(),
              base_stack.points_x_device, "send points_x buffer" );
  device_backend_->copy_async( points_y_pack.size(), points_y_pack.data(),
              base_stack.points_y_device, "send points_y buffer" );
  device_backend_->copy_async( points_z_pack.size(), points_z_pack.data(),
              base_stack.points_z_device, "send points_z buffer" );
  device_backend_->copy_async( weights_pack.size(), weights_pack.data(),
              base_stack.weights_device, "send weights buffer" );


  // Synchronize on the copy stream to keep host vecs in scope
  device_backend_->master_queue_synchronize(); 

}


void XCDeviceStackData::copy_weights_to_tasks( host_task_iterator task_begin, host_task_iterator task_end ) {

  if( not device_backend_ ) GAUXC_GENERIC_EXCEPTION("Invalid Device Backend");

  // Sanity check that npts is consistent
  size_t local_npts = std::accumulate( task_begin, task_end, 0ul, 
    []( const auto& a, const auto& b ) { return a + b.points.size(); } );

  if( local_npts != total_npts_task_batch )
    GAUXC_GENERIC_EXCEPTION("NPTS Mismatch");

  // Copy weights into contiguous host data
  std::vector<double> weights_host(local_npts);
  device_backend_->copy_async( local_npts, base_stack.weights_device, 
    weights_host.data(), "Weights D2H" );
  device_backend_->master_queue_synchronize(); 

  // Place into host memory 
  auto* weights_ptr = weights_host.data();
  for( auto it = task_begin; it != task_end; ++it ) {
    const auto npts = it->points.size();
    std::copy_n( weights_ptr, npts, it->weights.data() );
    weights_ptr += npts;
  }

}


}
