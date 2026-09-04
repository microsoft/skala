!> @brief SkalaXC public Fortran API.
!>
!> ABI isolation contract: this module binds, via iso_c_binding, ONLY to the
!> SkalaXC C API (opaque handles + status codes + raw arrays). No GauXC or
!> LibTorch type is ever referenced. A Fortran consumer needs only this module
!> and libskalaxc -- never GauXC.
!>
!> The module mirrors the SkalaXC C++ / C pipeline with one opaque derived type
!> per stage: runtime environment -> molecule / basis set -> molecular grid ->
!> load balancer -> molecular weights -> functional -> XC integrator.
module skalaxc
   use, intrinsic :: iso_c_binding
   use, intrinsic :: iso_fortran_env, only: error_unit
   implicit none
   private

   ! Public API ---------------------------------------------------------------
   public :: skalaxc_runtime_environment_t
   public :: skalaxc_molecule_t
   public :: skalaxc_basisset_t
   public :: skalaxc_molgrid_t
   public :: skalaxc_load_balancer_t
   public :: skalaxc_molecular_weights_t
   public :: skalaxc_functional_t
   public :: skalaxc_xc_integrator_t
   public :: SKALAXC_SUCCESS, SKALAXC_ERROR, SKALAXC_INVALID_ARGUMENT
   public :: skalaxc_version
   public :: skalaxc_last_error
   public :: skalaxc_grid_settings_t, skalaxc_grid_settings_default
   public :: skalaxc_device_runtime_settings_t, &
      & skalaxc_device_runtime_settings_default
   public :: skalaxc_timing_settings_t, skalaxc_timing_settings_default
   public :: skalaxc_integrator_settings_t, skalaxc_integrator_settings_default
   public :: skalaxc_timing_value_t, skalaxc_diagnostics_snapshot_t
   public :: skalaxc_runtime_environment_create

   integer(c_int), parameter :: SKALAXC_SUCCESS = 0
   integer(c_int), parameter :: SKALAXC_ERROR = 1
   integer(c_int), parameter :: SKALAXC_INVALID_ARGUMENT = 2

   ! Enumerations -------------------------------------------------------------
   public :: skalaxc_radialquad_becke, skalaxc_radialquad_mura_knowles, &
     & skalaxc_radialquad_murray_handy_laming, skalaxc_radialquad_treutler_ahlrichs
   enum, bind(c)
      !> @brief Becke radial quadrature
      enumerator :: skalaxc_radialquad_becke
      !> @brief Mura-Knowles radial quadrature (default)
      enumerator :: skalaxc_radialquad_mura_knowles
      !> @brief Murray-Handy-Laming radial quadrature
      enumerator :: skalaxc_radialquad_murray_handy_laming
      !> @brief Treutler-Ahlrichs radial quadrature
      enumerator :: skalaxc_radialquad_treutler_ahlrichs
   end enum

   !> @brief Named-constant bundle (e.g. skalaxc_radialquad%mura_knowles).
   type :: skalaxc_radialquad_enum
      integer(c_int) :: becke = skalaxc_radialquad_becke
      integer(c_int) :: mura_knowles = skalaxc_radialquad_mura_knowles
      integer(c_int) :: murray_handy_laming = skalaxc_radialquad_murray_handy_laming
      integer(c_int) :: treutler_ahlrichs = skalaxc_radialquad_treutler_ahlrichs
   end type skalaxc_radialquad_enum
   type(skalaxc_radialquad_enum), parameter, public :: &
     & skalaxc_radialquad = skalaxc_radialquad_enum()

   public :: skalaxc_atomicgridsize_finegrid, skalaxc_atomicgridsize_ultrafinegrid, &
     & skalaxc_atomicgridsize_superfinegrid, skalaxc_atomicgridsize_gm3, &
     & skalaxc_atomicgridsize_gm5
   enum, bind(c)
      !> @brief Fine grid (least accurate)
      enumerator :: skalaxc_atomicgridsize_finegrid
      !> @brief Ultrafine grid (default)
      enumerator :: skalaxc_atomicgridsize_ultrafinegrid
      !> @brief Superfine grid (most accurate)
      enumerator :: skalaxc_atomicgridsize_superfinegrid
      !> @brief Treutler-Ahlrichs GM3 grid
      enumerator :: skalaxc_atomicgridsize_gm3
      !> @brief Treutler-Ahlrichs GM5 grid
      enumerator :: skalaxc_atomicgridsize_gm5
   end enum

   !> @brief Named-constant bundle (e.g. skalaxc_atomicgridsize%ultrafinegrid).
   type :: skalaxc_atomicgridsize_enum
      integer(c_int) :: finegrid = skalaxc_atomicgridsize_finegrid
      integer(c_int) :: ultrafinegrid = skalaxc_atomicgridsize_ultrafinegrid
      integer(c_int) :: superfinegrid = skalaxc_atomicgridsize_superfinegrid
      integer(c_int) :: gm3 = skalaxc_atomicgridsize_gm3
      integer(c_int) :: gm5 = skalaxc_atomicgridsize_gm5
   end type skalaxc_atomicgridsize_enum
   type(skalaxc_atomicgridsize_enum), parameter, public :: &
     & skalaxc_atomicgridsize = skalaxc_atomicgridsize_enum()

   public :: skalaxc_pruningscheme_unpruned, skalaxc_pruningscheme_robust, &
     & skalaxc_pruningscheme_treutler
   enum, bind(c)
      !> @brief Unpruned atomic quadrature (default)
      enumerator :: skalaxc_pruningscheme_unpruned
      !> @brief The "Robust" scheme of Psi4
      enumerator :: skalaxc_pruningscheme_robust
      !> @brief The Treutler-Ahlrichs scheme
      enumerator :: skalaxc_pruningscheme_treutler
   end enum

   !> @brief Named-constant bundle (e.g. skalaxc_pruningscheme%unpruned).
   type :: skalaxc_pruningscheme_enum
      integer(c_int) :: unpruned = skalaxc_pruningscheme_unpruned
      integer(c_int) :: robust = skalaxc_pruningscheme_robust
      integer(c_int) :: treutler = skalaxc_pruningscheme_treutler
   end type skalaxc_pruningscheme_enum
   type(skalaxc_pruningscheme_enum), parameter, public :: &
     & skalaxc_pruningscheme = skalaxc_pruningscheme_enum()

   public :: skalaxc_executionspace_host, skalaxc_executionspace_device
   enum, bind(c)
      !> @brief Host (CPU) evaluation (supported)
      enumerator :: skalaxc_executionspace_host
      !> @brief CUDA device evaluation
      enumerator :: skalaxc_executionspace_device
   end enum

   !> @brief Named-constant bundle (e.g. skalaxc_executionspace%host).
   type :: skalaxc_executionspace_enum
      integer(c_int) :: host = skalaxc_executionspace_host
      integer(c_int) :: device = skalaxc_executionspace_device
   end type skalaxc_executionspace_enum
   type(skalaxc_executionspace_enum), parameter, public :: &
     & skalaxc_executionspace = skalaxc_executionspace_enum()

   public :: skalaxc_domainbatchmode_conservative, &
     & skalaxc_domainbatchmode_aggressive
   enum, bind(c)
      !> @brief Evaluate one complete atomic domain per model call.
      enumerator :: skalaxc_domainbatchmode_conservative
      !> @brief Batch all local domains having the same exact grid size.
      enumerator :: skalaxc_domainbatchmode_aggressive
   end enum

   !> @brief Named-constant bundle for complete-domain model batching.
   type :: skalaxc_domainbatchmode_enum
      integer(c_int) :: conservative = skalaxc_domainbatchmode_conservative
      integer(c_int) :: aggressive = skalaxc_domainbatchmode_aggressive
   end type skalaxc_domainbatchmode_enum
   type(skalaxc_domainbatchmode_enum), parameter, public :: &
     & skalaxc_domainbatchmode = skalaxc_domainbatchmode_enum()

   public :: skalaxc_xcweightalg_notpartitioned, skalaxc_xcweightalg_becke, &
     & skalaxc_xcweightalg_ssf, skalaxc_xcweightalg_lko
   enum, bind(c)
      !> @brief Weights are not partitioned
      enumerator :: skalaxc_xcweightalg_notpartitioned
      !> @brief Becke partitioning
      enumerator :: skalaxc_xcweightalg_becke
      !> @brief Stratmann-Scuseria-Frisch (default)
      enumerator :: skalaxc_xcweightalg_ssf
      !> @brief Laqua-Kussmann-Ochsenfeld
      enumerator :: skalaxc_xcweightalg_lko
   end enum

   !> @brief Named-constant bundle (e.g. skalaxc_xcweightalg%ssf).
   type :: skalaxc_xcweightalg_enum
      integer(c_int) :: notpartitioned = skalaxc_xcweightalg_notpartitioned
      integer(c_int) :: becke = skalaxc_xcweightalg_becke
      integer(c_int) :: ssf = skalaxc_xcweightalg_ssf
      integer(c_int) :: lko = skalaxc_xcweightalg_lko
   end type skalaxc_xcweightalg_enum
   type(skalaxc_xcweightalg_enum), parameter, public :: &
     & skalaxc_xcweightalg = skalaxc_xcweightalg_enum()

   integer(c_int32_t), parameter, public :: &
      & skalaxc_timingstatus_unavailable = 0
   integer(c_int32_t), parameter, public :: skalaxc_timingstatus_pending = 1
   integer(c_int32_t), parameter, public :: skalaxc_timingstatus_complete = 2

   ! One-based indices into skalaxc_diagnostics_snapshot_t%timings.
   integer(c_int), parameter, public :: skalaxc_timing_metric_count = 11
   integer(c_int), parameter, public :: skalaxc_timing_model_load = 1
   integer(c_int), parameter, public :: skalaxc_timing_feature_construction = 2
   integer(c_int), parameter, public :: skalaxc_timing_model_batch_packing = 3
   integer(c_int), parameter, public :: skalaxc_timing_model_forward = 4
   integer(c_int), parameter, public :: skalaxc_timing_model_backward = 5
   integer(c_int), parameter, public :: skalaxc_timing_potential_mapping = 6
   integer(c_int), parameter, public :: skalaxc_timing_ao_assembly = 7
   integer(c_int), parameter, public :: skalaxc_timing_gradient_assembly = 8
   integer(c_int), parameter, public :: skalaxc_timing_mpi_reduction = 9
   integer(c_int), parameter, public :: skalaxc_timing_total_exc_vxc = 10
   integer(c_int), parameter, public :: skalaxc_timing_total_exc_gradient = 11

   !> @brief Molecular integration-grid parameters. Interoperable (bind(C)) with
   !> the C skalaxc_grid_settings_t. Obtain the built-in preset from
   !> skalaxc_grid_settings_default(), then override individual components.
   type, bind(C) :: skalaxc_grid_settings_t
      !> @brief Pruning scheme (a skalaxc_pruningscheme_* value).
      integer(c_int)     :: pruning
      !> @brief Grid-point batch size (> 0).
      integer(c_int64_t) :: batch_size
      !> @brief Radial quadrature (a skalaxc_radialquad_* value).
      integer(c_int)     :: radial_quad
      !> @brief Atomic grid size preset (a skalaxc_atomicgridsize_* value).
      integer(c_int)     :: atomic_grid
   end type skalaxc_grid_settings_t

   !> @brief CUDA device selection and GauXC memory-pool settings.
   type, bind(C) :: skalaxc_device_runtime_settings_t
      integer(c_int32_t) :: device_id
      real(c_double)     :: memory_fraction
   end type skalaxc_device_runtime_settings_t

   !> @brief Lightweight, rank-local integrator diagnostics settings.
   type, bind(C) :: skalaxc_timing_settings_t
      !> Nonzero requests complete CUDA event timings when diagnostics are read.
      integer(c_int32_t) :: verbose
      !> Nonzero emits rank-local diagnostics to stderr.
      integer(c_int32_t) :: debug_logging
   end type skalaxc_timing_settings_t

   !> @brief XC-integrator construction settings.
   type, bind(C) :: skalaxc_integrator_settings_t
      type(skalaxc_timing_settings_t) :: timing
      integer(c_int)                  :: domain_batch_mode
   end type skalaxc_integrator_settings_t

   !> @brief Last and cumulative values for one timing phase.
   type, bind(C) :: skalaxc_timing_value_t
      integer(c_int64_t) :: last_nanoseconds
      integer(c_int64_t) :: total_nanoseconds
      integer(c_int64_t) :: call_count
      integer(c_int32_t) :: status
   end type skalaxc_timing_value_t

   !> @brief Rank-local diagnostics returned by an XC integrator.
   type, bind(C) :: skalaxc_diagnostics_snapshot_t
      integer(c_int32_t) :: backend
      integer(c_int32_t) :: rank
      integer(c_int32_t) :: communicator_size
      integer(c_int32_t) :: device_id
      integer(c_int32_t) :: openmp_threads
      real(c_double) :: device_memory_fraction
      integer(c_int32_t) :: domain_batch_mode
      type(skalaxc_timing_value_t) :: timings(skalaxc_timing_metric_count)
      integer(c_int64_t) :: exc_vxc_calls
      integer(c_int64_t) :: exc_gradient_calls
      integer(c_int64_t) :: model_batches
      integer(c_int64_t) :: domains
      integer(c_int64_t) :: tasks
      integer(c_int64_t) :: points
      integer(c_int64_t) :: local_atoms
      integer(c_int64_t) :: configured_model_batches
      integer(c_int64_t) :: task_points_min
      integer(c_int64_t) :: task_points_max
      integer(c_int64_t) :: task_basis_min
      integer(c_int64_t) :: task_basis_max
      integer(c_int64_t) :: model_batch_points_min
      integer(c_int64_t) :: model_batch_points_max
      integer(c_int64_t) :: max_domains_per_model_batch
   end type skalaxc_diagnostics_snapshot_t

   ! ---------------------------------------------------------------------------
   ! Pipeline handle types (each uniquely owns an opaque C handle).
   ! Assignment is forbidden because copying would duplicate ownership. Use
   ! destination%move_from(source) to transfer ownership and invalidate source.
   ! ---------------------------------------------------------------------------

   !> @brief Runtime environment handle (mirrors SkalaXC::RuntimeEnvironment).
   type :: skalaxc_runtime_environment_t
      private
      type(c_ptr) :: handle = c_null_ptr
   contains
      procedure :: comm_rank => rt_comm_rank
      procedure :: comm_size => rt_comm_size
      procedure :: is_valid => rt_is_valid
      procedure :: move_from => rt_move_from
      procedure, private :: rt_assign
      generic, public :: assignment(=) => rt_assign
      final :: rt_destroy
   end type skalaxc_runtime_environment_t

   !> @brief Molecule handle (mirrors SkalaXC::Molecule).
   type :: skalaxc_molecule_t
      private
      type(c_ptr) :: handle = c_null_ptr
   contains
      procedure :: create => mol_create
      procedure :: add_atom => mol_add_atom
      procedure :: from_arrays => mol_from_arrays
#ifdef SKALAXC_HAS_HDF5
      procedure :: from_hdf5 => mol_from_hdf5
#endif
      procedure :: natoms => mol_natoms
      procedure :: is_valid => mol_is_valid
      procedure :: move_from => mol_move_from
      procedure, private :: mol_assign
      generic, public :: assignment(=) => mol_assign
      final :: mol_destroy
   end type skalaxc_molecule_t

   !> @brief Basis-set handle (mirrors SkalaXC::BasisSet).
   type :: skalaxc_basisset_t
      private
      type(c_ptr) :: handle = c_null_ptr
   contains
      procedure :: create => bas_create
      procedure :: add_shell => bas_add_shell
      procedure :: from_arrays => bas_from_arrays
#ifdef SKALAXC_HAS_HDF5
      procedure :: from_hdf5 => bas_from_hdf5
#endif
      procedure :: nbf => bas_nbf
      procedure :: is_valid => bas_is_valid
      procedure :: move_from => bas_move_from
      procedure, private :: bas_assign
      generic, public :: assignment(=) => bas_assign
      final :: bas_destroy
   end type skalaxc_basisset_t

   !> @brief Molecular-grid handle (mirrors SkalaXC::MolGrid).
   type :: skalaxc_molgrid_t
      private
      type(c_ptr) :: handle = c_null_ptr
   contains
      procedure :: create_default => mg_create_default
      procedure :: is_valid => mg_is_valid
      procedure :: move_from => mg_move_from
      procedure, private :: mg_assign
      generic, public :: assignment(=) => mg_assign
      final :: mg_destroy
   end type skalaxc_molgrid_t

   !> @brief Load-balancer handle (mirrors SkalaXC::LoadBalancer).
   type :: skalaxc_load_balancer_t
      private
      type(c_ptr) :: handle = c_null_ptr
   contains
      procedure :: create => lb_create
      procedure :: is_valid => lb_is_valid
      procedure :: move_from => lb_move_from
      procedure, private :: lb_assign
      generic, public :: assignment(=) => lb_assign
      final :: lb_destroy
   end type skalaxc_load_balancer_t

   !> @brief Molecular-weights handle (mirrors SkalaXC::MolecularWeights).
   type :: skalaxc_molecular_weights_t
      private
      type(c_ptr) :: handle = c_null_ptr
   contains
      procedure :: create => mw_create
      procedure :: modify_weights => mw_modify_weights
      procedure :: is_valid => mw_is_valid
      procedure :: move_from => mw_move_from
      procedure, private :: mw_assign
      generic, public :: assignment(=) => mw_assign
      final :: mw_destroy
   end type skalaxc_molecular_weights_t

   !> @brief Functional handle (mirrors SkalaXC::functional_type).
   type :: skalaxc_functional_t
      private
      type(c_ptr) :: handle = c_null_ptr
   contains
      procedure :: create => func_create
      procedure :: is_valid => func_is_valid
      procedure :: move_from => func_move_from
      procedure, private :: func_assign
      generic, public :: assignment(=) => func_assign
      final :: func_destroy
   end type skalaxc_functional_t

   !> @brief XC-integrator handle (mirrors SkalaXC::XCIntegrator).
   !>
   !> One instance must not be used concurrently from multiple threads.
   !> Serialize access or use a separate integrator per calling thread.
   type :: skalaxc_xc_integrator_t
      private
      type(c_ptr) :: handle = c_null_ptr
   contains
      procedure :: create => xc_create
      procedure :: nbf => xc_nbf
      procedure :: natoms => xc_natoms
      procedure :: eval_exc_vxc_uks => xc_eval_exc_vxc_uks
      procedure :: eval_exc_grad_uks => xc_eval_exc_grad_uks
      procedure :: diagnostics => xc_diagnostics
      procedure :: reset_diagnostics => xc_reset_diagnostics
      procedure :: is_valid => xc_is_valid
      procedure :: move_from => xc_move_from
      procedure, private :: xc_assign
      generic, public :: assignment(=) => xc_assign
      final :: xc_destroy
   end type skalaxc_xc_integrator_t

   ! ---------------------------------------------------------------------------
   ! Raw C bindings (private).
   ! ---------------------------------------------------------------------------
   interface
      ! -- Runtime environment ------------------------------------------------
#ifdef SKALAXC_HAS_MPI
      function c_runtime_create_f(comm, out) &
         bind(C, name="skalaxc_runtime_environment_create_f") result(status)
         import :: c_int, c_ptr
         integer(c_int), value      :: comm
         type(c_ptr), intent(inout) :: out
         integer(c_int)             :: status
      end function c_runtime_create_f

      function c_device_runtime_create_f(comm, settings, out) &
         bind(C, name="skalaxc_device_runtime_environment_create_f") result(status)
         import :: c_int, c_ptr, skalaxc_device_runtime_settings_t
         integer(c_int), value :: comm
         type(skalaxc_device_runtime_settings_t), intent(in) :: settings
         type(c_ptr), intent(inout) :: out
         integer(c_int) :: status
      end function c_device_runtime_create_f
#else
      function c_runtime_create(out) &
         bind(C, name="skalaxc_runtime_environment_create") result(status)
         import :: c_int, c_ptr
         type(c_ptr), intent(inout) :: out
         integer(c_int)             :: status
      end function c_runtime_create

      function c_device_runtime_create(settings, out) &
         bind(C, name="skalaxc_device_runtime_environment_create") result(status)
         import :: c_int, c_ptr, skalaxc_device_runtime_settings_t
         type(skalaxc_device_runtime_settings_t), intent(in) :: settings
         type(c_ptr), intent(inout) :: out
         integer(c_int) :: status
      end function c_device_runtime_create
#endif

      function c_runtime_comm_rank(rt) &
         bind(C, name="skalaxc_runtime_environment_comm_rank") result(r)
         import :: c_ptr, c_int
         type(c_ptr), value :: rt
         integer(c_int)     :: r
      end function c_runtime_comm_rank

      function c_runtime_comm_size(rt) &
         bind(C, name="skalaxc_runtime_environment_comm_size") result(s)
         import :: c_ptr, c_int
         type(c_ptr), value :: rt
         integer(c_int)     :: s
      end function c_runtime_comm_size

      subroutine c_runtime_destroy(rt) &
         bind(C, name="skalaxc_runtime_environment_destroy")
         import :: c_ptr
         type(c_ptr), value :: rt
      end subroutine c_runtime_destroy

      ! -- Molecule -----------------------------------------------------------
      function c_molecule_create(out) &
         bind(C, name="skalaxc_molecule_create") result(status)
         import :: c_ptr, c_int
         type(c_ptr), intent(inout) :: out
         integer(c_int)             :: status
      end function c_molecule_create

      function c_molecule_add_atom(mol, atnum, x, y, z) &
         bind(C, name="skalaxc_molecule_add_atom") result(status)
         import :: c_ptr, c_int, c_int64_t, c_double
         type(c_ptr), value            :: mol
         integer(c_int64_t), value     :: atnum
         real(c_double), value         :: x, y, z
         integer(c_int)                :: status
      end function c_molecule_add_atom

      function c_molecule_from_arrays(natoms, Z, atom_xyz, out) &
         bind(C, name="skalaxc_molecule_from_arrays") result(status)
         import :: c_ptr, c_int, c_int64_t, c_double
         integer(c_int64_t), value  :: natoms
         integer(c_int64_t), intent(in) :: Z(*)
         real(c_double), intent(in)     :: atom_xyz(*)
         type(c_ptr), intent(inout) :: out
         integer(c_int)             :: status
      end function c_molecule_from_arrays

#ifdef SKALAXC_HAS_HDF5
      function c_molecule_from_hdf5(path, dset, out) &
         bind(C, name="skalaxc_molecule_from_hdf5") result(status)
         import :: c_ptr, c_int, c_char
         character(kind=c_char), intent(in) :: path(*)
         character(kind=c_char), intent(in) :: dset(*)
         type(c_ptr), intent(inout)         :: out
         integer(c_int)                     :: status
      end function c_molecule_from_hdf5
#endif

      function c_molecule_natoms(mol) &
         bind(C, name="skalaxc_molecule_natoms") result(n)
         import :: c_ptr, c_int64_t
         type(c_ptr), value :: mol
         integer(c_int64_t) :: n
      end function c_molecule_natoms

      subroutine c_molecule_destroy(mol) bind(C, name="skalaxc_molecule_destroy")
         import :: c_ptr
         type(c_ptr), value :: mol
      end subroutine c_molecule_destroy

      ! -- Basis set ----------------------------------------------------------
      function c_basisset_create(out) &
         bind(C, name="skalaxc_basisset_create") result(status)
         import :: c_ptr, c_int
         type(c_ptr), intent(inout) :: out
         integer(c_int)             :: status
      end function c_basisset_create

      function c_basisset_add_shell(basis, l, pure, center_xyz, nprim, &
                                    exponents, coefficients, normalize) &
         bind(C, name="skalaxc_basisset_add_shell") result(status)
         import :: c_ptr, c_int, c_int32_t, c_double
         type(c_ptr), value             :: basis
         integer(c_int32_t), value      :: l, pure, nprim, normalize
         real(c_double), intent(in)     :: center_xyz(*)
         real(c_double), intent(in)     :: exponents(*)
         real(c_double), intent(in)     :: coefficients(*)
         integer(c_int)                 :: status
      end function c_basisset_add_shell

      function c_basisset_from_arrays(nshells, shell_l, shell_pure, shell_xyz, &
                                      shell_nprim, prim_exp, prim_coeff, out) &
         bind(C, name="skalaxc_basisset_from_arrays") result(status)
         import :: c_ptr, c_int, c_int32_t, c_int64_t, c_double
         integer(c_int64_t), value      :: nshells
         integer(c_int32_t), intent(in) :: shell_l(*)
         integer(c_int32_t), intent(in) :: shell_pure(*)
         real(c_double), intent(in)     :: shell_xyz(*)
         integer(c_int32_t), intent(in) :: shell_nprim(*)
         real(c_double), intent(in)     :: prim_exp(*)
         real(c_double), intent(in)     :: prim_coeff(*)
         type(c_ptr), intent(inout)     :: out
         integer(c_int)                 :: status
      end function c_basisset_from_arrays

#ifdef SKALAXC_HAS_HDF5
      function c_basisset_from_hdf5(path, dset, out) &
         bind(C, name="skalaxc_basisset_from_hdf5") result(status)
         import :: c_ptr, c_int, c_char
         character(kind=c_char), intent(in) :: path(*)
         character(kind=c_char), intent(in) :: dset(*)
         type(c_ptr), intent(inout)         :: out
         integer(c_int)                     :: status
      end function c_basisset_from_hdf5
#endif

      function c_basisset_nbf(basis) &
         bind(C, name="skalaxc_basisset_nbf") result(n)
         import :: c_ptr, c_int64_t
         type(c_ptr), value :: basis
         integer(c_int64_t) :: n
      end function c_basisset_nbf

      subroutine c_basisset_destroy(basis) bind(C, name="skalaxc_basisset_destroy")
         import :: c_ptr
         type(c_ptr), value :: basis
      end subroutine c_basisset_destroy

      ! -- Molecular grid -----------------------------------------------------
      function c_molgrid_create_default(mol, grid, out) &
         bind(C, name="skalaxc_molgrid_create_default") result(status)
         import :: c_ptr, c_int
         type(c_ptr), value       :: mol
         type(c_ptr), value       :: grid
         type(c_ptr), intent(inout) :: out
         integer(c_int)           :: status
      end function c_molgrid_create_default

      subroutine c_molgrid_destroy(mg) bind(C, name="skalaxc_molgrid_destroy")
         import :: c_ptr
         type(c_ptr), value :: mg
      end subroutine c_molgrid_destroy

      ! -- Load balancer ------------------------------------------------------
      function c_load_balancer_create(ex, rt, mol, mg, basis, out) &
         bind(C, name="skalaxc_load_balancer_create") result(status)
         import :: c_ptr, c_int
         integer(c_int), value    :: ex
         type(c_ptr), value       :: rt, mol, mg, basis
         type(c_ptr), intent(inout) :: out
         integer(c_int)           :: status
      end function c_load_balancer_create

      subroutine c_load_balancer_destroy(lb) &
         bind(C, name="skalaxc_load_balancer_destroy")
         import :: c_ptr
         type(c_ptr), value :: lb
      end subroutine c_load_balancer_destroy

      ! -- Molecular weights --------------------------------------------------
      function c_molecular_weights_create(ex, weight_alg, out) &
         bind(C, name="skalaxc_molecular_weights_create") result(status)
         import :: c_ptr, c_int
         integer(c_int), value    :: ex, weight_alg
         type(c_ptr), intent(inout) :: out
         integer(c_int)           :: status
      end function c_molecular_weights_create

      function c_molecular_weights_modify_weights(mw, lb) &
         bind(C, name="skalaxc_molecular_weights_modify_weights") result(status)
         import :: c_ptr, c_int
         type(c_ptr), value :: mw, lb
         integer(c_int)     :: status
      end function c_molecular_weights_modify_weights

      subroutine c_molecular_weights_destroy(mw) &
         bind(C, name="skalaxc_molecular_weights_destroy")
         import :: c_ptr
         type(c_ptr), value :: mw
      end subroutine c_molecular_weights_destroy

      ! -- Functional ---------------------------------------------------------
      function c_functional_create(model, out) &
         bind(C, name="skalaxc_functional_create") result(status)
         import :: c_ptr, c_int, c_char
         character(kind=c_char), intent(in) :: model(*)
         type(c_ptr), intent(inout)         :: out
         integer(c_int)                     :: status
      end function c_functional_create

      subroutine c_functional_destroy(func) &
         bind(C, name="skalaxc_functional_destroy")
         import :: c_ptr
         type(c_ptr), value :: func
      end subroutine c_functional_destroy

      ! -- XC integrator ------------------------------------------------------
      function c_integrator_create(ex, func, lb, out) &
         bind(C, name="skalaxc_xc_integrator_create") result(status)
         import :: c_ptr, c_int
         integer(c_int), value      :: ex
         type(c_ptr), value         :: func, lb
         type(c_ptr), intent(inout) :: out
         integer(c_int)             :: status
      end function c_integrator_create

      function c_integrator_create_with_timing(ex, func, lb, settings, out) &
         bind(C, name="skalaxc_xc_integrator_create_with_timing") result(status)
         import :: c_ptr, c_int, skalaxc_timing_settings_t
         integer(c_int), value :: ex
         type(c_ptr), value    :: func, lb
         type(skalaxc_timing_settings_t), intent(in) :: settings
         type(c_ptr), intent(inout) :: out
         integer(c_int)             :: status
      end function c_integrator_create_with_timing

      function c_integrator_create_with_settings(ex, func, lb, settings, out) &
         bind(C, name="skalaxc_xc_integrator_create_with_settings") result(status)
         import :: c_ptr, c_int, skalaxc_integrator_settings_t
         integer(c_int), value :: ex
         type(c_ptr), value    :: func, lb
         type(skalaxc_integrator_settings_t), intent(in) :: settings
         type(c_ptr), intent(inout) :: out
         integer(c_int)             :: status
      end function c_integrator_create_with_settings

      function c_integrator_nbf(xc) &
         bind(C, name="skalaxc_xc_integrator_nbf") result(n)
         import :: c_ptr, c_int64_t
         type(c_ptr), value :: xc
         integer(c_int64_t) :: n
      end function c_integrator_nbf

      function c_integrator_natoms(xc) &
         bind(C, name="skalaxc_xc_integrator_natoms") result(n)
         import :: c_ptr, c_int64_t
         type(c_ptr), value :: xc
         integer(c_int64_t) :: n
      end function c_integrator_natoms

      function c_integrator_eval(xc, Ps, Pz, VXCs, VXCz, exc_out) &
         bind(C, name="skalaxc_xc_integrator_eval_exc_vxc_uks") result(status)
         import :: c_ptr, c_double, c_int
         type(c_ptr), value          :: xc
         real(c_double), intent(in)  :: Ps(*)
         real(c_double), intent(in)  :: Pz(*)
         real(c_double), intent(out) :: VXCs(*)
         real(c_double), intent(out) :: VXCz(*)
         real(c_double), intent(out) :: exc_out
         integer(c_int)              :: status
      end function c_integrator_eval

      function c_integrator_eval_grad(xc, Ps, Pz, gradient_out) &
         bind(C, name="skalaxc_xc_integrator_eval_exc_grad_uks") result(status)
         import :: c_ptr, c_double, c_int
         type(c_ptr), value          :: xc
         real(c_double), intent(in)  :: Ps(*)
         real(c_double), intent(in)  :: Pz(*)
         real(c_double), intent(out) :: gradient_out(*)
         integer(c_int)              :: status
      end function c_integrator_eval_grad

      function c_integrator_diagnostics(xc, snapshot) &
         bind(C, name="skalaxc_xc_integrator_get_diagnostics") result(status)
         import :: c_ptr, c_int, skalaxc_diagnostics_snapshot_t
         type(c_ptr), value :: xc
         type(skalaxc_diagnostics_snapshot_t), intent(out) :: snapshot
         integer(c_int) :: status
      end function c_integrator_diagnostics

      function c_integrator_reset_diagnostics(xc) &
         bind(C, name="skalaxc_xc_integrator_reset_diagnostics") result(status)
         import :: c_ptr, c_int
         type(c_ptr), value :: xc
         integer(c_int) :: status
      end function c_integrator_reset_diagnostics

      subroutine c_integrator_destroy(xc) &
         bind(C, name="skalaxc_xc_integrator_destroy")
         import :: c_ptr
         type(c_ptr), value :: xc
      end subroutine c_integrator_destroy

      ! -- Utilities ----------------------------------------------------------
      subroutine c_grid_defaults(settings) &
         bind(C, name="skalaxc_grid_settings_default")
         import :: skalaxc_grid_settings_t
         type(skalaxc_grid_settings_t), intent(out) :: settings
      end subroutine c_grid_defaults

      subroutine c_device_runtime_defaults(settings) &
         bind(C, name="skalaxc_device_runtime_settings_default")
         import :: skalaxc_device_runtime_settings_t
         type(skalaxc_device_runtime_settings_t), intent(out) :: settings
      end subroutine c_device_runtime_defaults

      subroutine c_timing_defaults(settings) &
         bind(C, name="skalaxc_timing_settings_default")
         import :: skalaxc_timing_settings_t
         type(skalaxc_timing_settings_t), intent(out) :: settings
      end subroutine c_timing_defaults

      subroutine c_integrator_defaults(settings) &
         bind(C, name="skalaxc_integrator_settings_default")
         import :: skalaxc_integrator_settings_t
         type(skalaxc_integrator_settings_t), intent(out) :: settings
      end subroutine c_integrator_defaults

      function c_version() bind(C, name="skalaxc_version") result(version)
         import :: c_ptr
         type(c_ptr) :: version
      end function c_version

      function c_last_error() bind(C, name="skalaxc_last_error_message") result(msg)
         import :: c_ptr
         type(c_ptr) :: msg
      end function c_last_error
   end interface

contains

   subroutine reject_handle_assignment(destination_valid, source_valid)
      logical, intent(in) :: destination_valid, source_valid
      write (error_unit, '(A,L1,A,L1,A)') &
         'SkalaXC handle assignment rejected (destination valid=', &
         destination_valid, ', source valid=', source_valid, ')'
      error stop "SkalaXC handles are non-copyable; use move_from"
   end subroutine reject_handle_assignment

   ! ==========================================================================
   ! Runtime environment
   ! ==========================================================================

   !> @brief Create the runtime environment.
   !> @param rt Runtime-environment handle to initialize.
   !> @param comm MPI communicator handle (MPI builds only).
   !> @return SKALAXC_SUCCESS on success, or an error status.
#ifdef SKALAXC_HAS_MPI
   function skalaxc_runtime_environment_create(rt, comm, device_settings) result(status)
      type(skalaxc_runtime_environment_t), intent(inout) :: rt
      integer, intent(in)                                :: comm
      type(skalaxc_device_runtime_settings_t), intent(in), optional :: device_settings
      integer(c_int)                                     :: status
      type(c_ptr)                                        :: new_handle
      new_handle = c_null_ptr
      if (present(device_settings)) then
         status = c_device_runtime_create_f(int(comm, c_int), &
           & device_settings, new_handle)
      else
         status = c_runtime_create_f(int(comm, c_int), new_handle)
      end if
      if (status == SKALAXC_SUCCESS) then
         call rt_destroy(rt)
         rt%handle = new_handle
      end if
   end function skalaxc_runtime_environment_create
#else
   function skalaxc_runtime_environment_create(rt, device_settings) result(status)
      type(skalaxc_runtime_environment_t), intent(inout) :: rt
      type(skalaxc_device_runtime_settings_t), intent(in), optional :: device_settings
      integer(c_int)                                     :: status
      type(c_ptr)                                        :: new_handle
      new_handle = c_null_ptr
      if (present(device_settings)) then
         status = c_device_runtime_create(device_settings, new_handle)
      else
         status = c_runtime_create(new_handle)
      end if
      if (status == SKALAXC_SUCCESS) then
         call rt_destroy(rt)
         rt%handle = new_handle
      end if
   end function skalaxc_runtime_environment_create
#endif

   !> @brief Calling rank within the environment (0 in serial builds).
   function rt_comm_rank(this) result(r)
      class(skalaxc_runtime_environment_t), intent(in) :: this
      integer(c_int)                                   :: r
      r = c_runtime_comm_rank(this%handle)
   end function rt_comm_rank

   !> @brief Number of cooperating ranks (1 in serial builds).
   function rt_comm_size(this) result(s)
      class(skalaxc_runtime_environment_t), intent(in) :: this
      integer(c_int)                                   :: s
      s = c_runtime_comm_size(this%handle)
   end function rt_comm_size

   !> @brief Finalize the handle (idempotent).
   subroutine rt_destroy(this)
      type(skalaxc_runtime_environment_t), intent(inout) :: this
      if (c_associated(this%handle)) then
         call c_runtime_destroy(this%handle)
         this%handle = c_null_ptr
      end if
   end subroutine rt_destroy

   subroutine rt_assign(lhs, rhs)
      class(skalaxc_runtime_environment_t), intent(inout) :: lhs
      class(skalaxc_runtime_environment_t), intent(in)    :: rhs
      call reject_handle_assignment(lhs%is_valid(), rhs%is_valid())
   end subroutine rt_assign

   subroutine rt_move_from(this, source)
      class(skalaxc_runtime_environment_t), intent(inout) :: this
      class(skalaxc_runtime_environment_t), intent(inout) :: source
      call rt_destroy(this)
      this%handle = source%handle
      source%handle = c_null_ptr
   end subroutine rt_move_from

   !> @brief True if the handle is initialized.
   logical function rt_is_valid(this)
      class(skalaxc_runtime_environment_t), intent(in) :: this
      rt_is_valid = c_associated(this%handle)
   end function rt_is_valid

   ! ==========================================================================
   ! Molecule
   ! ==========================================================================

   !> @brief Create an empty molecule.
   function mol_create(this) result(status)
      class(skalaxc_molecule_t), intent(inout) :: this
      integer(c_int)                           :: status
      type(c_ptr)                              :: new_handle
      new_handle = c_null_ptr
      status = c_molecule_create(new_handle)
      if (status == SKALAXC_SUCCESS) then
         call mol_destroy(this)
         this%handle = new_handle
      end if
   end function mol_create

   !> @brief Append an atom.
   !> @param atnum Atomic number (named atnum to avoid clashing with z).
   function mol_add_atom(this, atnum, x, y, z) result(status)
      class(skalaxc_molecule_t), intent(inout) :: this
      integer(c_int64_t), intent(in)           :: atnum
      real(c_double), intent(in)               :: x, y, z
      integer(c_int)                           :: status
      status = c_molecule_add_atom(this%handle, atnum, x, y, z)
   end function mol_add_atom

   !> @brief Create a molecule from native arrays.
   function mol_from_arrays(this, Z, atom_xyz) result(status)
      class(skalaxc_molecule_t), intent(inout) :: this
      integer(c_int64_t), intent(in)           :: Z(:)
      real(c_double), intent(in)               :: atom_xyz(:)
      integer(c_int)                           :: status
      type(c_ptr)                              :: new_handle
      if (.not. is_contiguous(Z) .or. .not. is_contiguous(atom_xyz) .or. &
          size(Z, kind=c_int64_t) > &
          (huge(0_c_int64_t) - modulo(huge(0_c_int64_t), 3_c_int64_t))/3_c_int64_t .or. &
          size(atom_xyz, kind=c_int64_t) /= 3_c_int64_t*size(Z, kind=c_int64_t)) then
         status = SKALAXC_INVALID_ARGUMENT
         return
      end if
      new_handle = c_null_ptr
      status = c_molecule_from_arrays(int(size(Z), c_int64_t), Z, atom_xyz, new_handle)
      if (status == SKALAXC_SUCCESS) then
         call mol_destroy(this)
         this%handle = new_handle
      end if
   end function mol_from_arrays

#ifdef SKALAXC_HAS_HDF5
   !> @brief Create a molecule from an HDF5 record.
   function mol_from_hdf5(this, path, dset) result(status)
      class(skalaxc_molecule_t), intent(inout) :: this
      character(len=*), intent(in)             :: path
      character(len=*), intent(in)             :: dset
      integer(c_int)                           :: status
      type(c_ptr)                              :: new_handle
      new_handle = c_null_ptr
      status = c_molecule_from_hdf5(trim(path)//c_null_char, &
                                    trim(dset)//c_null_char, new_handle)
      if (status == SKALAXC_SUCCESS) then
         call mol_destroy(this)
         this%handle = new_handle
      end if
   end function mol_from_hdf5
#endif

   !> @brief Number of atoms (-1 if the handle is null).
   function mol_natoms(this) result(n)
      class(skalaxc_molecule_t), intent(in) :: this
      integer(c_int64_t)                    :: n
      n = c_molecule_natoms(this%handle)
   end function mol_natoms

   !> @brief Destroy the handle (idempotent).
   subroutine mol_destroy(this)
      type(skalaxc_molecule_t), intent(inout) :: this
      if (c_associated(this%handle)) then
         call c_molecule_destroy(this%handle)
         this%handle = c_null_ptr
      end if
   end subroutine mol_destroy

   subroutine mol_assign(lhs, rhs)
      class(skalaxc_molecule_t), intent(inout) :: lhs
      class(skalaxc_molecule_t), intent(in)    :: rhs
      call reject_handle_assignment(lhs%is_valid(), rhs%is_valid())
   end subroutine mol_assign

   subroutine mol_move_from(this, source)
      class(skalaxc_molecule_t), intent(inout) :: this
      class(skalaxc_molecule_t), intent(inout) :: source
      call mol_destroy(this)
      this%handle = source%handle
      source%handle = c_null_ptr
   end subroutine mol_move_from

   !> @brief True if the handle is initialized.
   logical function mol_is_valid(this)
      class(skalaxc_molecule_t), intent(in) :: this
      mol_is_valid = c_associated(this%handle)
   end function mol_is_valid

   ! ==========================================================================
   ! Basis set
   ! ==========================================================================

   !> @brief Create an empty basis set.
   function bas_create(this) result(status)
      class(skalaxc_basisset_t), intent(inout) :: this
      integer(c_int)                           :: status
      type(c_ptr)                              :: new_handle
      new_handle = c_null_ptr
      status = c_basisset_create(new_handle)
      if (status == SKALAXC_SUCCESS) then
         call bas_destroy(this)
         this%handle = new_handle
      end if
   end function bas_create

   !> @brief Append a contracted Gaussian shell.
   !> @param normalize Optional; nonzero (default) normalizes the shell.
   function bas_add_shell(this, l, pure, center_xyz, exponents, coefficients, &
                          normalize) result(status)
      class(skalaxc_basisset_t), intent(inout) :: this
      integer(c_int32_t), intent(in)           :: l, pure
      real(c_double), intent(in)               :: center_xyz(:)
      real(c_double), intent(in)               :: exponents(:)
      real(c_double), intent(in)               :: coefficients(:)
      integer(c_int32_t), intent(in), optional :: normalize
      integer(c_int)                           :: status
      integer(c_int32_t)                       :: norm
      if (.not. is_contiguous(center_xyz) .or. &
          .not. is_contiguous(exponents) .or. &
          .not. is_contiguous(coefficients) .or. &
          size(center_xyz, kind=c_int64_t) /= 3_c_int64_t .or. &
          size(exponents, kind=c_int64_t) /= size(coefficients, kind=c_int64_t) .or. &
          size(exponents, kind=c_int64_t) > int(huge(0_c_int32_t), c_int64_t)) then
         status = SKALAXC_INVALID_ARGUMENT
         return
      end if
      norm = 1
      if (present(normalize)) norm = normalize
      status = c_basisset_add_shell(this%handle, l, pure, center_xyz, &
                                    int(size(exponents), c_int32_t), &
                                    exponents, coefficients, norm)
   end function bas_add_shell

   !> @brief Create a basis set from native arrays.
   function bas_from_arrays(this, shell_l, shell_pure, shell_xyz, shell_nprim, &
                            prim_exp, prim_coeff) result(status)
      class(skalaxc_basisset_t), intent(inout) :: this
      integer(c_int32_t), intent(in)           :: shell_l(:)
      integer(c_int32_t), intent(in)           :: shell_pure(:)
      real(c_double), intent(in)               :: shell_xyz(:)
      integer(c_int32_t), intent(in)           :: shell_nprim(:)
      real(c_double), intent(in)               :: prim_exp(:)
      real(c_double), intent(in)               :: prim_coeff(:)
      integer(c_int)                           :: status
      type(c_ptr)                              :: new_handle
      integer(c_int64_t)                       :: primitive_count
      integer                                  :: shell
      if (.not. is_contiguous(shell_l) .or. &
          .not. is_contiguous(shell_pure) .or. &
          .not. is_contiguous(shell_xyz) .or. &
          .not. is_contiguous(shell_nprim) .or. &
          .not. is_contiguous(prim_exp) .or. &
          .not. is_contiguous(prim_coeff) .or. &
          size(shell_pure, kind=c_int64_t) /= size(shell_l, kind=c_int64_t) .or. &
          size(shell_nprim, kind=c_int64_t) /= size(shell_l, kind=c_int64_t) .or. &
          size(shell_l, kind=c_int64_t) > &
          (huge(0_c_int64_t) - modulo(huge(0_c_int64_t), 3_c_int64_t))/3_c_int64_t .or. &
          size(shell_xyz, kind=c_int64_t) /= 3_c_int64_t*size(shell_l, kind=c_int64_t) .or. &
          size(prim_exp, kind=c_int64_t) /= size(prim_coeff, kind=c_int64_t)) then
         status = SKALAXC_INVALID_ARGUMENT
         return
      end if
      primitive_count = 0_c_int64_t
      do shell = 1, size(shell_nprim)
         if (shell_nprim(shell) < 0_c_int32_t .or. &
             primitive_count > huge(primitive_count) - int(shell_nprim(shell), c_int64_t)) then
            status = SKALAXC_INVALID_ARGUMENT
            return
         end if
         primitive_count = primitive_count + int(shell_nprim(shell), c_int64_t)
      end do
      if (primitive_count /= size(prim_exp, kind=c_int64_t)) then
         status = SKALAXC_INVALID_ARGUMENT
         return
      end if
      new_handle = c_null_ptr
      status = c_basisset_from_arrays(int(size(shell_l), c_int64_t), shell_l, &
                                      shell_pure, shell_xyz, shell_nprim, &
                                      prim_exp, prim_coeff, new_handle)
      if (status == SKALAXC_SUCCESS) then
         call bas_destroy(this)
         this%handle = new_handle
      end if
   end function bas_from_arrays

#ifdef SKALAXC_HAS_HDF5
   !> @brief Create a basis set from an HDF5 record.
   function bas_from_hdf5(this, path, dset) result(status)
      class(skalaxc_basisset_t), intent(inout) :: this
      character(len=*), intent(in)             :: path
      character(len=*), intent(in)             :: dset
      integer(c_int)                           :: status
      type(c_ptr)                              :: new_handle
      new_handle = c_null_ptr
      status = c_basisset_from_hdf5(trim(path)//c_null_char, &
                                    trim(dset)//c_null_char, new_handle)
      if (status == SKALAXC_SUCCESS) then
         call bas_destroy(this)
         this%handle = new_handle
      end if
   end function bas_from_hdf5
#endif

   !> @brief Number of basis functions (-1 if the handle is null).
   function bas_nbf(this) result(n)
      class(skalaxc_basisset_t), intent(in) :: this
      integer(c_int64_t)                    :: n
      n = c_basisset_nbf(this%handle)
   end function bas_nbf

   !> @brief Destroy the handle (idempotent).
   subroutine bas_destroy(this)
      type(skalaxc_basisset_t), intent(inout) :: this
      if (c_associated(this%handle)) then
         call c_basisset_destroy(this%handle)
         this%handle = c_null_ptr
      end if
   end subroutine bas_destroy

   subroutine bas_assign(lhs, rhs)
      class(skalaxc_basisset_t), intent(inout) :: lhs
      class(skalaxc_basisset_t), intent(in)    :: rhs
      call reject_handle_assignment(lhs%is_valid(), rhs%is_valid())
   end subroutine bas_assign

   subroutine bas_move_from(this, source)
      class(skalaxc_basisset_t), intent(inout) :: this
      class(skalaxc_basisset_t), intent(inout) :: source
      call bas_destroy(this)
      this%handle = source%handle
      source%handle = c_null_ptr
   end subroutine bas_move_from

   !> @brief True if the handle is initialized.
   logical function bas_is_valid(this)
      class(skalaxc_basisset_t), intent(in) :: this
      bas_is_valid = c_associated(this%handle)
   end function bas_is_valid

   ! ==========================================================================
   ! Molecular grid
   ! ==========================================================================

   !> @brief Create a default molecular grid for a molecule.
   !> @param grid Optional grid settings (default: built-in preset).
   function mg_create_default(this, mol, grid) result(status)
      class(skalaxc_molgrid_t), intent(inout)             :: this
      type(skalaxc_molecule_t), intent(in)                :: mol
      type(skalaxc_grid_settings_t), intent(in), optional :: grid
      integer(c_int)                                      :: status
      type(skalaxc_grid_settings_t), target :: grid_local
      type(c_ptr)                           :: grid_ptr, new_handle
      grid_ptr = c_null_ptr
      new_handle = c_null_ptr
      if (present(grid)) then
         grid_local = grid
         grid_ptr = c_loc(grid_local)
      end if
      status = c_molgrid_create_default(mol%handle, grid_ptr, new_handle)
      if (status == SKALAXC_SUCCESS) then
         call mg_destroy(this)
         this%handle = new_handle
      end if
   end function mg_create_default

   !> @brief Destroy the handle (idempotent).
   subroutine mg_destroy(this)
      type(skalaxc_molgrid_t), intent(inout) :: this
      if (c_associated(this%handle)) then
         call c_molgrid_destroy(this%handle)
         this%handle = c_null_ptr
      end if
   end subroutine mg_destroy

   subroutine mg_assign(lhs, rhs)
      class(skalaxc_molgrid_t), intent(inout) :: lhs
      class(skalaxc_molgrid_t), intent(in)    :: rhs
      call reject_handle_assignment(lhs%is_valid(), rhs%is_valid())
   end subroutine mg_assign

   subroutine mg_move_from(this, source)
      class(skalaxc_molgrid_t), intent(inout) :: this
      class(skalaxc_molgrid_t), intent(inout) :: source
      call mg_destroy(this)
      this%handle = source%handle
      source%handle = c_null_ptr
   end subroutine mg_move_from

   !> @brief True if the handle is initialized.
   logical function mg_is_valid(this)
      class(skalaxc_molgrid_t), intent(in) :: this
      mg_is_valid = c_associated(this%handle)
   end function mg_is_valid

   ! ==========================================================================
   ! Load balancer
   ! ==========================================================================

   !> @brief Create a load balancer for the given system.
   function lb_create(this, ex, rt, mol, mg, basis) result(status)
      class(skalaxc_load_balancer_t), intent(inout)      :: this
      integer(c_int), intent(in)                         :: ex
      type(skalaxc_runtime_environment_t), intent(in)    :: rt
      type(skalaxc_molecule_t), intent(in)               :: mol
      type(skalaxc_molgrid_t), intent(in)                :: mg
      type(skalaxc_basisset_t), intent(in)               :: basis
      integer(c_int)                                     :: status
      type(c_ptr)                                        :: new_handle
      new_handle = c_null_ptr
      status = c_load_balancer_create(ex, rt%handle, mol%handle, mg%handle, &
                                      basis%handle, new_handle)
      if (status == SKALAXC_SUCCESS) then
         call lb_destroy(this)
         this%handle = new_handle
      end if
   end function lb_create

   !> @brief Destroy the handle (idempotent).
   subroutine lb_destroy(this)
      type(skalaxc_load_balancer_t), intent(inout) :: this
      if (c_associated(this%handle)) then
         call c_load_balancer_destroy(this%handle)
         this%handle = c_null_ptr
      end if
   end subroutine lb_destroy

   subroutine lb_assign(lhs, rhs)
      class(skalaxc_load_balancer_t), intent(inout) :: lhs
      class(skalaxc_load_balancer_t), intent(in)    :: rhs
      call reject_handle_assignment(lhs%is_valid(), rhs%is_valid())
   end subroutine lb_assign

   subroutine lb_move_from(this, source)
      class(skalaxc_load_balancer_t), intent(inout) :: this
      class(skalaxc_load_balancer_t), intent(inout) :: source
      call lb_destroy(this)
      this%handle = source%handle
      source%handle = c_null_ptr
   end subroutine lb_move_from

   !> @brief True if the handle is initialized.
   logical function lb_is_valid(this)
      class(skalaxc_load_balancer_t), intent(in) :: this
      lb_is_valid = c_associated(this%handle)
   end function lb_is_valid

   ! ==========================================================================
   ! Molecular weights
   ! ==========================================================================

   !> @brief Create a molecular-weights partitioner.
   function mw_create(this, ex, weight_alg) result(status)
      class(skalaxc_molecular_weights_t), intent(inout) :: this
      integer(c_int), intent(in)                        :: ex
      integer(c_int), intent(in)                        :: weight_alg
      integer(c_int)                                    :: status
      type(c_ptr)                                       :: new_handle
      new_handle = c_null_ptr
      status = c_molecular_weights_create(ex, weight_alg, new_handle)
      if (status == SKALAXC_SUCCESS) then
         call mw_destroy(this)
         this%handle = new_handle
      end if
   end function mw_create

   !> @brief Partition the quadrature weights stored on a load balancer.
   function mw_modify_weights(this, lb) result(status)
      class(skalaxc_molecular_weights_t), intent(in) :: this
      type(skalaxc_load_balancer_t), intent(in)      :: lb
      integer(c_int)                                 :: status
      status = c_molecular_weights_modify_weights(this%handle, lb%handle)
   end function mw_modify_weights

   !> @brief Destroy the handle (idempotent).
   subroutine mw_destroy(this)
      type(skalaxc_molecular_weights_t), intent(inout) :: this
      if (c_associated(this%handle)) then
         call c_molecular_weights_destroy(this%handle)
         this%handle = c_null_ptr
      end if
   end subroutine mw_destroy

   subroutine mw_assign(lhs, rhs)
      class(skalaxc_molecular_weights_t), intent(inout) :: lhs
      class(skalaxc_molecular_weights_t), intent(in)    :: rhs
      call reject_handle_assignment(lhs%is_valid(), rhs%is_valid())
   end subroutine mw_assign

   subroutine mw_move_from(this, source)
      class(skalaxc_molecular_weights_t), intent(inout) :: this
      class(skalaxc_molecular_weights_t), intent(inout) :: source
      call mw_destroy(this)
      this%handle = source%handle
      source%handle = c_null_ptr
   end subroutine mw_move_from

   !> @brief True if the handle is initialized.
   logical function mw_is_valid(this)
      class(skalaxc_molecular_weights_t), intent(in) :: this
      mw_is_valid = c_associated(this%handle)
   end function mw_is_valid

   ! ==========================================================================
   ! Functional
   ! ==========================================================================

   !> @brief Create a functional from a Skala model selector.
   function func_create(this, model) result(status)
      class(skalaxc_functional_t), intent(inout) :: this
      character(len=*), intent(in)               :: model
      integer(c_int)                             :: status
      type(c_ptr)                                :: new_handle
      new_handle = c_null_ptr
      status = c_functional_create(trim(model)//c_null_char, new_handle)
      if (status == SKALAXC_SUCCESS) then
         call func_destroy(this)
         this%handle = new_handle
      end if
   end function func_create

   !> @brief Destroy the handle (idempotent).
   subroutine func_destroy(this)
      type(skalaxc_functional_t), intent(inout) :: this
      if (c_associated(this%handle)) then
         call c_functional_destroy(this%handle)
         this%handle = c_null_ptr
      end if
   end subroutine func_destroy

   subroutine func_assign(lhs, rhs)
      class(skalaxc_functional_t), intent(inout) :: lhs
      class(skalaxc_functional_t), intent(in)    :: rhs
      call reject_handle_assignment(lhs%is_valid(), rhs%is_valid())
   end subroutine func_assign

   subroutine func_move_from(this, source)
      class(skalaxc_functional_t), intent(inout) :: this
      class(skalaxc_functional_t), intent(inout) :: source
      call func_destroy(this)
      this%handle = source%handle
      source%handle = c_null_ptr
   end subroutine func_move_from

   !> @brief True if the handle is initialized.
   logical function func_is_valid(this)
      class(skalaxc_functional_t), intent(in) :: this
      func_is_valid = c_associated(this%handle)
   end function func_is_valid

   ! ==========================================================================
   ! XC integrator
   ! ==========================================================================

   !> @brief Create an XC integrator from a functional and weighted balancer.
   function xc_create(this, ex, func, lb, timing_settings, &
                      domain_batch_mode) result(status)
      class(skalaxc_xc_integrator_t), intent(inout) :: this
      integer(c_int), intent(in)                    :: ex
      type(skalaxc_functional_t), intent(in)        :: func
      type(skalaxc_load_balancer_t), intent(in)     :: lb
      type(skalaxc_timing_settings_t), intent(in), optional :: timing_settings
      integer(c_int), intent(in), optional          :: domain_batch_mode
      integer(c_int)                                :: status
      type(skalaxc_integrator_settings_t) :: integrator_settings
      type(c_ptr) :: new_handle
      new_handle = c_null_ptr
      if (present(timing_settings) .or. present(domain_batch_mode)) then
         integrator_settings = skalaxc_integrator_settings_default()
         if (present(timing_settings)) &
            & integrator_settings%timing = timing_settings
         if (present(domain_batch_mode)) &
            & integrator_settings%domain_batch_mode = domain_batch_mode
         status = c_integrator_create_with_settings(ex, func%handle, &
            & lb%handle, integrator_settings, new_handle)
      else
         status = c_integrator_create(ex, func%handle, lb%handle, new_handle)
      end if
      if (status == SKALAXC_SUCCESS) then
         call xc_destroy(this)
         this%handle = new_handle
      end if
   end function xc_create

   !> @brief Number of basis functions (-1 if the handle is null).
   function xc_nbf(this) result(n)
      class(skalaxc_xc_integrator_t), intent(in) :: this
      integer(c_int64_t)                         :: n
      n = c_integrator_nbf(this%handle)
   end function xc_nbf

   !> @brief Number of atoms (-1 if the handle is null).
   function xc_natoms(this) result(n)
      class(skalaxc_xc_integrator_t), intent(in) :: this
      integer(c_int64_t)                         :: n
      n = c_integrator_natoms(this%handle)
   end function xc_natoms

   !> @brief Evaluate the UKS ML exchange-correlation energy and potential.
   !> @param Ps Scalar spin-density matrix, length nbf*nbf, column-major.
   !> @param Pz Z spin-density matrix, length nbf*nbf, column-major.
   !> @param VXCs [out] Scalar XC potential, length nbf*nbf, column-major.
   !> @param VXCz [out] Z XC potential, length nbf*nbf, column-major.
   !> @param exc [out] Exchange-correlation energy.
   function xc_eval_exc_vxc_uks(this, Ps, Pz, VXCs, VXCz, exc) result(status)
      class(skalaxc_xc_integrator_t), intent(in) :: this
      real(c_double), intent(in)  :: Ps(:)
      real(c_double), intent(in)  :: Pz(:)
      real(c_double), intent(out) :: VXCs(:)
      real(c_double), intent(out) :: VXCz(:)
      real(c_double), intent(out) :: exc
      integer(c_int)              :: status
      integer(c_int64_t)          :: nbf, matrix_size
      nbf = this%nbf()
      if (nbf < 0_c_int64_t .or. &
          nbf > huge(0_c_int64_t)/max(1_c_int64_t, nbf)) then
         status = SKALAXC_INVALID_ARGUMENT
         return
      end if
      matrix_size = nbf*nbf
      if (.not. is_contiguous(Ps) .or. .not. is_contiguous(Pz) .or. &
          .not. is_contiguous(VXCs) .or. .not. is_contiguous(VXCz) .or. &
          size(Ps, kind=c_int64_t) /= matrix_size .or. &
          size(Pz, kind=c_int64_t) /= matrix_size .or. &
          size(VXCs, kind=c_int64_t) /= matrix_size .or. &
          size(VXCz, kind=c_int64_t) /= matrix_size) then
         status = SKALAXC_INVALID_ARGUMENT
         return
      end if
      status = c_integrator_eval(this%handle, Ps, Pz, VXCs, VXCz, exc)
   end function xc_eval_exc_vxc_uks

   !> @brief Evaluate the UKS ML exchange-correlation energy gradient.
   !> @param Ps Scalar spin-density matrix, length nbf*nbf, column-major.
   !> @param Pz Z spin-density matrix, length nbf*nbf, column-major.
   !> @param gradient [out] Atom-major Cartesian gradient, length 3*natoms.
   function xc_eval_exc_grad_uks(this, Ps, Pz, gradient) result(status)
      class(skalaxc_xc_integrator_t), intent(in) :: this
      real(c_double), intent(in)  :: Ps(:)
      real(c_double), intent(in)  :: Pz(:)
      real(c_double), intent(out) :: gradient(:)
      integer(c_int)              :: status
      integer(c_int64_t)          :: nbf, natoms, matrix_size
      nbf = this%nbf()
      natoms = this%natoms()
      if (nbf < 0_c_int64_t .or. natoms < 0_c_int64_t .or. &
          nbf > huge(0_c_int64_t)/max(1_c_int64_t, nbf) .or. &
          natoms > (huge(0_c_int64_t) - &
                    modulo(huge(0_c_int64_t), 3_c_int64_t))/3_c_int64_t) then
         status = SKALAXC_INVALID_ARGUMENT
         return
      end if
      matrix_size = nbf*nbf
      if (.not. is_contiguous(Ps) .or. .not. is_contiguous(Pz) .or. &
          .not. is_contiguous(gradient) .or. &
          size(Ps, kind=c_int64_t) /= matrix_size .or. &
          size(Pz, kind=c_int64_t) /= matrix_size .or. &
          size(gradient, kind=c_int64_t) /= 3_c_int64_t*natoms) then
         status = SKALAXC_INVALID_ARGUMENT
         return
      end if
      status = c_integrator_eval_grad(this%handle, Ps, Pz, gradient)
   end function xc_eval_exc_grad_uks

   !> @brief Retrieve rank-local diagnostics without an MPI collective.
   function xc_diagnostics(this, snapshot) result(status)
      class(skalaxc_xc_integrator_t), intent(in) :: this
      type(skalaxc_diagnostics_snapshot_t), intent(out) :: snapshot
      integer(c_int) :: status
      status = c_integrator_diagnostics(this%handle, snapshot)
   end function xc_diagnostics

   !> @brief Clear evaluation timings and counters.
   function xc_reset_diagnostics(this) result(status)
      class(skalaxc_xc_integrator_t), intent(in) :: this
      integer(c_int) :: status
      status = c_integrator_reset_diagnostics(this%handle)
   end function xc_reset_diagnostics

   !> @brief Destroy the handle (idempotent).
   subroutine xc_destroy(this)
      type(skalaxc_xc_integrator_t), intent(inout) :: this
      if (c_associated(this%handle)) then
         call c_integrator_destroy(this%handle)
         this%handle = c_null_ptr
      end if
   end subroutine xc_destroy

   subroutine xc_assign(lhs, rhs)
      class(skalaxc_xc_integrator_t), intent(inout) :: lhs
      class(skalaxc_xc_integrator_t), intent(in)    :: rhs
      call reject_handle_assignment(lhs%is_valid(), rhs%is_valid())
   end subroutine xc_assign

   subroutine xc_move_from(this, source)
      class(skalaxc_xc_integrator_t), intent(inout) :: this
      class(skalaxc_xc_integrator_t), intent(inout) :: source
      call xc_destroy(this)
      this%handle = source%handle
      source%handle = c_null_ptr
   end subroutine xc_move_from

   !> @brief True if the handle is initialized.
   logical function xc_is_valid(this)
      class(skalaxc_xc_integrator_t), intent(in) :: this
      xc_is_valid = c_associated(this%handle)
   end function xc_is_valid

   ! ==========================================================================
   ! Utilities
   ! ==========================================================================

   !> @brief Return the SkalaXC semantic version as a Fortran string.
   function skalaxc_version() result(version)
      character(len=:), allocatable :: version
      version = fortran_string(c_version())
   end function skalaxc_version

   !> @brief Most recent error message on the calling thread (Fortran string).
   function skalaxc_last_error() result(msg)
      character(len=:), allocatable :: msg
      msg = fortran_string(c_last_error())
   end function skalaxc_last_error

   function fortran_string(cptr) result(msg)
      type(c_ptr), intent(in), value :: cptr
      character(len=:), allocatable :: msg
      character(kind=c_char), pointer :: cstr(:)
      integer :: n
      if (.not. c_associated(cptr)) then
         msg = ''
         return
      end if
      call c_f_pointer(cptr, cstr, [huge(0)])
      n = 0
      do
         if (cstr(n + 1) == c_null_char) exit
         n = n + 1
      end do
      allocate (character(len=n) :: msg)
      block
         integer :: i
         do i = 1, n
            msg(i:i) = cstr(i)
         end do
      end block
   end function fortran_string

   !> @brief Grid settings pre-filled with the SkalaXC built-in preset.
   function skalaxc_grid_settings_default() result(grid)
      type(skalaxc_grid_settings_t) :: grid
      call c_grid_defaults(grid)
   end function skalaxc_grid_settings_default

   !> @brief Return the built-in CUDA runtime settings.
   function skalaxc_device_runtime_settings_default() result(settings)
      type(skalaxc_device_runtime_settings_t) :: settings
      call c_device_runtime_defaults(settings)
   end function skalaxc_device_runtime_settings_default

   !> @brief Return non-synchronizing timing settings.
   function skalaxc_timing_settings_default() result(settings)
      type(skalaxc_timing_settings_t) :: settings
      call c_timing_defaults(settings)
   end function skalaxc_timing_settings_default

   !> @brief Return conservative XC-integrator construction settings.
   function skalaxc_integrator_settings_default() result(settings)
      type(skalaxc_integrator_settings_t) :: settings
      call c_integrator_defaults(settings)
   end function skalaxc_integrator_settings_default

end module skalaxc
