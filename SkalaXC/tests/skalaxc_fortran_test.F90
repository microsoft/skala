! SkalaXC black-box Fortran test.
!
! A pure Fortran consumer: it `use`s ONLY the `skalaxc` module and links ONLY
! libskalaxc (via skalaxc_fortran), plus HDF5-Fortran to load its own density
! input. It has NO access to GauXC or LibTorch. Successful compilation and link
! prove the Fortran API is self-contained and ABI-isolated.
!
! It drives the per-stage pipeline (runtime -> molecule/basis -> molgrid ->
! load balancer -> molecular weights -> functional -> integrator), mirroring
! the C++ and C surfaces.
!
! Usage: skalaxc_fortran_test <ref_data_dir> <gauxc_ref_data_dir>

program skalaxc_fortran_test
   use, intrinsic :: iso_c_binding
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use skalaxc
#ifdef SKALAXC_HAS_MPI
   use mpi
#endif
   use hdf5
   implicit none

   character(len=1024) :: ref_dir, gauxc_dir
   integer :: nargs, herr, failures, total
#ifdef SKALAXC_HAS_MPI
   integer :: mpi_err
#endif
   character(len=1024) :: path

   nargs = command_argument_count()
   if (nargs < 2) then
      write (*, *) 'usage: skalaxc_fortran_test <ref_data_dir> <gauxc_ref_dir>'
      stop 2
   end if
   call get_command_argument(1, ref_dir)
   call get_command_argument(2, gauxc_dir)

#ifdef SKALAXC_HAS_MPI
   block
      logical :: inited
      call MPI_Initialized(inited, mpi_err)
      if (.not. inited) call MPI_Init(mpi_err)
   end block
#endif

   call h5open_f(herr)
   if (herr /= 0) stop 'h5open failed'

   failures = 0
   total = 0

   block
      character(len=:), allocatable :: version
      version = skalaxc_version()
      if (version == SKALAXC_EXPECTED_VERSION) then
         write (*, '(A)') '[PASS] SkalaXC version '//version
      else
         write (*, '(A)') '[FAIL] SkalaXC version: expected '// &
            SKALAXC_EXPECTED_VERSION//', got '//version
         failures = failures + 1
      end if
      total = total + 1
   end block

   block
      type(skalaxc_device_runtime_settings_t) :: device_settings
      type(skalaxc_timing_settings_t) :: timing_settings
      type(skalaxc_integrator_settings_t) :: integrator_settings
      device_settings = skalaxc_device_runtime_settings_default()
      timing_settings = skalaxc_timing_settings_default()
      integrator_settings = skalaxc_integrator_settings_default()
      if (device_settings%device_id == 0 .and. &
          abs(device_settings%memory_fraction - 0.75_c_double) < 1e-15_c_double &
          .and. timing_settings%verbose == 0_c_int32_t .and. &
          timing_settings%debug_logging == 0_c_int32_t .and. &
          integrator_settings%timing%verbose == 0_c_int32_t .and. &
          integrator_settings%timing%debug_logging == 0_c_int32_t .and. &
          integrator_settings%domain_batch_mode == &
          skalaxc_domainbatchmode%conservative) then
         write (*, '(A)') '[PASS] runtime and timing defaults'
      else
         write (*, '(A)') '[FAIL] runtime and timing defaults'
         failures = failures + 1
      end if
      total = total + 1
   end block

   failures = failures + run_native_case(.false., 'incremental')
   total = total + 1
   failures = failures + run_native_case(.true., 'arrays')
   total = total + 1

   failures = failures + run_handle_lifecycle_case()
   total = total + 1

   path = trim(ref_dir)//'/skala_he_def2qzvp_lda_uks.hdf5'
   failures = failures + run_case(trim(path), 'LDA', 'HE/def2-qzvp/lda', &
                                  enable_diagnostics=.true.)
   total = total + 1

   path = trim(ref_dir)//'/skala_he_def2qzvp_pbe_uks.hdf5'
   failures = failures + run_case(trim(path), 'PBE', 'HE/def2-qzvp/pbe')
   total = total + 1

   path = trim(ref_dir)//'/skala_he_def2qzvp_tpss_uks.hdf5'
   failures = failures + run_case(trim(path), 'TPSS', 'HE/def2-qzvp/tpss')
   total = total + 1

   path = trim(gauxc_dir)//'/h2o2_def2-tzvp.hdf5'
   failures = failures + run_gradient_case(trim(path), 'TPSS', &
                                           skalaxc_executionspace%host, 'H2O2 gradient')
   total = total + 1

#ifdef SKALAXC_HAS_CUDA
   path = trim(ref_dir)//'/skala_he_def2qzvp_tpss_uks.hdf5'
   failures = failures + run_case(trim(path), 'TPSS', &
                                  'HE/def2-qzvp/tpss [cuda]', skalaxc_executionspace%device)
   total = total + 1

   path = trim(gauxc_dir)//'/h2o2_def2-tzvp.hdf5'
   ! The current TPSS trace can exceed sm_120 resources during backward.
   ! PBE keeps this language-binding test focused on the public gradient API
   ! until TPSS is retraced with a smaller TensorExpr kernel.
   failures = failures + run_gradient_case(trim(path), 'PBE', &
                                           skalaxc_executionspace%device, 'H2O2 gradient [cuda]')
   total = total + 1
#endif

   write (*, '(/,I0,A,I0,A)') total - failures, ' / ', total, &
      ' Fortran public-API cases passed'

   call h5close_f(herr)

#ifdef SKALAXC_HAS_MPI
   block
      logical :: fin
      call MPI_Finalized(fin, mpi_err)
      if (.not. fin) call MPI_Finalize(mpi_err)
   end block
#endif

   if (failures /= 0) stop 1

contains

   integer function run_handle_lifecycle_case() result(rc)
      type(skalaxc_runtime_environment_t) :: rt, moved_rt
      type(skalaxc_molecule_t) :: mol, moved_mol
      type(skalaxc_basisset_t) :: basis, moved_basis
      type(skalaxc_molgrid_t) :: mg, moved_mg
      type(skalaxc_load_balancer_t) :: lb, moved_lb
      type(skalaxc_molecular_weights_t) :: mw, moved_mw
      type(skalaxc_functional_t) :: func, moved_func
      type(skalaxc_xc_integrator_t) :: xc, moved_xc
      integer(c_int) :: status
      integer(c_int64_t) :: moved_mol_natoms, moved_xc_nbf, moved_basis_nbf

      rc = 1
      status = mol%create()
      if (status /= SKALAXC_SUCCESS) goto 100
      status = mol%add_atom(2_c_int64_t, 0.0_c_double, 0.0_c_double, &
                            0.0_c_double)
      if (status /= SKALAXC_SUCCESS) goto 100
      status = basis%from_hdf5(trim(ref_dir)// &
                               '/skala_he_def2qzvp_lda_uks.hdf5', '/BASIS')
      if (status /= SKALAXC_SUCCESS) goto 100
#ifdef SKALAXC_HAS_MPI
      status = skalaxc_runtime_environment_create(rt, MPI_COMM_WORLD)
#else
      status = skalaxc_runtime_environment_create(rt)
#endif
      if (status /= SKALAXC_SUCCESS) goto 100
      status = mg%create_default(mol)
      if (status /= SKALAXC_SUCCESS) goto 100
      status = lb%create(skalaxc_executionspace%host, rt, mol, mg, basis)
      if (status /= SKALAXC_SUCCESS) goto 100
      status = mw%create(skalaxc_executionspace%host, skalaxc_xcweightalg%ssf)
      if (status /= SKALAXC_SUCCESS) goto 100
      status = mw%modify_weights(lb)
      if (status /= SKALAXC_SUCCESS) goto 100
      status = func%create('LDA')
      if (status /= SKALAXC_SUCCESS) goto 100
      status = xc%create(skalaxc_executionspace%host, func, lb)
      if (status /= SKALAXC_SUCCESS) goto 100

      status = moved_mol%create()
      if (status /= SKALAXC_SUCCESS) goto 100
      call moved_rt%move_from(rt)
      call moved_mol%move_from(mol)
      call moved_basis%move_from(basis)
      call moved_mg%move_from(mg)
      call moved_lb%move_from(lb)
      call moved_mw%move_from(mw)
      call moved_func%move_from(func)
      call moved_xc%move_from(xc)

      if (rt%is_valid() .or. mol%is_valid() .or. basis%is_valid() .or. &
          mg%is_valid() .or. lb%is_valid() .or. mw%is_valid() .or. &
          func%is_valid() .or. xc%is_valid()) goto 100
      if (.not. moved_rt%is_valid() .or. .not. moved_mol%is_valid() .or. &
          .not. moved_basis%is_valid() .or. .not. moved_mg%is_valid() .or. &
          .not. moved_lb%is_valid() .or. .not. moved_mw%is_valid() .or. &
          .not. moved_func%is_valid() .or. .not. moved_xc%is_valid()) goto 100
      moved_mol_natoms = moved_mol%natoms()
      moved_xc_nbf = moved_xc%nbf()
      moved_basis_nbf = moved_basis%nbf()
      if (moved_mol_natoms /= 1_c_int64_t .or. &
          moved_xc_nbf /= moved_basis_nbf) goto 100
      rc = 0

100   continue
      if (rc == 0) then
         write (*, '(A)') '[PASS] Fortran unique handle ownership'
      else if (status /= SKALAXC_SUCCESS) then
         write (*, '(A,A)') '[FAIL] Fortran unique handle ownership: ', &
            trim(skalaxc_last_error())
      else
         write (*, '(A)') '[FAIL] Fortran unique handle ownership'
      end if
   end function run_handle_lifecycle_case

   integer function run_native_case(use_arrays, name) result(rc)
      logical, intent(in) :: use_arrays
      character(len=*), intent(in) :: name
      type(skalaxc_runtime_environment_t) :: rt
      type(skalaxc_molecule_t)            :: mol
      type(skalaxc_basisset_t)            :: basis
      type(skalaxc_molgrid_t)             :: mg
      type(skalaxc_load_balancer_t)       :: lb
      type(skalaxc_molecular_weights_t)   :: mw
      type(skalaxc_functional_t)          :: func
      type(skalaxc_xc_integrator_t)       :: xc
      integer(c_int64_t) :: atomic_numbers(2)
      integer(c_int64_t) :: strided_atomic_numbers(4)
      integer(c_int32_t) :: shell_l(2), shell_pure(2), shell_nprim(2)
      integer(c_int32_t) :: strided_shell_l(4)
      real(c_double) :: atom_xyz(6), exponents(3), coefficients(3)
      real(c_double) :: primitive_exponents(6), primitive_coefficients(6)
      real(c_double) :: Ps(4), Pz(4), VXCs(4), VXCz(4), exc
      real(c_double) :: strided_matrix_storage(8), strided_gradient_storage(12)
      real(c_double) :: short_matrix(3)
      integer(c_int) :: status
      integer(c_int64_t) :: molecule_atoms, basis_functions
      integer(c_int64_t) :: integrator_atoms, integrator_basis_functions
#ifdef SKALAXC_HAS_MPI
      integer(c_int) :: runtime_rank, runtime_size
      integer :: world_rank, world_size, mpi_error
#endif

      atomic_numbers = [1_c_int64_t, 1_c_int64_t]
      strided_atomic_numbers = [1_c_int64_t, 0_c_int64_t, 1_c_int64_t, 0_c_int64_t]
      shell_l = [0_c_int32_t, 0_c_int32_t]
      strided_shell_l = [0_c_int32_t, 0_c_int32_t, 0_c_int32_t, 0_c_int32_t]
      shell_pure = [0_c_int32_t, 0_c_int32_t]
      shell_nprim = [3_c_int32_t, 3_c_int32_t]
      atom_xyz = [-0.7_c_double, 0.0_c_double, 0.0_c_double, &
                  0.7_c_double, 0.0_c_double, 0.0_c_double]
      exponents = [3.42525091_c_double, 0.62391373_c_double, &
                   0.16885540_c_double]
      coefficients = [0.15432897_c_double, 0.53532814_c_double, &
                      0.44463454_c_double]
      primitive_exponents = [exponents, exponents]
      primitive_coefficients = [coefficients, coefficients]
      rc = 1

      if (use_arrays) then
         status = mol%from_arrays(atomic_numbers, atom_xyz(1:5))
         if (status /= SKALAXC_INVALID_ARGUMENT) goto 100
         status = mol%from_arrays(strided_atomic_numbers(1:3:2), atom_xyz)
         if (status /= SKALAXC_INVALID_ARGUMENT) goto 100
         status = basis%from_arrays(shell_l, shell_pure, atom_xyz, &
                                    shell_nprim, primitive_exponents(1:5), &
                                    primitive_coefficients)
         if (status /= SKALAXC_INVALID_ARGUMENT) goto 100
         status = basis%from_arrays(strided_shell_l(1:3:2), shell_pure, &
                                    atom_xyz, shell_nprim, primitive_exponents, &
                                    primitive_coefficients)
         if (status /= SKALAXC_INVALID_ARGUMENT) goto 100
         status = mol%from_arrays(atomic_numbers, atom_xyz)
         if (status == SKALAXC_SUCCESS) then
            status = basis%from_arrays(shell_l, shell_pure, atom_xyz, &
                                       shell_nprim, primitive_exponents, &
                                       primitive_coefficients)
         end if
      else
         status = mol%create()
         if (status == SKALAXC_SUCCESS) then
            status = mol%add_atom(1_c_int64_t, atom_xyz(1), atom_xyz(2), atom_xyz(3))
         end if
         if (status == SKALAXC_SUCCESS) then
            status = mol%add_atom(1_c_int64_t, atom_xyz(4), atom_xyz(5), atom_xyz(6))
         end if
         if (status == SKALAXC_SUCCESS) status = basis%create()
         if (status == SKALAXC_SUCCESS) then
            status = basis%add_shell(0_c_int32_t, 0_c_int32_t, atom_xyz(1:5:2), &
                                     exponents, coefficients)
            if (status /= SKALAXC_INVALID_ARGUMENT) goto 100
            status = basis%add_shell(0_c_int32_t, 0_c_int32_t, atom_xyz(1:3), &
                                     exponents, coefficients)
         end if
         if (status == SKALAXC_SUCCESS) then
            status = basis%add_shell(0_c_int32_t, 0_c_int32_t, atom_xyz(4:6), &
                                     exponents, coefficients)
         end if
      end if
      if (status /= SKALAXC_SUCCESS) goto 100

#ifdef SKALAXC_HAS_MPI
      status = skalaxc_runtime_environment_create(rt, MPI_COMM_WORLD)
#else
      status = skalaxc_runtime_environment_create(rt)
#endif
      if (status /= SKALAXC_SUCCESS) goto 100
#ifdef SKALAXC_HAS_MPI
      call MPI_Comm_rank(MPI_COMM_WORLD, world_rank, mpi_error)
      call MPI_Comm_size(MPI_COMM_WORLD, world_size, mpi_error)
      runtime_rank = rt%comm_rank()
      runtime_size = rt%comm_size()
      if (runtime_rank /= world_rank .or. runtime_size /= world_size) goto 100
#endif
      status = mg%create_default(mol)
      if (status /= SKALAXC_SUCCESS) goto 100
      status = lb%create(skalaxc_executionspace%host, rt, mol, mg, basis)
      if (status /= SKALAXC_SUCCESS) goto 100
      status = mw%create(skalaxc_executionspace%host, skalaxc_xcweightalg%ssf)
      if (status /= SKALAXC_SUCCESS) goto 100
      status = mw%modify_weights(lb)
      if (status /= SKALAXC_SUCCESS) goto 100
      status = func%create('LDA')
      if (status /= SKALAXC_SUCCESS) goto 100
      status = xc%create(skalaxc_executionspace%host, func, lb)
      if (status /= SKALAXC_SUCCESS) goto 100

      Ps = 0.5_c_double
      Pz = 0.0_c_double
      status = xc%eval_exc_vxc_uks(Ps, Pz, VXCs, VXCz, exc)
      if (status /= SKALAXC_SUCCESS) goto 100
      status = xc%eval_exc_vxc_uks(short_matrix, Pz, VXCs, VXCz, exc)
      if (status /= SKALAXC_INVALID_ARGUMENT) goto 100
      strided_matrix_storage = 0.0_c_double
      status = xc%eval_exc_vxc_uks(strided_matrix_storage(1:8:2), Pz, &
                                   VXCs, VXCz, exc)
      if (status /= SKALAXC_INVALID_ARGUMENT) goto 100
      strided_gradient_storage = 0.0_c_double
      status = xc%eval_exc_grad_uks(Ps, Pz, strided_gradient_storage(1:12:2))
      if (status /= SKALAXC_INVALID_ARGUMENT) goto 100
      status = SKALAXC_SUCCESS
      molecule_atoms = mol%natoms()
      basis_functions = basis%nbf()
      integrator_atoms = xc%natoms()
      integrator_basis_functions = xc%nbf()
      if (molecule_atoms /= 2_c_int64_t .or. &
          basis_functions /= 2_c_int64_t .or. &
          integrator_atoms /= 2_c_int64_t .or. &
          integrator_basis_functions /= 2_c_int64_t) goto 100
      if (.not. ieee_is_finite(exc) .or. .not. all(ieee_is_finite(VXCs)) .or. &
          .not. all(ieee_is_finite(VXCz))) goto 100
      if (abs(VXCs(2) - VXCs(3)) >= 1e-10_c_double .or. &
          abs(VXCz(2) - VXCz(3)) >= 1e-10_c_double) goto 100
      rc = 0

100   continue
      if (rc == 0) then
         write (*, '(A,A)') '[PASS] Fortran native construction: ', trim(name)
      else if (status /= SKALAXC_SUCCESS) then
         write (*, '(A,A,A,A)') '[FAIL] Fortran native construction: ', &
            trim(name), ' : ', trim(skalaxc_last_error())
      else
         write (*, '(A,A,A)') '[FAIL] Fortran native construction: ', &
            trim(name), ' : validation failed'
      end if
   end function run_native_case

   !> @brief Build the full pipeline for one fixture.
   subroutine build(path, model, rt, mol, basis, mg, lb, mw, func, xc, status, &
                    execution_space, timing_settings)
      character(len=*), intent(in)                       :: path, model
      type(skalaxc_runtime_environment_t), intent(inout) :: rt
      type(skalaxc_molecule_t), intent(inout)            :: mol
      type(skalaxc_basisset_t), intent(inout)            :: basis
      type(skalaxc_molgrid_t), intent(inout)             :: mg
      type(skalaxc_load_balancer_t), intent(inout)       :: lb
      type(skalaxc_molecular_weights_t), intent(inout)   :: mw
      type(skalaxc_functional_t), intent(inout)          :: func
      type(skalaxc_xc_integrator_t), intent(inout)       :: xc
      integer(c_int), intent(out)                        :: status
      integer(c_int), intent(in), optional               :: execution_space
      type(skalaxc_timing_settings_t), intent(in), optional :: timing_settings
      integer(c_int) :: backend
      type(skalaxc_device_runtime_settings_t) :: device_settings

      backend = skalaxc_executionspace%host
      if (present(execution_space)) backend = execution_space

#ifdef SKALAXC_HAS_MPI
      if (backend == skalaxc_executionspace%device) then
         device_settings = skalaxc_device_runtime_settings_default()
         status = skalaxc_runtime_environment_create(rt, MPI_COMM_WORLD, &
                                                     device_settings)
      else
         status = skalaxc_runtime_environment_create(rt, MPI_COMM_WORLD)
      end if
#else
      if (backend == skalaxc_executionspace%device) then
         device_settings = skalaxc_device_runtime_settings_default()
         status = skalaxc_runtime_environment_create(rt, device_settings)
      else
         status = skalaxc_runtime_environment_create(rt)
      end if
#endif
      if (status /= SKALAXC_SUCCESS) return
      status = mol%from_hdf5(path, '/MOLECULE')
      if (status /= SKALAXC_SUCCESS) return
      status = basis%from_hdf5(path, '/BASIS')
      if (status /= SKALAXC_SUCCESS) return
      status = mg%create_default(mol)
      if (status /= SKALAXC_SUCCESS) return
      status = lb%create(backend, rt, mol, mg, basis)
      if (status /= SKALAXC_SUCCESS) return
      status = mw%create(backend, skalaxc_xcweightalg%ssf)
      if (status /= SKALAXC_SUCCESS) return
      status = mw%modify_weights(lb)
      if (status /= SKALAXC_SUCCESS) return
      status = func%create(model)
      if (status /= SKALAXC_SUCCESS) return
      if (present(timing_settings)) then
         status = xc%create(backend, func, lb, timing_settings=timing_settings)
      else
         status = xc%create(backend, func, lb)
      end if
   end subroutine build

   !> @brief Read a double dataset into a flat buffer. Returns .true. on success.
   logical function read_dset(file_id, name, buf, dims)
      integer(hid_t), intent(in)   :: file_id
      character(len=*), intent(in) :: name
      real(c_double), intent(out)  :: buf(:)
      integer(hsize_t), intent(in) :: dims(1)
      integer(hid_t) :: dset_id
      integer :: e
      read_dset = .false.
      call h5dopen_f(file_id, name, dset_id, e)
      if (e /= 0) return
      call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, buf, dims, e)
      call h5dclose_f(dset_id, e)
      read_dset = (e == 0)
   end function read_dset

   !> @brief Evaluate one EXC/VXC reference case. Returns 0 on pass, 1 on fail.
   integer function run_case(path, model, name, execution_space, &
                             enable_diagnostics) result(rc)
      character(len=*), intent(in) :: path, model, name
      integer(c_int), intent(in), optional :: execution_space
      logical, intent(in), optional :: enable_diagnostics
      type(skalaxc_runtime_environment_t) :: rt
      type(skalaxc_molecule_t)            :: mol
      type(skalaxc_basisset_t)            :: basis
      type(skalaxc_molgrid_t)             :: mg
      type(skalaxc_load_balancer_t)       :: lb
      type(skalaxc_molecular_weights_t)   :: mw
      type(skalaxc_functional_t)          :: func
      type(skalaxc_xc_integrator_t)       :: xc
      real(c_double), allocatable :: Ps(:), Pz(:), VXCs(:), VXCz(:), one(:)
      real(c_double) :: exc, exc_ref, rel_err, denom, sym_err, d
      integer(c_int) :: status
      integer(c_int64_t) :: nbf, n2, i, j
      integer(hid_t) :: file_id
      integer :: e
      logical :: ok, active_rank, idle_rank
      logical :: timing_enabled
      type(skalaxc_timing_settings_t) :: timing_settings
      type(skalaxc_diagnostics_snapshot_t) :: diagnostics

      rc = 1
      timing_enabled = .false.
      if (present(enable_diagnostics)) timing_enabled = enable_diagnostics
      if (timing_enabled) then
         timing_settings = skalaxc_timing_settings_default()
         call build(path, model, rt, mol, basis, mg, lb, mw, func, xc, status, &
                    execution_space, timing_settings)
      else
         call build(path, model, rt, mol, basis, mg, lb, mw, func, xc, status, &
                    execution_space)
      end if
      if (status /= SKALAXC_SUCCESS) then
         write (*, '(A,A,A,A)') '[FAIL] ', trim(name), ' : build failed: ', &
            trim(skalaxc_last_error())
         return
      end if

      nbf = xc%nbf()
      n2 = nbf*nbf
      allocate (Ps(n2), Pz(n2), VXCs(n2), VXCz(n2), one(1))

      call h5fopen_f(path, H5F_ACC_RDONLY_F, file_id, e)
      ok = (e == 0)
      if (ok) ok = read_dset(file_id, '/DENSITY_SCALAR', Ps, [int(n2, hsize_t)])
      if (ok) ok = read_dset(file_id, '/DENSITY_Z', Pz, [int(n2, hsize_t)])
      if (ok) ok = read_dset(file_id, '/EXC', one, [1_hsize_t])
      if (e == 0) call h5fclose_f(file_id, e)
      if (.not. ok) then
         write (*, '(A,A,A)') '[FAIL] ', trim(name), ' : HDF5 read failed'
         return
      end if
      exc_ref = one(1)

      status = xc%eval_exc_vxc_uks(Ps, Pz, VXCs, VXCz, exc)
      if (status /= SKALAXC_SUCCESS) then
         write (*, '(A,A,A,A)') '[FAIL] ', trim(name), ' : eval failed: ', &
            trim(skalaxc_last_error())
         return
      end if

      if (timing_enabled) then
         status = xc%diagnostics(diagnostics)
         active_rank = diagnostics%tasks > 0_c_int64_t .and. &
                       diagnostics%points > 0_c_int64_t .and. &
                       diagnostics%local_atoms == 1_c_int64_t .and. &
                       diagnostics%configured_model_batches == 1_c_int64_t .and. &
                       diagnostics%task_points_min > 0_c_int64_t .and. &
                       diagnostics%task_points_max > 0_c_int64_t .and. &
                       diagnostics%model_batches == 1_c_int64_t .and. &
                       diagnostics%domains == 1_c_int64_t .and. &
                       diagnostics%timings(skalaxc_timing_model_forward)%status == &
                       skalaxc_timingstatus_complete .and. &
                       diagnostics%timings(skalaxc_timing_model_forward)%call_count == &
                       1_c_int64_t
         idle_rank = diagnostics%tasks == 0_c_int64_t .and. &
                     diagnostics%points == 0_c_int64_t .and. &
                     diagnostics%local_atoms == 0_c_int64_t .and. &
                     diagnostics%configured_model_batches == 0_c_int64_t .and. &
                     diagnostics%task_points_min == 0_c_int64_t .and. &
                     diagnostics%task_points_max == 0_c_int64_t .and. &
                     diagnostics%model_batches == 0_c_int64_t .and. &
                     diagnostics%domains == 0_c_int64_t .and. &
                     diagnostics%timings(skalaxc_timing_model_forward)%status == &
                     skalaxc_timingstatus_unavailable .and. &
                     diagnostics%timings(skalaxc_timing_model_forward)%call_count == &
                     0_c_int64_t
         if (status /= SKALAXC_SUCCESS .or. &
             diagnostics%communicator_size < 1_c_int32_t .or. &
             diagnostics%device_id /= -1_c_int32_t .or. &
             diagnostics%openmp_threads < 1_c_int32_t .or. &
             diagnostics%exc_vxc_calls /= 1_c_int64_t .or. &
             (.not. active_rank .and. .not. idle_rank) .or. &
             diagnostics%timings(skalaxc_timing_total_exc_vxc)%call_count /= &
             1_c_int64_t) return
         status = xc%reset_diagnostics()
         if (status /= SKALAXC_SUCCESS) return
         status = xc%diagnostics(diagnostics)
         if (status /= SKALAXC_SUCCESS .or. &
             diagnostics%exc_vxc_calls /= 0_c_int64_t .or. &
             diagnostics%model_batches /= 0_c_int64_t .or. &
             diagnostics%domains /= 0_c_int64_t .or. &
             diagnostics%configured_model_batches /= &
             merge(1_c_int64_t, 0_c_int64_t, &
                   diagnostics%tasks > 0_c_int64_t) .or. &
             .not. ((diagnostics%tasks > 0_c_int64_t .and. &
                     diagnostics%points > 0_c_int64_t) .or. &
                    (diagnostics%tasks == 0_c_int64_t .and. &
                     diagnostics%points == 0_c_int64_t)) .or. &
             diagnostics%timings(skalaxc_timing_model_load)%call_count /= &
             1_c_int64_t .or. &
             diagnostics%timings(skalaxc_timing_model_forward)%status /= &
             skalaxc_timingstatus_unavailable) return
      end if

      sym_err = 0.0_c_double
      do j = 1, nbf
         do i = 1, nbf
            d = abs(VXCs((j - 1)*nbf + i) - VXCs((i - 1)*nbf + j))
            if (d > sym_err) sym_err = d
         end do
      end do

      denom = max(1.0_c_double, abs(exc_ref))
      rel_err = abs(exc - exc_ref)/denom

      if (rel_err < 1e-5_c_double .and. sym_err < 1e-10_c_double) then
         write (*, '(A,A,A,I0,A,ES20.10,A,ES10.2,A,ES10.2)') '[PASS] ', &
            trim(name), ' : nbf=', nbf, ' EXC=', exc, ' rel=', rel_err, &
            ' sym=', sym_err
         rc = 0
      else
         write (*, '(A,A,A,ES20.10,A,ES10.2,A,ES10.2)') '[FAIL] ', trim(name), &
            ' : EXC=', exc, ' rel=', rel_err, ' sym=', sym_err
      end if

   end function run_case

   !> @brief Evaluate the H2O2 gradient case. Returns 0 on pass, 1 on fail.
   integer function run_gradient_case(path, model, execution_space, name) result(rc)
      character(len=*), intent(in) :: path, model, name
      integer(c_int), intent(in) :: execution_space
      type(skalaxc_runtime_environment_t) :: rt
      type(skalaxc_molecule_t)            :: mol
      type(skalaxc_basisset_t)            :: basis
      type(skalaxc_molgrid_t)             :: mg
      type(skalaxc_load_balancer_t)       :: lb
      type(skalaxc_molecular_weights_t)   :: mw
      type(skalaxc_functional_t)          :: func
      type(skalaxc_xc_integrator_t)       :: xc
      real(c_double), allocatable :: Ps(:), Pz(:), gradient(:)
      real(c_double) :: squared_norm, translation(3)
      integer(c_int) :: status
      integer(c_int64_t) :: nbf, natoms, n2, i
      integer(hid_t) :: file_id
      integer :: e
      logical :: ok

      rc = 1
      call build(path, model, rt, mol, basis, mg, lb, mw, func, xc, status, &
                 execution_space)
      if (status /= SKALAXC_SUCCESS) then
         write (*, '(A,A,A)') '[FAIL] ', trim(name), ' : build failed: '// &
            trim(skalaxc_last_error())
         return
      end if

      nbf = xc%nbf()
      natoms = xc%natoms()
      n2 = nbf*nbf
      allocate (Ps(n2), Pz(n2), gradient(3*natoms))
      Pz = 0.0_c_double

      call h5fopen_f(path, H5F_ACC_RDONLY_F, file_id, e)
      ok = (e == 0)
      if (ok) ok = read_dset(file_id, '/DENSITY', Ps, [int(n2, hsize_t)])
      if (e == 0) call h5fclose_f(file_id, e)
      if (.not. ok) then
         write (*, '(A,A,A)') '[FAIL] ', trim(name), ' : HDF5 read failed'
         return
      end if

      status = xc%eval_exc_grad_uks(Ps, Pz, gradient)
      if (status /= SKALAXC_SUCCESS) then
         write (*, '(A,A,A)') '[FAIL] ', trim(name), ' : eval failed: '// &
            trim(skalaxc_last_error())
         return
      end if

      squared_norm = 0.0_c_double
      translation = 0.0_c_double
      do i = 1, 3*natoms
         if (.not. ieee_is_finite(gradient(i))) then
            return
         end if
         squared_norm = squared_norm + gradient(i)*gradient(i)
         translation(int(mod(i - 1, 3_c_int64_t)) + 1) = &
            translation(int(mod(i - 1, 3_c_int64_t)) + 1) + gradient(i)
      end do

      if (squared_norm > 1e-6_c_double .and. abs(translation(1)) < 1e-10_c_double &
          .and. abs(translation(2)) < 1e-10_c_double &
          .and. abs(translation(3)) < 1e-10_c_double) then
         write (*, '(A,A,A,I0,A,ES20.10)') &
            '[PASS] ', trim(name), ' : natoms=', natoms, &
            ' squared_norm=', squared_norm
         rc = 0
      else
         write (*, '(A,A,A)') '[FAIL] ', trim(name), &
            ' : validation failed'
      end if

   end function run_gradient_case

end program skalaxc_fortran_test
