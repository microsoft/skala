! SkalaXC Fortran example: evaluate a UKS machine-learning exchange-correlation
! functional on a system + density loaded from an HDF5 file.
!
! Consumption contract: this program `use`s ONLY the `skalaxc` module and links
! ONLY libskalaxc (through skalaxc_fortran). HDF5-Fortran is used purely to load
! the caller's own density input.
!
! The program follows the SkalaXC pipeline (one opaque derived type per stage):
!   runtime environment -> molecule / basis set -> molecular grid ->
!   load balancer -> molecular weights -> functional -> XC integrator
!
! Usage: skalaxc_eval_fortran <system.hdf5> [model]
!   <system.hdf5>  HDF5 file with /MOLECULE, /BASIS, /DENSITY_SCALAR, /DENSITY_Z
!   [model]        "LDA", "PBE" (default), "TPSS", or a path to a .fun model

program skalaxc_eval_fortran
   use, intrinsic :: iso_c_binding
   use skalaxc
#ifdef SKALAXC_HAS_MPI
   use mpi
#endif
   use hdf5
   implicit none

   character(len=1024) :: path, model
   integer :: nargs, ierr

   nargs = command_argument_count()
   if (nargs < 1) then
      write (*, *) 'usage: skalaxc_eval_fortran <system.hdf5> [model]'
      stop 2
   end if
   call get_command_argument(1, path)
   if (nargs >= 2) then
      call get_command_argument(2, model)
   else
      model = 'PBE'
   end if

#ifdef SKALAXC_HAS_MPI
   call MPI_Init(ierr)
   if (ierr /= MPI_SUCCESS) stop 'MPI_Init failed'
#endif

   call h5open_f(ierr)
   if (ierr /= 0) stop 'h5open failed'

   block
      type(skalaxc_runtime_environment_t) :: rt
      type(skalaxc_molecule_t)            :: mol
      type(skalaxc_basisset_t)            :: basis
      type(skalaxc_molgrid_t)             :: mg
      type(skalaxc_load_balancer_t)       :: lb
      type(skalaxc_molecular_weights_t)   :: mw
      type(skalaxc_functional_t)          :: func
      type(skalaxc_xc_integrator_t)       :: xc
      integer(c_int64_t) :: nbf, n2, i
      integer(hid_t)     :: file_id, dset_id
      integer(hsize_t)   :: dims(1)
      real(c_double), allocatable :: Ps(:), Pz(:), VXCs(:), VXCz(:)
      real(c_double) :: exc, vs

      ! 1. Runtime environment.
#ifdef SKALAXC_HAS_MPI
      call check(skalaxc_runtime_environment_create(rt, MPI_COMM_WORLD))
#else
      call check(skalaxc_runtime_environment_create(rt))
#endif

      ! 2. Load the molecule and basis set from the file.
      call check(mol%from_hdf5(trim(path), '/MOLECULE'))
      call check(basis%from_hdf5(trim(path), '/BASIS'))

      ! 3. Build the default molecular integration grid.
      call check(mg%create_default(mol))

      ! 4. Distribute the quadrature tasks.
      call check(lb%create(skalaxc_executionspace%host, rt, mol, mg, basis))

      ! 5. Partition the quadrature weights in place.
      call check(mw%create(skalaxc_executionspace%host, skalaxc_xcweightalg%ssf))
      call check(mw%modify_weights(lb))

      ! 6. Select the Skala ML functional model.
      call check(func%create(trim(model)))

      ! 7. Build the XC integrator.
      call check(xc%create(skalaxc_executionspace%host, func, lb))

      nbf = xc%nbf()
      n2 = nbf*nbf
      allocate (Ps(n2), Pz(n2), VXCs(n2), VXCz(n2))

      ! 8. Load the input spin densities (your data; here from the same file).
      call h5fopen_f(trim(path), H5F_ACC_RDONLY_F, file_id, ierr)
      if (ierr /= 0) stop 'h5fopen failed'
      dims(1) = int(n2, hsize_t)
      call h5dopen_f(file_id, '/DENSITY_SCALAR', dset_id, ierr)
      call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, Ps, dims, ierr)
      call h5dclose_f(dset_id, ierr)
      call h5dopen_f(file_id, '/DENSITY_Z', dset_id, ierr)
      call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, Pz, dims, ierr)
      call h5dclose_f(dset_id, ierr)
      call h5fclose_f(file_id, ierr)

      ! 9. Evaluate the ML exchange-correlation energy and potential (UKS).
      call check(xc%eval_exc_vxc_uks(Ps, Pz, VXCs, VXCz, exc))

      ! 10. Report the energy and a summary of the potential.
      vs = 0.0_c_double
      do i = 1, n2
         vs = vs + VXCs(i)*VXCs(i)
      end do
      write (*, '(A,A,A,I0,A,ES20.10,A,ES14.6)') 'model=', trim(model), &
         ' nbf=', nbf, ' EXC=', exc, ' |VXC_scalar|_F=', sqrt(vs)
   end block

   call h5close_f(ierr)

#ifdef SKALAXC_HAS_MPI
   call MPI_Finalize(ierr)
#endif

contains

   !> @brief Abort with the SkalaXC error message if status is not success.
   subroutine check(status)
      integer(c_int), intent(in) :: status
      if (status /= SKALAXC_SUCCESS) then
         write (*, '(A,A)') 'SkalaXC error: ', trim(skalaxc_last_error())
         stop 1
      end if
   end subroutine check

end program skalaxc_eval_fortran
