program installed_consumer_fortran
   use, intrinsic :: iso_c_binding, only: c_double, c_int, c_int32_t, c_int64_t
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use skalaxc
#ifdef SKALAXC_HAS_MPI
   use mpi, only: MPI_COMM_WORLD, MPI_Finalize, MPI_Init, MPI_SUCCESS
#endif
   implicit none

   type(skalaxc_runtime_environment_t) :: runtime
   type(skalaxc_molecule_t) :: molecule
   type(skalaxc_basisset_t) :: basis
   type(skalaxc_molgrid_t) :: grid
   type(skalaxc_load_balancer_t) :: load_balancer
   type(skalaxc_molecular_weights_t) :: weights
   type(skalaxc_functional_t) :: functional
   type(skalaxc_xc_integrator_t) :: integrator
   integer(c_int64_t) :: atomic_numbers(2)
   integer(c_int32_t) :: shell_l(2), shell_pure(2), shell_nprim(2)
   real(c_double) :: atom_xyz(6), exponents(6), coefficients(6)
   real(c_double) :: scalar_density(4), spin_density(4)
   real(c_double) :: scalar_potential(4), spin_potential(4), energy
   integer(c_int) :: status
#ifdef SKALAXC_HAS_MPI
   integer :: mpi_error
#endif

#ifdef SKALAXC_HAS_MPI
   call MPI_Init(mpi_error)
   if (mpi_error /= MPI_SUCCESS) stop 1
#endif

   block
      character(len=:), allocatable :: version
      version = skalaxc_version()
      if (len(version) == 0) stop 1
   end block

   atomic_numbers = [1_c_int64_t, 1_c_int64_t]
   shell_l = [0_c_int32_t, 0_c_int32_t]
   shell_pure = [0_c_int32_t, 0_c_int32_t]
   shell_nprim = [3_c_int32_t, 3_c_int32_t]
   atom_xyz = [-0.7_c_double, 0.0_c_double, 0.0_c_double, &
               0.7_c_double, 0.0_c_double, 0.0_c_double]
   exponents = [3.42525091_c_double, 0.62391373_c_double, 0.16885540_c_double, &
                3.42525091_c_double, 0.62391373_c_double, 0.16885540_c_double]
   coefficients = [0.15432897_c_double, 0.53532814_c_double, 0.44463454_c_double, &
                   0.15432897_c_double, 0.53532814_c_double, 0.44463454_c_double]
   scalar_density = 0.5_c_double
   spin_density = 0.0_c_double

#ifdef SKALAXC_HAS_MPI
   status = skalaxc_runtime_environment_create(runtime, MPI_COMM_WORLD)
#else
   status = skalaxc_runtime_environment_create(runtime)
#endif
   if (status /= SKALAXC_SUCCESS) stop 1
   status = molecule%from_arrays(atomic_numbers, atom_xyz)
   if (status /= SKALAXC_SUCCESS) stop 1
   status = basis%from_arrays(shell_l, shell_pure, atom_xyz, shell_nprim, &
                              exponents, coefficients)
   if (status /= SKALAXC_SUCCESS) stop 1
   status = grid%create_default(molecule)
   if (status /= SKALAXC_SUCCESS) stop 1
   status = load_balancer%create(skalaxc_executionspace%host, runtime, &
                                 molecule, grid, basis)
   if (status /= SKALAXC_SUCCESS) stop 1
   status = weights%create(skalaxc_executionspace%host, skalaxc_xcweightalg%ssf)
   if (status /= SKALAXC_SUCCESS) stop 1
   status = weights%modify_weights(load_balancer)
   if (status /= SKALAXC_SUCCESS) stop 1
   status = functional%create('LDA')
   if (status /= SKALAXC_SUCCESS) stop 1
   status = integrator%create(skalaxc_executionspace%host, functional, load_balancer)
   if (status /= SKALAXC_SUCCESS) stop 1
   status = integrator%eval_exc_vxc_uks(scalar_density, spin_density, &
                                        scalar_potential, spin_potential, energy)
   if (status /= SKALAXC_SUCCESS) stop 1
   if (.not. ieee_is_finite(energy) .or. &
       .not. all(ieee_is_finite(scalar_potential)) .or. &
       .not. all(ieee_is_finite(spin_potential))) stop 1

#ifdef SKALAXC_HAS_MPI
   call MPI_Finalize(mpi_error)
   if (mpi_error /= MPI_SUCCESS) stop 1
#endif
end program installed_consumer_fortran
