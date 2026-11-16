program skalaxc_fortran_array_lifecycle
   use, intrinsic :: iso_c_binding
   use skalaxc
#ifdef SKALAXC_HAS_MPI
   use mpi
#endif
   implicit none

#if HANDLE_RANK == 1
#define SHAPE (2)
#define DIMS (:)
#define AT(i) (i)
#else
#define SHAPE (2,1)
#define DIMS (:,:)
#define AT(i) (i,1)
#endif
   type handles
      type(skalaxc_runtime_environment_t), allocatable :: rt DIMS
      type(skalaxc_molecule_t), allocatable :: mol DIMS
      type(skalaxc_basisset_t), allocatable :: basis DIMS
      type(skalaxc_molgrid_t), allocatable :: grid DIMS
      type(skalaxc_load_balancer_t), allocatable :: lb DIMS
      type(skalaxc_molecular_weights_t), allocatable :: weights DIMS
      type(skalaxc_functional_t), allocatable :: func DIMS
      type(skalaxc_xc_integrator_t), allocatable :: xc DIMS
   end type handles
   interface
      integer(c_int) function counts_equal(expected) bind(C, name="skalaxc_test_destroy_counts_equal")
         import c_int
         integer(c_int), value :: expected
      end function counts_equal
   end interface
#ifdef SKALAXC_HAS_MPI
   integer :: ierr
   call MPI_Init(ierr)
#endif

   call exercise_arrays()
   if (counts_equal(2_c_int) /= 1) error stop "scope exit must destroy each remaining handle once"
#ifdef SKALAXC_HAS_MPI
   call MPI_Finalize(ierr)
#endif

contains

   subroutine check(status)
      integer(c_int), intent(in) :: status
      if (status /= SKALAXC_SUCCESS) then
         print *, skalaxc_last_error()
         error stop "handle construction failed"
      end if
   end subroutine check

   subroutine exercise_arrays()
      type(handles) :: source
      type(handles), allocatable :: destination
      integer :: i
      allocate (destination)
      allocate (source%rt SHAPE, source%mol SHAPE, source%basis SHAPE, source%grid SHAPE, &
                source%lb SHAPE, source%weights SHAPE, source%func SHAPE, source%xc SHAPE)
      allocate (destination%rt SHAPE, destination%mol SHAPE, destination%basis SHAPE, &
                destination%grid SHAPE, destination%lb SHAPE, destination%weights SHAPE, &
                destination%func SHAPE, destination%xc SHAPE)
      do i = 1, 2
#ifdef SKALAXC_HAS_MPI
         call check(skalaxc_runtime_environment_create(source%rt AT(i), MPI_COMM_SELF))
#else
         call check(skalaxc_runtime_environment_create(source%rt AT(i)))
#endif
         call check(source%mol AT(i)%create())
         call check(source%mol AT(i)%add_atom(2_c_int64_t, 0._c_double, 0._c_double, 0._c_double))
         call check(source%basis AT(i)%create())
         call check(source%basis AT(i)%add_shell(0_c_int32_t, 0_c_int32_t, &
                                                 [0._c_double, 0._c_double, 0._c_double], &
                                                 [1._c_double], [1._c_double]))
         call check(source%grid AT(i)%create_default(source%mol AT(i)))
         call check(source%lb AT(i)%create(skalaxc_executionspace%host, source%rt AT(i), &
                                           source%mol AT(i), source%grid AT(i), source%basis AT(i)))
         call check(source%weights AT(i)%create(skalaxc_executionspace%host, skalaxc_xcweightalg%ssf))
         call check(source%weights AT(i)%modify_weights(source%lb AT(i)))
         call check(source%func AT(i)%create('LDA'))
         call check(source%xc AT(i)%create(skalaxc_executionspace%host, source%func AT(i), source%lb AT(i)))
      end do
      if (counts_equal(0_c_int) /= 1) error stop "unexpected destruction during construction"
      call destination%rt AT(1)%move_from(source%rt AT(1))
      call destination%mol AT(1)%move_from(source%mol AT(1))
      call destination%basis AT(1)%move_from(source%basis AT(1))
      call destination%grid AT(1)%move_from(source%grid AT(1))
      call destination%lb AT(1)%move_from(source%lb AT(1))
      call destination%weights AT(1)%move_from(source%weights AT(1))
      call destination%func AT(1)%move_from(source%func AT(1))
      call destination%xc AT(1)%move_from(source%xc AT(1))
      deallocate (destination%xc, destination%func, destination%weights, destination%lb, &
                  destination%grid, destination%basis, destination%mol, destination%rt)
      deallocate (destination)
      if (counts_equal(1_c_int) /= 1) error stop "deallocation must destroy each moved handle once"
   end subroutine exercise_arrays
end program skalaxc_fortran_array_lifecycle
