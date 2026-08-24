#include "gauxc/gauxc_config.f"

program skala
  use iso_c_binding, only : c_int, c_int64_t, c_double, c_bool
  use flap, only : command_line_interface
  use gauxc_status, only : gauxc_status_type, gauxc_status_message
  use gauxc_enums, only : gauxc_executionspace, gauxc_radialquad, &
    & gauxc_atomicgridsizedefault, gauxc_pruningscheme
  use gauxc_runtime_environment, only : gauxc_runtime_environment_type, &
    & gauxc_runtime_environment_comm_rank, gauxc_runtime_environment_comm_size, &
    & gauxc_runtime_environment_new, gauxc_delete
  use gauxc_molecule, only : gauxc_molecule_type, gauxc_molecule_new, gauxc_delete
  use gauxc_basisset, only : gauxc_basisset_type, gauxc_basisset_new, gauxc_delete
  use gauxc_molgrid, only : gauxc_molgrid_type, gauxc_molgrid_new_default, &
    & gauxc_delete
  use gauxc_load_balancer, only : gauxc_load_balancer_type, gauxc_load_balancer_factory_type, &
    & gauxc_load_balancer_factory_new, gauxc_get_instance, gauxc_delete
  use gauxc_molecular_weights, only : gauxc_molecular_weights_type, &
    & gauxc_molecular_weights_factory_type, gauxc_molecular_weights_factory_new, &
    & gauxc_molecular_weights_modify_weights, gauxc_get_instance, gauxc_delete, &
    & gauxc_molecular_weights_settings
  use gauxc_xc_functional, only : gauxc_functional_type, gauxc_functional_from_string, gauxc_delete
  use gauxc_integrator, only : gauxc_integrator_type, gauxc_integrator_new, &
    & gauxc_eval_exc_vxc, gauxc_delete
  use gauxc_external_hdf5_read, only : gauxc_read_hdf5_record
#ifdef GAUXC_HAS_MPI
  use mpi
#endif

  implicit none(type, external)

  type(gauxc_status_type) :: status
  type(gauxc_runtime_environment_type) :: rt
  type(gauxc_molecule_type) :: mol
  type(gauxc_basisset_type) :: basis
  type(gauxc_molgrid_type) :: grid
  type(gauxc_load_balancer_factory_type) :: lbf
  type(gauxc_load_balancer_type) :: lb
  type(gauxc_molecular_weights_factory_type) :: mwf
  type(gauxc_molecular_weights_type) :: mw
  type(gauxc_functional_type) :: func
  type(gauxc_integrator_type) :: integrator
  real(c_double), allocatable :: p_s(:,:), p_z(:,:), vxc_s(:,:), vxc_z(:,:)

  character(len=:), allocatable :: input_file, model, grid_spec, rad_quad_spec, prune_spec, &
    & lb_exec_space_str, int_exec_space_str
  integer(c_int) :: grid_type, radial_quad, pruning_scheme
  integer(c_int) :: lb_exec_space, int_exec_space
  integer(c_int) :: world_rank, world_size
  integer(c_int64_t) :: batch_size
  real(c_double) :: exc, t_exc, t_start, t_end
  type(command_line_interface) :: cli
  integer :: error

#ifdef GAUXC_HAS_MPI
  call MPI_Init(error)
#endif

  grid_spec = "fine"
  rad_quad_spec = "muraknowles"
  prune_spec = "robust"
  lb_exec_space_str = "host"
  int_exec_space_str = "host"
  batch_size = 512_c_int64_t

  input: block
  character(len=512) :: dummy
  call cli%init(description="Driver for using Skala")
  call cli%add(position=1, required=.true., act="store", error=error, positional=.true., &
    & help="Input HDF5 file containing molecule, basis set and density matrix")
  if (error /= 0) exit input
  call cli%add(switch="--model", required=.true., act="store", error=error, &
    & help="Model to use for the calculation")
  if (error /= 0) exit input
  call cli%add(switch="--grid", act="store", error=error, def=grid_spec, required=.false., &
    & help="Molecular grid specification", choices="fine,ultrafine,superfine,gm3,gm5")
  if (error /= 0) exit input
  call cli%add(switch="--radial-quadrature", act="store", error=error, def=rad_quad_spec, &
    & help="Radial quadrature to use", choices="becke,muraknowles,treutlerahlrichs,murrayhandylaming")
  if (error /= 0) exit input
  call cli%add(switch="--pruning-scheme", act="store", error=error, def=prune_spec, &
    & help="Pruning scheme to use", choices="unpruned,robust,treutler")
  if (error /= 0) exit input
  call cli%add(switch="--lb-exec-space", act="store", error=error, def=lb_exec_space_str, &
    & help="Execution space for load balancer", choices="host,device")
  if (error /= 0) exit input
  call cli%add(switch="--int-exec-space", act="store", error=error, def=int_exec_space_str, &
    & help="Execution space for integrator", choices="host,device")
  if (error /= 0) exit input
  call cli%add(switch="--batch-size", act="store", error=error, def="512", &
    & help="Batch size for grid point processing")
  if (error /= 0) exit input
    
  call cli%parse(error=error)
  if (error /= 0) exit input

  call cli%get(position=1, val=dummy, error=error)
  input_file = trim(dummy)
  if (error /= 0) exit input
  call cli%get(switch="--model", val=dummy, error=error)
  model = trim(dummy)
  if (error /= 0) exit input
  if (cli%is_passed(switch="--grid")) then
    call cli%get(switch="--grid", val=dummy, error=error)
    if (error /= 0) exit input
    grid_spec = trim(dummy)
  end if
  if (cli%is_passed(switch="--radial-quadrature")) then
    call cli%get(switch="--radial-quadrature", val=dummy, error=error)
    if (error /= 0) exit input
    rad_quad_spec = trim(dummy)
  end if
  if (cli%is_passed(switch="--pruning-scheme")) then
    call cli%get(switch="--pruning-scheme", val=dummy, error=error)
    if (error /= 0) exit input
    prune_spec = trim(dummy)
  end if
  if (cli%is_passed(switch="--lb-exec-space")) then
    call cli%get(switch="--lb-exec-space", val=dummy, error=error)
    if (error /= 0) exit input
    lb_exec_space_str = trim(dummy)
  end if
  if (cli%is_passed(switch="--int-exec-space")) then
    call cli%get(switch="--int-exec-space", val=dummy, error=error)
    if (error /= 0) exit input
    int_exec_space_str = trim(dummy)
  end if
  if (cli%is_passed(switch="--batch-size")) then
    call cli%get(switch="--batch-size", val=batch_size, error=error)
    if (error /= 0) exit input
  end if
  end block input
  if (error /= 0) then
    print '(a)', cli%error_message
#ifdef GAUXC_HAS_MPI
    call MPI_Abort(MPI_COMM_WORLD, 1, error)
#else
    stop 1
#endif
  end if

#ifdef GAUXC_HAS_MPI
  call MPI_Barrier(MPI_COMM_WORLD, error)
#endif

  main: block
  ! Create runtime
#ifdef GAUXC_HAS_MPI
  rt = gauxc_runtime_environment_new(status, MPI_COMM_WORLD)
#else
  rt = gauxc_runtime_environment_new(status)
#endif
  if (status%code /= 0) exit main
  world_rank = gauxc_runtime_environment_comm_rank(status, rt)
  if (status%code /= 0) exit main
  world_size = gauxc_runtime_environment_comm_size(status, rt)
  if (status%code /= 0) exit main

  if (world_rank == 0) then
    print '(a)', &
      & "Configuration", &
      & "-> Input file        : "//input_file, &
      & "-> Model             : "//model, &
      & "-> Grid              : "//grid_spec, &
      & "-> Radial quadrature : "//rad_quad_spec, &
      & "-> Pruning scheme    : "//prune_spec, &
      & ""
  end if

  ! Get molecule (atomic numbers and cartesian coordinates)
  mol = gauxc_molecule_new(status)
  if (status%code /= 0) exit main
  ! Load molecule from HDF5 dataset
  call gauxc_read_hdf5_record(status, mol, input_file, "/MOLECULE")
  if (status%code /= 0) exit main

  ! Get basis set
  basis = gauxc_basisset_new(status)
  if (status%code /= 0) exit main
  ! Load basis set from HDF5 dataset
  call gauxc_read_hdf5_record(status, basis, input_file, "/BASIS")
  if (status%code /= 0) exit main

  ! Define molecular grid from grid size, radial quadrature and pruning scheme
  grid_type = read_atomic_grid_size(grid_spec)
  radial_quad = read_radial_quad(rad_quad_spec)
  pruning_scheme = read_pruning_scheme(prune_spec)
  grid = gauxc_molgrid_new_default(status, mol, pruning_scheme, batch_size, radial_quad, grid_type)
  if (status%code /= 0) exit main

  ! Choose whether we run on host or device
  lb_exec_space = read_execution_space(lb_exec_space_str)
  int_exec_space = read_execution_space(int_exec_space_str)

  ! Setup load balancer based on molecule, grid and basis set
  lbf = gauxc_load_balancer_factory_new(status, lb_exec_space, "Replicated")
  if (status%code /= 0) exit main
  lb = gauxc_get_instance(status, lbf, rt, mol, grid, basis)
  if (status%code /= 0) exit main

  ! Apply partitioning weights to the molecule grid
  mwf = gauxc_molecular_weights_factory_new(status, int_exec_space)
  if (status%code /= 0) exit main
  mw = gauxc_get_instance(status, mwf)
  if (status%code /= 0) exit main
  call gauxc_molecular_weights_modify_weights(status, mw, lb)
  if (status%code /= 0) exit main

  ! Setup exchange-correlation integrator
  func = gauxc_functional_from_string(status, "PBE", .true._c_bool)
  integrator = gauxc_integrator_new(status, func, lb, int_exec_space)
  if (status%code /= 0) exit main

  ! Load density matrix from input
  call read_matrix_from_hdf5_record(status, p_s, input_file, "/DENSITY_SCALAR")
  if (status%code /= 0) exit main
  call read_matrix_from_hdf5_record(status, p_z, input_file, "/DENSITY_Z")
  if (status%code /= 0) exit main

#ifdef GAUXC_HAS_MPI
  call MPI_Barrier(MPI_COMM_WORLD, error)
#endif
  t_start = timing()

  ! Integrate exchange-correlation energy
  allocate(vxc_s(size(p_s,1), size(p_s,2)))
  allocate(vxc_z(size(p_z,1), size(p_z,2)))
  call gauxc_eval_exc_vxc(status, integrator, p_s, p_z, model, exc, vxc_s, vxc_z)
  if (status%code /= 0) exit main

#ifdef GAUXC_HAS_MPI
  call MPI_Barrier(MPI_COMM_WORLD, error)
#endif
  t_end = timing()
  t_exc = t_end - t_start

  if (world_rank == 0) then
    print '(a)', "Results"
    print '(a,1x,es17.10,:1x,a)', "Exc          =", exc, "Eh"
    print '(a,1x,es17.10,:1x,a)', "|VXC(a+b)|_F =", sqrt(sum(vxc_s**2))
    print '(a,1x,es17.10,:1x,a)', "|VXC(a-b)|_F =", sqrt(sum(vxc_z**2))
    print '(a,1x,es17.10,:1x,a)', "Runtime XC   =", t_exc
  end if

  end block main
  if (world_rank == 0 .and. status%code /= 0) then
    print '(a,1x,i0)', "GauXC returned with status code", status%code
    print '(a)', gauxc_status_message(status)
  end if

  call gauxc_delete(status, rt)
  call gauxc_delete(status, mol)
  call gauxc_delete(status, basis)
  call gauxc_delete(status, grid)
  call gauxc_delete(status, lbf)
  call gauxc_delete(status, lb)
  call gauxc_delete(status, mwf)
  call gauxc_delete(status, mw)
  call gauxc_delete(status, func)
  call gauxc_delete(status, integrator)

#ifdef GAUXC_HAS_MPI
  call MPI_Finalize(error)
#endif

contains

  pure function read_execution_space(spec) result(val)
    character(len=*), intent(in) :: spec
    integer(c_int) :: val
    select case(spec)
    case("host")
      val = gauxc_executionspace%host
    case("device")
      val = gauxc_executionspace%device
    end select
  end function read_execution_space

  pure function read_atomic_grid_size(spec) result(val)
    character(len=*), intent(in) :: spec
    integer(c_int) :: val
    select case(spec)
    case("fine")
      val = gauxc_atomicgridsizedefault%finegrid
    case("ultrafine")
      val = gauxc_atomicgridsizedefault%ultrafinegrid
    case("superfine")
      val = gauxc_atomicgridsizedefault%superfinegrid
    case("gm3")
      val = gauxc_atomicgridsizedefault%gm3
    case("gm5")
      val = gauxc_atomicgridsizedefault%gm5
    end select
  end function read_atomic_grid_size
  
  pure function read_radial_quad(spec) result(val)
    character(len=*), intent(in) :: spec
    integer(c_int) :: val
    select case(spec)
    case("becke")
      val = gauxc_radialquad%becke
    case("muraknowles")
      val = gauxc_radialquad%mura_knowles
    case("treutlerahlrichs")
      val = gauxc_radialquad%treutler_ahlrichs
    case("murrayhandylaming")
      val = gauxc_radialquad%murray_handy_laming
    end select
  end function read_radial_quad
  
  pure function read_pruning_scheme(spec) result(val)
    character(len=*), intent(in) :: spec
    integer(c_int) :: val
    select case(spec)
    case("unpruned")
      val = gauxc_pruningscheme%unpruned
    case("robust")
      val = gauxc_pruningscheme%robust
    case("treutler")
      val = gauxc_pruningscheme%treutler
    end select
  end function read_pruning_scheme

  subroutine read_matrix_from_hdf5_record(status, mat, file, dataset)
    use hdf5, only : hid_t, hsize_t, H5F_ACC_RDONLY_F, H5T_NATIVE_DOUBLE, &
      & h5open_f, h5close_f, h5fopen_f, h5dopen_f, h5dget_space_f, &
      & h5sget_simple_extent_dims_f, h5dread_f, h5fclose_f, h5dclose_f, h5sclose_f
    type(gauxc_status_type), intent(out) :: status
    real(c_double), allocatable :: mat(:,:)
    character(len=*), intent(in) :: file, dataset

    logical :: file_opened, dataset_opened, dataspace_obtained
    integer(hid_t) :: file_id, dataset_id, dataspace_id
    integer :: hdf5_status
    integer(hsize_t), dimension(2) :: dims, maxdims

    file_opened = .false.
    dataset_opened = .false.
    dataspace_obtained = .false.
    status%code = 0

    h5: block
    ! Initialize HDF5 Fortran interface
    call h5open_f(hdf5_status)
    if (hdf5_status < 0) exit h5

    ! Open the HDF5 file and dataset
    call h5fopen_f(trim(file), H5F_ACC_RDONLY_F, file_id, hdf5_status)
    if (hdf5_status < 0) exit h5
    file_opened = .true.

    call h5dopen_f(file_id, trim(dataset), dataset_id, hdf5_status)
    if (hdf5_status < 0) exit h5
    dataset_opened = .true.

    call h5dget_space_f(dataset_id, dataspace_id, hdf5_status)
    if (hdf5_status < 0) exit h5
    dataspace_obtained = .true.

    ! Get the dimensions of the dataset
    call h5sget_simple_extent_dims_f(dataspace_id, dims, maxdims, hdf5_status)
    if (hdf5_status < 0) exit h5

    ! Allocate the matrix and read the data
    allocate(mat(dims(1), dims(2)))
    call h5dread_f(dataset_id, H5T_NATIVE_DOUBLE, mat, dims, hdf5_status)
    if (hdf5_status < 0) exit h5
    end block h5

    if (hdf5_status < 0) then
      status%code = 1
      if (allocated(mat)) deallocate(mat)
    end if

    ! Close HDF5 resources (reverse order)
    if (dataspace_obtained) call h5sclose_f(dataspace_id, hdf5_status)
    if (dataset_opened) call h5dclose_f(dataset_id, hdf5_status)
    if (file_opened) call h5fclose_f(file_id, hdf5_status)
    call h5close_f(hdf5_status)
  end subroutine read_matrix_from_hdf5_record

  function timing() result(time)
    real(c_double) :: time
    integer(c_int64_t) :: time_count, time_rate, time_max
    call system_clock(time_count, time_rate, time_max)
    time = real(time_count, c_double) / real(time_rate, c_double)
  end function timing
end program skala