program main
  use iso_c_binding, only : c_double
  use ftorch, only : torch_tensor
  use skala_ftorch, only : skala_model, skala_model_load, skala_feature, skala_tensor_load, &
    & skala_tensor_sum, skala_tensor_mean, skala_tensor_mul, skala_tensor_item_double, &
    & skala_tensor_to_array, skala_dict, skala_dict_new

  implicit none

  type(skala_model) :: model
  type(skala_dict) :: input, vxc
  type(torch_tensor) :: exc
  type(torch_tensor) :: density, grad, kin, grid_coords, grid_weights, coarse_0_atomic_coords
  type(torch_tensor) :: dexc_ddensity, dexc_dgrad, dexc_dkin, dexc_dgrid_coords, &
    & dexc_dgrid_weights, dexc_dcoarse_0_atomic_coords, vxc_norm

  character(len=:), allocatable :: path, feature_dir

  cli_input: block
  call get_argument(1, path)
  if (.not. allocated(path)) exit cli_input
  call get_argument(2, feature_dir)
  if (.not. allocated(feature_dir)) exit cli_input
  end block cli_input
  if (.not. allocated(path) .or. .not.allocated(feature_dir)) then
    call get_argument(0, path)
    print '(a)', "Usage: "//path//" <model_path> <feature_dir>"
    stop 1
  end if

  ! Load the model
  print '(a)', "[1] Loading model from "//path
  call skala_model_load(model, path)

  print '(a)', "[2] Loading features from "//feature_dir
  get_features: block
  integer :: ift
  do ift = 1, size(model%features)
    select case(model%features(ift))
    case(skala_feature%density)
      print '(a)', " -> Loading density"
      call skala_tensor_load(density, feature_dir//"/density.pt")
    case(skala_feature%grad)
      print '(a)', " -> Loading grad"
      call skala_tensor_load(grad, feature_dir//"/grad.pt")
    case(skala_feature%kin)
      print '(a)', " -> Loading kin"
      call skala_tensor_load(kin, feature_dir//"/kin.pt")
    case(skala_feature%grid_coords)
      print '(a)', " -> Loading grid_coords"
      call skala_tensor_load(grid_coords, feature_dir//"/grid_coords.pt")
    case(skala_feature%grid_weights)
      print '(a)', " -> Loading grid_weights"
      call skala_tensor_load(grid_weights, feature_dir//"/grid_weights.pt")
    case(skala_feature%coarse_0_atomic_coords)
      print '(a)', " -> Loading coarse_0_atomic_coords"
      call skala_tensor_load(coarse_0_atomic_coords, feature_dir//"/coarse_0_atomic_coords.pt")
    end select
  end do
  end block get_features

  ! Prepare the input dictionary for the model
  print '(a)', "[3] Preparing input dictionary"
  call skala_dict_new(input)
  if (model%needs_feature(skala_feature%density)) &
    call input%insert(skala_feature%density, density)
  if (model%needs_feature(skala_feature%grad)) &
    call input%insert(skala_feature%grad, grad)
  if (model%needs_feature(skala_feature%kin)) &
    call input%insert(skala_feature%kin, kin)
  if (model%needs_feature(skala_feature%grid_coords)) &
    call input%insert(skala_feature%grid_coords, grid_coords)
  if (model%needs_feature(skala_feature%grid_weights)) &
    call input%insert(skala_feature%grid_weights, grid_weights)
  if (model%needs_feature(skala_feature%coarse_0_atomic_coords)) &
    call input%insert(skala_feature%coarse_0_atomic_coords, coarse_0_atomic_coords)

  ! Request exc and vxc from the model
  print '(a)', "[4] Running model inference"
  call model%get_exc_and_vxc(input, exc, vxc)

  ! Print the exchange-correlation energy
  print '(a)', "[5] Computing XC energy = sum(exc * grid_weights)"
  exc_weighted: block
  type(torch_tensor) :: weighted, weighted_sum
  call skala_tensor_mul(exc, grid_weights, weighted)
  call skala_tensor_sum(weighted, weighted_sum)
  print '(a, es22.14)', " -> E_xc = ", skala_tensor_item_double(weighted_sum)
  end block exc_weighted

  print '(a)', "[6] Extracting vxc components"
  if (model%needs_feature(skala_feature%density)) &
    call vxc%at(skala_feature%density, dexc_ddensity)
  if (model%needs_feature(skala_feature%grad)) &
    call vxc%at(skala_feature%grad, dexc_dgrad)
  if (model%needs_feature(skala_feature%kin)) &
    call vxc%at(skala_feature%kin, dexc_dkin)
  if (model%needs_feature(skala_feature%grid_coords)) &
    call vxc%at(skala_feature%grid_coords, dexc_dgrid_coords)
  if (model%needs_feature(skala_feature%grid_weights)) &
    call vxc%at(skala_feature%grid_weights, dexc_dgrid_weights)
  if (model%needs_feature(skala_feature%coarse_0_atomic_coords)) &
    call vxc%at(skala_feature%coarse_0_atomic_coords, dexc_dcoarse_0_atomic_coords)

  ! Print mean of each gradient component
  print '(a)', "[7] Gradient means (dexc/dx)"
  print_gradients: block
  type(torch_tensor) :: grad_mean
  if (model%needs_feature(skala_feature%density)) then
    call skala_tensor_mean(dexc_ddensity, grad_mean)
    print '(a, es22.14)', " -> mean(dexc/d_density)                = ", &
      skala_tensor_item_double(grad_mean)
  end if
  if (model%needs_feature(skala_feature%grad)) then
    call skala_tensor_mean(dexc_dgrad, grad_mean)
    print '(a, es22.14)', " -> mean(dexc/d_grad)                   = ", &
      skala_tensor_item_double(grad_mean)
  end if
  if (model%needs_feature(skala_feature%kin)) then
    call skala_tensor_mean(dexc_dkin, grad_mean)
    print '(a, es22.14)', " -> mean(dexc/d_kin)                    = ", &
      skala_tensor_item_double(grad_mean)
  end if
  if (model%needs_feature(skala_feature%grid_coords)) then
    call skala_tensor_mean(dexc_dgrid_coords, grad_mean)
    print '(a, es22.14)', " -> mean(dexc/d_grid_coords)            = ", &
      skala_tensor_item_double(grad_mean)
  end if
  if (model%needs_feature(skala_feature%grid_weights)) then
    call skala_tensor_mean(dexc_dgrid_weights, grad_mean)
    print '(a, es22.14)', " -> mean(dexc/d_grid_weights)           = ", &
      skala_tensor_item_double(grad_mean)
  end if
  if (model%needs_feature(skala_feature%coarse_0_atomic_coords)) then
    call skala_tensor_mean(dexc_dcoarse_0_atomic_coords, grad_mean)
    print '(a, es22.14)', " -> mean(dexc/d_coarse_0_atomic_coords) = ", &
      skala_tensor_item_double(grad_mean)
  end if
  end block print_gradients

  ! Demonstrate direct Fortran array access to tensor data
  print '(a)', "[8] Accessing tensor data as Fortran arrays"
  array_access: block
  real(c_double), pointer :: arr1d(:), arr2d(:,:), arr3d(:,:,:)

  ! exc is 1-D (npts)
  call skala_tensor_to_array(exc, arr1d)
  print '(a, i0, a)', " -> exc: shape = (", size(arr1d), ")"
  print '(a, 3es22.14, a)', &
    "      [", arr1d(:3), " ...]"

  ! density is 2-D (nspin, npts)
  if (model%needs_feature(skala_feature%density)) then
    call skala_tensor_to_array(dexc_ddensity, arr2d)
    print '(a, i0, a, i0, a)', " -> dexc/d_density: shape = (", &
      size(arr2d, 1), ", ", size(arr2d, 2), ")"
    print '(a, 3es22.14, a)', &
      "     [[", arr2d(:3, 1), " ...]", &
      "      [", arr2d(:3, 2), " ...]]"
  end if

  ! grad is 3-D (nspin, 3, npts)
  if (model%needs_feature(skala_feature%grad)) then
    call skala_tensor_to_array(dexc_dgrad, arr3d)
    print '(a, i0, a, i0, a, i0, a)', " -> dexc/d_grad: shape = (", &
      size(arr3d, 1), ", ", size(arr3d, 2), ", ", size(arr3d, 3), ")"
    print '(a, 3es22.14, a)', &
      "    [[[", arr3d(:3, 1, 1), " ...]", &
      "      [", arr3d(:3, 2, 1), " ...]]", &
      "     [[  ...                    ]]]"
  end if

  ! kin is 2-D (nspin, npts)
  if (model%needs_feature(skala_feature%kin)) then
    call skala_tensor_to_array(dexc_dkin, arr2d)
    print '(a, i0, a, i0, a)', " -> dexc/d_kin: shape = (", &
      size(arr2d, 1), ", ", size(arr2d, 2), ")"
    print '(a, 3es22.14, a)', &
      "     [[", arr2d(:3, 1), " ...]", &
      "      [", arr2d(:3, 2), " ...]]"
  end if

  ! grid_coords is 2-D (npts, 3)
  if (model%needs_feature(skala_feature%grid_coords)) then
    call skala_tensor_to_array(dexc_dgrid_coords, arr2d)
    print '(a, i0, a, i0, a)', " -> dexc/d_grid_coords: shape = (", &
      size(arr2d, 1), ", ", size(arr2d, 2), ")"
    print '(a, 3es22.14, a)', &
      "     [[", arr2d(:3, 1), " ...]", &
      "      [", arr2d(:3, 2), " ...]]"
  end if

  ! grid_weights is 1-D (npts)
  if (model%needs_feature(skala_feature%grid_weights)) then
    call skala_tensor_to_array(dexc_dgrid_weights, arr1d)
    print '(a, i0, a)', " -> dexc/d_grid_weights: shape = (", size(arr1d), ")"
    print '(a, 3es22.14, a)', &
      "      [", arr1d(:3), " ...]"
  end if

  ! coarse_0_atomic_coords is 2-D (natoms, 3)
  if (model%needs_feature(skala_feature%coarse_0_atomic_coords)) then
    call skala_tensor_to_array(dexc_dcoarse_0_atomic_coords, arr2d)
    print '(a, i0, a, i0, a)', " -> dexc/d_coarse_0_atomic_coords: shape = (", &
      size(arr2d, 1), ", ", size(arr2d, 2), ")"
    print '(a, 3es22.14, a)', &
      "     [[", arr2d(:3, 1), "]", "      [  ...]]"
  end if
  end block array_access

contains
  subroutine get_argument(idx, arg)
    integer, intent(in) :: idx
    character(len=:), allocatable, intent(out) :: arg

    integer :: length, stat

    call get_command_argument(idx, length=length, status=stat)
    if (stat /= 0) return

    allocate(character(len=length) :: arg, stat=stat)
    if (stat /= 0) return

    if (length > 0) then
      call get_command_argument(idx, arg, status=stat)
      if (stat /= 0) then
        deallocate(arg)
        return
      end if
    end if
  end subroutine get_argument
end program main