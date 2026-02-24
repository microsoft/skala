program main
  use ftorch, only : torch_tensor, torch_tensor_print, operator(*)
  use skala_ftorch, only : skala_model, skala_model_load, skala_feature, skala_tensor_load, &
    & skala_tensor_sum, skala_tensor_item_double, skala_dict, skala_dict_new
  use flap, only : command_line_interface
  implicit none

  type(skala_model) :: model
  type(skala_dict) :: input, vxc
  type(torch_tensor) :: exc, exc_sum
  type(torch_tensor) :: density, grad, kin, grid_coords, grid_weights, coarse_0_atomic_coords
  type(torch_tensor) :: dexc_ddensity, dexc_dgrad, dexc_dkin, dexc_dgrid_coords, &
    & dexc_dgrid_weights, dexc_dcoarse_0_atomic_coords, vxc_norm

  type(command_line_interface) :: cli

  character(len=:), allocatable :: path, feature_dir
  integer :: error

  cli_input: block
  character(len=512) :: dummy
  call cli%init(description="Test Skala model inference from Fortran")
  call cli%add(position=1, required=.true., act="store", error=error, positional=.true.)
  if (error /= 0) exit cli_input
  call cli%add(position=2, required=.true., act="store", error=error, positional=.true.)
  if (error /= 0) exit cli_input
  call cli%parse(error=error)
  if (error /= 0) exit cli_input
  call cli%get(position=1, val=dummy, error=error)
  if (error /= 0) exit cli_input
  path = trim(dummy)
  call cli%get(position=2, val=dummy, error=error)
  if (error /= 0) exit cli_input
  feature_dir = trim(dummy)
  end block cli_input
  if (error /= 0) then
    print '(a)', cli%error_message
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

  print '(a)', "[5] Extracting vxc components"
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
end program main