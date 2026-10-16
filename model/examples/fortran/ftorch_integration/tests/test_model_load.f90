program test_model_load
  use iso_c_binding, only : c_associated
  use skala_ftorch, only : skala_model, skala_model_load
  implicit none
  integer :: status

  block
    type(skala_model) :: model
    call skala_model_load(model, "nonexistent-model.fun", repeat("0", 64), status=status)
    if (status == 0) error stop "Missing model must return a failure status"
    if (c_associated(model%p)) error stop "Failed model must not expose a usable handle"
    if (allocated(model%features)) error stop "Failed model must not expose feature metadata"
  end block
end program test_model_load
