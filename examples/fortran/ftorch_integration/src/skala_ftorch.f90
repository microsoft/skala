module skala_ftorch
  use iso_c_binding, only : c_ptr, c_char, c_int, c_int64_t, c_size_t, c_bool, &
    & c_null_ptr, c_null_char, c_double, c_f_pointer
  use ftorch, only : torch_model, torch_model_delete, torch_tensor
  implicit none
  private

  public :: skala_model, skala_model_load, skala_feature, skala_tensor_load, &
    & skala_tensor_sum, skala_tensor_mean, skala_tensor_mul, skala_tensor_item_double, &
    & skala_tensor_to_array, &
    & skala_tensor_numel, skala_tensor_ndim, skala_tensor_size, &
    & skala_dict, skala_dict_new

  interface skala_tensor_to_array
    module procedure skala_tensor_to_array_1d
    module procedure skala_tensor_to_array_2d
    module procedure skala_tensor_to_array_3d
  end interface skala_tensor_to_array

  type :: skala_feature_enum
    integer :: density = 1
    integer :: grad = 2
    integer :: kin = 3
    integer :: grid_coords = 4
    integer :: grid_weights = 5
    integer :: coarse_0_atomic_coords = 6
    integer :: max_feature = 6
  end type skala_feature_enum
  type(skala_feature_enum), parameter :: skala_feature = skala_feature_enum()

  type :: skala_dict
    type(c_ptr) :: p = c_null_ptr
  contains
    generic :: at => at_one, at_vec
    procedure, private :: at_one => skala_dict_at_one
    procedure, private :: at_vec => skala_dict_at_vec
    generic :: insert => insert_vec, insert_one
    procedure, private :: insert_vec => skala_dict_insert_vec
    procedure, private :: insert_one => skala_dict_insert_one
    final :: skala_dict_delete
  end type skala_dict

  type, extends(torch_model) :: skala_model
    integer(c_int), allocatable :: features(:)
  contains
    procedure :: get_exc => skala_model_get_exc
    procedure :: get_exc_and_vxc => skala_model_get_exc_and_vxc
    procedure :: needs_feature => skala_model_needs_feature
    final :: skala_model_delete
  end type skala_model

contains

  subroutine skala_model_load(model, path, requires_grad)
    type(skala_model), intent(out) :: model
    character(len=*), intent(in) :: path
    logical, intent(in), optional :: requires_grad

    interface
      function load_skala_model_c(path, requires_grad, features) result(model) &
        & bind(c, name="skala_model_load")
        import :: c_ptr, c_char, c_int, c_bool
        character(kind=c_char), intent(in) :: path(*)
        logical(c_bool), value :: requires_grad
        integer(c_int), intent(out) :: features(*)
        type(c_ptr) :: model
      end function load_skala_model_c
    end interface

    logical(c_bool) :: requires_grad_c
    integer(c_int) :: features(skala_feature%max_feature)

    requires_grad_c = .false.
    if (present(requires_grad)) requires_grad_c = requires_grad

    features(:) = 0
    model%p = load_skala_model_c(to_c_str(path), requires_grad_c, features)
    model%features = pack(features, features > 0)
  end subroutine skala_model_load

  subroutine skala_tensor_load(tensor, path)
    type(torch_tensor), intent(out) :: tensor
    character(len=*), intent(in) :: path

    interface
      function skala_tensor_load_c(path) result(tensor) bind(c, name="skala_tensor_load")
        import :: c_ptr, c_char
        character(kind=c_char), intent(in) :: path(*)
        type(c_ptr) :: tensor
      end function skala_tensor_load_c
    end interface

    tensor%p = skala_tensor_load_c(to_c_str(path))
  end subroutine skala_tensor_load

  subroutine skala_tensor_sum(tensor, sum)
    type(torch_tensor), intent(in) :: tensor
    type(torch_tensor), intent(out) :: sum

    interface
      function skala_tensor_sum_c(tensor) result(sum) bind(c, name="skala_tensor_sum")
        import :: c_ptr, c_double
        type(c_ptr), value :: tensor
        type(c_ptr) :: sum
      end function skala_tensor_sum_c
    end interface

    sum%p = skala_tensor_sum_c(tensor%p)
  end subroutine skala_tensor_sum

  function skala_tensor_item_double(tensor) result(value)
    type(torch_tensor), intent(in) :: tensor
    real(c_double) :: value

    interface
      function skala_tensor_item_double_c(tensor) result(value) bind(c, name="skala_tensor_item_double")
        import :: c_ptr, c_double
        type(c_ptr), value :: tensor
        real(c_double) :: value
      end function skala_tensor_item_double_c
    end interface

    value = skala_tensor_item_double_c(tensor%p)
  end function skala_tensor_item_double

  subroutine skala_tensor_mul(a, b, result)
    type(torch_tensor), intent(in) :: a, b
    type(torch_tensor), intent(out) :: result

    interface
      function skala_tensor_mul_c(a, b) result(r) bind(c, name="skala_tensor_mul")
        import :: c_ptr
        type(c_ptr), value :: a, b
        type(c_ptr) :: r
      end function skala_tensor_mul_c
    end interface

    result%p = skala_tensor_mul_c(a%p, b%p)
  end subroutine skala_tensor_mul

  subroutine skala_tensor_mean(tensor, mean)
    type(torch_tensor), intent(in) :: tensor
    type(torch_tensor), intent(out) :: mean

    interface
      function skala_tensor_mean_c(tensor) result(mean) bind(c, name="skala_tensor_mean")
        import :: c_ptr
        type(c_ptr), value :: tensor
        type(c_ptr) :: mean
      end function skala_tensor_mean_c
    end interface

    mean%p = skala_tensor_mean_c(tensor%p)
  end subroutine skala_tensor_mean

  function skala_tensor_numel(tensor) result(n)
    type(torch_tensor), intent(in) :: tensor
    integer(c_int64_t) :: n

    interface
      function skala_tensor_numel_c(tensor) result(n) bind(c, name="skala_tensor_numel")
        import :: c_ptr, c_int64_t
        type(c_ptr), value :: tensor
        integer(c_int64_t) :: n
      end function skala_tensor_numel_c
    end interface

    n = skala_tensor_numel_c(tensor%p)
  end function skala_tensor_numel

  function skala_tensor_ndim(tensor) result(n)
    type(torch_tensor), intent(in) :: tensor
    integer(c_int64_t) :: n

    interface
      function skala_tensor_ndim_c(tensor) result(n) bind(c, name="skala_tensor_ndim")
        import :: c_ptr, c_int64_t
        type(c_ptr), value :: tensor
        integer(c_int64_t) :: n
      end function skala_tensor_ndim_c
    end interface

    n = skala_tensor_ndim_c(tensor%p)
  end function skala_tensor_ndim

  function skala_tensor_size(tensor, dim) result(n)
    type(torch_tensor), intent(in) :: tensor
    integer, intent(in) :: dim
    integer(c_int64_t) :: n

    interface
      function skala_tensor_size_c(tensor, dim) result(n) bind(c, name="skala_tensor_size")
        import :: c_ptr, c_int64_t
        type(c_ptr), value :: tensor
        integer(c_int64_t), value :: dim
        integer(c_int64_t) :: n
      end function skala_tensor_size_c
    end interface

    n = skala_tensor_size_c(tensor%p, int(dim, c_int64_t))
  end function skala_tensor_size

  subroutine skala_tensor_to_array_1d(tensor, array)
    type(torch_tensor), intent(in) :: tensor
    real(c_double), pointer, intent(out) :: array(:)

    interface
      function skala_tensor_data_ptr_c(tensor) result(ptr) bind(c, name="skala_tensor_data_ptr")
        import :: c_ptr
        type(c_ptr), value :: tensor
        type(c_ptr) :: ptr
      end function skala_tensor_data_ptr_c
    end interface

    type(c_ptr) :: data_ptr
    integer(c_int64_t) :: n

    n = skala_tensor_numel(tensor)
    data_ptr = skala_tensor_data_ptr_c(tensor%p)
    call c_f_pointer(data_ptr, array, [n])
  end subroutine skala_tensor_to_array_1d

  subroutine skala_tensor_to_array_2d(tensor, array)
    type(torch_tensor), intent(in) :: tensor
    real(c_double), pointer, intent(out) :: array(:,:)

    interface
      function skala_tensor_data_ptr_c(tensor) result(ptr) bind(c, name="skala_tensor_data_ptr")
        import :: c_ptr
        type(c_ptr), value :: tensor
        type(c_ptr) :: ptr
      end function skala_tensor_data_ptr_c
    end interface

    type(c_ptr) :: data_ptr
    integer(c_int64_t) :: nrow, ncol

    ! C is row-major, Fortran is column-major: swap dimensions
    nrow = skala_tensor_size(tensor, 1)
    ncol = skala_tensor_size(tensor, 0)
    data_ptr = skala_tensor_data_ptr_c(tensor%p)
    call c_f_pointer(data_ptr, array, [nrow, ncol])
  end subroutine skala_tensor_to_array_2d

  subroutine skala_tensor_to_array_3d(tensor, array)
    type(torch_tensor), intent(in) :: tensor
    real(c_double), pointer, intent(out) :: array(:,:,:)

    interface
      function skala_tensor_data_ptr_c(tensor) result(ptr) bind(c, name="skala_tensor_data_ptr")
        import :: c_ptr
        type(c_ptr), value :: tensor
        type(c_ptr) :: ptr
      end function skala_tensor_data_ptr_c
    end interface

    type(c_ptr) :: data_ptr
    integer(c_int64_t) :: n0, n1, n2

    ! C is row-major, Fortran is column-major: reverse dimension order
    n0 = skala_tensor_size(tensor, 2)
    n1 = skala_tensor_size(tensor, 1)
    n2 = skala_tensor_size(tensor, 0)
    data_ptr = skala_tensor_data_ptr_c(tensor%p)
    call c_f_pointer(data_ptr, array, [n0, n1, n2])
  end subroutine skala_tensor_to_array_3d

  subroutine skala_model_get_exc(model, input, exc)
    class(skala_model), intent(inout) :: model
    type(skala_dict), intent(in) :: input
    type(torch_tensor), intent(out) :: exc

    interface
      subroutine skala_model_get_exc_c(model_c, input_c, output_c) bind(c, name="skala_model_get_exc")
        import :: c_ptr
        type(c_ptr), value :: model_c
        type(c_ptr), value :: input_c
        type(c_ptr), intent(out) :: output_c
      end subroutine skala_model_get_exc_c
    end interface

    call skala_model_get_exc_c(model%torch_model%p, input%p, exc%p)
  end subroutine skala_model_get_exc

  subroutine skala_model_get_exc_and_vxc(model, input, exc, vxc)
    class(skala_model), intent(inout) :: model
    type(skala_dict), intent(in) :: input
    type(torch_tensor), intent(out) :: exc
    type(skala_dict), intent(out) :: vxc

    interface
      subroutine skala_model_get_exc_and_vxc_c(model_c, input_c, exc_c, vxc_c) bind(c, name="skala_model_get_exc_and_vxc")
        import :: c_ptr
        type(c_ptr), value :: model_c
        type(c_ptr), value :: input_c
        type(c_ptr), intent(out) :: exc_c
        type(c_ptr), intent(out) :: vxc_c
      end subroutine skala_model_get_exc_and_vxc_c
    end interface

    call skala_model_get_exc_and_vxc_c(model%torch_model%p, input%p, exc%p, vxc%p)
  end subroutine skala_model_get_exc_and_vxc

  function skala_model_needs_feature(model, feature) result(needs)
    class(skala_model), intent(in) :: model
    integer, intent(in) :: feature
    logical :: needs

    needs = any(model%features == feature)
  end function skala_model_needs_feature

  subroutine skala_model_delete(model)
    type(skala_model), intent(inout) :: model
    
    call torch_model_delete(model%torch_model)
  end subroutine skala_model_delete

  subroutine skala_dict_new(input)
    type(skala_dict), intent(out) :: input

    interface
      function skala_dict_new_c() result(input) bind(c, name="skala_dict_new")
        import :: c_ptr
        type(c_ptr) :: input
      end function skala_dict_new_c
    end interface

    input%p = skala_dict_new_c()
  end subroutine skala_dict_new

  subroutine skala_feature_key(feature, key)
    integer, intent(in) :: feature
    character(kind=c_char, len=:), allocatable, intent(out) :: key
    select case(feature)
    case default
      error stop "Unknown feature"
    case (skala_feature%density)
      key = "density"
    case (skala_feature%grad)
      key = "grad"
    case (skala_feature%kin)
      key = "kin"
    case (skala_feature%grid_coords)
      key = "grid_coords"
    case (skala_feature%grid_weights)
      key = "grid_weights"
    case (skala_feature%coarse_0_atomic_coords)
      key = "coarse_0_atomic_coords"
    end select
  end subroutine skala_feature_key

  pure function to_c_str(str) result(c_str)
    character(len=*), intent(in) :: str
    character(kind=c_char) :: c_str(len(str)+1)

    c_str = transfer(str // c_null_char, [character(kind=c_char)::], len(str)+1)
  end function to_c_str

  subroutine skala_dict_insert_one(dict, feature, tensor)
    class(skala_dict), intent(inout) :: dict
    integer, intent(in) :: feature
    type(torch_tensor), intent(in) :: tensor

    interface
      subroutine skala_dict_insert_c(dict, feature, tensors, ntensors) bind(c, name="skala_dict_insert")
        import :: c_ptr, c_size_t, c_char
        type(c_ptr), value :: dict
        character(kind=c_char), intent(in) :: feature(*)
        type(c_ptr), intent(in) :: tensors(*)
        integer(c_size_t), value :: ntensors
      end subroutine skala_dict_insert_c
    end interface

    character(kind=c_char, len=:), allocatable :: key
    type(c_ptr) :: tensor_ptr(1)

    tensor_ptr(1) = tensor%p

    call skala_feature_key(feature, key)
    call skala_dict_insert_c(dict%p, to_c_str(key), tensor_ptr, 1_c_size_t)
  end subroutine skala_dict_insert_one

  subroutine skala_dict_insert_vec(dict, feature, tensors)
    class(skala_dict), intent(inout) :: dict
    integer, intent(in) :: feature
    type(torch_tensor), intent(in) :: tensors(:)

    interface
      subroutine skala_dict_insert_c(dict, feature, tensors, ntensors) bind(c, name="skala_dict_insert")
        import :: c_ptr, c_size_t, c_char
        type(c_ptr), value :: dict
        character(kind=c_char), intent(in) :: feature(*)
        type(c_ptr), intent(in) :: tensors(*)
        integer(c_size_t), value :: ntensors
      end subroutine skala_dict_insert_c
    end interface

    character(kind=c_char, len=:), allocatable :: key
    type(c_ptr), allocatable :: tensor_ptrs(:)
    integer :: iptr

    allocate(tensor_ptrs(size(tensors)))
    do iptr = 1, size(tensors)
      tensor_ptrs(iptr) = tensors(iptr)%p
    end do

    call skala_feature_key(feature, key)
    call skala_dict_insert_c(dict%p, to_c_str(key), tensor_ptrs, size(tensor_ptrs, kind=c_size_t))
  end subroutine skala_dict_insert_vec

  subroutine skala_dict_at_one(dict, feature, tensor)
    class(skala_dict), intent(in) :: dict
    integer, intent(in) :: feature
    type(torch_tensor), intent(out) :: tensor

    interface
      function skala_dict_at_c(dict, key) result(list) bind(c, name="skala_dict_at")
        import :: c_ptr, c_char
        type(c_ptr), value :: dict
        character(kind=c_char), intent(in) :: key(*)
        type(c_ptr) :: list
      end function skala_dict_at_c

      function skala_list_at_c(list, index) result(tensor) bind(c, name="skala_list_at")
        import :: c_ptr, c_size_t
        type(c_ptr), value :: list
        integer(c_size_t), value :: index
        type(c_ptr) :: tensor
      end function skala_list_at_c

      subroutine skala_list_delete_c(list) bind(c, name="skala_list_delete")
        import :: c_ptr
        type(c_ptr), value :: list
      end subroutine skala_list_delete_c
    end interface

    character(kind=c_char, len=:), allocatable :: key
    type(c_ptr) :: list

    call skala_feature_key(feature, key)
    list = skala_dict_at_c(dict%p, to_c_str(key))
    tensor%p = skala_list_at_c(list, 0_c_size_t)
    call skala_list_delete_c(list)
  end subroutine skala_dict_at_one

  subroutine skala_dict_at_vec(dict, feature, tensors, ntensors)
    class(skala_dict), intent(in) :: dict
    integer, intent(in) :: feature
    type(torch_tensor), intent(out) :: tensors(:)
    integer, intent(out) :: ntensors

    interface
      function skala_dict_at_c(dict, key) result(list) bind(c, name="skala_dict_at")
        import :: c_ptr, c_char
        type(c_ptr), value :: dict
        character(kind=c_char), intent(in) :: key(*)
        type(c_ptr) :: list
      end function skala_dict_at_c

      function skala_list_size_c(list) result(n) bind(c, name="skala_list_size")
        import :: c_ptr, c_size_t
        type(c_ptr), value :: list
        integer(c_size_t) :: n
      end function skala_list_size_c

      function skala_list_at_c(list, index) result(tensor) bind(c, name="skala_list_at")
        import :: c_ptr, c_size_t
        type(c_ptr), value :: list
        integer(c_size_t), value :: index
        type(c_ptr) :: tensor
      end function skala_list_at_c

      subroutine skala_list_delete_c(list) bind(c, name="skala_list_delete")
        import :: c_ptr
        type(c_ptr), value :: list
      end subroutine skala_list_delete_c
    end interface

    character(kind=c_char, len=:), allocatable :: key
    type(c_ptr) :: list
    integer :: i

    call skala_feature_key(feature, key)
    list = skala_dict_at_c(dict%p, to_c_str(key))
    ntensors = int(skala_list_size_c(list))
    do i = 1, min(ntensors, size(tensors))
      tensors(i)%p = skala_list_at_c(list, int(i - 1, c_size_t))
    end do
    call skala_list_delete_c(list)
  end subroutine skala_dict_at_vec

  subroutine skala_dict_delete(dict)
    type(skala_dict), intent(inout) :: dict

    interface
      subroutine skala_dict_delete_c(dict) bind(c, name="skala_dict_delete")
        import :: c_ptr
        type(c_ptr), value :: dict
      end subroutine skala_dict_delete_c
    end interface

    call skala_dict_delete_c(dict%p)
  end subroutine skala_dict_delete

end module skala_ftorch