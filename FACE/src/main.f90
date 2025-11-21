program main

  use config
  use initial_conditions
  ! import the FTorch procedures that are used in this worked example
  use ftorch, only: &
    torch_kCPU, &
    torch_model, &
    torch_tensor, &
    torch_delete, &
    torch_kFloat32, &
    torch_model_load, &
    torch_tensor_zeros, &
    torch_tensor_print, &
    torch_model_forward, &
    torch_tensor_from_array

  use, intrinsic :: iso_c_binding, only: c_int64_t

  implicit none

  integer, parameter :: ndims = 4
  integer(c_int64_t), dimension(4), parameter :: tensor_shape = [1, 44, n_lat, n_lon]

  ! Set up Torch data structures
  type(torch_model) :: ace_model
  type(torch_tensor), dimension(1) :: input_tensor
  type(torch_tensor), dimension(1) :: output_tensor

  real(wp), dimension(:,:,:,:), allocatable, target :: initial_condition

  call torch_model_load(ace_model, torchscript_file, torch_kCPU)

  allocate(initial_condition(in_shape(1), in_shape(2), in_shape(3), in_shape(4)))
  call ic_read(initial_condition)

  ! Create a tensor based off an array
  call torch_tensor_from_array(input_tensor(1), initial_condition, torch_kCPU)
  call torch_tensor_zeros(output_tensor(1), ndims, tensor_shape, torch_kFloat32, torch_kCPU)

  print *, "Shape of the input tensor:", input_tensor(1) % get_shape()
  print *, "Shape of the output tensor:", output_tensor(1) % get_shape()

  ! Infer
  call torch_model_forward(ace_model, input_tensor, output_tensor)

  call torch_tensor_print(output_tensor(1))

  ! Clean up
  ! --------
  call torch_delete(ace_model)
  call torch_delete(input_tensor)
  call torch_delete(output_tensor)
  deallocate(initial_condition)

  print *,"*** SUCCESS! ***"

end program main
