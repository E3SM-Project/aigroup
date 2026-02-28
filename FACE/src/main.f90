program main

  use io
  use norm

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
  use, intrinsic :: iso_fortran_env, only : sp => real32

  implicit none

  character(len=256) :: fname, fullpath
  character(len=*), parameter :: ic_file_name="/global/cfs/cdirs/e3sm/anolan/ACE2-E3SMv3/initial_conditions/1971010100_time_1.nc"
  character(len=*), parameter :: torchscript_file="/global/cfs/cdirs/e3sm/anolan/ACE2-E3SMv3/ace2_EAMv3_ckpt_traced.tar"
  character(len=*), parameter :: output_directory="/global/cfs/cdirs/e3sm/anolan/aigroup/FACE/output"

  character(len=*), parameter :: norm_file="/global/cfs/cdirs/e3sm/anolan/ACE2-E3SMv3/ace2_EAMv3_normalize.nc"
  character(len=*), parameter :: denorm_file="/global/cfs/cdirs/e3sm/anolan/ACE2-E3SMv3/ace2_EAMv3_denormalize.nc"

  integer, parameter :: n_lat = 180
  integer, parameter :: n_lon = 360
  integer, parameter :: n_input_channels=39
  integer, parameter :: n_output_channels=44

  ! Set up Torch data structures
  type(torch_model) :: ace_model
  type(torch_tensor), dimension(1) :: input_tensor
  type(torch_tensor), dimension(1) :: output_tensor

  type(t_normalizer) :: normalizer
  type(t_denormalizer) :: denormalizer

  real(sp), target :: net_inputs(1, n_input_channels, n_lat, n_lon)
  real(sp), target :: net_outputs(1, n_output_channels, n_lat, n_lon)
  real(sp), target :: net_inputs_nn(1, n_input_channels, n_lat, n_lon)

  integer :: i

  call torch_model_load(ace_model, torchscript_file, torch_kCPU)

  print *, "reading initial condition"
  call ace_read_initial_condition(ic_file_name, net_inputs)

  print *, "reading normalizer"
  call init_normalizer(normalizer, norm_file, n_input_channels)
  print *, "reading denormalizer"
  call init_normalizer(denormalizer, denorm_file, n_output_channels)

  do i = 1, 6
    net_inputs_nn = net_inputs
    call normalizer%normalize(net_inputs_nn)

    if (.not. is_contiguous(net_inputs_nn))  error stop "net_inputs not contiguous"
    if (.not. is_contiguous(net_outputs)) error stop "net_outputs not contiguous"

    ! Create a tensor based off an array
    call torch_tensor_from_array(input_tensor(1), net_inputs_nn, device_type=torch_kCPU)
    call torch_tensor_from_array(output_tensor(1), net_outputs, device_type=torch_kCPU)

    if (i .eq. 1) then
      print *, "****************************************************"
      print *, "Input shape : ", input_tensor(1)%get_shape()
      print *, "Input stride: ", input_tensor(1)%get_stride()
      print *, "Output shape: ", output_tensor(1)%get_shape()
      print *, "Output stride:", output_tensor(1)%get_stride()
      print *, "****************************************************"
    endif

    ! Infer
    call torch_model_forward(ace_model, input_tensor, output_tensor)

     if (i .eq. 1) then
        print *, "net_inputs(1,1,1,1)= ", net_inputs(1, 1, 1, 1)
        print *, "net_outputs(1,1,1,1)=", net_outputs(1, 1, 1, 1)
     endif

    call torch_delete(input_tensor)
    call torch_delete(output_tensor)

    call denormalizer%denormalize(net_outputs)

    write(fname,'(A,I0,A)') 'ftorch_output_', i, '.nc'
    fullpath = trim(output_directory)//'/'//trim(fname)

    call ace_write_restart(trim(fullpath), net_outputs)

    call advance(net_inputs, net_outputs)
  enddo

  ! Clean up
  ! --------
  call torch_delete(ace_model)

  call finalize_normalizer(normalizer)
  call finalize_normalizer(denormalizer)

  print *,"*** SUCCESS! ***"

  contains

    subroutine advance(net_inputs, net_outputs)
      real(sp), dimension(:, :, :, :), intent(inout) :: net_inputs
      real(sp), dimension(:, :, :, :), intent(in)    :: net_outputs

      ! !LOCAL VARIABLES:
      integer :: i, j

      do i = 1, n_lat
        do j = 1, n_lon
          net_inputs(1,  6, i, j) = net_outputs(1, 1, i, j)  ! PS
          net_inputs(1,  7, i, j) = net_outputs(1, 2, i, j)  ! TS
          net_inputs(1,  8, i, j) = net_outputs(1,  3, i, j) ! T_0
          net_inputs(1,  9, i, j) = net_outputs(1,  4, i, j) ! T_1
          net_inputs(1, 10, i, j) = net_outputs(1,  5, i, j) ! T_2
          net_inputs(1, 11, i, j) = net_outputs(1,  6, i, j) ! T_3
          net_inputs(1, 12, i, j) = net_outputs(1,  7, i, j) ! T_4
          net_inputs(1, 13, i, j) = net_outputs(1,  8, i, j) ! T_5
          net_inputs(1, 14, i, j) = net_outputs(1,  9, i, j) ! T_6
          net_inputs(1, 15, i, j) = net_outputs(1, 10, i, j) ! T_7
          net_inputs(1, 16, i, j) = net_outputs(1, 11, i, j) ! specific_total_water_0
          net_inputs(1, 17, i, j) = net_outputs(1, 12, i, j) ! specific_total_water_1
          net_inputs(1, 18, i, j) = net_outputs(1, 13, i, j) ! specific_total_water_2
          net_inputs(1, 19, i, j) = net_outputs(1, 14, i, j) ! specific_total_water_3
          net_inputs(1, 20, i, j) = net_outputs(1, 15, i, j) ! specific_total_water_4
          net_inputs(1, 21, i, j) = net_outputs(1, 16, i, j) ! specific_total_water_5
          net_inputs(1, 22, i, j) = net_outputs(1, 17, i, j) ! specific_total_water_6
          net_inputs(1, 23, i, j) = net_outputs(1, 18, i, j) ! specific_total_water_7
          net_inputs(1, 24, i, j) = net_outputs(1, 19, i, j) ! U_0
          net_inputs(1, 25, i, j) = net_outputs(1, 20, i, j) ! U_1
          net_inputs(1, 26, i, j) = net_outputs(1, 21, i, j) ! U_2
          net_inputs(1, 27, i, j) = net_outputs(1, 22, i, j) ! U_3
          net_inputs(1, 28, i, j) = net_outputs(1, 23, i, j) ! U_4
          net_inputs(1, 29, i, j) = net_outputs(1, 24, i, j) ! U_5
          net_inputs(1, 30, i, j) = net_outputs(1, 25, i, j) ! U_6
          net_inputs(1, 31, i, j) = net_outputs(1, 26, i, j) ! U_7
          net_inputs(1, 32, i, j) = net_outputs(1, 27, i, j) ! V_0
          net_inputs(1, 33, i, j) = net_outputs(1, 28, i, j) ! V_1
          net_inputs(1, 34, i, j) = net_outputs(1, 29, i, j) ! V_2
          net_inputs(1, 35, i, j) = net_outputs(1, 30, i, j) ! V_3
          net_inputs(1, 36, i, j) = net_outputs(1, 31, i, j) ! V_4
          net_inputs(1, 37, i, j) = net_outputs(1, 32, i, j) ! V_5
          net_inputs(1, 38, i, j) = net_outputs(1, 33, i, j) ! V_6
          net_inputs(1, 39, i, j) = net_outputs(1, 34, i, j) ! V_7
        enddo
      enddo
    end subroutine advance
end program main
