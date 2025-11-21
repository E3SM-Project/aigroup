module config
  ! either shared stuff or stuff to read from a namelist
  use, intrinsic :: iso_fortran_env, only : sp => real32

  implicit none

  !private

  integer, parameter :: wp = sp

  ! We are reading 2D data, on a 180 x 360 grid.
  integer, parameter :: n_lat = 180, n_lon = 360, n_channels=39

  integer, parameter :: in_dims = 4
  integer, parameter :: in_shape(in_dims) = [1, n_channels, n_lat, n_lon]

  character(len=*), parameter :: ic_file_name="ACE2-EAMv3/initial_conditions/1971010100.nc"
  character(len=*), parameter :: torchscript_file="ACE2-EAMv3/ace2_EAMv3_ckpt_traced.tar"

end module config
