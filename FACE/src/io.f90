module io

  use, intrinsic :: iso_fortran_env, only : sp => real32

  use netcdf

  implicit none
  private

  public ncio_read_var
  public ace_read_initial_condition
  public ace_write_restart
  public check

  interface ncio_read_var
    module procedure ncio_read_1d_var
    module procedure ncio_read_2d_var
  end interface

  contains

  subroutine ace_read_initial_condition(ic_file_name, net_inputs)

    character(len=*), intent(in) :: ic_file_name
    real(sp), dimension(:, :, :, :), intent(inout) :: net_inputs

    integer :: ncid

    ! open netcdf file
    call check( nf90_open(trim(ic_file_name), nf90_nowrite, ncid) )

    call ncio_read_var(net_inputs(1,  1, :, :), ncid, "LANDFRAC")
    call ncio_read_var(net_inputs(1,  2, :, :), ncid, "OCNFRAC")
    call ncio_read_var(net_inputs(1,  3, :, :), ncid, "ICEFRAC")
    call ncio_read_var(net_inputs(1,  4, :, :), ncid, "PHIS")
    call ncio_read_var(net_inputs(1,  5, :, :), ncid, "SOLIN")
    call ncio_read_var(net_inputs(1,  6, :, :), ncid, "PS")
    call ncio_read_var(net_inputs(1,  7, :, :), ncid, "TS")
    call ncio_read_var(net_inputs(1,  8, :, :), ncid, "T_0")
    call ncio_read_var(net_inputs(1,  9, :, :), ncid, "T_1")
    call ncio_read_var(net_inputs(1, 10, :, :), ncid, "T_2")
    call ncio_read_var(net_inputs(1, 11, :, :), ncid, "T_3")
    call ncio_read_var(net_inputs(1, 12, :, :), ncid, "T_4")
    call ncio_read_var(net_inputs(1, 13, :, :), ncid, "T_5")
    call ncio_read_var(net_inputs(1, 14, :, :), ncid, "T_6")
    call ncio_read_var(net_inputs(1, 15, :, :), ncid, "T_7")
    call ncio_read_var(net_inputs(1, 16, :, :), ncid, "specific_total_water_0")
    call ncio_read_var(net_inputs(1, 17, :, :), ncid, "specific_total_water_1")
    call ncio_read_var(net_inputs(1, 18, :, :), ncid, "specific_total_water_2")
    call ncio_read_var(net_inputs(1, 19, :, :), ncid, "specific_total_water_3")
    call ncio_read_var(net_inputs(1, 20, :, :), ncid, "specific_total_water_4")
    call ncio_read_var(net_inputs(1, 21, :, :), ncid, "specific_total_water_5")
    call ncio_read_var(net_inputs(1, 22, :, :), ncid, "specific_total_water_6")
    call ncio_read_var(net_inputs(1, 23, :, :), ncid, "specific_total_water_7")
    call ncio_read_var(net_inputs(1, 24, :, :), ncid, "U_0")
    call ncio_read_var(net_inputs(1, 25, :, :), ncid, "U_1")
    call ncio_read_var(net_inputs(1, 26, :, :), ncid, "U_2")
    call ncio_read_var(net_inputs(1, 27, :, :), ncid, "U_3")
    call ncio_read_var(net_inputs(1, 28, :, :), ncid, "U_4")
    call ncio_read_var(net_inputs(1, 29, :, :), ncid, "U_5")
    call ncio_read_var(net_inputs(1, 30, :, :), ncid, "U_6")
    call ncio_read_var(net_inputs(1, 31, :, :), ncid, "U_7")
    call ncio_read_var(net_inputs(1, 32, :, :), ncid, "V_0")
    call ncio_read_var(net_inputs(1, 33, :, :), ncid, "V_1")
    call ncio_read_var(net_inputs(1, 34, :, :), ncid, "V_2")
    call ncio_read_var(net_inputs(1, 35, :, :), ncid, "V_3")
    call ncio_read_var(net_inputs(1, 36, :, :), ncid, "V_4")
    call ncio_read_var(net_inputs(1, 37, :, :), ncid, "V_5")
    call ncio_read_var(net_inputs(1, 38, :, :), ncid, "V_6")
    call ncio_read_var(net_inputs(1, 39, :, :), ncid, "V_7")

    ! Clean up
    call check( nf90_close(ncid) )
  end subroutine ace_read_initial_condition


  subroutine ace_write_restart(restart_file_name, net_outputs)

    character(len=*), intent(in) :: restart_file_name
    real(sp), dimension(:, :, :, :), intent(in) :: net_outputs

    integer :: ncid, varid, dimids(2)
    integer :: lat_dimid, lon_dimid

    ! open netcdf file
    call check( nf90_create(trim(restart_file_name), nf90_clobber, ncid) )

    call check( nf90_def_dim(ncid, "nlat", size(net_outputs, dim=3), lat_dimid) )
    call check( nf90_def_dim(ncid, "nlon", size(net_outputs, dim=4), lon_dimid) )

    dimids =  (/ lon_dimid, lat_dimid /)

    call check( nf90_def_var(ncid, 'PS', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'TS', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'T_0', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'T_1', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'T_2', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'T_3', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'T_4', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'T_5', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'T_6', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'T_7', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'specific_total_water_0', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'specific_total_water_1', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'specific_total_water_2', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'specific_total_water_3', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'specific_total_water_4', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'specific_total_water_5', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'specific_total_water_6', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'specific_total_water_7', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'U_0', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'U_1', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'U_2', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'U_3', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'U_4', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'U_5', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'U_6', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'U_7', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'V_0', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'V_1', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'V_2', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'V_3', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'V_4', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'V_5', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'V_6', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'V_7', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'LHFLX', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'SHFLX', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'surface_precipitation_rate', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'surface_upward_longwave_flux', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'FLUT', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'FLDS', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'FSDS', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'surface_upward_shortwave_flux', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'top_of_atmos_upward_shortwave_flux', NF90_REAL, dimids, varid) )
    call check( nf90_def_var(ncid, 'tendency_of_total_water_path_due_to_advection', NF90_REAL, dimids, varid) )
    call check( nf90_enddef(ncid) )

    call ncio_write_2d_var(transpose(net_outputs(1,  1, :, :)), ncid, 'PS')
    call ncio_write_2d_var(transpose(net_outputs(1,  2, :, :)), ncid, 'TS')
    call ncio_write_2d_var(transpose(net_outputs(1,  3, :, :)), ncid, 'T_0')
    call ncio_write_2d_var(transpose(net_outputs(1,  4, :, :)), ncid, 'T_1')
    call ncio_write_2d_var(transpose(net_outputs(1,  5, :, :)), ncid, 'T_2')
    call ncio_write_2d_var(transpose(net_outputs(1,  6, :, :)), ncid, 'T_3')
    call ncio_write_2d_var(transpose(net_outputs(1,  7, :, :)), ncid, 'T_4')
    call ncio_write_2d_var(transpose(net_outputs(1,  8, :, :)), ncid, 'T_5')
    call ncio_write_2d_var(transpose(net_outputs(1,  9, :, :)), ncid, 'T_6')
    call ncio_write_2d_var(transpose(net_outputs(1, 10, :, :)), ncid, 'T_7')
    call ncio_write_2d_var(transpose(net_outputs(1, 11, :, :)), ncid, 'specific_total_water_0')
    call ncio_write_2d_var(transpose(net_outputs(1, 12, :, :)), ncid, 'specific_total_water_1')
    call ncio_write_2d_var(transpose(net_outputs(1, 13, :, :)), ncid, 'specific_total_water_2')
    call ncio_write_2d_var(transpose(net_outputs(1, 14, :, :)), ncid, 'specific_total_water_3')
    call ncio_write_2d_var(transpose(net_outputs(1, 15, :, :)), ncid, 'specific_total_water_4')
    call ncio_write_2d_var(transpose(net_outputs(1, 16, :, :)), ncid, 'specific_total_water_5')
    call ncio_write_2d_var(transpose(net_outputs(1, 17, :, :)), ncid, 'specific_total_water_6')
    call ncio_write_2d_var(transpose(net_outputs(1, 18, :, :)), ncid, 'specific_total_water_7')
    call ncio_write_2d_var(transpose(net_outputs(1, 19, :, :)), ncid, 'U_0')
    call ncio_write_2d_var(transpose(net_outputs(1, 20, :, :)), ncid, 'U_1')
    call ncio_write_2d_var(transpose(net_outputs(1, 21, :, :)), ncid, 'U_2')
    call ncio_write_2d_var(transpose(net_outputs(1, 22, :, :)), ncid, 'U_3')
    call ncio_write_2d_var(transpose(net_outputs(1, 23, :, :)), ncid, 'U_4')
    call ncio_write_2d_var(transpose(net_outputs(1, 24, :, :)), ncid, 'U_5')
    call ncio_write_2d_var(transpose(net_outputs(1, 25, :, :)), ncid, 'U_6')
    call ncio_write_2d_var(transpose(net_outputs(1, 26, :, :)), ncid, 'U_7')
    call ncio_write_2d_var(transpose(net_outputs(1, 27, :, :)), ncid, 'V_0')
    call ncio_write_2d_var(transpose(net_outputs(1, 28, :, :)), ncid, 'V_1')
    call ncio_write_2d_var(transpose(net_outputs(1, 29, :, :)), ncid, 'V_2')
    call ncio_write_2d_var(transpose(net_outputs(1, 30, :, :)), ncid, 'V_3')
    call ncio_write_2d_var(transpose(net_outputs(1, 31, :, :)), ncid, 'V_4')
    call ncio_write_2d_var(transpose(net_outputs(1, 32, :, :)), ncid, 'V_5')
    call ncio_write_2d_var(transpose(net_outputs(1, 33, :, :)), ncid, 'V_6')
    call ncio_write_2d_var(transpose(net_outputs(1, 34, :, :)), ncid, 'V_7')
    call ncio_write_2d_var(transpose(net_outputs(1, 35, :, :)), ncid, 'LHFLX')
    call ncio_write_2d_var(transpose(net_outputs(1, 36, :, :)), ncid, 'SHFLX')
    call ncio_write_2d_var(transpose(net_outputs(1, 37, :, :)), ncid, 'surface_precipitation_rate')
    call ncio_write_2d_var(transpose(net_outputs(1, 38, :, :)), ncid, 'surface_upward_longwave_flux')
    call ncio_write_2d_var(transpose(net_outputs(1, 39, :, :)), ncid, 'FLUT')
    call ncio_write_2d_var(transpose(net_outputs(1, 10, :, :)), ncid, 'FLDS')
    call ncio_write_2d_var(transpose(net_outputs(1, 41, :, :)), ncid, 'FSDS')
    call ncio_write_2d_var(transpose(net_outputs(1, 42, :, :)), ncid, 'surface_upward_shortwave_flux')
    call ncio_write_2d_var(transpose(net_outputs(1, 43, :, :)), ncid, 'top_of_atmos_upward_shortwave_flux')
    call ncio_write_2d_var(transpose(net_outputs(1, 44, :, :)), ncid, 'tendency_of_total_water_path_due_to_advection')

    call check( nf90_close(ncid) )
  end subroutine ace_write_restart

  subroutine ncio_read_1d_var(data, ncid, var)
    implicit none

    real(sp), dimension(:), intent(inout) :: data
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: var

    integer :: varid = 0
    integer, dimension(1) :: start = (/ 1 /)
    integer, dimension(1) :: count

    count = (/ size(data, dim=1) /)

    ! Get the varid of the data variable, based on its name.
    call check( nf90_inq_varid(ncid, trim(var), varid) )
    ! read the data.
    call check( nf90_get_var(ncid, varid, data, start=start, count=count) )

  end subroutine ncio_read_1d_var

  subroutine ncio_read_2d_var(data, ncid, var)
    implicit none

    real(sp), dimension(:, :), intent(inout) :: data
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: var

    integer :: varid = 0
    integer, dimension(2) :: start = (/ 1, 1 /)
    integer, dimension(2) :: count

    count = (/ size(data, dim=2), size(data, dim=1) /)

    ! Get the varid of the data variable, based on its name.
    call check( nf90_inq_varid(ncid, trim(var), varid) )
    ! read the data.
    call check( nf90_get_var(ncid, varid, data, start=start, count=count) )

  end subroutine ncio_read_2d_var

  subroutine ncio_write_2d_var(data, ncid, var)
    implicit none

    real(sp), dimension(:, :), intent(in) :: data
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: var
    integer :: nlat, varid = 0

    nlat = size(data, dim=2)

    ! Get the varid of the data variable, based on its name.
    call check( nf90_inq_varid(ncid, trim(var), varid) )
    ! read the data.
    call check( nf90_put_var(ncid, varid, data) ) !, map=(/ nlat, 1 /)) )

  end subroutine ncio_write_2d_var

  subroutine check(status)
    integer, intent ( in) :: status
      if(status /= nf90_noerr) then
        print *, trim(nf90_strerror(status))
        stop 2
      end if
  end subroutine check
end module io
