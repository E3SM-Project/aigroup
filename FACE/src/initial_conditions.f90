module initial_conditions

  use config
  use netcdf

  implicit none
  private

  public ic_read

  contains

  subroutine ic_read(initial_condition)

    real(wp), dimension(:, :, :, :), intent(inout) :: initial_condition

    integer :: ncid

    !
    call check( nf90_open(ic_file_name, nf90_nowrite, ncid) )

    initial_condition(1,  1, :, :) = read_ic_var(ncid, "LANDFRAC")
    initial_condition(1,  2, :, :) = read_ic_var(ncid, "OCNFRAC")
    initial_condition(1,  3, :, :) = read_ic_var(ncid, "ICEFRAC")
    initial_condition(1,  4, :, :) = read_ic_var(ncid, "PHIS")
    initial_condition(1,  5, :, :) = read_ic_var(ncid, "SOLIN")
    initial_condition(1,  6, :, :) = read_ic_var(ncid, "PS")
    initial_condition(1,  7, :, :) = read_ic_var(ncid, "TS")
    initial_condition(1,  8, :, :) = read_ic_var(ncid, "T_0")
    initial_condition(1,  9, :, :) = read_ic_var(ncid, "T_1")
    initial_condition(1, 10, :, :) = read_ic_var(ncid, "T_2")
    initial_condition(1, 11, :, :) = read_ic_var(ncid, "T_3")
    initial_condition(1, 12, :, :) = read_ic_var(ncid, "T_4")
    initial_condition(1, 13, :, :) = read_ic_var(ncid, "T_5")
    initial_condition(1, 14, :, :) = read_ic_var(ncid, "T_6")
    initial_condition(1, 15, :, :) = read_ic_var(ncid, "T_7")
    initial_condition(1, 16, :, :) = read_ic_var(ncid, "specific_total_water_0")
    initial_condition(1, 17, :, :) = read_ic_var(ncid, "specific_total_water_1")
    initial_condition(1, 18, :, :) = read_ic_var(ncid, "specific_total_water_2")
    initial_condition(1, 19, :, :) = read_ic_var(ncid, "specific_total_water_3")
    initial_condition(1, 20, :, :) = read_ic_var(ncid, "specific_total_water_4")
    initial_condition(1, 21, :, :) = read_ic_var(ncid, "specific_total_water_5")
    initial_condition(1, 22, :, :) = read_ic_var(ncid, "specific_total_water_6")
    initial_condition(1, 23, :, :) = read_ic_var(ncid, "specific_total_water_7")
    initial_condition(1, 24, :, :) = read_ic_var(ncid, "U_0")
    initial_condition(1, 25, :, :) = read_ic_var(ncid, "U_1")
    initial_condition(1, 26, :, :) = read_ic_var(ncid, "U_2")
    initial_condition(1, 27, :, :) = read_ic_var(ncid, "U_3")
    initial_condition(1, 28, :, :) = read_ic_var(ncid, "U_4")
    initial_condition(1, 29, :, :) = read_ic_var(ncid, "U_5")
    initial_condition(1, 30, :, :) = read_ic_var(ncid, "U_6")
    initial_condition(1, 31, :, :) = read_ic_var(ncid, "U_7")
    initial_condition(1, 32, :, :) = read_ic_var(ncid, "V_0")
    initial_condition(1, 33, :, :) = read_ic_var(ncid, "V_1")
    initial_condition(1, 34, :, :) = read_ic_var(ncid, "V_2")
    initial_condition(1, 35, :, :) = read_ic_var(ncid, "V_3")
    initial_condition(1, 36, :, :) = read_ic_var(ncid, "V_4")
    initial_condition(1, 37, :, :) = read_ic_var(ncid, "V_5")
    initial_condition(1, 38, :, :) = read_ic_var(ncid, "V_6")
    initial_condition(1, 39, :, :) = read_ic_var(ncid, "V_7")

    ! Clean up
    call check( nf90_close(ncid) )
  end subroutine ic_read

  function read_ic_var(ncid, var) result(data_c)
    implicit none

    integer, intent(in) :: ncid
    character(len=*), intent(in) :: var
    real(wp) :: data_f(n_lat, n_lon)
    real(wp) :: data_c(n_lon, n_lat)

    integer :: varid = 0
    integer, dimension(3) :: start = (/ 1, 1, 1 /)
    integer, dimension(3) :: count = (/ n_lon, n_lat, 1 /)

    ! Get the varid of the data variable, based on its name.
    call check( nf90_inq_varid(ncid, trim(var), varid) )
    ! read the data.
    call check( nf90_get_var(ncid, varid, data_f, start=start, count=count) )

    ! (n_lon, n_lat) --> (n_lat, n_lon)
    data_c = transpose(data_f)

  end function read_ic_var

  subroutine check(status)
    integer, intent ( in) :: status
      if(status /= nf90_noerr) then
        print *, trim(nf90_strerror(status))
        stop 2
      end if
  end subroutine check
end module initial_conditions
