program fortran_main
    use iso_c_binding
    implicit none

    ! Interface to C++ functions
    interface
        function initialize_python() bind(C, name="initialize_python")
            use iso_c_binding
            integer(c_int) :: initialize_python
        end function initialize_python

        function call_main_python(input_data, size, output_data) bind(C, name="call_main_python")
            use iso_c_binding
            real(c_float), dimension(*), intent(in) :: input_data
            integer(c_int), value, intent(in) :: size
            real(c_float), intent(out) :: output_data
            integer(c_int) :: call_main_python
        end function call_main_python

        subroutine finalize_python() bind(C, name="finalize_python")
            use iso_c_binding
        end subroutine finalize_python
    end interface

    ! Local variables
    integer(c_int), parameter :: array_size = 5
    integer(c_int) :: status
    real(c_float), dimension(array_size) :: input_data
    real(c_float) :: output_data
    integer(c_int) :: i

    print *, "========================================"
    print *, "Fortran Main - Entry Point"
    print *, "========================================"

    ! Initialize input data
    do i = 1, array_size
        input_data(i) = real(i, kind=c_float)
    end do

    ! Initialize Python
    print *, ""
    print *, "[Fortran] Initializing..."
    status = initialize_python()
    if (status /= 0) then
        print *, "[Fortran] ERROR: Initialization failed"
        stop 1
    end if
    print *, "[Fortran] Initialization successful"

    ! Test call python operation
    print *, ""
    print *, "--- Testing Call Python Operation ---"
    print *, "[Fortran] Sending data:"
    do i = 1, array_size
        print *, "  data(", i, ") =", input_data(i)
    end do

    status = call_main_python(input_data, array_size, output_data)
    if (status /= 0) then
        print *, "[Fortran] ERROR: Operation failed"
    else
        print *, "[Fortran] Result:", output_data
    end if

    ! Cleanup
    print *, ""
    print *, "--- Cleanup ---"
    call finalize_python()

    print *, ""
    print *, "========================================"
    print *, "Fortran Main - Complete"
    print *, "========================================"

end program fortran_main
