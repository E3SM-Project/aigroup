module norm

  use, intrinsic :: iso_fortran_env, only : sp => real32

  use netcdf
  use io

  implicit none

  type, public :: t_normalization_struct
    real(kind=sp), dimension(:), allocatable :: means
    real(kind=sp), dimension(:), allocatable :: stds
  end type t_normalization_struct

  type, extends(t_normalization_struct) :: t_normalizer
    contains
      procedure :: normalize
  end type t_normalizer

  type, extends(t_normalization_struct) :: t_denormalizer
    contains
      procedure :: denormalize
  end type t_denormalizer

contains

  subroutine init_normalizer(norm, norm_file, n_channels)
    implicit none

    class(t_normalization_struct), intent(inout) :: norm
    character(len=*),   intent(in) :: norm_file
    integer,            intent(in) :: n_channels

    integer :: ncid    ! netcdf file id

    allocate(norm%stds(n_channels))
    allocate(norm%means(n_channels))

    call check( nf90_open(trim(norm_file), nf90_nowrite, ncid) )

    call ncio_read_var(norm%stds, ncid, 'stds')
    call ncio_read_var(norm%means, ncid, 'means')

    call check( nf90_close(ncid) )

  end subroutine init_normalizer

  subroutine finalize_normalizer(norm)
    implicit none
    class(t_normalization_struct), intent(inout) :: norm

    deallocate(norm%stds)
    deallocate(norm%means)

  end subroutine finalize_normalizer

  subroutine normalize(self, inputs)
    class(t_normalizer) :: self
    real(kind=sp), intent(inout) :: inputs(:, :, :, :)

    integer :: i, j, k
    integer :: nx, ny, nc

    nc = SIZE(inputs, dim=2)
    ny = SIZE(inputs, dim=3)
    nx = SIZE(inputs, dim=4)

    do k = 1, nc
      do i = 1, ny
        do j = 1, nx
          inputs(1, k, i, j) = (inputs(1, k, i, j) - self%means(k)) / self%stds(k)
        enddo
      enddo
    enddo

  end subroutine normalize

  subroutine denormalize(self, outputs)
    class(t_denormalizer) :: self
    real(kind=sp), intent(inout) :: outputs(:, :, :, :)

    integer :: i, j, k
    integer :: nx, ny, nc

    nc = SIZE(outputs, dim=2)
    ny = SIZE(outputs, dim=3)
    nx = SIZE(outputs, dim=4)

    do k = 1, nc
      do i = 1, ny
        do j = 1, nx
          outputs(1, k, i, j) = outputs(1, k, i, j) * self%stds(k) + self%means(k)
        enddo
      enddo
    enddo

  end subroutine denormalize

end module norm
