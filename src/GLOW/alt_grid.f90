subroutine alt_grid(Nalt, minalt, dmin, dmax, z)
  implicit none
  ! * Nalt: number of altitude points
  ! * altmin: minimum altitude [km]
  ! * dmin: minimum grid spacing at minimum altitude [km]
  ! * dmax: maximum grid spacing at maximum altitude [km]
  !
  ! Example: alt = alt_grid(286, 90, 1.5, 11.1475)

  real, intent(in) :: dmin, dmax, minalt
  integer, intent(in) :: Nalt
  real :: dz(Nalt)
  real, intent(out) :: z(Nalt)
  integer :: i

  call ztanh(dmin, dmax, Nalt, dz)

  call cumsum(Nalt, dz, z)

  do i = 1, Nalt
    z(i) = z(i) + minalt - dmin
  end do

  end subroutine alt_grid

subroutine ztanh(gridmin, gridmax, N, dz)

  real, intent(in) :: gridmin, gridmax
  integer, intent(in) :: N
  real, intent(out) :: dz(N)
  real :: x(N)

  call linspace(0., 3.14, N, x)
  !! arbitrarily picking 3.14 as where tanh gets to 99% of asymptote
  
  dz = tanh(x) * gridmax + gridmin

  end subroutine ztanh


subroutine linspace(Vmin, Vmax, Nv, x)

  real, intent(in) :: Vmin, Vmax
  integer, intent(in) :: Nv
  integer :: i
  real :: step
  real, intent(out) :: x(Nv)

  step = (Vmax - Vmin) / Nv
  x(1) = Vmin
  do i = 2,Nv
    x(i) = x(i-1) + step
  enddo

  end subroutine linspace


subroutine cumsum(Nalt, A, out)

  integer, intent(in) :: Nalt
  real, intent(in) :: A(Nalt)
  real, intent(out) :: out(Nalt)
  integer :: i

  out(1) = A(1)
  do i = 2,Nalt
    out(i) = out(i-1) + A(i)
  enddo

  end subroutine cumsum
