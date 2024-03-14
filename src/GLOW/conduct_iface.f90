! This is the interface file for the conduct subroutine
! It is used to call the conduct subroutine from the main program
! The conduct subroutine is used to calculate the Pedersen and Hall conductivities.
!
! Input:
!   alt(jmax) - The altitude at each grid point
!   glat - The latitude of the grid
!   glong - The longitude of the grid
!   zo(jmax) - Atomic oxygen density at each grid point
!   zo2(jmax) - Molecular oxygen density at each grid point
!   zn2(jmax) - Molecular nitrogen density at each grid point
!   zxden(3, jmax) - O+(4S) density at each grid point
!   zxden(6, jmax) - O2+ density at each grid point
!   zxden(7, jmax) - NO+ density at each grid point
!   ztn(jmax) - Neutral temperature at each grid point
!   zti(jmax) - Ion temperature at each grid point
!   zte(jmax) - Electron temperature at each grid point
!
! Output:
!   pedcond(jmax) - Pedersen conductivity at each grid point
!   hallcond(jmax) - Hall conductivity at each grid point


subroutine conduct_iface(jmax, alt, pedcond, hallcond)
    use cglow, only: glat, glong 
    use cglow, only: zo, zo2, zn2, zxden, ztn, zti, zte
    implicit none
    integer, intent(in) :: jmax
    real, intent(in) :: alt(jmax)
    real, intent(inout) :: pedcond(jmax), hallcond(jmax)
    integer :: j
    do j = 1, jmax
        call conduct(glat, glong, alt(j), &
                     zo(j), zo2(j), zn2(j), &
                     zxden(3, j), zxden(6, j), &
                     zxden(7, j), &
                     ztn(j), zti(j), zte(j), &
                     pedcond(j), hallcond(j))
    end do


end subroutine conduct_iface
