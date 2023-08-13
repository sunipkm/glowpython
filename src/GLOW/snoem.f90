! Subroutine SNOEM calculates nitric oxide zonal mean altitude profile
! as function of magnetic latitude for specified day of year, Kp, and F10.7.

! The NOEM empirical model is based on data from the SNOE ultraviolet
! spectrometer during 1998-2000, using empirical orthogonal function analysis.
! Altitude range is from 100 to 150 km.

! Marsh et al., JGR, 109, A07301, doi:10.1029/2003JA010199, 2004.

! Adapted by Stan Solomon, 5/2014, from IDL and F90 code supplied by Dan Marsh. 
! Minor revisions to make compatible with gfortran, SCS, 4/2017

    subroutine snoem(doy, kp, f107, z, mlat, nozm)

      use cglow,only: snoem_zin, snoem_mlatin, snoem_no_mean, snoem_eofs

      implicit none

      integer,intent(in) :: doy
      real,intent(in) :: kp, f107
      real,intent(out) :: z(16), mlat(33), nozm(33,16)

      real :: theta0                 ! day number in degrees
      real :: dec                    ! solar declination angle
      real :: m1, m2, m3             ! coefficients for first 3 eofs
      real, parameter :: pi=3.1415926536
      integer :: j, k

!... calculate coefficients (m1 to m3) for eofs based on geophysical parameters

!... eof1 - kp 

      m1 =  kp * 0.689254 - 1.53366

!... eof2 - declination

      theta0 = 2.*pi * float(doy - 1) / 365.

      dec = 0.006918 &
          - 0.399912 * cos(theta0)   + 0.070257 * sin(theta0) &
          - 0.006758 * cos(2*theta0) + 0.000907 * sin(2*theta0) & 
          - 0.002697 * cos(3*theta0) + 0.001480 * sin(3*theta0)

      dec = dec * 180./pi

      m2 = -0.31978 + dec*0.097309 + dec**2*0.00048979 - dec**3*0.00010360
      
!... eof3 - f107 

      m3 =  alog10(f107) * 6.35777 - 13.8163 

!... zonal mean distrib. is sum of mean and eofs

      do k=1,16
        do j=1,33
          nozm(j,k) = snoem_no_mean(j,k)-m1*snoem_eofs(j,k,1)+m2*snoem_eofs(j,k,2)-m3*snoem_eofs(j,k,3) 
        end do
      end do

      z(:) = snoem_zin(:)
      mlat(:) = snoem_mlatin(:)

      return

    end subroutine snoem
