! Subroutine EPHOTO_INIT

! This software is part of the GLOW model.  Use is governed by the Open Source
! Academic Research License Agreement contained in the file glowlicense.txt.
! For more information see the file glow.txt.

! Adapted from Banks & Nagy 2-stream input code by Stan Solomon, 6/1988
! Modified to handle Auger electrons, Stan Solomon, 7/1990
! Reads cross sectons from files (for 1-nm bins), Scott Bailey, ~1994
! Modified bin structure, fixed CIII problem, Stan Solomon, 12/2000
! Corrected additional Auger problem, Liying Qian, 11/2002
! Converged above three branches, Stan Solomon, 3/2005
! Removed LIMIN, wavelength loop now runs from 1 to LMAX, SCS, 3/2005
! Converted common blocks to use-associated variables, Ben Foster, 2015
! Refactored to f90, SCS, 6/2016

! This subroutine calculates photoionization, rates, certain
! photodissociative excitation rates, and the photoelectron production
! spectrum as a function of altitude.  Uses continuously variable energy
! grid.  Three major species: O, O2, N2; NO is treated as a minor (non-
! absorbing) specie.

! Input supplied through use-associated variables defined in module cglow.f90:
! WAVE1   wavelength array, upper bound; Angstroms
! WAVE2   wavelength array, lower bound; Angstroms
! SFLUX   solar flux array; photons cm-2 sec-1
! ZZ      altitude array; cm above earth
! ZMAJ    density array for species O, O2, N2, altitude; cm-3
! ZNO     density of NO at each altitude; cm-3
! ZCOL    slant column density for species O, O2, N2, altitude; cm-2
! ENER    energy grid for photoelectrons; eV
! EDEL     array of energy grid increments; eV

! Output provided through use-associated variables defined in module cglow.f90:
! PESPEC  photoelectron production spectrum for each altitude; cm-3 s-1
! PHOTOI  photoionization rates for state, species, altitude; cm-3 s-1
! PHOTOD  photodissoc./exc. rates for state, species, alt.; cm-3 s-1
! PHONO   photoionization/dissoc./exc. rates for NO; cm-3 s-1

! Other definitions:
! DSPECT  ionization rate in particular wavelength bin; cm-3 s-1
! TAU     optical depth, dimensionless
! FLUX    solar flux at altitude; cm-2 s-1
! SIGABS  photoabsorption cross sections, O, O2, N2; cm2
! SIGION  photoionization cross sections, O, O2, N2; cm2
! SIGAO, SIGAO2, SIGAN2, SIGIO, SIGIO2, SIGIN2; cross sect. data arrays
! NNN     number of states for each species
! TPOT    ionization potentials for each species, state; eV
! PROB    branching ratios for each state, species, and wavelength bin:
!         O+ states: 4S, 2Do, 2Po, 4Pe, 2Pe
!         O2+ states: X, a+A, b, dissoc.
!         N2+ states: X, A, B, C, F, dissoc.
! PROBO, PROBO2, PROBN2; branching ratio data arrays
! BSO2    yield of O(1S) from dissociation of O2
! EPSIL1  energy loss lower bound for state, species, wavelength; eV
! EPSIL2  energy loss upper bound for state, species, wavelength; eV
! SIGNO   NO photoionization xsect at Ly-alpha
! AUGE    Mean energy of Auger electrons for each species; eV
! AUGL    Wavelength threshold for Auger electrons; Angstroms

! Array dimensions:
! JMAX    number of altitude levels
! NBINS   number of energetic electron energy bins
! LMAX    number of wavelength intervals for solar flux
! NMAJ    number of major species
! NST     number of states produced by photoionization/dissociation


subroutine ephoto_init

  use cglow,only: nbins,lmax,nmaj,nst
  use cglow,only: wave1,wave2,epsil1,epsil2,sigion,sigabs,ephoto_prob
  use cglow,only: data_dir

  implicit none
  save

  integer :: nnn(nmaj)
  real ::   tpot(nst,nmaj), &
            sigao(lmax), sigao2(lmax), sigan2(lmax), &
            sigio(lmax), sigio2(lmax), sigin2(lmax), &
            probo(nst,lmax), probo2(nst,lmax), probn2(nst,lmax), &
            bso2(lmax), auge(nmaj), augl(nmaj)

  real,parameter :: signo = 2.0e-18
  integer :: l,n,k,i
  real :: aa,bb
  character(len=1024) :: filepath

  nnn = (/5,4,6/)
  tpot(1:nst,1) = (/13.61, 16.93, 18.63, 28.50, 40.00,  0.00/)
  tpot(1:nst,2) = (/12.07, 16.10, 18.20, 20.00,  0.00,  0.00/)
  tpot(1:nst,3) = (/15.60, 16.70, 18.80, 30.00, 34.80, 25.00/)
  auge = (/500.,500.,360./)
  augl = (/24.,24.,33./)
  bso2(1:12) = 0.
  bso2(13) = .01
  bso2(14) = .03
  bso2(15:21) = .10
  bso2(22:29) = .07
  bso2(30:34) = .03
  bso2(35:39) = .01
  bso2(40:lmax) = 0.

  ! Initialize the output arrays
  epsil1(:,:,:) = 0.
  epsil2(:,:,:) = 0.
  sigion(:,:) = 0.
  sigabs(:,:) = 0.
  ephoto_prob(:,:,:) = 0.

! First time only: Read cross section data from files, convert to cm2,
! calculate energy losses:

    filepath = trim(data_dir)//'ephoto_xn2.dat'
    open(unit=1,file=filepath,status='old',action='read')
    read(1,*)
    read(1,*)
    read(1,*)
    read(1,*)
    do l=lmax,1,-1
      read(1,*) aa,bb,(probn2(n,l),n=1,nst),sigin2(l),sigan2(l)
    enddo
    close(1)

    filepath = trim(data_dir)//'ephoto_xo2.dat'
    open(unit=1,file=filepath,status='old',action='read')
    read(1,*)
    read(1,*)
    read(1,*)
    read(1,*)
    do l=lmax,1,-1
      read(1,*) aa,bb,(probo2(n,l),n=1,nst),sigio2(l),sigao2(l)
    enddo
    close(1)

    filepath = trim(data_dir)//'ephoto_xo.dat'
    open(unit=1,file=filepath,status='old',action='read')
    read(1,*)
    read(1,*)
    read(1,*)
    read(1,*)
    do l=lmax,1,-1
      read(1,*) aa,bb,(probo(n,l),n=1,nst),sigio(l),sigao(l)
    enddo
    close(1)

    do l=1,lmax
      sigabs(1,l) = sigao(l)  * 1.e-18
      sigabs(2,l) = sigao2(l) * 1.e-18
      sigabs(3,l) = sigan2(l) * 1.e-18
      sigion(1,l) = sigio(l)  * 1.e-18
      sigion(2,l) = sigio2(l) * 1.e-18
      sigion(3,l) = sigin2(l) * 1.e-18
    enddo

    do l=1,lmax
      do k=1,nst
        ephoto_prob(k,1,l) = probo(k,l)
        ephoto_prob(k,2,l) = probo2(k,l)
        ephoto_prob(k,3,l) = probn2(k,l)
      enddo
    enddo

    do l=1,lmax 
      do i=1,nmaj 
        do k=1,nnn(i) 
          epsil1(k,i,l)=12397.7/wave1(l)-tpot(k,i) 
          epsil2(k,i,l)=12397.7/wave2(l)-tpot(k,i) 
          if (wave1(l) <= augl(i)) then
            epsil1(k,i,l) = epsil1(k,i,l) - auge(i)
            epsil2(k,i,l) = epsil2(k,i,l) - auge(i)
          endif
        enddo
      enddo
    enddo 


end subroutine ephoto_init

