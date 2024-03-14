module cglow

! This software is part of the GLOW model.  Use is governed by the Open Source
! Academic Research License Agreement contained in the file glowlicense.txt.
! For more information see the file glow.txt.

! Version 0.981, 6/2017

! Stan Solomon and Ben Foster, 1/2015
! Stan Solomon, 1/2016, 3/2016, consolidated with cxglow
! Stan Solomon, 6/2017, zeroed out arrays

! CGLOW Defines array dimensions and use-associated variables for the GLOW model.
! Replaces the header file glow.h and common blocks /CGLOW/, /CXSECT/, and /CXPARS/
! that were used in older versions of the model (v. 0.973 and earlier).

! For variable definitions, see subroutine GLOW and subroutine EXSECT.

! Old common blocks, for reference:

!     COMMON /CGLOW/
!    >    IDATE, UT, GLAT, GLONG, ISCALE, JLOCAL, KCHEM,
!    >    F107, F107A, HLYBR, FEXVIR, HLYA, HEIEW, XUVFAC,
!    >    ZZ(JMAX), ZO(JMAX), ZN2(JMAX), ZO2(JMAX), ZNO(JMAX),
!    >    ZNS(JMAX), ZND(JMAX), ZRHO(JMAX), ZE(JMAX),
!    >    ZTN(JMAX), ZTI(JMAX), ZTE(JMAX),
!    >    PHITOP(NBINS), EFLUX(NF), EZERO(NF),
!    >    SZA, DIP, EFRAC, IERR,
!    >    ZMAJ(NMAJ,JMAX), ZCOL(NMAJ,JMAX),
!    >    WAVE1(LMAX), WAVE2(LMAX), SFLUX(LMAX),
!    >    ENER(NBINS), EDEL(NBINS),
!    >    PESPEC(NBINS,JMAX), SESPEC(NBINS,JMAX),
!    >    PHOTOI(NST,NMAJ,JMAX), PHOTOD(NST,NMAJ,JMAX), PHONO(NST,JMAX),
!    >    QTI(JMAX), AURI(NMAJ,JMAX), PIA(NMAJ,JMAX), SION(NMAJ,JMAX),
!    >    UFLX(NBINS,JMAX), DFLX(NBINS,JMAX), AGLW(NEI,NMAJ,JMAX),
!    >    EHEAT(JMAX), TEZ(JMAX), ECALC(JMAX),
!    >    ZXDEN(NEX,JMAX), ZETA(NW,JMAX), ZCETA(NC,NW,JMAX), VCB(NW)

!     COMMON /CXSECT/ SIGS(NMAJ,NBINS), PE(NMAJ,NBINS), PIN(NMAJ,NBINS),
!    >                SIGA(NMAJ,NBINS,NBINS), SEC(NMAJ,NBINS,NBINS),
!    >                SIGEX(NEI,NMAJ,NBINS), SIGIX(NEI,NMAJ,NBINS),
!    >                IIMAXX(NBINS)

!     COMMON /CXPARS/ WW(NEI,NMAJ), AO(NEI,NMAJ), OMEG(NEI,NMAJ),
!    >                ANU(NEI,NMAJ), BB(NEI,NMAJ), AUTO(NEI,NMAJ),
!    >                THI(NEI,NMAJ),  AK(NEI,NMAJ),   AJ(NEI,NMAJ),
!    >                TS(NEI,NMAJ),   TA(NEI,NMAJ),   TB(NEI,NMAJ),
!    >                GAMS(NEI,NMAJ), GAMB(NEI,NMAJ)
  use, intrinsic :: iso_fortran_env, only: wp=>real32

  implicit none
  save

!! Array dimensions, configurable:

  integer :: jmax=0              ! number of vertical levels
  integer :: nbins=0             ! number of energetic electron energy bins
  
  !! Array dimensions, non-configurable:
  
  integer,parameter :: lmax=123  ! number of wavelength intervals for solar flux
  integer,parameter :: nmaj=3    ! number of major species
  integer,parameter :: nst=6     ! number of states produced by photoionization/dissociation
  integer,parameter :: nei=10    ! number of states produced by electron impact
  integer,parameter :: nex=12    ! number of excited/ionized species
  integer,parameter :: nw=15     ! number of airglow emission wavelengths
  integer,parameter :: nc=10     ! number of component production terms for each emission

! Directory containing data files needed by glow subroutines:

  character(1024) :: data_dir

  integer :: idate,iscale,jlocal,kchem,ierr
  real    :: ut,glat,glong,f107,f107a,f107p,ap,ef,ec
  real    :: xuvfac, sza, dip, efrac
  real,dimension(nw) :: vcb

  real,allocatable,dimension(:) ::             &                      ! (jmax)
    zz, zo, zn2, zo2, zno, zns, znd, zrho, ze, &
    ztn, zti, zte, eheat, tez, ecalc, tei, tpi, tir
  real(wp),allocatable,dimension(:) :: phitop, ener, edel             ! (nbins)
  real,allocatable,dimension(:)     :: wave1, wave2, sflux            ! (lmax)
  real,allocatable,dimension(:)     :: sf_rflux, sf_scale1, sf_scale2 ! (lmax)
  real,allocatable,dimension(:,:)   :: pespec, sespec, uflx, dflx     ! (nbins,jmax)
  real,allocatable,dimension(:,:)   :: zmaj, zcol, pia, sion          ! (nmaj,jmax)
  real,allocatable,dimension(:,:,:) :: photoi, photod                 ! (nst,nmaj,jmax)
  real,allocatable,dimension(:,:)   :: phono                          ! (nst,jmax)
  real,allocatable,dimension(:,:,:) :: epsil1, epsil2, ephoto_prob    ! (nst,nmaj,lmax)
  real,allocatable,dimension(:,:)   :: sigion, sigabs                 ! (nmaj,lmax)
  real,allocatable,dimension(:,:,:) :: aglw                           ! (nei,nmaj,jmax)
  real,allocatable,dimension(:,:)   :: zxden                          ! (nex,jmax)
  real(wp),allocatable,dimension(:,:)   :: zeta                       ! (nw,jmax)
  real,allocatable,dimension(:,:,:) :: zceta                          ! (nc,nw,jmax)
  real,allocatable,dimension(:,:)   :: zlbh                           ! (nc,jmax)
  real,allocatable,dimension(:,:)   :: sigs,pe,pin                    ! (nmaj,nbins)
  real,allocatable,dimension(:,:,:) :: sigex,sigix                    ! (nei,nmaj,nbins)
  real,allocatable,dimension(:,:,:) :: siga,sec                       ! (nei,nbins,nbins)
  integer,allocatable,dimension(:)  :: iimaxx                         ! (nbins)
  real,allocatable,dimension(:,:) :: &                                ! (nei,nmaj)
    ww,ao,omeg,anu,bb,auto,thi,ak,aj,ts,ta,tb,gams,gamb

  real(wp), allocatable, dimension(:,:) :: production, loss  ! gchem.f90

  real :: snoem_zin(16)          ! altitude grid
  real :: snoem_mlatin(33)       ! magnetic latitude grid
  real :: snoem_no_mean(33,16)   ! mean nitric oxide distribution
  real :: snoem_eofs(33,16,3)    ! empirical orthogonal functions

  contains

!-----------------------------------------------------------------------

  subroutine cglow_static_init
    ! These arrays do not change size, hence they are allocated here
    ! and remain allocated for the duration of the program.
    if (.not.allocated(ww)) then
    allocate( &
      ww(nei,nmaj), &
      ao(nei,nmaj), &
      omeg(nei,nmaj), &
      anu(nei,nmaj), &
      bb(nei,nmaj), &
      auto(nei,nmaj), &
      thi(nei,nmaj), &
      ak(nei,nmaj), &
      aj(nei,nmaj), &
      ts(nei,nmaj), &
      ta(nei,nmaj), &
      tb(nei,nmaj), &
      gams(nei,nmaj), &
      gamb(nei,nmaj), &
    )
    endif

    WW(1:nei,1)=&
    (/1.96, 4.17, 9.29, 9.53,10.76,10.97,12.07,12.54, 0., 0./)
     WW(1:nei,2)=&
    (/0.98, 1.64, 4.50, 8.44, 9.90,13.50, 0.25, 0.00, 0., 0./)
     WW(1:nei,3)=&
     (/6.17, 8.16,11.03, 8.40,12.85,14.00,13.75, 1.85, 0., 0./)

     AO(1:nei,1)=&
     (/.0100,.0042,.1793,.3565,.0327,.0245,.0293,.1221, 0.,0./)
     AO(1:nei,2)=&
     (/.0797,.0211,.0215,.3400,.0657,1.110,3.480, 0.00, 0.,0./)
     AO(1:nei,3)=&
     (/2.770,.1140,.1790,.0999,.8760,.6010,1.890,1.350, 0.,0./)

     OMEG(1:nei,1)=&
     (/1.00, 1.00, 3.00, 0.75, 3.00, 0.85, 0.75, 0.75, 0.,0./)
     OMEG(1:nei,2)=&
     (/2.00, 2.00, 1.15, 0.75, 0.75, 0.75, 7.00, 0.00, 0.,0./)
     OMEG(1:nei,3)=&
     (/3.00, 3.00, 3.00, 1.00, 0.75, 0.75, 0.75, 8.00, 0.,0./)

     ANU(1:nei,1)=&
     (/2.00, 1.04, 2.53, 0.54, 2.43, 2.87, 0.93, 0.72, 0.,0./)
     ANU(1:nei,2)=&
     (/6.18, 4.14, 1.00, 1.05, 1.60, 3.00,10.87, 0.00, 0.,0./)
     ANU(1:nei,3)=&
     (/4.53, 4.78, 4.32, 4.05, 1.47, 1.27, 3.00, 1.58, 0.,0./)

     BB(1:nei,1)=&
     (/1.00, 0.50, 1.02, 0.01, 4.19, 4.88, 0.66, 0.17, 0.,0./)
     BB(1:nei,2)=&
     (/0.53, 0.51, 0.98, 0.99, 1.86, 1.00, 1.00, 0.00, 0.,0./)
     BB(1:nei,3)=&
     (/1.42, 3.54,12.70, 5.20, 0.86, 0.45, 1.00, 1.00, 0.,0./)

     AUTO(1:nei,1) = (/0.,0.,0.,0.,0.,0.,0.,0.,0.,0./)
     AUTO(1:nei,2) = (/0.,0.,0.,0.,0.,0.,0.,0.,0.,0./)
     AUTO(1:nei,3) = (/0.,0.,0.,0.,0.,0.,0.,0.,0.,0./)

     THI(1:nei,1)=&
     (/13.60,16.90,18.50, 0.00, 0.00, 0.00, 0.00, 0.,0.,0./)
     THI(1:nei,2)=&
     (/12.10,16.10,16.90,18.20,20.00,23.00,37.00, 0.,0.,0./)
     THI(1:nei,3)=&
     (/15.58,16.73,18.75,22.00,23.60,40.00, 0.00, 0.,0.,0./)

     AK(1:nei,1)=&
     (/ 1.13, 1.25, 0.67, 0.00, 0.00, 0.00, 0.00, 0.,0.,0./)
     AK(1:nei,2)=&
     (/0.47, 1.13, 1.13, 1.01, 0.65, 0.95, 0.59, 0.,0.,0./)
     AK(1:nei,3)=&
     (/2.42, 1.06, 0.55, 0.37, 0.37, 0.53, 0.00, 0.,0.,0./)

     AJ(1:nei,1)=&
     (/1.81, 1.79, 1.78, 0.00, 0.00, 0.00, 0.00, 0.,0.,0./)
     AJ(1:nei,2)=&
     (/3.76, 3.76, 3.76, 3.76, 3.76, 3.76, 3.76, 0.,0.,0./)
     AJ(1:nei,3)=&
     (/1.74, 1.74, 1.74, 1.74, 1.74, 1.74, 0.00, 0.,0.,0./)

     TS(1:nei,1)=&
     (/6.41, 6.41, 6.41, 0.00, 0.00, 0.00, 0.00, 0.,0.,0./)
     TS(1:nei,2)=&
     (/1.86, 1.86, 1.86, 1.86, 1.86, 1.86, 1.86, 0.,0.,0./)
     TS(1:nei,3)=&
     (/4.71, 4.71, 4.71, 4.71, 4.71, 4.71, 0.00, 0.,0.,0./)

     TA(1:nei,1)=&
     (/3450.,3450.,3450.,   0.,   0.,   0.,   0., 0.,0.,0./)
     TA(1:nei,2)=&
     (/1000.,1000.,1000.,1000.,1000.,1000.,1000., 0.,0.,0./)
     TA(1:nei,3)=&
     (/1000.,1000.,1000.,1000.,1000.,1000.,   0., 0.,0.,0./)

     TB(1:nei,1)=&
     (/162.00,162.0,162.0, 0.00, 0.00, 0.00, 0.00, 0.,0.,0./)
     TB(1:nei,2)=&
     (/24.20,32.20,33.80,36.40,40.60,46.00,74.00, 0.,0.,0./)
     TB(1:nei,3)=&
     (/31.16,33.46,37.50,44.00,47.20,80.00, 0.00, 0.,0.,0./)

     GAMS(1:nei,1)=&
     (/13.00,13.0,13.00, 0.00, 0.00, 0.00, 0.00, 0.,0.,0./)
     GAMS(1:nei,2)=&
     (/18.50,18.5,18.50,18.50,18.50,18.50,18.50, 0.,0.,0./)
     GAMS(1:nei,3)=&
     (/13.80,13.8,13.80,13.80,13.80,13.80, 0.00, 0.,0.,0./)

     GAMB(1:nei,1)=&
     (/-.815,-.815,-.815,0.00, 0.00, 0.00, 0.00, 0.,0.,0./)
     GAMB(1:nei,2)=&
     (/12.10,16.10,16.90,18.2,20.30,23.00,37.00, 0.,0.,0./)
     GAMB(1:nei,3)=&
     (/15.58,16.73,18.75,22.0,23.60,40.00, 0.00, 0.,0.,0./)

    if (.not.allocated(wave1)) then
    allocate(&
      wave1(lmax), &
      wave2(lmax), &
      sflux(lmax), &
      sf_rflux(lmax), &
      sf_scale1(lmax), &
      sf_scale2(lmax), &
    )


    wave1(:) = 0.
    wave2(:) = 0.
    sflux(:) = 0.
    sf_rflux(:) = 0.
    sf_scale1(:) = 0.
    sf_scale2(:) = 0.
    endif
    if (.not.allocated(epsil1)) then
    allocate(&
      epsil1(nst,nmaj,lmax), &
      epsil2(nst,nmaj,lmax), &
      ephoto_prob(nst,nmaj,lmax), &
    )

    epsil1(:,:,:) = 0.
    epsil2(:,:,:) = 0.
    ephoto_prob(:,:,:) = 0.
    endif

    if (.not.allocated(sigion)) then
    allocate(&
      sigion(nmaj,lmax), &
      sigabs(nmaj,lmax), &
    )

    sigion(:,:) = 0.
    sigabs(:,:) = 0.
    endif

    ! Does not depend on anything
    call snoem_init()              ! initialize snoem
  end subroutine cglow_static_init

  subroutine cglow_static_deinit
    if (.not.allocated(ww)) return
    deallocate(ww,ao,omeg,anu,bb,auto,thi,ak,aj,ts,ta,tb,gams,gamb)
    deallocate(wave1, wave2, sflux, sf_rflux, sf_scale1, sf_scale2)
    deallocate(epsil1, epsil2, ephoto_prob)
    deallocate(sigion, sigabs)
  end subroutine cglow_static_deinit

  subroutine cglow_dynamic_alloc
    ! These arrays are allocated and deallocated as needed.
    ! They are zeroed out in cglow_init.
    allocate(zz   (jmax), &
       zo   (jmax), &
       zn2  (jmax), &
       zo2  (jmax), &
       zno  (jmax), &
       zns  (jmax), &
       znd  (jmax), &
       zrho (jmax), &
       ze   (jmax), &
       ztn  (jmax), &
       zti  (jmax), &
       zte  (jmax), &
       eheat(jmax), &
       tez  (jmax), &
       tei  (jmax), &
       tpi  (jmax), &
       tir  (jmax), &
       ecalc(jmax) &
    )

    allocate(zxden(nex,jmax), &
       zeta(nw,jmax),   &
       zceta(nc,nw,jmax), &
       zlbh(nc,jmax))

    allocate(aglw(nei,nmaj,jmax))

    if (.not.allocated(production))allocate(production(nex,jmax))
    if (.not.allocated(loss))allocate(loss(nex,jmax))

    if (.not.allocated(phitop)) allocate(phitop(nbins))
    if (.not.allocated(ener)) allocate(ener(nbins))
    if (.not.allocated(edel)) allocate(edel(nbins))

    allocate(pespec(nbins,jmax), &
       sespec(nbins,jmax), &
       uflx  (nbins,jmax), &
       dflx  (nbins,jmax))

    allocate(zmaj(nmaj,jmax), &
       zcol(nmaj,jmax), &
       pia (nmaj,jmax), &
       sion(nmaj,jmax))

    allocate(sigs(nmaj,nbins), &
        pe  (nmaj,nbins), &
        pin (nmaj,nbins))
      
    allocate(phono(nst,jmax))
    allocate(photoi(nst,nmaj,jmax), &
             photod(nst,nmaj,jmax))
 
     allocate(sigex(nei,nmaj,nbins), &
        sigix(nei,nmaj,nbins))
 
     allocate(siga(nei,nbins,nbins), &
        sec (nei,nbins,nbins))
 
     allocate(iimaxx(nbins))

    !  call cglow_dynamic_zero
  end subroutine cglow_dynamic_alloc

  subroutine cglow_dynamic_dealloc
    if (allocated(zz)) deallocate(zz)
    if (allocated(zo)) deallocate(zo)
    if (allocated(zn2)) deallocate(zn2)
    if (allocated(zo2)) deallocate(zo2)
    if (allocated(zno)) deallocate(zno)
    if (allocated(zns)) deallocate(zns)
    if (allocated(znd)) deallocate(znd)
    if (allocated(zrho)) deallocate(zrho)
    if (allocated(ze)) deallocate(ze)
    if (allocated(ztn)) deallocate(ztn)
    if (allocated(zti)) deallocate(zti)
    if (allocated(zte)) deallocate(zte)
    if (allocated(eheat)) deallocate(eheat)
    if (allocated(tez)) deallocate(tez)
    if (allocated(tei)) deallocate(tei)
    if (allocated(tpi)) deallocate(tpi)
    if (allocated(tir)) deallocate(tir)
    if (allocated(ecalc)) deallocate(ecalc)
    if (allocated(zxden)) deallocate(zxden)
    if (allocated(zeta)) deallocate(zeta)
    if (allocated(zceta)) deallocate(zceta)
    if (allocated(zlbh)) deallocate(zlbh)
    if (allocated(phitop)) deallocate(phitop)
    if (allocated(ener)) deallocate(ener)
    if (allocated(edel)) deallocate(edel)
    if (allocated(pespec)) deallocate(pespec)
    if (allocated(sespec)) deallocate(sespec)
    if (allocated(uflx)) deallocate(uflx)
    if (allocated(dflx)) deallocate(dflx)
    if (allocated(zmaj)) deallocate(zmaj)
    if (allocated(zcol)) deallocate(zcol)
    if (allocated(pia)) deallocate(pia)
    if (allocated(sion)) deallocate(sion)
    if (allocated(sigs)) deallocate(sigs)
    if (allocated(aglw)) deallocate(aglw)
    if (allocated(pe)) deallocate(pe)
    if (allocated(pin)) deallocate(pin)
    if (allocated(sigex)) deallocate(sigex)
    if (allocated(sigix)) deallocate(sigix)
    if (allocated(siga)) deallocate(siga)
    if (allocated(sec)) deallocate(sec)
    if (allocated(iimaxx)) deallocate(iimaxx)
    if (allocated(phono)) deallocate(phono)
    if (allocated(photoi)) deallocate(photoi)
    if (allocated(photod)) deallocate(photod)
  end subroutine cglow_dynamic_dealloc

  subroutine cglow_dynamic_zero
       production(:,:) = 0.
       loss(:,:) = 0.

       zz   (:)     =0.
       zo   (:)     =0.
       zn2  (:)     =0.
       zo2  (:)     =0.
       zno  (:)     =0.
       zns  (:)     =0.
       znd  (:)     =0.
       zrho (:)     =0.
       ze   (:)     =0.
       ztn  (:)     =0.
       zti  (:)     =0.
       zte  (:)     =0.
       eheat(:)     =0.
       tez  (:)     =0.
       tei  (:)     =0.
       tpi  (:)     =0.
       tir  (:)     =0.
       ecalc(:)     =0.
       zxden(:,:)   =0.
       zeta (:,:)   =0.
       zceta(:,:,:) =0.
       zlbh  (:,:)  =0.
       phitop(:)    =0.
       ener  (:)    =0.
       edel  (:)    =0.
       pespec(:,:)  =0.
       sespec(:,:)  =0.
       uflx  (:,:)  =0.
       dflx  (:,:)  =0.
       zmaj(:,:)    =0.
       zcol(:,:)    =0.
       pia (:,:)    =0.
       sion(:,:)    =0.
       aglw  (:,:,:)=0.
       photoi(:,:,:)=0.
       photod(:,:,:)=0.
       phono(:,:)   =0.
       sigs(:,:)    =0.
       pe  (:,:)    =0.
       pin (:,:)    =0.
       sigex(:,:,:) =0.
       sigix(:,:,:) =0.
       siga(:,:,:)  =0.
       sec (:,:,:)  =0.
       iimaxx(:)    =0.

  end subroutine cglow_dynamic_zero

  subroutine sflux_init()
    ! Initialize the solar flux model
    call ssflux_init(iscale)       ! initialize ssflux
    call ephoto_init()         ! initialize ephoto
  end subroutine sflux_init
  
  
  subroutine egrid_init
      ! Depends on nbins
       call egrid(ener, edel, nbins)  ! initialize energy grid
       call EXSECT(ener, edel, nbins) ! call exsect

  end subroutine egrid_init

  subroutine cglow_init
    ! Initialize the GLOW model
    call cglow_static_init
    call cglow_dynamic_alloc
    call sflux_init
    call egrid_init
  end subroutine cglow_init

!-----------------------------------------------------------------------

end module cglow
