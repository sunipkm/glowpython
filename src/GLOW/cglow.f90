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

  real,allocatable,dimension(:) ::             &                   ! (jmax)
    zz, zo, zn2, zo2, zno, zns, znd, zrho, ze, &
    ztn, zti, zte, eheat, tez, ecalc, tei, tpi, tir
  real(wp),allocatable,dimension(:)     :: phitop, ener, edel           ! (nbins)
  real,allocatable,dimension(:)     :: wave1, wave2, sflux         ! (lmax)
  real,allocatable,dimension(:)     :: sf_rflux, sf_scale1, sf_scale2 ! (lmax)
  real,allocatable,dimension(:,:)   :: pespec, sespec, uflx, dflx  ! (nbins,jmax)
  real,allocatable,dimension(:,:)   :: zmaj, zcol, pia, sion       ! (nmaj,jmax)
  real,allocatable,dimension(:,:,:) :: photoi, photod              ! (nst,nmaj,jmax)
  real,allocatable,dimension(:,:)   :: phono                       ! (nst,jmax)
  real,allocatable,dimension(:,:,:) :: epsil1, epsil2, ephoto_prob ! (nst,nmaj,lmax)
  real,allocatable,dimension(:,:)   :: sigion, sigabs              ! (nmaj,lmax)
  real,allocatable,dimension(:,:,:) :: aglw                        ! (nei,nmaj,jmax)
  real,allocatable,dimension(:,:)   :: zxden                       ! (nex,jmax)
  real(wp),allocatable,dimension(:,:)   :: zeta                        ! (nw,jmax)
  real,allocatable,dimension(:,:,:) :: zceta                       ! (nc,nw,jmax)
  real,allocatable,dimension(:,:)   :: zlbh                        ! (nc,jmax)
  real,allocatable,dimension(:,:)   :: sigs,pe,pin                 ! (nmaj,nbins)
  real,allocatable,dimension(:,:,:) :: sigex,sigix                 ! (nei,nmaj,nbins)
  real,allocatable,dimension(:,:,:) :: siga,sec                    ! (nei,nbins,nbins)
  integer,allocatable,dimension(:)  :: iimaxx                      ! (nbins)
  real,allocatable,dimension(:,:) :: &                             ! (nei,nmaj)
    ww,ao,omeg,anu,bb,auto,thi,ak,aj,ts,ta,tb,gams,gamb

  real(wp), allocatable, dimension(:,:) :: production, loss  ! gchem.f90

  real :: snoem_zin(16)          ! altitude grid
  real :: snoem_mlatin(33)       ! magnetic latitude grid
  real :: snoem_no_mean(33,16)   ! mean nitric oxide distribution
  real :: snoem_eofs(33,16,3)    ! empirical orthogonal functions

  contains

!-----------------------------------------------------------------------

  subroutine cglow_init

! Allocate variable arrays:

    allocate        &
      (zz   (jmax), &
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
       ecalc(jmax))

    allocate            &
      (zxden(nex,jmax), &
       zeta(nw,jmax),   &
       zceta(nc,nw,jmax), &
       zlbh(nc,jmax))

    if (.not.allocated(production))allocate(production(NEX,JMAX))
    if (.not.allocated(loss))allocate(loss(NEX,JMAX))

    if (.not.allocated(phitop)) allocate(phitop(nbins))
    if (.not.allocated(ener)) allocate(ener(nbins))
    if (.not.allocated(edel)) allocate(edel(nbins))

    allocate           &
      (wave1(lmax),    &
       wave2(lmax),    &
       sflux(lmax),    &
       sf_rflux(lmax),    &
       sf_scale1(lmax),&
       sf_scale2(lmax))

    allocate               &
      (pespec(nbins,jmax), &
       sespec(nbins,jmax), &
       uflx  (nbins,jmax), &
       dflx  (nbins,jmax))

    allocate &
      (zmaj(nmaj,jmax), &
       zcol(nmaj,jmax), &
       pia (nmaj,jmax), &
       sion(nmaj,jmax))

    allocate &
      (aglw  (nei,nmaj,jmax), &
       photoi(nst,nmaj,jmax), &
       photod(nst,nmaj,jmax), &
       epsil1(nst,nmaj,lmax), &
       epsil2(nst,nmaj,lmax), &
       ephoto_prob(nst,nmaj,lmax), &
       sigion(nmaj,lmax),     &
       sigabs(nmaj,lmax),     &
       phono(nst,jmax))

    allocate             &
      (sigs(nmaj,nbins), &
       pe  (nmaj,nbins), &
       pin (nmaj,nbins))

    allocate                  &
      (sigex(nei,nmaj,nbins), &
       sigix(nei,nmaj,nbins))

    allocate                  &
      (siga(nei,nbins,nbins), &
       sec (nei,nbins,nbins))

    allocate(iimaxx(nbins))

    allocate           &
      (ww  (nei,nmaj), &
       ao  (nei,nmaj), &
       omeg(nei,nmaj), &
       anu (nei,nmaj), &
       bb  (nei,nmaj), &
       auto(nei,nmaj), &
       thi (nei,nmaj), &
       ak  (nei,nmaj), &
       aj  (nei,nmaj), &
       ts  (nei,nmaj), &
       ta  (nei,nmaj), &
       tb  (nei,nmaj), &
       gams(nei,nmaj), &
       gamb(nei,nmaj))

! Zero all allocated variable arrays:

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
       zeta(:,:)    =0.
       zceta(:,:,:) =0.
       zlbh(:,:)    =0.
       phitop(:)    =0.
       ener  (:)    =0.
       edel   (:)    =0.
       wave1(:)     =0.
       wave2(:)     =0.
       sflux(:)     =0.
       sf_rflux(:)  =0.
       sf_scale1(:) =0.
       sf_scale2(:) =0.
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
       epsil1(:,:,:) =0.
       epsil2(:,:,:) =0.
       ephoto_prob(:,:,:) =0.
       sigabs(:,:)    =0.
       sigion(:,:)    =0.
       sigs(:,:)    =0.
       pe  (:,:)    =0.
       pin (:,:)    =0.
       sigex(:,:,:) =0.
       sigix(:,:,:) =0.
       siga(:,:,:)  =0.
       sec (:,:,:)  =0.
       iimaxx(:)    =0.
       ww  (:,:)    =0.
       ao  (:,:)    =0.
       omeg(:,:)    =0.
       anu (:,:)    =0.
       bb  (:,:)    =0.
       auto(:,:)    =0.
       thi (:,:)    =0.
       ak  (:,:)    =0.
       aj  (:,:)    =0.
       ts  (:,:)    =0.
       ta  (:,:)    =0.
       tb  (:,:)    =0.
       gams(:,:)    =0.
       gamb(:,:)    =0.

       call egrid(ener, edel, nbins)  ! initialize energy grid
       call snoem_init()              ! initialize snoem
       call ssflux_init(iscale)       ! initialize ssflux
       call ephoto_init()             ! initialize ephoto
       call EXSECT(ener, edel, nbins) ! call exsect

  end subroutine cglow_init

!-----------------------------------------------------------------------

end module cglow
