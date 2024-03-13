subroutine cglow_release

  use cglow

  if (allocated (zz)) deallocate (zz)
  if (allocated (zo)) deallocate (zo)
  if (allocated (zn2)) deallocate (zn2)
  if (allocated (zo2)) deallocate (zo2)  
  if (allocated (zno)) deallocate (zno)  
  if (allocated (zns)) deallocate (zns)  
  if (allocated (znd)) deallocate (znd)  
  if (allocated (zrho)) deallocate (zrho) 
  if (allocated (ze)) deallocate (ze)   
  if (allocated (ztn)) deallocate (ztn)  
  if (allocated (zti)) deallocate (zti)  
  if (allocated (zte)) deallocate (zte)  
  if (allocated (eheat)) deallocate (eheat)
  if (allocated (tez)) deallocate (tez)  
  if (allocated (tei)) deallocate (tei)  
  if (allocated (tpi)) deallocate (tpi)  
  if (allocated (tir)) deallocate (tir)  
  if (allocated (ecalc)) deallocate (ecalc)

  if (allocated (zxden)) deallocate (zxden)
  if (allocated (zeta)) deallocate (zeta)
  if (allocated (zceta)) deallocate (zceta)
  if (allocated (zlbh)) deallocate (zlbh)

  if (allocated (phitop)) deallocate (phitop)
  if (allocated (ener)) deallocate (ener)
  if (allocated (edel)) deallocate (edel)

  if (allocated (wave1)) deallocate (wave1)
  if (allocated (wave2)) deallocate (wave2)
  if (allocated (sflux)) deallocate (sflux)
  if (allocated (sf_rflux)) deallocate (sf_rflux)
  if (allocated (sf_scale1)) deallocate (sf_scale1)
  if (allocated (sf_scale2)) deallocate (sf_scale2)

  if (allocated (pespec)) deallocate (pespec)
  if (allocated (sespec)) deallocate (sespec)
  if (allocated (uflx)) deallocate (uflx)
  if (allocated (dflx)) deallocate (dflx)

  if (allocated (zmaj)) deallocate (zmaj)
  if (allocated (zcol)) deallocate (zcol)
  if (allocated (pia)) deallocate (pia)
  if (allocated (sion)) deallocate (sion)

  if (allocated (aglw)) deallocate (aglw)
  if (allocated (photoi)) deallocate (photoi)
  if (allocated (photod)) deallocate (photod)
  if (allocated (phono)) deallocate (phono)
  if (allocated (epsil1)) deallocate (epsil1)
  if (allocated (epsil2)) deallocate (epsil2)
  if (allocated (ephoto_prob)) deallocate (ephoto_prob)
  if (allocated (sigion)) deallocate (sigion)
  if (allocated (sigabs)) deallocate (sigabs)

  if (allocated (sigs)) deallocate (sigs)
  if (allocated (pe)) deallocate (pe)
  if (allocated (pin)) deallocate (pin)
  
  if (allocated (sigex)) deallocate (sigex)
  if (allocated (sigix)) deallocate (sigix)
  
  if (allocated (siga)) deallocate (siga)
  if (allocated (sec)) deallocate (sec)

  if (allocated (iimaxx)) deallocate (iimaxx)

  if (allocated (ww)) deallocate (ww)
  if (allocated (ao)) deallocate (ao)
  if (allocated (omeg)) deallocate (omeg)
  if (allocated (anu)) deallocate (anu)
  if (allocated (bb)) deallocate (bb)
  if (allocated (auto)) deallocate (auto)
  if (allocated (thi)) deallocate (thi)
  if (allocated (ak)) deallocate (ak)
  if (allocated (aj)) deallocate (aj)
  if (allocated (ts)) deallocate (ts)
  if (allocated (ta)) deallocate (ta)
  if (allocated (tb)) deallocate (tb)
  if (allocated (gams)) deallocate (gams)
  if (allocated (gamb)) deallocate (gamb)

end subroutine cglow_release
