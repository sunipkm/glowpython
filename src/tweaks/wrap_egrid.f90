! DEL is a reserved word in python, so this module is necessary to
! make sure cglow del is initialized correctly.

subroutine wrap_egrid(e, de)

  use cglow, only: ener,edel,nbins

  real, dimension(nbins), intent(out) :: e, de

  call egrid(ener, edel, nbins)

  e = ener
  de = edel

end subroutine wrap_egrid
