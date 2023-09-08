from __future__ import annotations
from typing import Iterable, Sequence, SupportsFloat as Numeric
import atexit
import warnings

from . import fortran
from .fortran import cglow, cglow as cg, mzgrid, maxt, glow, conduct # type: ignore

import xarray
from xarray import Dataset, Variable
from numpy import array, asarray, float32, ones, zeros, copy
from datetime import datetime, timedelta
import geomagdata as gi
import pytz
from os import path

import matplotlib.pyplot as plt

DATA_DIR = path.join(path.dirname(__file__), 'data', '')
IRI90_DIR = path.join(DATA_DIR, 'iri90/')

cglow.jmax = 0 # initialize this to zero
cglow.nbins = 0 # initialize this to zero

def init_cglow(jmax: int, nbins: int, force_realloc: bool = False) -> None:
    """Initialize FORTRAN CGLOW module.

    Args:
        jmax (int): Number of altitude bins.
        nbins (int): Number of energy bins.
        force_realloc (bool, optional): Force reallocation of FORTRAN arrays. Defaults to False.
    """
    if force_realloc: release_cglow()
    if cglow.jmax == jmax and cglow.nbins == nbins: # no reallocation required
        return
    if (cglow.jmax != jmax or cglow.nbins != nbins) and (cglow.nbins != 0 and cglow.jmax != 0): # reallocation required
        release_cglow()
    cglow.jmax = jmax
    cglow.nbins = nbins
    cglow.data_dir.put(0, '{: <1024s}'.format(DATA_DIR))
    cglow.cglow_init()
    # global __CGLOW_INIT__
    # if __CGLOW_INIT__:
    #     # raise RuntimeError("cglow_init has already been called!")
    #     release_cglow()
    #     init_cglow(jmax)
    # else:
    #     cglow.data_dir.put(0, '{: <1024s}'.format(DATA_DIR))
    #     cglow.jmax = jmax
    #     cglow.cglow_init()
    #     __CGLOW_INIT__ = True

@atexit.register
def release_cglow():
    'Deallocate all allocatable cglow arrays'
    fortran.cglow_release()
    cglow.jmax = 0
    cglow.nbins = 0

def reset_cglow(jmax: int=None, nbins: int=None) -> None:
    """Reset CGLOW module and reallocate all arrays.

    Args:
        jmax (int, optional): Number of altitude bins. Defaults to None, uses previous value from CGLOW.
        nbins (int, optional): Number of energy bins. Defaults to None, uses previous value from CGLOW.

    Raises:
        ValueError: jmax OR nbins is zero, which is an impossible state.
    """
    if jmax is None:
        jmax = cglow.jmax
    if nbins is None:
        nbins = cglow.nbins
    if jmax == 0 and nbins == 0:
        warnings.warn('jmax or nbins is zero, nothing to reset', RuntimeWarning)
        return
    if jmax == 0 or nbins == 0:
        raise ValueError('jmax or nbins is zero, cannot reset, impossible state')
    release_cglow()
    init_cglow(jmax, nbins)

def set_standard_switches(iscale: int=1, xuvfac: Numeric = 3, kchem: int = 4, jlocal: bool = False, itail: bool = False, fmono: Numeric = 0, emono: Numeric = 0) -> None:
    """Set standard switches for CGLOW model.

    Args:
        iscale (int, optional): Solar flux model. Defaults to 1 (EUVAC model). 
        Use 0 for Hinteregger, and 2 for user-supplied solar flux model. 
        In that case, `data/ssflux_user.dat` must be contain the user-supplied solar flux model.

        xuvfac (Numeric, optional): XUV enhancement factor. Defaults to 3.

        kchem (int, optional): CGLOW chemical calculations. Defaults to 4.
        `kchem = 0: no calculations at all are performed.`
        `kchem = 1: electron density, O+(4S), N+, N2+, O2+, NO+ supplied at all altitudes; O+(2P), O+(2D), excited neutrals, and emission rates are calculated.`
        `kchem = 2: electron density, O+(4S), N+, N2+, O2+, NO+ supplied at all altitudes; O+(2P), O+(2D), excited neutrals, and emission rates are calculated.`
        `kchem = 3: electron density supplied at all altitudes; everything else calculated.  Note that this may violate charge neutrality and/or lead to other unrealistic results below 200 km, if the electron density supplied is significantly different from what the model thinks it should be.  If it is desired to use a specified ionosphere, KCHEM=2 is probably a better option.`
        `kchem = 4: electron density supplied above 200 km; electron density below 200 km is calculated, everything else calculated at all altitudes. Electron density for the next two levels above J200 is log interpolated between E(J200) and E(J200+3).`

        jlocal (bool, optional): Set to False for electron transport calculations, set to True for local calculations only. Defaults to False.
        itail (bool, optional): Disable/enable low-energy tail. Defaults to False.
        fmono (float, optional): Monoenergetic energy flux, erg/cm^2. Defaults to 0.
        emono (float, optional): Monoenergetic characteristic energy, keV. Defaults to 0.

    Raises:
        ValueError: iscale must be between 0 and 2.
        ValueError: kchem must be between 0 and 4.
    """
    reinit = False
    if cglow.jmax != 0 and cglow.nbins != 0 and cglow.iscale != iscale: # re-initialization required
        reinit = True

    if iscale < 0 or iscale > 2:
        raise ValueError('iscale must be between 0 and 2')

    if kchem < 0 or kchem > 4:
        raise ValueError('kchem must be between 0 and 4')

    cglow.iscale = iscale
    cglow.xuvfac = xuvfac
    cglow.kchem = kchem
    cglow.jlocal = 1 if jlocal else 0
    cglow.itail = 1 if itail else 0
    cglow.fmono = fmono
    cglow.emono = emono

    if reinit:
        reset_cglow()

def runglow() -> None:
    'Make sure `cglow.zz` is set before running subroutine `glow`'
    cglow.zz = cglow.z * 1e5
    fortran.glow()

def generic(time: datetime, glat: Numeric, glon: Numeric, Nbins: int, Q: Numeric = None, Echar: Numeric = None, density_perturbation: Sequence = None, *, geomag_params: dict | Iterable = None, tzaware: bool = False, jmax: int = 250, metadata = None, iscale: int=1, xuvfac: Numeric = 3, kchem: int = 4, jlocal: bool = False, itail: bool = False, fmono: float = 0, emono: float = 0) -> xarray.Dataset:
    """GLOW model with optional electron precipitation assuming Maxwellian distribution.
    Defaults to no precipitation.

    Args:
        time (datetime): Model evaluation time.
        glat (Numeric): Location latitude.
        glon (Numeric): Location longitude.
        Nbins (int): Number of energy bins (must be >= 10).
        Q (Numeric, optional): Flux of precipitating electrons. Setting to None or < 0.001 makes it equivalent to no-precipitation. Defaults to None.
        Echar (Numeric, optional): Energy of precipitating electrons. Setting to None or < 1 makes it equivalent to no-precipitation. Defaults to None.
        density_perturbation (Sequence, optional): Density perturbations of O, O2, N2, NO, N(4S), N(2D) and e-. Supply as a sequence of length 7. Values must be positive. Defaults to None (all ones, i.e. no modification).
        geomag_params (dict | Iterable, optional): Custom geomagnetic parameters 'f107a' (average f10.7 over 81 days), 'f107' (current day f10.7), 'f107p' (previous day f10.7) and 'Ap' (the global 3 hour ap index). Must be present in this order for list or tuple, and use these keys for the dictionary. Defaults to None.
        tzaware (bool, optional): If time is time zone aware. If true, `time` is recast to 'UTC' using `time.astimezone(pytz.utc)`.
        jmax (int, optional): Maximum number of altitude levels. Defaults to 250.
        metadata (dict, optional): Metadata to be added to the output dataset. Defaults to None.
        iscale (int, optional): Solar flux model. Defaults to 1 (EUVAC model).
        xuvfac (Numeric, optional): XUV enhancement factor. Defaults to 3.
        kchem (int, optional): CGLOW chemical calculations. Defaults to 4. See `cglow.kchem` for details.
        jlocal (bool, optional): Set to False for electron transport calculations, set to True for local calculations only. Defaults to False.
        itail (bool, optional): Disable/enable low-energy tail. Defaults to False.
        fmono (float, optional): Monoenergetic energy flux, erg/cm^2. Defaults to 0.
        emono (float, optional): Monoenergetic characteristic energy, keV. Defaults to 0.

    Raises:
        RuntimeError: Invalid type for geomag_params.
        KeyError: Any of the geomagnetic parameter keys absent.
        IndexError: geomag_params does not have at least 4 elements.

    Returns:
        xarray.Dataset: GLOW model output dataset.
    """
    if Nbins < 10:
        raise ValueError("Number of energy bins must be >= 10")
    
    if tzaware:
        time = time.astimezone(pytz.utc)

    idate, utsec = glowdate(time)

    if Q is None or Echar is None: # no precipitation case
        Q = 0.0001
        Echar = 0.1

    if geomag_params is None:
        ip = gi.get_indices([time - timedelta(days=1), time], 81, tzaware=tzaware)
        f107a = float(ip["f107s"][1])
        f107 = float(ip['f107'][1])
        f107p = float(ip['f107'][0])
        ap = float(ip["Ap"][1])
    elif isinstance(geomag_params, dict):
        f107a = float(geomag_params['f107a'])
        f107 = float(geomag_params['f107'])
        f107p = float(geomag_params['f107p'])
        ap = float(geomag_params['Ap'])
    elif isinstance(geomag_params, Iterable):
        f107a = float(geomag_params[0])
        f107 = float(geomag_params[1])
        f107p = float(geomag_params[2])
        ap = float(geomag_params[3])
    else:
        raise RuntimeError('Invalid type %s for geomag params %s' % (str(type(geomag_params), str(geomag_params))))
    
    ip = {}
    ip['f107a'] = (f107a)
    ip['f107'] = (f107)
    ip['f107p'] = (f107p)
    ip['ap'] = (ap)

    if density_perturbation is None:
        density_perturbation = ones(7, dtype=float32)
    else:
        density_perturbation = asarray(density_perturbation, dtype=float32)
        if density_perturbation.ndim != 1 or len(density_perturbation) != 7:
            raise ValueError('Density perturbation must be a sequence of length 7')
        if any(density_perturbation <= 0):
            raise ValueError('Density perturbation must be positive.')
    
    _glon = glon
    glon = glon % 360

    set_standard_switches(iscale, xuvfac, kchem, jlocal, itail, fmono, emono)
    init_cglow(jmax, Nbins)
    # wrap_egrid()

    (cg.idate, cg.ut, cg.glat, cg.glong, cg.f107a,
        cg.f107, cg.f107p, cg.ap, cg.ef, cg.ec) = \
        (idate, utsec, glat, glon, f107a, f107, f107p, ap, Q, Echar)
    # !
    # ! Calculate local solar time:
    # !
    stl = (cg.ut/3600. + cg.glong/15.) % 24

    (cg.z, cg.zo, cg.zo2, cg.zn2, cg.zns, cg.znd, cg.zno, cg.ztn,
    cg.zun, cg.zvn, cg.ze, cg.zti, cg.zte, cg.zxden) = \
        mzgrid(cg.jmax, cg.nex, cg.idate, cg.ut, cg.glat, cg.glong,
            stl, cg.f107a, cg.f107, cg.f107p, cg.ap, IRI90_DIR)
    
    # Apply the density perturbations
    cg.zo *= density_perturbation[0]
    cg.zo2 *= density_perturbation[1]
    cg.zn2 *= density_perturbation[2]
    cg.zno *= density_perturbation[3]
    cg.zns *= density_perturbation[4]
    cg.znd *= density_perturbation[5]
    cg.ze *= density_perturbation[6]
    
    # !
    # ! Fill altitude array, converting to cm:
    # !
    cg.zz = cg.z * 1.e5     # ! km to cm at all jmax levels
    
    # !
    # ! Call MAXT to put auroral electron flux specified by namelist input into phitop array:
    # !
    cg.phitop[:] = 0.
    if cg.ef > 0.001 and cg.ec > 1:
        cg.phitop = maxt(cg.ef, cg.ec, cg.ener, cg.edel, cg.itail,
                            cg.fmono, cg.emono)
        
    # ! call GLOW to calculate ionized and excited species, airglow emission rates,
    # ! and vertical column brightnesses:
    # !
    glow()
    # !
    # ! Call CONDUCT to calculate Pederson and Hall conductivities:
    # !
    pedcond = zeros((jmax,), cg.z.dtype)
    hallcond = pedcond.copy()
    for j in range(jmax):
        pedcond[j], hallcond[j] = conduct(cg.glat, cg.glong, cg.z[j], cg.zo[j], cg.zo2[j], cg.zn2[j],
                                        cg.zxden[2, j], cg.zxden[5, j], cg.zxden[6, j], cg.ztn[j], cg.zti[j], cg.zte[j])
        
    try:
        ds = Dataset(coords={'alt_km': ('alt_km', copy(cg.zz)*1e-5, {'standard_name': 'altitude',
                                                    'long_name': 'altitude',
                                                    'units': 'km'}),
                            'energy': ('energy', copy(cg.ener), {'long_name': 'precipitation energy',
                                                        'units': 'eV'})},
                    data_vars={'precip': ('energy', copy(cg.phitop), {'long_name': 'auroral electron flux'})
                            })
    except ValueError:
        raise ImportError('cglow has not been initialized yet!')

    density_attrs = {
        'long_name': 'number density',
        'units': 'cm^{-3}',
    }

    # Neutral Densities
    ds['O']        = Variable('alt_km', copy(cg.zo),          density_attrs)
    ds['O2']       = Variable('alt_km', copy(cg.zo2),         density_attrs)
    ds['N2']       = Variable('alt_km', copy(cg.zn2),         density_attrs)
    ds['NO']       = Variable('alt_km', copy(cg.zno),         density_attrs)
    ds['NS']       = Variable('alt_km', copy(cg.zns),         density_attrs)
    ds['ND']       = Variable('alt_km', copy(cg.znd),         density_attrs)
    ds['NeIn']     = Variable('alt_km', copy(cg.ze),          density_attrs)
    ds['NeOut']    = Variable('alt_km', copy(cg.ecalc),       density_attrs)
    ds['ionrate']  = Variable('alt_km', copy(cg.tir),         density_attrs)
    ds['O+']       = Variable('alt_km', copy(cg.zxden[2,:]),  density_attrs)
    ds['O2+']      = Variable('alt_km', copy(cg.zxden[5,:]),  density_attrs)
    ds['NO+']      = Variable('alt_km', copy(cg.zxden[6,:]),  density_attrs)
    ds['N2D']      = Variable('alt_km', copy(cg.zxden[9,:]),  density_attrs)
    ds['pederson'] = Variable('alt_km', pedcond,      {'units': 'S m^{-1}'})
    ds['hall']     = Variable('alt_km', hallcond,     {'units': 'S m^{-1}'})

    # Temperatures
    temperature_attrs = {
        'long_name': 'Temperature',
        'units': 'K',
    }

    ds['Tn'] = Variable('alt_km', copy(cg.ztn), temperature_attrs)
    ds['Tn'].attrs['comment'] = 'Netural Temperature'

    ds['Ti'] = Variable('alt_km', copy(cg.zti), temperature_attrs)
    ds['Ti'].attrs['comment'] = 'Ion Temperature'

    ds['Te'] = Variable('alt_km', copy(cg.zte), temperature_attrs)
    ds['Te'].attrs['comment'] = 'Electron Temperature'

    # Emissions
    ds['ver'] = Variable(('alt_km', 'wavelength'), copy(cg.zeta).T, {
        'long_name': 'Volume Emission Rate',
        'units': 'cm^{-3} s^{-1}',
    })
    
    wavelen = [
        '3371',
        '4278',
        '5200',
        '5577',
        '6300',
        '7320',
        '10400',
        '3644',
        '7774',
        '8446',
        '3726',
        "LBH",
        '1356',
        '1493',
        '1304'
    ]

    ds.coords['wavelength'] = wavelen

    state = [
        "O+(2P)",
        "O+(2D)",
        "O+(4S)",
        "N+",
        "N2+",
        "O2+",
        "NO+",
        "N2(A)",
        "N(2P)",
        "N(2D)",
        "O(1S)",
        "O(1D)",
    ]

    ds['production'] = Variable(('alt_km', 'state'), copy(cg.production).T, {'units': 'cm^{-3} s^{-1}'})
    ds['loss'] = Variable(('alt_km', 'state'), copy(cg.loss).T, {'units': 's^{-1}'})
    ds.coords['state'] = state

    ds['excitedDensity'] = Variable(('alt_km', 'state'), copy(cg.zxden).T, {'units': 'cm^{-3}'})
    ds['eHeat'] = Variable('alt_km', copy(cg.eheat), {'units': 'ergs cm^{-3} s^{-1}'})
    ds['Tez'] = Variable('alt_km', copy(cg.tez), {'units': 'eV cm^{-3} s^{-1}', 'comment': 'total energetic electron energy deposition', 'long_name': 'energy deposition'})
    ds['sflux'] = Variable('wave', cg.sflux,
                        {'long_name': 'solar flux',
                            'units': 'cm^{-2} s^{-1}',
                            'comment': 'scaled solar flux'})
    wave_attrs = {'long_name': 'wavelength',
                'units': 'Å'}
    ds.coords['wave1'] = ('wave', cg.wave1, wave_attrs)
    ds.coords['wave1'].attrs['comment'] = 'longwave edge of solar flux wavelength range'
    ds.coords['wave2'] = ('wave', cg.wave2, wave_attrs)
    ds.coords['wave2'].attrs['comment'] = 'shortwave edge of solar flux wavelength range'

    ds.attrs["geomag_params"] = ip
    if Q is not None and Echar is not None:
        ds.attrs['precip'] = {'Q': Q, 'Echar': Echar}
    else:
        ds.attrs['precip'] ={'Q': 0, 'Echar': 0}
    ds.attrs["time"] = time.isoformat()
    ds.attrs["glatlon"] = (glat, _glon)

    if metadata is not None:
        ds.attrs["metadata"] = metadata

    # ds.ver.loc[dict(wavelength='5577')].plot()
    # plt.show()

    return ds

def maxwellian(time: datetime, glat: Numeric, glon: Numeric, Nbins: int, Q: Numeric, Echar: Numeric, density_perturbation: Sequence = None, *, geomag_params: dict | Iterable = None, tzaware: bool = False) -> xarray.Dataset:
    """GLOW model with electron precipitation assuming Maxwellian distribution.

    Args:
        time (datetime): Model evaluation time.
        glat (Numeric): Location latitude.
        glon (Numeric): Location longitude.
        Nbins (int): Number of energy bins (must be >= 10).
        Q (Numeric): Flux of precipitating electrons.
        Echar (Numeric): Energy of precipitating electrons.
        density_perturbation (Sequence, optional): Density perturbations of O, O2, N2, NO, N(4S), N(2D) and e-. Supply as a sequence of length 7. Values must be positive. Defaults to None (all ones, i.e. no modification).
        geomag_params (dict | Iterable, optional): Custom geomagnetic parameters 'f107a' (average f10.7 over 81 days), 'f107' (current day f10.7), 'f107p' (previous day f10.7) and 'Ap' (the global 3 hour ap index). Must be present in this order for list or tuple, and use these keys for the dictionary. Defaults to None.
        tzaware (bool, optional): If time is time zone aware. If true, `time` is recast to 'UTC' using `time.astimezone(pytz.utc)`.

    Raises:
        RuntimeError: Invalid type for geomag_params.
        KeyError: Any of the geomagnetic parameter keys absent.
        IndexError: geomag_params does not have at least 4 elements.

    Returns:
        xarray.Dataset: GLOW model output dataset.
    """
    return generic(time, glat, glon, Nbins, Q, Echar, density_perturbation, geomag_params=geomag_params, tzaware=tzaware)

def no_precipitation(time: datetime, glat: Numeric, glon: Numeric, Nbins: int, density_perturbation: Sequence = None, *, geomag_params: dict | Iterable = None, tzaware: bool = False) -> xarray.Dataset:
    """GLOW model without electron precipitation.

    Args:
        time (datetime): Model evaluation time.
        glat (Numeric): Location latitude.
        glon (Numeric): Location longitude.
        Nbins (int): Number of energy bins (must be >= 10).
        density_perturbation (Sequence, optional): Density perturbations of O, O2, N2, NO, N(4S), N(2D) and e-. Supply as a sequence of length 7. Values must be positive. Defaults to None (all ones, i.e. no modification).
        geomag_params (dict | Iterable, optional): Custom geomagnetic parameters 'f107a' (average f10.7 over 81 days), 'f107' (current day f10.7), 'f107p' (previous day f10.7) and 'Ap' (the global 3 hour ap index). Must be present in this order for list or tuple, and use these keys for the dictionary. Defaults to None.
        tzaware (bool, optional): If time is time zone aware. If true, `time` is recast to 'UTC' using `time.astimezone(pytz.utc)`.

    Raises:
        RuntimeError: Invalid type for geomag_params.
        KeyError: Any of the geomagnetic parameter keys absent.
        IndexError: geomag_params does not have at least 4 elements.

    Returns:
        xarray.Dataset: GLOW model output dataset.
    """
    return generic(time, glat, glon, Nbins, Q=None, Echar=None, density_perturbation=density_perturbation, geomag_params=geomag_params, tzaware=tzaware)


def glowdate(t: datetime) -> tuple[str, str]:
    idate = int(f'{t.year}{t.strftime("%j")}')
    utsec = (t.hour * 3600 + t.minute * 60 + t.second)

    return idate, utsec
