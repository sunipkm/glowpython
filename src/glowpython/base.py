from __future__ import annotations
from typing import Iterable, SupportsFloat as Numeric

from . import fortran
from .fortran import cglow, wrap_egrid, cglow as cg, mzgrid, maxt, glow, conduct

import xarray
from xarray import Dataset, Variable
from numpy import zeros, copy
from datetime import datetime, timedelta
import geomagdata as gi
import pytz
from os import path

import matplotlib.pyplot as plt

__CGLOW_INIT__ = False

IRI90_DIR = path.join(path.dirname(__file__), 'data', 'iri90/')

def init_cglow(jmax: int) -> None:
    'Initialize cglow variables.'
    global __CGLOW_INIT__
    if __CGLOW_INIT__:
        # raise RuntimeError("cglow_init has already been called!")
        release_cglow()
        init_cglow(jmax)
    else:
        data_dir = path.join(path.dirname(__file__), 'data', '')
        cglow.data_dir.put(0, '{: <1024s}'.format(data_dir))
        cglow.jmax = jmax
        cglow.cglow_init()
        __CGLOW_INIT__ = True

def release_cglow():
    global __CGLOW_INIT__
    # Deallocate all allocatable cglow arrays
    fortran.cglow_release()

    __CGLOW_INIT__ = False

def reset_cglow(jmax=None) -> None:
    if jmax is None:
        jmax = cglow.jmax
    release_cglow()
    init_cglow(jmax)

def set_egrid(ds: Dataset) -> None:
    'Initialize E and dE in both fortran.cglow and ds.'
    ds.E.data, ds.dE.data = fortran.wrap_egrid()

def set_standard_switches() -> None:
    'Set `cglow` switches to standard values.'
    cglow.iscale = 1
    cglow.xuvfac = 3
    cglow.kchem = 4
    cglow.jlocal = 0
    cglow.itail = 0
    cglow.fmono = 0
    cglow.emono = 0

def runglow() -> None:
    'Make sure `cglow.zz` is set before running subroutine `glow`'
    cglow.zz = cglow.z * 1e5
    fortran.glow()

def exec_glow(idate, utsec, glat, glon, Q, Echar, f107a, f107, f107p, ap, jmax: int = 250) -> Dataset:
    """Run GLOW with the given parameters and return the result as an xarray.Dataset.
    This replicates the behavior of the `glowbasic.f90` driver program."""
    set_standard_switches()
    init_cglow(jmax)
    wrap_egrid()

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
                             'energy': ('energy', copy(cg.ener), {'long_name': 'energy',
                                                        'units': 'eV'}),
                             'precip': ('energy', copy(cg.phitop), {'long_name': 'auroral electron flux'})
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
    ds['N2D']      = Variable('alt_km', copy(cg.zxden[9,:]), density_attrs)
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

    # ds['production'] = Variable(('alt_km', 'state'), copy(cg.production).T, {'units': 'cm^{-3} s^{-1}'})
    # ds['loss'] = Variable(('alt_km', 'state'), copy(cg.loss).T, {'units': 's^{-1}'})
    # ds.coords['state'] = state

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

    reset_cglow()
    return ds

def generic(time: datetime, glat: Numeric, glon: Numeric, Nbins: int, Q: Numeric = None, Echar: Numeric = None, *, geomag_params: dict | Iterable = None, tzaware: bool = False, jmax: int = 250) -> xarray.Dataset:
    """GLOW model with optional electron precipitation assuming Maxwellian distribution.
    Defaults to no precipitation.

    Args:
        time (datetime): Model evaluation time.
        glat (Numeric): Location latitude.
        glon (Numeric): Location longitude.
        Q (Numeric, optional): Flux of precipitating electrons. Setting to None or < 0.001 makes it equivalent to no-precipitation. Defaults to None.
        Echar (Numeric, optional): Energy of precipitating electrons. Setting to None or < 1 makes it equivalent to no-precipitation. Defaults to None.
        Nbins (int): Number of energy bins (unused, set to 250).
        geomag_params (dict | Iterable, optional): Custom geomagnetic parameters 'f107a' (average f10.7 over 81 days), 'f107' (current day f10.7), 'f107p' (previous day f10.7) and 'Ap' (the global 3 hour ap index). Must be present in this order for list or tuple, and use these keys for the dictionary. Defaults to None.
        tzaware (bool, optional): If time is time zone aware. If true, `time` is recast to 'UTC' using `time.astimezone(pytz.utc)`.

    Raises:
        RuntimeError: Invalid type for geomag_params.
        KeyError: Any of the geomagnetic parameter keys absent.
        IndexError: geomag_params does not have at least 4 elements.

    Returns:
        xarray.Dataset: GLOW model output dataset.
    """
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

    
    ds = exec_glow(idate, utsec, glat, glon, Q, Echar, f107a, f107, f107p, ap, jmax)

    ds.attrs["geomag_params"] = ip
    if Q is not None and Echar is not None:
        ds.attrs['precip'] = {'Q': Q, 'Echar': Echar}
    else:
        ds.attrs['precip'] ={'Q': 0, 'Echar': 0}
    ds.attrs["time"] = time.isoformat()
    ds.attrs["glatlon"] = (glat, glon)

    return ds

def maxwellian(time: datetime, glat: Numeric, glon: Numeric, Nbins: int, Q: Numeric, Echar: Numeric, *, geomag_params: dict | Iterable = None, tzaware: bool = False) -> xarray.Dataset:
    """GLOW model with electron precipitation assuming Maxwellian distribution.

    Args:
        time (datetime): Model evaluation time.
        glat (Numeric): Location latitude.
        glon (Numeric): Location longitude.
        Q (Numeric): Flux of precipitating electrons.
        Echar (Numeric): Energy of precipitating electrons.
        Nbins (int): Number of energy bins (unused, set to 250).
        geomag_params (dict | Iterable, optional): Custom geomagnetic parameters 'f107a' (average f10.7 over 81 days), 'f107' (current day f10.7), 'f107p' (previous day f10.7) and 'Ap' (the global 3 hour ap index). Must be present in this order for list or tuple, and use these keys for the dictionary. Defaults to None.
        tzaware (bool, optional): If time is time zone aware. If true, `time` is recast to 'UTC' using `time.astimezone(pytz.utc)`.

    Raises:
        RuntimeError: Invalid type for geomag_params.
        KeyError: Any of the geomagnetic parameter keys absent.
        IndexError: geomag_params does not have at least 4 elements.

    Returns:
        xarray.Dataset: GLOW model output dataset.
    """
    return generic(time, glat, glon, Nbins, Q, Echar, geomag_params=geomag_params, tzaware=tzaware)

def no_precipitation(time: datetime, glat: Numeric, glon: Numeric, Nbins: int, *, geomag_params: dict | Iterable = None, tzaware: bool = False) -> xarray.Dataset:
    """GLOW model without electron precipitation.

    Args:
        time (datetime): Model evaluation time.
        glat (Numeric): Location latitude.
        glon (Numeric): Location longitude.
        Nbins (int): Number of energy bins (unused, set to 250).
        geomag_params (dict | Iterable, optional): Custom geomagnetic parameters 'f107a' (average f10.7 over 81 days), 'f107' (current day f10.7), 'f107p' (previous day f10.7) and 'Ap' (the global 3 hour ap index). Must be present in this order for list or tuple, and use these keys for the dictionary. Defaults to None.
        tzaware (bool, optional): If time is time zone aware. If true, `time` is recast to 'UTC' using `time.astimezone(pytz.utc)`.

    Raises:
        RuntimeError: Invalid type for geomag_params.
        KeyError: Any of the geomagnetic parameter keys absent.
        IndexError: geomag_params does not have at least 4 elements.

    Returns:
        xarray.Dataset: GLOW model output dataset.
    """
    return generic(time, glat, glon, Nbins, Q=None, Echar=None, geomag_params=geomag_params, tzaware=tzaware)


def glowdate(t: datetime) -> tuple[str, str]:
    idate = f'{t.year}{t.strftime("%j")}'
    utsec = str(t.hour * 3600 + t.minute * 60 + t.second)

    return idate, utsec
