from __future__ import annotations
from os import path
import pytz
import geomagdata as gi
from datetime import datetime, timedelta
from numpy import allclose, array, asarray, float32, full, isnan, ndarray, ones, trapz, zeros, copy
import numpy as np
from xarray import Dataset, Variable
import xarray
from .utils import Singleton, alt_grid, glowdate, geocent_to_geodet, interpolate_nan
from .glowfort import cglow, cglow as cg, mzgrid, maxt, glow, conduct_iface  # type: ignore
from typing import Any, Iterable, Optional, Sequence, SupportsFloat as Numeric, Tuple, Literal
import atexit
import warnings

from multiprocessing import current_process

HmFSource = Literal[
    'CCIR',
    'URSI',
]

# Suppress FutureWarnings
warnings.simplefilter(action='once', category=FutureWarning)

DATA_DIR = path.join(path.dirname(__file__), 'data', '')  # GLOW model ancillary data directory
IRI90_DIR = path.join(DATA_DIR, 'iri90/')                 # IRI-90 model data directory

cglow.jmax = 0   # initialize this to zero
cglow.nbins = 0  # initialize this to zero


def init_cglow() -> None:
    """## Initialize FORTRAN `cglow` module.
    This function must be called ONCE and ONLY ONCE before using the GLOW model.
    """
    cglow.data_dir.put(0, '{: <1024s}'.format(DATA_DIR))
    cglow.cglow_static_init()
    cglow.jmax = 0
    cglow.nbins = 0


@atexit.register
def release_cglow():
    '## Deallocate all allocatable `cglow` arrays.'
    cglow.cglow_dynamic_dealloc()
    cglow.cglow_static_deinit()
    cglow.jmax = 0
    cglow.nbins = 0


def reset_cglow(jmax: int = None, nbins: int = None) -> None:
    """## Reset `cglow` module and reallocate all arrays.

    ### Args:
        - `jmax (int, optional)`: Number of altitude bins. Defaults to None, uses previous value from CGLOW.
        - `nbins (int, optional)`: Number of energy bins. Defaults to None, uses previous value from CGLOW.

    ### Raises:
        - `ValueError`: jmax OR nbins is zero, which is an impossible state.
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
    realloc = False
    if cglow.jmax != jmax or cglow.nbins != nbins:
        realloc = True
    # set the values
    cglow.jmax = jmax
    cglow.nbins = nbins
    # perform the allocation
    if realloc:
        cglow.cglow_dynamic_dealloc()  # deallocate if allocated
        cglow.cglow_dynamic_alloc()  # allocate
        cglow.cglow_dynamic_zero()  # zero out the arrays
        cglow.egrid_init()  # initialize energy grid


class GlowModel(Singleton):
    """## GLOW Model Singleton Class.
    This class is a singleton and should be used to evaluate the GLOW model.
    The class is a singleton because the FORTRAN module `cglow` is not thread-safe.
    The class provides methods to initialize the model, set up the model for evaluation,
    evaluate the atmosphere, optionally model the precipitating electrons 
    (in case of aurora), run the radiative transfer model, and retrieve the result.

    This class is compatible with the `multiprocessing` module, and can be used in a
    multi-process environment.

    ### Example (single evaluation):
    >>> from glowpython import GlowModel
    >>> from datetime import datetime
    >>> mod = GlowModel()
    >>> mod.setup(datetime(2020, 1, 1, 0, 0), 65, 0) # setup the model
    >>> mod.precipitation() # set the precipitation parameters (optional)
    >>> mod.atmosphere() # evaluate the atmosphere
    >>> mod.radtrans() # run the radiative transfer model
    >>> ds = mod.result() # get the result

    ### Example (multi-process evaluation):
    >>> from multiprocessing import Pool
    >>> def eval_glow(time, lat, lon):
    >>>     mod = GlowModel()
    >>>     mod.setup(time, lat, lon)
    >>>     res = mod()
    >>>     return res
    >>> times = [datetime(2020, 1, 1)]*4
    >>> lat = [0, 10, 20, 30]
    >>> lon = [0, 10, 20, 30]
    >>> with Pool(4) as pool:
    >>>     results = pool.starmap(eval_glow, zip(times, lat, lon))
    """

    def _init(self) -> None:
        """## Create an instance of the GLOW Model Singleton Class.
        On creation, the model is reset and initialized with standard
        values for the switches. This method is private and should
        not be called directly. Use the `GlowModel` class to get an instance.
        The creation of the first instance triggers this function.
        """
        init_cglow()  # initialize the cglow module
        self._reset = True
        self._initd = False
        self._ready = False
        self._atm = False
        self._evaluated = False
        self._z = zeros(0, dtype=float32, order='F')
        self.initialize()
        return self

    def initialize(self, alt_km: int | Iterable = 250, nbins: int = 100, iscale: int = 1, *, wave1: Sequence = None, wave2: Sequence = None, rflux: Sequence = None) -> None:
        """## Initialize Inputs
        Initializes the altitude grid, energy bins and solar flux model.

       ### Args:
            - `alt_km (int | Iterable, optional)`: Altitude grid length/altitude grid. Defaults to 250.
                - `None`: Uses the previous value of `jmax` and altitude grid.
                - `int`: Length of altitude grid. The altitude grid is then evaluated using the `alt_grid` function.
                - `Iterable`: Custom altitude grid. Must be a 1-D array, and between 60 and 1000 km.
            - `nbins (int, optional)`: Number of energy bins. Defaults to 100.
            - `iscale (int, optional)`: Solar flux model. Defaults to 1 (EUVAC model). 
                - `0`: Hinteregger Model
                - `1`: EUVAC Model
                - `2`: User-supplied solar flux model. In this case, `wave1`, `wave2` and `rflux` must be supplied.
            - `wave1 (Sequence, optional)`: Starting points of the solar flux wavelength bins (`length=123`). Defaults to None. Required if `iscale` is 2.
            - `wave2 (Sequence, optional)`: Ends of the solar flux wavelength bins. Defaults to None. Required if `iscale` is 2.
            - `rflux (Sequence, optional)`: Solar flux values. Defaults to None. Required if `iscale` is 2.
        ### Raises:
            - `RuntimeError`: Reset the model before initializing.
            - `ValueError`: `alt_km` must be specified if not already initialized.
            - `ValueError`: Altitude grid must be between 60 and 1000 km.
            - `ValueError`: Altitude grid must be a 1-D array.
            - `ValueError`: `nbins` must be specified if not already initialized.
            - `ValueError`: Number of energy bins must be >= 10.
            - `ValueError`: `iscale` must be between `0` and `2`.
            - `ValueError`: `wave1`, `wave2` and `rflux` must be supplied if `iscale` is 2.
            - `ValueError`: `wave1`, `wave2` and `rflux` must be 1-D arrays.
            - `ValueError`: `wave1`, `wave2` and `rflux` must be of the same length.
            - `ValueError`: `wave1`, `wave2` and `rflux` must be of length 123.
            - `ValueError`: `wave1` and `wave2` must be sorted.
            - `ValueError`: `wave1` and `wave2` must be sorted in the same order.
        """
        if not self._reset:
            raise RuntimeError('Reset the model before initializing')
        # deal with altitude grid
        if alt_km is None:
            if cglow.jmax == 0:
                raise ValueError('alt_km must be specified if not already initialized')
            jmax = cglow.jmax
        elif isinstance(alt_km, int):
            jmax = alt_km
            if jmax != len(self._z):
                self._z = alt_grid(jmax, 60., 0.5, 4.)
            else:
                tmp = alt_grid(jmax, 60., 0.5, 4.)
                if not allclose(self._z, tmp):
                    self._z = tmp
        elif isinstance(alt_km, Iterable):
            alt_km = array(alt_km, dtype=float32, order='F')  # always make a copy
            if any(alt_km < 60) or any(alt_km > 1000):
                raise ValueError('Altitude grid must be between 60 and 1000 km')
            if alt_km.ndim != 1:
                raise ValueError('alt_km must be a 1-D array')
            jmax = len(alt_km)
            self._z = alt_km
        # deal with energy grid
        if nbins is None and cglow.nbins == 0:
            raise ValueError('nbins must be specified if not already initialized')
        if nbins is not None and nbins < 10:
            raise ValueError('Number of energy bins must be >= 10')
        # do realloc
        reset_cglow(jmax, nbins)
        cg.zz = self._z * 1e5  # convert to cm

        if not 0 <= iscale <= 2:
            raise ValueError('iscale must be between 0 and 2')

        if iscale == 2:
            if wave1 is None or wave2 is None or rflux is None:
                raise ValueError('wave1, wave2 and rflux must be supplied if iscale is 2')
            wave1: ndarray = array(wave1, dtype=float32, order='F')
            wave2: ndarray = array(wave2, dtype=float32, order='F')
            rflux: ndarray = array(rflux, dtype=float32, order='F')
            if wave1.ndim != 1 or wave2.ndim != 1 or rflux.ndim != 1:
                raise ValueError('wave1, wave2 and rflux must be 1-D arrays')
            if len(wave1) != len(wave2) or len(wave2) != len(rflux):
                raise ValueError('wave1, wave2 and rflux must be of the same length')
            if len(wave1) != cg.lmax:
                raise ValueError('wave1, wave2 and rflux must be of length %d' % cg.lmax)

            def ascending(a): return int(all(a[:-1] <= a[1:]))*1
            def descending(a): return int(all(a[:-1] >= a[1:]))*(-1)
            def is_sorted(a): return all(a[:-1] <= a[1:]) or all(a[:-1] >= a[1:])
            if not is_sorted(wave1) or not is_sorted(wave2):
                raise ValueError('wave1 and wave2 must be sorted')
            if ascending(wave1) + descending(wave2) == 0:
                raise ValueError('wave1 and wave2 must be sorted in the same order')
            if ascending(wave1):  # in ascending order, reverse
                wave1 = wave1[::-1]
                wave2 = wave2[::-1]
                rflux = rflux[::-1]
            cg.iscale = iscale
            cg.wave1 = wave1
            cg.wave2 = wave2
            cg.sflux = rflux
        else:
            cg.iscale = iscale
            cg.sflux_init()

        self._initd = True
        self._ready = False
        self._atm = False
        self._evaluated = False

    def reset(self, clear: bool = False) -> None:
        """## Reset the GLOW model.
        Resets the GLOW model to its initial state and clears previous result.
        Call this method after retrieving the result to zero
        out the arrays and re-initialize the energy grid.

        ### Args:
            - `clear (bool, optional)`: Clear internal FORTRAN arrays to zero, and recalculate energy grid. Defaults to False.
        """
        if clear:
            cglow.cglow_dynamic_zero()  # zero out the arrays
            cglow.egrid_init()  # re-initialize energy grid

        self._ds = Dataset()  # empty dataset
        self._ready = False
        self._atm = False
        self._evaluated = False

    def setup(self, time: datetime,
              glat: Numeric,
              glon: Numeric,
              *,
              geomag_params: dict | Iterable = None,
              tzaware: bool = False,) -> None:
        """## Setup the GLOW model for evaluation.
        Initializes the `cglow` module if not initialized, and sets the input parameters for the GLOW model. Additionally, calculates the auroral electron flux using the `glowfort.maxt` subroutine if needed.

        Calls :method:`GlowModel.initialize` to initialize the altitude grid, energy bins and solar flux model.

        ### Args:
            - `time (datetime)`: Model evaluation time.
            - `glat (Numeric)`: Location latitude (degrees).
            - `glon (Numeric)`: Location longitude (degrees).
            - `geomag_params (dict | Iterable, optional)`: Custom geomagnetic parameters.
                - `f107a` (running average F10.7 over 81 days), 
                - `f107` (current day F10.7), 
                - `f107p` (previous day F10.7), and
                - `Ap` (the global 3 hour $a_p$ index). 

                Must be present in this order for list or tuple, and use these keys for the dictionary. Defaults to None.
            - `tzaware (bool, optional)`: If time is time zone aware. If true, `time` is recast to 'UTC' using `time.astimezone(pytz.utc)`. Defaults to False.

        ### Raises:
        - `RuntimeError`: Initialize the model before setting up.
            - `RuntimeError`: Invalid type for geomagnetic parameters.
        """
        if not self._initd:
            raise RuntimeError('Initialize the model before setting up')

        if tzaware:
            time = time.astimezone(pytz.utc)

        self._time = time

        idate, utsec = glowdate(time)

        if geomag_params is None:
            ip = gi.get_indices([time - timedelta(days=1), time], 81, tzaware=tzaware)
            f107a = float(ip["f107s"].iloc[1])
            f107 = float(ip['f107'].iloc[1])
            f107p = float(ip['f107'].iloc[0])
            ap = float(ip["Ap"].iloc[1])
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

        _glon = glon  # unmodified for dataset
        glon = glon % 360

        (cg.idate, cg.ut, cg.glat, cg.glong, cg.f107a,
         cg.f107, cg.f107p, cg.ap) = \
            (idate, utsec, glat, glon, f107a, f107, f107p, ap)

        self._stl = (cg.ut/3600. + cg.glong/15.) % 24

        cg.phitop[:] = 0.

        ds = Dataset(coords={'alt_km': ('alt_km', self._z, {'standard_name': 'altitude',
                                                            'long_name': 'altitude',
                                                            'units': 'km'}),
                             'energy': ('energy', cg.ener, {'long_name': 'precipitation energy',
                                                            'units': 'eV'})},
                     data_vars={'precip': ('energy', cg.phitop,
                                           {
                                               'long_name': 'auroral electron flux',
                                               'units': 'cm^{-2} s^{-1} eV^{-1}'}
                                           )
                                }
                     )

        ds.coords['sflux'] = Variable('wave', cg.sflux,
                                      {'long_name': 'solar flux',
                                       'units': 'cm^{-2} s^{-1}',
                                       'comment': 'scaled solar flux'})
        wave_attrs = {'long_name': 'wavelength',
                      'units': 'Å'}
        ds.coords['wave'] = ('wave', (cg.wave1 + cg.wave2)*0.5, wave_attrs)
        ds.coords['wave'].attrs['description'] = 'Center of solar flux bins'
        ds.coords['dwave'] = ('wave', cg.wave2 - cg.wave1, wave_attrs)
        ds.coords['dwave'].attrs['description'] = 'Width of solar flux bins'

        ds.attrs["time"] = time.isoformat()
        ds.attrs["glatlon"] = (glat, _glon)

        for k, v in ip.items():
            ds.attrs[k] = v

        cglow.ef = 0
        cglow.ec = 0
        cglow.itail = 0
        cglow.fmono = 0
        cglow.emono = 0

        ds.attrs['Q'] = cg.ef
        ds.attrs['Echar'] = cg.ec
        ds.attrs['itail'] = cglow.itail
        ds.attrs['fmono'] = cglow.fmono
        ds.attrs['emono'] = cglow.emono

        ds.attrs['iscale'] = cglow.iscale

        self._ds = ds

        self._initd = True
        self._ready = True
        self._atm = False
        self._evaluated = False
        return

    def precipitation(self, Q: Numeric = None, Echar: Numeric = None, *,
                      itail: bool = False,
                      fmono: Numeric = 0,
                      emono: Numeric = 0):
        """## Calculate electron precipitation.

        ### Args:
            - `Q (Numeric, optional)`: Flux of precipitating electrons (erg/cm^2/s). Setting to None or < 0.001 makes it equivalent to no-precipitation. Defaults to None.
            - `Echar (Numeric, optional)`: Energy of precipitating electrons. Setting to None or < 1 makes it equivalent to no-precipitation. Defaults to None.
            - `itail (bool, optional)`: Disable/enable low-energy tail. Defaults to False.
            - `fmono (float, optional)`: Monoenergetic energy flux, erg/cm^2. Defaults to 0.
            - `emono (float, optional)`: Monoenergetic characteristic energy, keV. Defaults to 0.
        """
        if not self._initd:
            raise RuntimeError('Initialize the model before setting up')
        ds = self._ds

        if Q is not None and Echar is not None:
            ds.attrs['Q'] = Q
            ds.attrs['Echar'] = Echar
        if Q is None or Echar is None:  # no precipitation case
            Q = 0.0001
            Echar = 0.1

        cglow.ef = Q
        cglow.ec = Echar
        cglow.itail = 1 if itail else 0
        cglow.fmono = fmono
        cglow.emono = emono

        # ! Call MAXT to put auroral electron flux specified by namelist input into phitop array:
        cg.phitop = maxt(cg.ef, cg.ec, cg.ener, cg.edel, cg.itail,
                         cg.fmono, cg.emono)

        ds.attrs['itail'] = cglow.itail
        ds.attrs['fmono'] = cglow.fmono
        ds.attrs['emono'] = cglow.emono
        return

    def atmosphere(self, density_perturbation: Sequence = None, tec: Numeric | Dataset = None, hmf2: Optional[Numeric] = None, nmf2: Optional[Numeric] = None, f2_peak: HmFSource = 'URSI') -> None:
        """## Evaluate the atmosphere
        Uses the MSISE-00 and IRI-90 models to calculate the neutral and ion densities, and temperatures.

        The following subroutines are called in order:
         - `glowfort.gtd7`: Calculate the neutral atmosphere parameters.
         - `glowfort.snoemint`: Obtain the NO profile using the Nitric Oxide Empirical Model (NOEM).
         - `glowfort.iri90`: Calculate the ionosphere using the IRI-90 model.

        ### Args:
            - `density_perturbation (Sequence, optional)`: Density perturbations of O, O2, N2, NO, N(4S), N(2D) and e-. 
               Supply as a sequence of length 7. Values must be positive. Defaults to None (all ones, i.e. no modification).
            - `tec (Numeric | Dataset, optional)`: Total Electron Content (TEC) in TECU. 
              Defaults to None. Used to scale IRI-90 derived electron density.
              If `Dataset`, must contain the following coordinates and variable:
                  - `timestamps` (`datetime64[ns]`): Timestamp of the TEC map,
                  - `gdlat` (`float64`): Geodetic latitude in degrees,
                  - `glon` (`float64`): Geocentric longitude in degrees,
                  - `tec` (`float64`): TEC in TECU.

              The nearest spatio-temporal TEC value is used to scale the electron density. This 
              dataset format is compatible with the GPS TEC maps from the Madrigal database.
            - `hmf2 (Numeric, optional)`: Height of the F2 peak (km). Defaults to None.
            - `nmf2 (Numeric, optional)`: Density of the F2 peak (m^-3). Defaults to None.
            - `f2_peak (IRISRC, optional)`: F2 peak model. Defaults to 'URSI'.

        ### Raises:
            - `RuntimeError`: GLOW model not ready for evaluation. Run `setup` first.
            - `ValueError`: Density perturbation must be a sequence of length 7.
            - `ValueError`: Density perturbation must be positive.
            - `ValueError`: TEC dataset does not contain the timestamp.
            - `ValueError`: Supplied TEC must be positive.
            - `ValueError`: Supplied TEC must be in TECU. 1 TECU = 10^16 electrons/m^2.
        """
        if not self._ready:
            raise RuntimeError('GLOW model not ready for evaluation. Run `setup` first.')
        # Used constants
        time = self._time
        glat, glon = self._ds.attrs['glatlon']

        # Sanitize the inputs
        if density_perturbation is None:
            density_perturbation = ones(7, dtype=float32)
        else:
            density_perturbation = asarray(density_perturbation, dtype=float32)
            if density_perturbation.ndim != 1 or len(density_perturbation) != 7:
                raise ValueError('Density perturbation must be a sequence of length 7')
            if any(density_perturbation <= 0):
                raise ValueError('Density perturbation must be positive.')

        self._denpert = density_perturbation

        if tec is not None:
            if isinstance(tec, Dataset):
                tmin = float(tec['timestamps'].min())
                tmax = float(tec['timestamps'].max())
                if not (tmin <= time.timestamp() <= tmax):
                    raise ValueError('TEC dataset does not contain the time %s' % time)
                gdlat = geocent_to_geodet(glat)
                tec = float(tec.interp(coords={'timestamps': time.timestamp(), 'gdlat': gdlat, 'glon': glon}).tec)
            if isnan(tec):
                tec = 1
                warnings.warn(RuntimeWarning(f'<{glat:.2f}, {glon:.2f}> TEC is NaN, using 1 TECU instead'))
            if tec <= 0:
                raise ValueError('TEC must be positive.')
            if tec > 200:
                raise ValueError('TEC must be in TECU. 1 TECU = 10^16 electrons/m^2')

        # ! Calculate local solar time:
        stl = (cg.ut/3600. + cg.glong/15.) % 24

        jf = full(12, 1, dtype=np.int32, order='F')
        oarr = zeros(30, dtype=float32, order='F')
        if f2_peak == 'URSI':
            jf[4] = 0
        if hmf2 is not None:
            jf[8] = 0
            oarr[1] = hmf2
        if nmf2 is not None:
            jf[7] = 0
            oarr[0] = nmf2

        # ! Call MZGRID to calculate neutral atmosphere parameters:
        (cg.zo, cg.zo2, cg.zn2, cg.zns, cg.znd, cg.zno, cg.ztn,
         cg.zun, cg.zvn, cg.ze, cg.zti, cg.zte, cg.zxden) = \
            mzgrid(cg.nex, cg.idate, cg.ut, cg.glat, cg.glong,
                   stl, cg.f107a, cg.f107, cg.f107p, cg.ap, self._z,
                   jf, oarr, IRI90_DIR)

        # Apply the density perturbations
        cg.zo *= density_perturbation[0]
        cg.zo2 *= density_perturbation[1]
        cg.zn2 *= density_perturbation[2]
        cg.zno *= density_perturbation[3]
        cg.zns *= density_perturbation[4]
        cg.znd *= density_perturbation[5]
        cg.ze *= density_perturbation[6]

        tecscale = 1

        # Scale the electron density to match the TEC
        if tec is not None:
            tec *= 1e12  # convert to num / cm^-2
            ne = interpolate_nan(cg.ze, inplace=False)  # Filter out NaNs
            iritec = trapz(ne, cg.zz)  # integrate the electron density
            tecscale = tec / iritec  # scale factor
            cg.ze = cg.ze * tecscale  # scale the electron density to match the TEC
            cg.zxden[:, :] = cg.zxden[:, :] * tecscale  # scale the ion densities to match the TEC

        ds = self._ds
        density_attrs = {
            'long_name': 'number density',
            'units': 'cm^{-3}',
            'source': 'MSISE-00',
            'description': 'Neutral densities from the MSISE-00 model'
        }

        # Neutral Densities
        ds['O'] = Variable('alt_km',       cg.zo,           density_attrs)
        ds['O2'] = Variable('alt_km',      cg.zo2,          density_attrs)
        ds['N2'] = Variable('alt_km',      cg.zn2,          density_attrs)
        ds['NO'] = Variable('alt_km',      cg.zno,          density_attrs)
        ds['NS'] = Variable('alt_km',      cg.zns,          density_attrs)
        ds['ND'] = Variable('alt_km',      cg.znd,          density_attrs)

        density_attrs['source'] = 'IRI-90'
        density_attrs['description'] = 'Ion density from the IRI-90 model'
        ds['NeIn'] = Variable('alt_km',    cg.ze,           density_attrs)
        ds['NeIn'].attrs['description'] = 'Electron density from the IRI-90 model'
        ds['O+'] = Variable('alt_km',      cg.zxden[2, :],  density_attrs)
        ds['O+'].attrs['description'] = 'O+(4S) ion density from the IRI-90 model'
        ds['O2+'] = Variable('alt_km',     cg.zxden[5, :],  density_attrs)
        ds['NO+'] = Variable('alt_km',     cg.zxden[6, :],  density_attrs)

        # Temperatures
        temperature_attrs = {
            'long_name': 'Temperature',
            'units': 'K',
            'description': 'Species temperatures in Kelvin',
        }

        ds['Tn'] = Variable('alt_km', cg.ztn, temperature_attrs)
        ds['Tn'].attrs['comment'] = 'Netural Temperature'
        ds['Tn'].attrs['source'] = 'MSISE-00'

        ds['Ti'] = Variable('alt_km', cg.zti, temperature_attrs)
        ds['Ti'].attrs['comment'] = 'Ion Temperature'
        ds['Ti'].attrs['source'] = 'MSISE-00'

        ds['Te'] = Variable('alt_km', cg.zte, temperature_attrs)
        ds['Te'].attrs['comment'] = 'Electron Temperature'
        ds['Te'].attrs['source'] = 'MSISE-00'

        ds.coords['tecscale'] = ('tecscale', [tecscale], {
            'long_name': 'TEC scaling factor',
            'description': 'Scaling factor applied to the electron density to match the TEC. 1.0 means no scaling.',
        })

        ds.coords['denperturb'] = ('denperturb', density_perturbation, {
            'long_name': 'Density Perturbation',
            'description': 'Density perturbation applied to the atmospheric densities. 1.0 means no perturbation.',
        })

        ds.attrs['nmf2'] = (oarr[0], {'long_name': 'F2 peak density', 'units': 'm^-3'})
        ds.attrs['hmf2'] = (oarr[1], {'long_name': 'F2 peak height', 'units': 'km'})
        ds.attrs['nmf1'] = (oarr[2], {'long_name': 'F1 peak density', 'units': 'm^-3'})
        ds.attrs['hmf1'] = (oarr[3], {'long_name': 'F1 peak height', 'units': 'km'})
        ds.attrs['nme'] = (oarr[4], {'long_name': 'E peak density', 'units': 'm^-3'})
        ds.attrs['hme'] = (oarr[5], {'long_name': 'E peak height', 'units': 'km'})
        ds.attrs['nmd'] = (oarr[6], {'long_name': 'D peak density', 'units': 'm^-3'})
        ds.attrs['hmd'] = (oarr[7], {'long_name': 'D peak height', 'units': 'km'})

        self._atm = True
        return

    def radtrans(self, xuvfac: int = 3, jlocal: bool = False, kchem: int = 4, *,
                 ion_n: Iterable = None,
                 ion_n2: Iterable = None,
                 ion_o: Iterable = None,
                 ion_o2: Iterable = None,
                 ion_no: Iterable = None):
        """## Run the radiative transfer model.
        Executes the following subroutines in order:
         - `glowfort.fieldm`: Calculate the magnetic dip angle and SZA (radians).
         - `glowfort.ssflux`: Scale the solar flux using scaling mode and F10.7 values.
         - `glowfort.rcolum`: Calculate slant path column densities of major species in
            the direction of the sun.
         - `glowfort.ephoto`: Calculate the photoelectron production spectrum and 
            photoionization rates as a function of altitude, unless all altitudes
            are dark.
         - `glowfort.qback`: Add background ionization to photoionization.
         - `glowfort.etrans`: Calculate photoelectron and auroral electron transport 
           and electron impact excitation rates, unless there are no energetic
           electrons, in which case zero arrays.
         - `glowfort.gchem`: Calculate the densities of excited and ionized
           constituents, airglow emission rates, and vertical column brightness.
         - `glowfort.bands`: Calculate the LBH specific airglow emission rates.
         - `glowfort.conduct`: Calculate the Pederson and Hall conductivities.

        ### Args:
            - `xuvfac (int, optional)`: XUV enhancement factor. Defaults to 3.
            - `jlocal (bool, optional)`: Set to False for electron transport calculations, set to True for local calculations only. Defaults to False.
            - `kchem (int, optional)`: `kchem (int, optional)`: CGLOW chemical calculations. Defaults to 4.
                - `0`: no calculations at all are performed.
                - `1`: electron density, O+(4S), N+, N2+, O2+, NO+ supplied at all altitudes; O+(2P), O+(2D), excited neutrals, and emission rates are calculated.
                - `2`: electron density, O+(4S), N+, N2+, O2+, NO+ supplied at all altitudes; O+(2P), O+(2D), excited neutrals, and emission rates are calculated.
                - `3`: electron density supplied at all altitudes; everything else calculated.  Note that this may violate charge neutrality and/or lead to other unrealistic results below 200 km, if the electron density supplied is significantly different from what the model thinks it should be. If it is desired to use a specified ionosphere, `kchem = 2` is probably a better option.
                - `4`: electron density supplied above 200 km; electron density below 200 km is calculated, everything else calculated at all altitudes. Electron density for the next two levels above `J200` is log interpolated between `E(J200)` and `E(J200+3)`.
            - `ion_n (Iterable, optional)`: N+ density profile. Must be specified if `kchem = 2`. Defaults to None.
            - `ion_n2 (Iterable, optional)`: N2+ density profile. Must be specified if `kchem = 2`. Defaults to None.
            - `ion_o (Iterable, optional)`: O+ density profile. If None, IRI-90 profile is used. Defaults to None.
            - `ion_o2 (Iterable, optional)`: O2+ density profile. If None, IRI-90 profile is used. Defaults to None.
            - `ion_no (Iterable, optional)`: NO+ density profile. If None, IRI-90 profile is used. Defaults to None.

        ### Raises:
            - `RuntimeError`: GLOW model not ready for evaluation. Run `setup` first.
            - `RuntimeError`: Atmosphere not evaluated. Run `atmosphere` (and `precipitation` if needed) first.
            - `ValueError`: `kchem` must be between `0` and `4`.
            - `ValueError`: `ion_n` and `ion_n2` must be specified for `kchem = 2`.
            - `ValueError`: `Invalid dimensions for ion density profiles`.
        """
        if not self._atm:
            raise RuntimeError('Atmosphere not evaluated. Run `atmosphere` (and `precipitation` if needed) first.')
        cglow.xuvfac = xuvfac
        # ! Set switch for ETRANS
        cglow.jlocal = 1 if jlocal else 0

        # Conditions for kchem
        kchemsrc = {
            'N+': 'Custom',
            'N2+': 'Custom',
            'O+': 'IRI-90',
            'O2+': 'IRI-90',
            'NO+': 'IRI-90',
        }

        if kchem < 0 or kchem > 4:
            raise ValueError('kchem must be between 0 and 4')
        if kchem == 1:
            if ion_n is None or ion_n2 is None:
                raise ValueError('N+ and N2+ density profiles must be specified for kchem = 1')
            ion_n = asarray(ion_n, dtype=float32, order='F')
            ion_n2 = asarray(ion_n2, dtype=float32, order='F')
            if ion_o.ndim != 1 or ion_o2.ndim != 1 or len(ion_o) != cglow.jmax or len(ion_o2) != cglow.jmax:
                raise ValueError('Invalid ion density profiles')
            cg.zxden[3, :] = ion_n
            cg.zxden[4, :] = ion_n2

        if 0 < kchem < 3:
            if ion_o is not None:
                ion_o = asarray(ion_o, dtype=float32, order='F')
                if ion_o.ndim != 1 or len(ion_o) != cglow.jmax:
                    raise ValueError('Invalid O+ density profile.')
                cg.zxden[2, :] = ion_o
                kchemsrc['O+'] = 'Custom'
            if ion_o2 is not None:
                ion_o2 = asarray(ion_o2, dtype=float32, order='F')
                if ion_o2.ndim != 1 or len(ion_o2) != cglow.jmax:
                    raise ValueError('Invalid O2+ density profile.')
                cg.zxden[5, :] = ion_o2
                kchemsrc['O2+'] = 'Custom'
            if ion_no is not None:
                ion_no = asarray(ion_no, dtype=float32, order='F')
                if ion_no.ndim != 1 or len(ion_no) != cglow.jmax:
                    raise ValueError('Invalid NO+ density profile.')
                cg.zxden[6, :] = ion_no
                kchemsrc['NO+'] = 'Custom'

        cglow.kchem = kchem

        # ! call GLOW to calculate ionized and excited species, airglow emission rates,
        # ! and vertical column brightnesses:
        glow()
        # ! Call CONDUCT to calculate Pederson and Hall conductivities:
        pedcond = zeros((cg.jmax,), self._z.dtype)
        hallcond = pedcond.copy()
        conduct_iface(self._z, pedcond, hallcond)  # loop in FORTRAN
        # for j in range(cg.jmax):
        #     pedcond[j], hallcond[j] = conduct(cg.glat, cg.glong, self._z[j], cg.zo[j], cg.zo2[j], cg.zn2[j],
        #                                       cg.zxden[2, j], cg.zxden[5, j], cg.zxden[6, j], cg.ztn[j], cg.zti[j], cg.zte[j])

        ds = self._ds

        density_attrs = {
            'long_name': 'number density',
            'units': 'cm^{-3}',
            'source': 'GLOW',
        }
        ds['NeOut'] = Variable('alt_km',   cg.ecalc, density_attrs)
        ds['ionrate'] = Variable('alt_km', cg.tir,   density_attrs)

        ds['pederson'] = \
            Variable('alt_km',
                     pedcond,
                     {'units': 'S m^{-1}',
                      'long_name': 'Pederson Conductivity',
                      'description': 'Pederson Conductivity',
                      'source': 'IRI-90'}
                     )
        ds['hall'] = \
            Variable('alt_km',
                     hallcond,
                     {'units': 'S m^{-1}',
                      'long_name': 'Hall Conductivity',
                      'description': 'Hall Conductivity',
                      'source': 'IRI-90'}
                     )
        # Emissions
        ds['ver'] = Variable(('alt_km', 'wavelength'), cg.zeta.T, {
            'long_name': 'Volume Emission Rate',
            'units': 'cm^{-3} s^{-1}',
            'description': 'Volume Emission Rate',
            'source': 'GLOW'
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

        ds.coords['wavelength'] = ('wavelength', wavelen, {
            'long_name': 'wavelength',
            'description': 'Emission Wavelengths. LBH is the total emission in the Lyman-Birge-Hopfield band.',
            'units': 'Å',
            'source': 'GLOW'
        })

        state = [
            "O+(2P)",
            "O+(2D)",
            "O+",
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

        ds['production'] = Variable(('alt_km', 'state'), cg.production.T, {'units': 'cm^{-3} s^{-1}'})
        ds['loss'] = Variable(('alt_km', 'state'), cg.loss.T, {'units': 's^{-1}'})
        ds.coords['state'] = ('state', state, {
            'long_name': 'state',
            'description': 'Ionized and excited species'
        })

        ds['ionrate'] = Variable('alt_km', cg.tir, density_attrs)
        ds['ionrate'].attrs['description'] = 'Total ionization rate'

        ds['NeOut'] = Variable('alt_km', cg.ecalc, density_attrs)
        if kchem > 3:
            ds['NeOut'].attrs['description'] = 'Electron density calculated below 200 km using iterative method to account for charge neutrality'
        else:
            ds['NeOut'].attrs['description'] = 'Electron density from IRI-90'

        statenam = [
            "O^+(^2P)",
            "O^+(^2D)",
            "O^+(^4S)",
            "N^+",
            "N_2^+",
            "O_2^+",
            "N_O^+",
            "N_2(A)",
            "N(^2P)",
            "N(^2D)",
            "O(^1S)",
            "O(^1D)",
        ]

        for idx, key in enumerate(state):
            ds[key] = Variable('alt_km', cg.zxden[idx, :], density_attrs)
            ds[key].attrs['description'] = f'{key} ion density'
            ds[key].attrs['formatted'] = statenam[idx]
            if kchem == 0:
                ds[key].attrs['source'] = 'IRI-90'
            elif kchem == 1:
                if key in ['O+', 'O2+', 'NO+', 'N+', 'N2+']:
                    ds[key].attrs['source'] = kchemsrc[key]
                else:
                    ds[key].attrs['source'] = 'GLOW'
            elif kchem == 2:
                if key in ['O+', 'O2+', 'NO+']:
                    ds[key].attrs['source'] = kchemsrc[key]
                else:
                    ds[key].attrs['source'] = 'GLOW'

        ds['eHeat'] = Variable('alt_km', cg.eheat, {'units': 'eV cm^{-3} s^{-1}',
                                                    'description': 'Ambient electron heating rate'})
        ds['Tez'] = Variable('alt_km', cg.tez, {'units': 'eV cm^{-3} s^{-1}',
                                                'comment': 'total energetic electron energy deposition', 'long_name': 'energy deposition'})

        ds.attrs['xuvfac'] = xuvfac
        ds.attrs['kchem'] = kchem
        ds.attrs['jlocal'] = cglow.jlocal

        self._evaluated = True
        return

    def evaluate(self, xuvfac: int = 3, jlocal: bool = False, kchem: int = 4, *,
                 density_perturbation: Sequence = None,
                 tec: Numeric | Dataset = None,
                 hmf2: Optional[Numeric] = None,
                 nmf2: Optional[Numeric] = None,
                 f2_peak: HmFSource = 'URSI',
                 ion_n: Iterable = None,
                 ion_n2: Iterable = None,
                 ion_o: Iterable = None,
                 ion_o2: Iterable = None,
                 ion_no: Iterable = None) -> xarray.Dataset:
        """## Evaluate the GLOW model.
        This function runs the atmosphere and radiative transfer models in sequence.
        If electron precipitation is required, use the :method:`GlowModel.precipitation` method before calling this function.

        ### Args:
            - `xuvfac (int, optional)`: XUV enhancement factor. Defaults to 3.
            - `jlocal (bool, optional)`: Set to False for electron transport calculations, set to True for local calculations only. Defaults to False.
            - `kchem (int, optional)`: CGLOW chemical calculations. Defaults to 4.
            - `density_perturbation (Sequence, optional)`: Density perturbations of O, O2, N2, NO, N(4S), N(2D) and e-. Defaults to None.
            - `tec (Numeric | Dataset, optional)`: Total Electron Content (TEC) in TECU. Defaults to None. Used to scale IRI-90 derived electron density.
            - `hmf2 (Numeric, optional)`: Height of the F2 peak (km). Defaults to None.
            - `nmf2 (Numeric, optional)`: Density of the F2 peak (m^-3). Defaults to None.
            - `f2_peak (IRISRC, optional)`: F2 peak model. Defaults to 'URSI'.
            - `ion_n (Iterable, optional)`: N+ density profile. Must be specified if `kchem = 2`. Defaults to None.
            - `ion_n2 (Iterable, optional)`: N2+ density profile. Must be specified if `kchem = 2`. Defaults to None.
            - `ion_o (Iterable, optional)`: O+ density profile. If None, IRI-90 profile is used. Defaults to None.
            - `ion_o2 (Iterable, optional)`: O2+ density profile. If None, IRI-90 profile is used. Defaults to None.
            - `ion_no (Iterable, optional)`: NO+ density profile. If None, IRI-90 profile is used. Defaults to None.

        ### Returns:
            - `xarray.Dataset`: GLOW model output dataset.
        """
        self.atmosphere(density_perturbation, tec, hmf2, nmf2, f2_peak)
        self.radtrans(xuvfac, jlocal, kchem, ion_n=ion_n, ion_n2=ion_n2, ion_o=ion_o, ion_o2=ion_o2, ion_no=ion_no)
        return self.result()

    def result(self, copy: bool = True) -> xarray.Dataset:
        """## Get the GLOW model output dataset.
        This function returns a deep or shallow copy of the GLOW model output dataset.
        - The shallow copy is **THREAD LOCAL** and not `Send`, i.e. can not be sent across or shared between threads. 
        - Use `copy=True` to get a deep copy.
        - Deep copies of result are available **ONLY AFTER** evaluation.

        ### Args:
            - `copy (bool, optional)`: Return a deep copy. Defaults to True.

        ### Returns:
            - `xarray.Dataset`: GLOW model output dataset.

        ### Raises:
            - `RuntimeError`: GLOW model not initialized.
            - `RuntimeError`: GLOW model not evaluated, in case a deep copy is requested without evaluation.
        """
        if not self._initd:
            raise RuntimeError('GLOW model not initialized')
        if not self._evaluated and copy:
            raise RuntimeError('GLOW model not evaluated')
        return self._ds.copy(deep=copy)

    def altitude_grid(self) -> xarray.DataArray:
        """## Get the altitude grid.

        ### Raises:
            - `RuntimeError`: GLOW model not initialized.

        ### Returns:
            - `xarray.DataArray`: Altitude grid (km).
        """
        if not self._initd:
            raise RuntimeError('GLOW model not initialized')
        return self._ds['alt_km'].copy(deep=True)

    def __call__(self, xuvfac: int = 3, jlocal: bool = False, kchem: int = 4, *,
                 density_perturbation: Sequence = None,
                 tec: Numeric | Dataset = None,
                 hmf2: Optional[Numeric] = None,
                 nmf2: Optional[Numeric] = None,
                 f2_peak: HmFSource = 'URSI',
                 ion_n: Iterable = None,
                 ion_n2: Iterable = None,
                 ion_o: Iterable = None,
                 ion_o2: Iterable = None,
                 ion_no: Iterable = None) -> xarray.Dataset:
        """## Evaluate the GLOW model.
        This function runs the atmosphere and radiative transfer models in sequence.
        If electron precipitation is required, use the :method:`GlowModel.precipitation` method before calling this function.

        ### Args:
            - `xuvfac (int, optional)`: XUV enhancement factor. Defaults to 3.
            - `jlocal (bool, optional)`: Set to False for electron transport calculations, set to True for local calculations only. Defaults to False.
            - `kchem (int, optional)`: CGLOW chemical calculations. Defaults to 4.
            - `density_perturbation (Sequence, optional)`: Density perturbations of O, O2, N2, NO, N(4S), N(2D) and e-. Defaults to None.
            - `tec (Numeric | Dataset, optional)`: Total Electron Content (TEC) in TECU. Defaults to None. Used to scale IRI-90 derived electron density.
            - `ion_n (Iterable, optional)`: N+ density profile. Must be specified if `kchem = 2`. Defaults to None.
            - `ion_n2 (Iterable, optional)`: N2+ density profile. Must be specified if `kchem = 2`. Defaults to None.
            - `ion_o (Iterable, optional)`: O+ density profile. If None, IRI-90 profile is used. Defaults to None.
            - `ion_o2 (Iterable, optional)`: O2+ density profile. If None, IRI-90 profile is used. Defaults to None.
            - `ion_no (Iterable, optional)`: NO+ density profile. If None, IRI-90 profile is used. Defaults to None.

        ### Returns:
            - `xarray.Dataset`: GLOW model output dataset.
        """
        return self.evaluate(xuvfac, jlocal, kchem, density_perturbation=density_perturbation, tec=tec, ion_n=ion_n, ion_n2=ion_n2, ion_o=ion_o, ion_o2=ion_o2, ion_no=ion_no, hmf2=hmf2, nmf2=nmf2, f2_peak=f2_peak)


def generic(time: datetime,
            glat: Numeric,
            glon: Numeric,
            Nbins: int = 100,
            Q: Numeric = None,
            Echar: Numeric = None,
            density_perturbation: Sequence = None,
            *,
            geomag_params: dict | Iterable = None,
            tzaware: bool = False,
            tec: Numeric | Dataset = None,
            hmf2: Numeric = None,
            nmf2: Numeric = None,
            f2_peak: HmFSource = 'URSI',
            metadata: dict = None,
            jmax: int = 250,
            iscale: int = 1,
            xuvfac: Numeric = 3,
            kchem: int = 4,
            jlocal: bool = False,
            itail: bool = False,
            fmono: float = 0,
            emono: float = 0) -> xarray.Dataset:
    """## GLOW model with optional electron precipitation assuming Maxwellian distribution.
    Defaults to no precipitation.

    ### Args:
        - `time (datetime)`: Model evaluation time.
        - `glat (Numeric)`: Location latitude (degrees).
        - `glon (Numeric)`: Location longitude (degrees).
        - `Nbins (int, optional)`: Number of energy bins (must be >= 10). Defaults to 100.
        - `Q (Numeric, optional)`: Flux of precipitating electrons (erg/cm^2/s). Setting to None or < 0.001 makes it equivalent to no-precipitation. Defaults to None.
        - `Echar (Numeric, optional)`: Energy of precipitating electrons. Setting to None or < 1 makes it equivalent to no-precipitation. Defaults to None.
        - `density_perturbation (Sequence, optional)`: Density perturbations of O, O2, N2, NO, N(4S), N(2D) and e-. 
           Supply as a sequence of length 7. Values must be positive. Defaults to None (all ones, i.e. no modification).
        - `geomag_params (dict | Iterable, optional)`: Custom geomagnetic parameters.
            - `f107a` (running average F10.7 over 81 days), 
            - `f107` (current day F10.7), 
            - `f107p` (previous day F10.7), and
            - `Ap` (the global 3 hour $a_p$ index). 

            Must be present in this order for list or tuple, and use these keys for the dictionary. Defaults to None.
        - `tzaware (bool, optional)`: If time is time zone aware. If true, `time` is recast to 'UTC' using `time.astimezone(pytz.utc)`. Defaults to False.
        - `tec (Numeric | Dataset, optional)`: Total Electron Content (TEC) in TECU. Defaults to None. Used to scale IRI-90 derived electron density.

            If `Dataset`, must contain the following coordinates:
                - `timestamps` (`datetime64[ns]`): Timestamp of the TEC map,
                - `gdlat` (`float64`): Geodetic latitude in degrees,
                - `glon` (`float64`): Geocentric longitude in degrees,
            and the following data variable:
                - `tec` (`float64`): TEC in TECU.

            The nearest spatio-temporal TEC value is used to scale the electron density. This 
            dataset format is compatible with the GPS TEC maps from the Madrigal database.

        - `hmf2 (Numeric, optional)`: Height of the F2 peak (km). Defaults to None.
        - `nmf2 (Numeric, optional)`: Density of the F2 peak (m^-3). Defaults to None.
        - `f2_peak (HmFSource, optional)`: F2 peak model. Defaults to 'URSI'.
        - `metadata (dict, optional)`: Metadata to be added to the output dataset. Defaults to None.
        - `jmax (int, optional)`: Maximum number of altitude levels. Defaults to 250.
        - `iscale (int, optional)`: Solar flux model. Defaults to 1 (EUVAC model).
        - `xuvfac (Numeric, optional)`: XUV enhancement factor. Defaults to 3.
        - `kchem (int, optional)`: CGLOW chemical calculations. Defaults to 4. See `cglow.kchem` for details.
        - `jlocal (bool, optional)`: Set to False for electron transport calculations, set to True for local calculations only. Defaults to False.
        - `itail (bool, optional)`: Disable/enable low-energy tail. Defaults to False.
        - `fmono (float, optional)`: Monoenergetic energy flux, erg/cm^2. Defaults to 0.
        - `emono (float, optional)`: Monoenergetic characteristic energy, keV. Defaults to 0.

    ### Raises:
        - `ValueError`: Number of energy bins must be >= 10.
        - `RuntimeError`: Invalid type for geomag params.
        - `ValueError`: Density perturbation must be a sequence of length 7.
        - `ValueError`: Density perturbation must be positive.
        - `ValueError`: TEC must be positive.
        - `ValueError`: TEC must be in TECU. 1 TECU = 10^16 electrons/m^2.
        - `ValueError`: TEC dataset is not valid for the timestamp.
        - `ImportError`: Module `cglow` is uninitialized.

    ### Returns:
        - `xarray.Dataset`: GLOW model output dataset.
    """
    mod = GlowModel()  # Get an instance of the GLOW model
    mod.initialize(jmax, Nbins, iscale)
    mod.setup(time, glat, glon, geomag_params=geomag_params, tzaware=tzaware)
    mod.precipitation(Q, Echar, itail=itail, fmono=fmono, emono=emono)
    ds = mod.evaluate(xuvfac, jlocal, kchem, density_perturbation=density_perturbation,
                      tec=tec, hmf2=hmf2, nmf2=nmf2, f2_peak=f2_peak)
    _ = metadata
    return ds


def maxwellian(time: datetime,
               glat: Numeric,
               glon: Numeric,
               Nbins: int,
               Q: Numeric,
               Echar: Numeric,
               density_perturbation: Sequence = None,
               *,
               geomag_params: dict | Iterable = None,
               tzaware: bool = False,
               tec: Numeric | Dataset = None,
               hmf2: Optional[Numeric] = None,
               nmf2: Optional[Numeric] = None,
               f2_peak: HmFSource = 'URSI',
               metadata: dict = None) -> xarray.Dataset:
    """## GLOW model with electron precipitation assuming Maxwellian distribution.

    ### Args:
        - `time (datetime)`: Model evaluation time.
        - `glat (Numeric)`: Location latitude (degrees).
        - `glon (Numeric)`: Location longitude (degrees).
        - `Nbins (int)`: Number of energy bins (must be >= 10).
        - `Q (Numeric, optional)`: Flux of precipitating electrons (erg/cm^2/s). Setting to None or < 0.001 makes it equivalent to no-precipitation. Defaults to None.
        - `Echar (Numeric, optional)`: Energy of precipitating electrons. Setting to None or < 1 makes it equivalent to no-precipitation. Defaults to None.
        - `density_perturbation (Sequence, optional)`: Density perturbations of O, O2, N2, NO, N(4S), N(2D) and e-. 
           Supply as a sequence of length 7. Values must be positive. Defaults to None (all ones, i.e. no modification).
        - `geomag_params (dict | Iterable, optional)`: Custom geomagnetic parameters.
            - `f107a` (running average F10.7 over 81 days), 
            - `f107` (current day F10.7), 
            - `f107p` (previous day F10.7), and
            - `Ap` (the global 3 hour $a_p$ index). 

            Must be present in this order for list or tuple, and use these keys for the dictionary. Defaults to None.
        - `tzaware (bool, optional)`: If time is time zone aware. If true, `time` is recast to 'UTC' using `time.astimezone(pytz.utc)`. Defaults to False.
        - `tec (Numeric | Dataset, optional)`: Total Electron Content (TEC) in TECU. Defaults to None. Used to scale IRI-90 derived electron density.

            If `Dataset`, must contain the following coordinates:
                - `timestamps` (`datetime64[ns]`): Timestamp of the TEC map,
                - `gdlat` (`float64`): Geodetic latitude in degrees,
                - `glon` (`float64`): Geocentric longitude in degrees,
            and the following data variable:
                - `tec` (`float64`): TEC in TECU.

            The nearest spatio-temporal TEC value is used to scale the electron density. This
            dataset format is compatible with the GPS TEC maps from the Madrigal database.
        - `hmf2 (Numeric, optional)`: Height of the F2 peak (km). Defaults to None.
        - `nmf2 (Numeric, optional)`: Density of the F2 peak (m^-3). Defaults to None.
        - `f2_peak (IRISRC, optional)`: F2 peak model. Defaults to 'URSI'.
        - `metadata (dict, optional)`: Metadata to be added to the output dataset. Defaults to None.

    ### Raises:
        Refer to `generic` for details.

    ### Returns:
        - `xarray.Dataset`: GLOW model output dataset.
    """
    mod = GlowModel()  # Get an instance of the GLOW model
    if cg.jmax == 0:
        alt_km = 250
    else:
        alt_km = None
    mod.initialize(alt_km, Nbins)
    mod.setup(time, glat, glon, geomag_params=geomag_params, tzaware=tzaware)
    mod.precipitation(Q, Echar)
    ds = mod(density_perturbation=density_perturbation, tec=tec, hmf2=hmf2, nmf2=nmf2, f2_peak=f2_peak)
    _ = metadata
    return ds


def no_precipitation(time: datetime,
                     glat: Numeric,
                     glon: Numeric,
                     Nbins: int = 100,
                     density_perturbation: Sequence = None,
                     *,
                     geomag_params: dict | Iterable = None,
                     tzaware: bool = False,
                     tec: Numeric | Dataset = None,
                     hmf2: Optional[Numeric] = None,
                     nmf2: Optional[Numeric] = None,
                     f2_peak: HmFSource = 'URSI',
                     metadata: dict = None) -> xarray.Dataset:
    """## GLOW model with no electron precipitation.

    ### Args:
        - `time (datetime)`: Model evaluation time.
        - `glat (Numeric)`: Location latitude (degrees).
        - `glon (Numeric)`: Location longitude (degrees).
        - `Nbins (int)`: Number of energy bins (must be >= 10). Defaults to 100.
        - `density_perturbation (Sequence, optional)`: Density perturbations of O, O2, N2, NO, N(4S), N(2D) and e-. 
           Supply as a sequence of length 7. Values must be positive. Defaults to None (all ones, i.e. no modification).
        - `geomag_params (dict | Iterable, optional)`: Custom geomagnetic parameters.
            - `f107a` (running average F10.7 over 81 days), 
            - `f107` (current day F10.7), 
            - `f107p` (previous day F10.7), and
            - `Ap` (the global 3 hour $a_p$ index). 

            Must be present in this order for list or tuple, and use these keys for the dictionary. Defaults to None.
        - `tzaware (bool, optional)`: If time is time zone aware. If true, `time` is recast to 'UTC' using `time.astimezone(pytz.utc)`. Defaults to False.
        - `tec (Numeric | Dataset, optional)`: Total Electron Content (TEC) in TECU. Defaults to None. Used to scale IRI-90 derived electron density.

            If `Dataset`, must contain the following coordinates:
                - `timestamps` (`datetime64[ns]`): Timestamp of the TEC map,
                - `gdlat` (`float64`): Geodetic latitude in degrees,
                - `glon` (`float64`): Geocentric longitude in degrees,
            and the following data variable:
                - `tec` (`float64`): TEC in TECU.

            The nearest spatio-temporal TEC value is used to scale the electron density. This
            dataset format is compatible with the GPS TEC maps from the Madrigal database.
        - `hmf2 (Numeric, optional)`: Height of the F2 peak (km). Defaults to None.
        - `nmf2 (Numeric, optional)`: Density of the F2 peak (m^-3). Defaults to None.
        - `f2_peak (IRISRC, optional)`: F2 peak model. Defaults to 'URSI'.
        - `metadata (dict, optional)`: Metadata to be added to the output dataset. Defaults to None.

    ### Raises:
        Refer to `generic` for details.

    ### Returns:
        - `xarray.Dataset`: GLOW model output dataset.
    """
    mod = GlowModel()  # Get an instance of the GLOW model
    if cg.jmax == 0:
        alt_km = 250
    else:
        alt_km = None
    mod.initialize(alt_km, Nbins)
    mod.setup(time, glat, glon, geomag_params=geomag_params, tzaware=tzaware)
    ds = mod(density_perturbation=density_perturbation, tec=tec, hmf2=hmf2, nmf2=nmf2, f2_peak=f2_peak)
    _ = metadata
    return ds


def monoenergetic(time: datetime,
                  glat: Numeric,
                  glon: Numeric,
                  Nbins: int = 100,
                  fmono: float = 0,
                  emono: float = 0,
                  itail: bool = False,
                  density_perturbation: Sequence = None,
                  *,
                  geomag_params: dict | Iterable = None,
                  tzaware: bool = False,
                  tec: Numeric | Dataset = None,
                  hmf2: Optional[Numeric] = None,
                  nmf2: Optional[Numeric] = None,
                  f2_peak: HmFSource = 'URSI',
                  metadata: dict = None) -> xarray.Dataset:
    """## GLOW model with monoenergetic precipitation.

    ### Args:
        - `time (datetime)`: Model evaluation time.
        - `glat (Numeric)`: Location latitude (degrees).
        - `glon (Numeric)`: Location longitude (degrees).
        - `Nbins (int)`: Number of energy bins (must be >= 10). Defaults to 100.
        - `fmono (float, optional)`: Monoenergetic energy flux, erg/cm^2. Defaults to 0.
        - `emono (float, optional)`: Monoenergetic characteristic energy, keV. Defaults to 0.
        - `itail (bool, optional)`: Disable/enable low-energy tail. Defaults to False.
        - `density_perturbation (Sequence, optional)`: Density perturbations of O, O2, N2, NO, N(4S), N(2D) and e-. 
           Supply as a sequence of length 7. Values must be positive. Defaults to None (all ones, i.e. no modification).
        - `geomag_params (dict | Iterable, optional)`: Custom geomagnetic parameters.
            - `f107a` (running average F10.7 over 81 days), 
            - `f107` (current day F10.7), 
            - `f107p` (previous day F10.7), and
            - `Ap` (the global 3 hour $a_p$ index). 

            Must be present in this order for list or tuple, and use these keys for the dictionary. Defaults to None.
        - `tzaware (bool, optional)`: If time is time zone aware. If true, `time` is recast to 'UTC' using `time.astimezone(pytz.utc)`. Defaults to False.
        - `tec (Numeric | Dataset, optional)`: Total Electron Content (TEC) in TECU. Defaults to None. Used to scale IRI-90 derived electron density.

            If `Dataset`, must contain the following coordinates:
                - `timestamps` (`datetime64[ns]`): Timestamp of the TEC map,
                - `gdlat` (`float64`): Geodetic latitude in degrees,
                - `glon` (`float64`): Geocentric longitude in degrees,
            and the following data variable:
                - `tec` (`float64`): TEC in TECU.

            The nearest spatio-temporal TEC value is used to scale the electron density. This
            dataset format is compatible with the GPS TEC maps from the Madrigal database.
        - `hmf2 (Numeric, optional)`: Height of the F2 peak (km). Defaults to None.
        - `nmf2 (Numeric, optional)`: Density of the F2 peak (m^-3). Defaults to None.
        - `f2_peak (IRISRC, optional)`: F2 peak model. Defaults to 'URSI'.
        - `metadata (dict, optional)`: Metadata to be added to the output dataset. Defaults to None.

    ### Raises:
        Refer to `generic` for details.

    ### Returns:
        - `xarray.Dataset`: GLOW model output dataset.
    """
    mod = GlowModel()  # Get an instance of the GLOW model
    if cg.jmax == 0:
        alt_km = 250
    else:
        alt_km = None
    mod.initialize(alt_km, Nbins)
    mod.setup(time, glat, glon, geomag_params=geomag_params, tzaware=tzaware)
    mod.precipitation(0.001, 0.1, fmono=fmono, emono=emono, itail=itail)
    ds = mod(density_perturbation=density_perturbation, tec=tec, hmf2=hmf2, nmf2=nmf2, f2_peak=f2_peak)
    _ = metadata
    return ds
