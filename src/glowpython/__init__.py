"""
GlowPython
==========

GlowPython is a Python package for evaluating the Global Airglow (GLOW) model.
The FORTRAN backend of the GLOW model is NOT thread-safe. This package provides
a singleton class to ensure that only one instance of the GLOW model is used at
a time. It is recommended to only use **ONE** instance of the `GlowModel` class
at a time (in a loop) for repeated evaluations.

Provides
    1. `GlowModel`: Base class for GLOW model. This is a singleton class.
    2. `generic`: GLOW model with generic interface.
    3. `no_precipitation`: GLOW model without precipitation.
    4. `maxwellian`: GLOW model with maxwellian precipitation.

Usage:
    1. Class based usage:
        >>> from glowpython import GlowModel
        >>> glow = GlowModel() # Get an instance of the GLOW model
        >>> glow.setup(time, lat, lon) # Setup the GLOW model for a given time and location
        >>> ds = glow() # Evaluate the GLOW model and get the dataset```
    2. Function based usage:
        >>> from glowpython import no_precipitation
        >>> ds = no_precipitation(time, lat, lon) # Evaluate the GLOW model and get the dataset
    3. Advanced scenarios:
        The user can inherit from the `GlowModel` class and insert custom hooks to
        perform operations between atmosphere and radiation calculations.

Output Dataset:
    - Coordinates:
        - `alt_km`: Altitude grid (km)
        - `energy`: Energy grid (eV)
        - `state`: Neutral/Ionic excitation states (string)
        - `wavelength`: Wavelength of emission features in Angstrom (string). Represented as strings to accommodate the LBH band.
        - `wave`: Solar flux wavelength (Angstrom)
        - `dwave`: Solar flux wavelength bin width (Angstrom)
        - `sflux`: Solar flux (W/m^2/Angstrom)
        - `tecscale`: TEC scale factor (float)
    - Data:
        - Dimension (`alt_km`):
            - `O`: Neutral atomic oxygen density (cm^-3)
            - `O2`: Neutral molecular oxygen density (cm^-3)
            - `N2`: Neutral molecular nitrogen density (cm^-3)
            - `NO`: Neutral nitric oxide density (cm^-3)
            - `NS`: N(4S) density (cm^-3)
            - `ND`: N(2D) density (not used, set to zero) (cm^-3)
            - `NeIn`: Electron density (IRI-90), input to GLOW radiative transfer model. (cm^-3)
            - `O+`: Ion atomic oxygen (4S) density (cm^-3)
            - `O+(2P)`: Ion atomic oxygen (2P) density (cm^-3)
            - `O+(2D)`: Ion atomic oxygen (2D) density (cm^-3)
            - `O2+`: Ion molecular oxygen density (cm^-3)
            - `N+`: Ion molecular nitrogen density (cm^-3)
            - `N2+`: Ion molecular nitrogen density (cm^-3)
            - `NO+`: Ion nitric oxide density (cm^-3)
            - `N2(A)`: Molecular nitrogen (GLOW) density (cm^-3)
            - `N(2P)`: N(2P) density (cm^-3)
            - `N(2D)`: N(2D) density (cm^-3)
            - `O(1S)`: O(1S) density (cm^-3)
            - `O(1D)`: O(1D) density (cm^-3)
            - `NeOut`: Electron density (calculated below 200 km for `kchem=4` using GLOW model, cm^-3) 
            - `Te`: Electron temperature (K)
            - `Ti`: Ion temperature (K)
            - `Tn`: Neutral temperature (K)
            - `ionrate`: Ionization rate (s^-1)
            - `pederson`: Pederson conductivity (S/m)
            - `hall`: Hall conductivity (S/m)
            - `eHeat`: Ambient electron heating rate (eV/cm^3/s)
            - `Tez`: Total energetic electron energy deposition (eV/cm^3/s)
        - Dimension (`alt_km`, `wavelength`):
            - `ver`: Volume emission rate of various features (1/cm^3/s)
        - Dimension (`alt_km`, `state`):
            - `production`: Production rate of various species  (1/cm^3/s)
            - `loss`: Fractional loss rate of various species (1/s)
        - Dimension (`energy`):
            - `precip`: Precipitation flux (cm^-2/s/eV)
    - Attributes:
        - `time`: Time of evaluation (ISO 8601 formatted string)
        - `glatlon`: Geographic latitude and longitude of evaluation (degrees)
        - `Q`: Flux of precipitating electrons (erg/cm^2/s)
        - `Echar`: Characteristic energy of precipitating electrons (eV)
        - 'f107a': 81-day average of F10.7 solar flux
        - 'f107': Present day F10.7 solar flux
        - 'f107p': Previous day F10.7 solar flux
        - 'ap': Magnetic index
        - `iscale`: Solar flux model. 0: Hinteregger, 1: EUVAC
        - `xuvfac`: XUV enhancement factor. 
        - `itail`: Low energy tail enabled/disabled. 0: disabled, 1: enabled
        - `fmono`: Monoenergetic energy flux (erg/cm^2).
        - `emono`: Monoenergetic characteristic energy (keV).
        - `jlocal`: Local calculation only (disable electron transport).
        - `kchem`: GLOW chemistry level.
"""
from .base import GlowModel, generic, no_precipitation, maxwellian
from .utils import nan_helper, interpolate_nan, geocent_to_geodet, WGS74_ELL, WGS84_ELL
from .version import __version__

__all__ = ['GlowModel', 'generic', 'no_precipitation', 'maxwellian',
           'interpolate_nan', 'nan_helper', 
           'geocent_to_geodet', 'WGS74_ELL', 'WGS84_ELL',
           '__version__']
