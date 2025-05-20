# GlowPython

The FORTRAN GLobal airglOW ([GLOW](https://github.com/NCAR/GLOW)) Model in Python &ge; 3.7.

A Fortran compiler is **REQUIRED**.

<b>Note:</b> This version uses `meson` and `ninja` as the build system, and does not rely on `distutils`,
and is Python 3.12 compatible.

<b>This library also allows parallelization of model evaluation using the [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) module.</b>

Additionally, the `GlowModel` class exposes complete control over model evaluation, which is divided
into the following fundamental steps:
1. `initialize`: Set the altitude, energy grids and solar flux model (Hinteregger, EUVAC, custom).
2. `setup`: Set the model up for evaluation, specifying time, location, and optionally geomagnetic parameters. If geomagnetic parameters are not specified, they are retrieved using [`geomagdata`](https://pypi.org/project/geomagdata/).
3. `precipitation` (optional): Evaluate the energy and flux of precipitating electrons (Maxwellian or monoenergetic).
4. `atmosphere`: Evaluate the neutral atmosphere (MSISE-00) and ionosphere (IRI-90).
5. `radtrans`: Evaluate the GLOW radiative transfer model to calculate volume emission rates and ion composition.
6. `result`: Retrieve the model output as a [xarray.Dataset](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html).

<b>Note:</b> Between the `atmosphere` and `radtrans` steps, the intermediate dataset can be accessed using the `result` method by passing `copy=False`. This returns a reference to the internal dataset. Modifying the dataset in place allows for modifying the atmosphere and ionosphere presented to the `radtrans` step.

## Installation

<b>Note:</b> For macOS Big Sur and above, you may need to add the following line to your environment script (`~/.zshrc` if using ZSH, or the relevant shell init script):
```sh
export LIBRARY_PATH="$LIBRARY_PATH:/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib"
```
Then reopen the terminal. This fixes the issue where `-lSystem` fails for `gfortran`. Additionally,
installation of `gfortran` using `homebrew` is required.

Direct installation using pip:
```sh
$ pip install glowpython
```

Install from source repository by:

```sh
$ git clone https://github.com/sunipkm/glowpython -b development
$ cd glowpython && pip install .
```

Requires (and installs) [geomagdata](https://pypi.org/project/geomagdata/) for timezone aware geomagnetic parameter retrieval.

Then use examples such as:

* `Examples/Maxwellian.py`: Maxwellian precipitation, specify Q (flux) and E0 (characteristic energy).
* `Examples/NoPrecipitation.py`: No precipitating electrons.

These examples are also available on the command line: `GlowMaxwellian` and `GlowNoPrecip`.

Or, use GLOW in your own Python program by:
```python
import glowpython as glow

iono = glow.maxwellian(time, glat, glon, Nbins, Q, Echar)
```

Read the module documentation for more information.

`iono` is a
[xarray.Dataset](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html)
containing outputs from GLOW:
- Coordinates:
    - `alt_km`: Altitude grid (km)
    - `energy`: Energy grid (eV)
    - `state`: Neutral/Ionic excitation states (string)
    - `wavelength`: Wavelength of emission features in Angstrom (string). Represented as strings to accommodate the LBH band.
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
    - `iscale`: Solar flux model. 0: Hinteregger, 1: EUVAC
    - `xuvfac`: XUV enhancement factor. 
    - `itail`: Low energy tail enabled/disabled. 0: disabled, 1: enabled
    - `fmono`: Monoenergetic energy flux (erg/cm^2).
    - `emono`: Monoenergetic characteristic energy (keV).
    - `jlocal`: Local calculation only (disable electron transport).
    - `kchem`: GLOW chemistry level.

All available keys carry unit and description.

## Development Path
This repository started its life as a fork of the [NCAR-GLOW](https://github.com/space-physics/NCAR-GLOW) model to upgrade from [geomagindices](https://github.com/space-physics/geomagindices) that does not provide flux and Ap data post-SWPC.
The [NCAR-GLOW](https://github.com/space-physics/NCAR-GLOW) implementation runs the GLOW model as a separate process, invoked from a compiled GLOW binary, and runtime parameters are passed to this binary over `STDIN` and `STDOUT` is parsed to produce the [`Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) containing the results of the simulation.
This context switching, and text data parsing adds significant runtime overhead, and loss of precision.
Starting v3.0.0, the core of [`glowpython`](https://github.com/sunipkm/glowpython) was changed to use F2PY, and pass inputs and get outputs from the FORTRAN library directly.
Apart from the public-facing API (`no_precipitation`, `maxwellian`), the internal implementation of this repository is completely independent from NCAR-GLOW.

