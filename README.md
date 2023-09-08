# GLOW

The GLobal airglOW Model, independently and easily accessed from **Fortran 2003** compiler.
Optionally available from Python &ge; 3.7.

A Fortran compiler is REQUIRED.

<b>Note:</b> This version uses `meson` and `ninja` as the build system, and does not rely on `distutils`,
and is Python 3.12 compatible.

## Installation

<b>Note:</b> For macOS Big Sur and above, add the following line to your environment script (`~/.zshrc` if using ZSH, or the relevant shell init script):
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
$ git clone https://github.com/sunipkm/glowpython
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

`iono` is a
[xarray.Dataset](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html)
containing outputs from GLOW, including:

* number densities of neutrals, ions and electrons
* Pedersen and Hall currents
* volume emission rate vs. wavelength and altitude
* precipitating flux vs. energy
* production and loss rates
* solar flux and wavelength ranges
* electron energy deposition

All available keys carry unit and description.



