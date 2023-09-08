# GLOW

The GLobal airglOW Model, independently and easily accessed from **Fortran 2003** compiler.
Optionally available from Python &ge; 3.7.

A Fortran compiler is REQUIRED.

## Installation

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

* Maxwellian.py: Maxwellian precipitation, specify Q (flux) and E0 (characteristic energy).
* NoPrecipitation.py: No precipitating electrons.

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



