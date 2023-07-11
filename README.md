# GLOW

The GLobal airglOW Model, independently and easily accessed from **Fortran 2003** compiler.
Optionally available from Python &ge; 3.7.

A Fortran compiler is REQUIRED.

## Installation

Direct install using pip:
```sh
$ pip install glowpython
```

Install from source repository by:

```sh
$ git clone https://github.com/sunipkm/glowpython --recurse-submodules
$ cd glowpython && pip install .
```

Requires (and installs) [geomagdata](https://pypi.org/project/geomagdata/) for timezone aware geomagnetic parameters retrieval.

Then use examples such as:

* Maxwellian.py:  Maxwellian precipiation, specify Q and E0.
* Monoenergetic.py: Specify unit flux for one energy bin, all other energy bins are zero flux.
* NoPrecipitation.py: No precipitating electrons.

or use GLOW in your own Python program by:
```python
import glowpython as glow

iono = glow.maxwellian(time, glat, glon, Q, Echar, Nbins)
```

`iono` is an
[xarray.Dataset](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html)
containing outputs from GLOW, including:

* number densities of neutrals, ions and electrons
* Pedersen and Hall currents
* volume emssion rate vs. wavelength and altitude
* precipitating flux vs. energy
* many more, request if you want it.

All available keys carry unit and description.



