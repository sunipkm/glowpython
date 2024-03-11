# %%
from __future__ import annotations
from typing import Iterable, SupportsAbs, Tuple, SupportsFloat as Numeric, Callable
from numpy import arctan, interp, isnan, ndarray, tan, pi as M_PI, asarray, all
from datetime import datetime

#: WGS84 ellipsoid major and minor axes.
WGS84_ELL = (6378137, 6356752.3142)  # WGS84 ellipsoid
#: WGS74 ellipsoid major and minor axes.
WGS74_ELL = (6378135, 6356750.5)  # WGS74 ellipsoid


def geocent_to_geodet(lat: Numeric | Iterable[Numeric], ell: Tuple[Numeric, Numeric] = WGS84_ELL) -> Numeric | ndarray:
    """## Convert geocentric latitude to geodetic latitude.

    ### Args:
        - `lat (Numeric | Iterable[Numeric])`: Geocentric latitude in degrees.
        - `ell (Tuple[Numeric, Numeric], optional)`: Reference ellipsoid major and minor axes. Defaults to WGS84 ellipsoid.

    ### Asserts:
        - `-90 <= lat <= 90`: Latitude is within bounds.
        - `ell[0] > 0`: Major axis length is positive.
        - `ell[1] > 0`: Minor axis length is positive.
        - `ell[0] > ell[1]`: Major and minor axes convention.
        - `lat.ndim == 1`: Latitude is 1-D array.

    ### Returns:
        - `Numeric`: Geodetic latitude in degrees.
    """
    a, b = ell
    if isinstance(lat, Iterable):
        lat = asarray(lat)
        assert (lat.ndim == 1)
    assert (a > 0 and b > 0 and a > b and all((-90 <= lat) & (lat <= 90)))
    return arctan(b*tan(lat*M_PI/180)/a)*180/M_PI


def glowdate(t: datetime) -> Tuple[int, Numeric]:
    """## Convert datetime to GLOW date and UT seconds.

    ### Args:
        - `t (datetime)`: Datetime object.

    ### Returns:
        - `Tuple[int, Numeric]`: GLOW date (YYYYDDD) and UT seconds.
    """
    idate = int(f'{t.year:04}{t:%j}')
    utsec = (t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6)

    return idate, utsec


def nan_helper(y: ndarray) -> Tuple[ndarray, Callable[[ndarray], ndarray]]:
    """## Helper function to return NaN indices and a function to return non-NaN indices.

    ### Args:
        - `y (ndarray)`: Input 1-D array with NaNs.

    ### Returns:
        - `Tuple[ndarray, Callable[[ndarray], ndarray]]`: Tuple of NaN indices and a function to return non-NaN indices.

    ### Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return isnan(y), lambda z: z.nonzero()[0]


def interpolate_nan(y: ndarray, *, inplace: bool = True, left: SupportsAbs = None, right: SupportsAbs = None, period: Numeric = None) -> ndarray:
    """## Interpolate NaNs in a 1-D array.

    ### Args:
        - `y (ndarray)`: 1-D Array.
        - `inplace (bool, optional)`: Change input array in place. Defaults to True.
        - `left (Numeric, optional)`: Left boundary value. Defaults to 0.
        - `right (Numeric, optional)`: Right boundary value. Defaults to None.
        - `period (Numeric, optional)`: Period of the array. Defaults to None.

    ### Returns:
        - `ndarray`: Interpolated array.
    """
    if not inplace:
        y = y.copy()
    nans, x = nan_helper(y)
    y[nans] = interp(x(nans), x(~nans), y[~nans], left=left, right=right, period=period)
    return y


# %% Test functions
if __name__ == '__main__':
    import numpy as np
    y = np.array([0, 1, np.nan, np.nan, 4, 5, 6, np.nan, 8, 9])
    assert np.allclose(interpolate_nan(y),  [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    y = np.array([1, 1, 1, np.nan, np.nan, 2, 2, np.nan, 0])
    assert np.allclose(np.round(interpolate_nan(y), 2), [1., 1., 1., 1.33, 1.67, 2., 2., 1., 0.])
    lats = np.arange(-80, 80, 20)
    glats = geocent_to_geodet(lats)
    gglats = list(map(geocent_to_geodet, lats))
    assert np.allclose(glats, gglats)

# %%
