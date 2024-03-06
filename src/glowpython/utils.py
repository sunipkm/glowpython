from __future__ import annotations
from typing import Tuple, SupportsFloat as Numeric
from numpy import arctan, tan, pi as M_PI
from datetime import datetime

WGS84_ELL = (6378137, 6356752.3142)  # WGS84 ellipsoid
WGS74_ELL = (6378135, 6356750.5)  # WGS74 ellipsoid


def geocent_to_geodet(lat: Numeric, ell: Tuple[Numeric, Numeric] = WGS84_ELL) -> Numeric:
    """## Convert geocentric latitude to geodetic latitude.

    ### Args:
        - `lat (Numeric)`: Geocentric latitude in degrees.
        - `ell (Tuple[Numeric, Numeric], optional)`: _description_. Defaults to WGS84_ELL.

    ### Asserts:
        - `-90 <= lat <= 90`: Latitude is within bounds.
        - `ell[0] > 0`: Major axis length is positive.
        - `ell[1] > 0`: Minor axis length is positive.
        - `ell[0] > ell[1]`: Major and minor axes convention.

    ### Returns:
        - `Numeric`: Geodetic latitude in degrees.
    """
    a, b = ell
    assert (a > 0 and b > 0 and a > b and -90 <= lat <= 90)
    return arctan(b*tan(lat*M_PI/180)/a)*180/M_PI

def glowdate(t: datetime) -> Tuple[int, Numeric]:
    """## Convert datetime to GLOW date and UT seconds.

    ### Args:
        - `t (datetime)`: Datetime object.

    ### Returns:
        - `Tuple[int, Numeric]`: GLOW date (YYYYDDD) and UT seconds.
    """
    idate = int(f'{t.year}{t:%j}')
    utsec = (t.hour * 3600 + t.minute * 60 + t.second)

    return idate, utsec