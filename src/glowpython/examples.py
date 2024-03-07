import numpy as np

from glowpython import interpolate_nan
from .base import maxwellian, no_precipitation
from . import plots as plot
from datetime import datetime
from matplotlib.pyplot import show
from dateutil.parser import parse
import sys
import argparse


def Maxwellian():
    parser = argparse.ArgumentParser(
        prog='Maxwellian Precipitation',
        description='Evaluate upper atmosphere VER for a given time and location.',
        epilog='Uses MSIS00 for atmosphere, IRI90 for ionosphere, GLOW for emissions.')

    parser.add_argument('--time', type=str, help='Time in ISO 8601 format.', default='2015-12-13T10:00:00', required=False)
    parser.add_argument('--glat', type=float, help='Geographic latitude in degrees.', default=42.6, required=False)
    parser.add_argument('--glon', type=float, help='Geographic longitude in degrees.', default=-71.2, required=False)
    parser.add_argument('--Q', type=float, help='Flux in erg cm^-2 s^-1.', default=1, required=False)
    parser.add_argument('--Echar', type=float, help='Characteristic energy in eV.', default=100e3, required=False)
    parser.add_argument('--Nbins', type=int, help='Number of energy bins.', default=250, required=False)
    parser.add_argument('--tec', type=float, help='Total electron content in TECU.', default=None, required=False)

    args = parser.parse_args()

    time = parse(args.time)
    glat = args.glat
    glon = args.glon
    # %% flux [erg cm-2 s-1 == mW m-2 s-1]
    Q = args.Q
    # %% characteristic energy [eV]
    Echar = args.Echar
    # %% Number of energy bins
    Nbins = args.Nbins
    tec = args.tec

    iono = maxwellian(time, glat, glon, Nbins, Q, Echar, tec=tec)

    ne = interpolate_nan(iono["NeIn"].values, inplace=False)

    print(f'TEC: {np.trapz(ne, iono.alt_km.values*1e5)*1e-12:.2f} TECU')
    # %% plots
    plot.precip(iono["precip"])
    plot.ver(iono)
    plot.density(iono)
    plot.temperature(iono)

    show()


def NoPrecipitation():
    parser = argparse.ArgumentParser(
        prog='No Electron Precipitation',
        description='Evaluate upper atmosphere VER for a given time and location.',
        epilog='Uses MSIS00 for atmosphere, IRI90 for ionosphere, GLOW for emissions.')

    parser.add_argument('--time', type=str, help='Time in ISO 8601 format.', default='2015-12-13T10:00:00', required=False)
    parser.add_argument('--glat', type=float, help='Geographic latitude in degrees.', default=42.6, required=False)
    parser.add_argument('--glon', type=float, help='Geographic longitude in degrees.', default=-71.2, required=False)
    parser.add_argument('--Nbins', type=int, help='Number of energy bins.', default=250, required=False)
    parser.add_argument('--tec', type=float, help='Total electron content in TECU.', default=None, required=False)

    args = parser.parse_args()

    time = parse(args.time)
    glat = args.glat
    glon = args.glon
    Nbins = args.Nbins
    tec = args.tec

    iono = no_precipitation(time, glat, glon, Nbins, tec=tec)

    ne = interpolate_nan(iono["NeIn"].values, inplace=False)

    print(f'TEC: {np.trapz(ne, iono.alt_km.values*1e5)*1e-12:.2f} TECU')

    plot.density(iono)

    plot.temperature(iono)

    plot.ver(iono)

    show()
