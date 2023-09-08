from .base import maxwellian, no_precipitation
from . import plots as plot
from datetime import datetime
from matplotlib.pyplot import show

def Maxwellian():
    time = datetime(2015, 12, 13, 10, 0, 0)
    glat = 65.1
    glon = -147.5
    # %% flux [erg cm-2 s-1 == mW m-2 s-1]
    Q = 1
    # %% characteristic energy [eV]
    Echar = 100e3
    # %% Number of energy bins
    Nbins = 250

    iono = maxwellian(time, glat, glon, Nbins, Q, Echar)
    # %% plots
    plot.precip(iono["precip"])
    plot.ver(iono)
    plot.density(iono)
    plot.temperature(iono)

    show()

def NoPrecipitation():
    time = datetime(2015, 12, 13, 10, 0, 0)
    glat = 65.1
    glon = -147.5
    
    Nbins = 250

    iono = no_precipitation(time, glat, glon, Nbins)

    plot.density(iono)

    plot.temperature(iono)

    plot.ver(iono)

    show()
