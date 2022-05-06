import numpy as np
from datarep.matplotlibsettings import *

phi0 = 0.4
phi1 = 1.8

phi2 = np.linspace(0., 2*np.pi, 1000)

def func(phi):
    return - np.cos(phi2 - phi0)**2 - np.cos(phi2 - phi1)**2

pl.axvline(phi0)
pl.axvline(phi1)

pl.plot(phi2, func(phi2))

pl.show()