import numpy as np

import copy

from datarep.matplotlibsettings import *

def cel(y, t=1, r=5.):
    t = (t+1)/2
    s = 1. / (1. + np.exp(-r*y))
    return t * np.log(1. + np.exp(-r*y)) / r + (1. - t) * np.log(1. + np.exp(r*y)) / r

def hardtanh(x):
    x = copy.deepcopy(x)
    x[x>1] = 1
    x[x<-1] = -1
    return x

def hmsel(y, t=1):
    s = hardtanh(y)
    return (s - t)**2 / 4.
    # return np.abs(s-t)

def ll(y, t=1):
    return np.maximum(np.zeros_like(y), -t*y)


def fff(y):
    return np.log((1.+np.exp(-10*y))) / 10.

y = np.linspace(-10, 10, 1000)

print(np.log(2))

pl.figure()
ax = pl.gca()

ax.plot(y, cel(y, t=1), 'r', label='t=1')
ax.plot(y, cel(y, t=-1), 'b', label='t=-1')

ax.plot(y, hmsel(y, t=1), 'r--', label='t=1')
ax.plot(y, hmsel(y, t=-1), 'b--', label='t=-1')

ax.plot(y, ll(y, t=1), 'r:', label='t=1')
ax.plot(y, ll(y, t=-1), 'b:', label='t=-1')

ax.plot(y, fff(y), 'g')

ax.set_ylim((-0.1,10.))
myLegend(ax, loc=0)

pl.show()