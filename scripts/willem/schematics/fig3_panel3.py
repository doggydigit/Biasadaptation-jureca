import numpy as np

from datarep import paths
from datarep.matplotlibsettings import *


def plot_schema(axes=None):

    mu0 = np.array([8.,2.])

    m0 = np.array([[6.,2.], [2.,3.]])
    cov0 = np.dot(m0, m0.T)

    mu1 = np.array([2.,7.])

    m1 = np.array([[2.,1.], [1.,6.]])
    cov1 = np.dot(m1, m1.T)

    xdata0 = np.random.multivariate_normal(mu0, cov0, size=200)
    xdata1 = np.random.multivariate_normal(mu1, cov1, size=200)

    pl.figure(figsize=(3,3))
    ax = noFrameAx(pl.gca())

    ax.scatter(xdata0[:,0], xdata0[:,1], c='DarkGrey', s=markersize/2.)
    ax.scatter(xdata1[:,0], xdata1[:,1], c='DarkGrey', s=markersize/2.)

    ax.set_xlim((-10,30))
    ax.set_ylim((-10,30))

    pl.show()


if __name__ == "__main__":
    plot_schema()
    # plot_panel()