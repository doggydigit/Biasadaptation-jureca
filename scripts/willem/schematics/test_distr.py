import numpy as np

from datarep import paths
from datarep.matplotlibsettings import *


def plot_schema(axes=None, n_samples=200):


    mu0 = np.array([3.,0.])

    m0 = np.array([[1.,6.], [6.,5.]])
    cov0 = np.dot(m0, m0.T)

    mu1 = np.array([-13.,6.])

    m1 = np.array([[1.,-5.], [-5.,2.]])
    cov1 = np.dot(m1, m1.T)

    mu2 = np.array([-3.,20.])

    m2 = np.array([[-7.,-.1], [-.1,1.]])
    cov2 = np.dot(m2, m2.T)

    # m2 = np.array([[-7.,-7.], [-7.,7.]])
    # cov2 = np.dot(m2, m2.T)

    mu3 = np.array([12.,22.])

    m3 = np.array([[2.,3.], [3.,-1.]])
    cov3 = np.dot(m3, m3.T)

    # xdata = np.random.multivariate_normal(mu0, cov0, size=n_samples)
    # xdata = np.random.multivariate_normal(mu1, cov1, size=n_samples)
    xdata = np.random.multivariate_normal(mu2, cov0, size=n_samples)
    # xdata = np.random.multivariate_normal(mu3, cov3, size=n_samples)

    if axes is None:
        pl.figure(figsize=(12, 7))
        ax0 = pl.subplot(121)
        ax1 = pl.subplot(122)
        pshow = True
    else:
        ax0 = axes[0]
        ax1 = axes[1]
        pshow = False

    # ax0 = noFrameAx(ax0)

    # ax0.scatter(xdata0[:,0], xdata0[:,1], c='DarkGrey', s=markersize/2.)
    ax0.scatter(xdata[:,0], xdata[:,1], c='k', s=markersize/2.)
    # ax0.scatter(xdata2[:,0], xdata2[:,1], c='DarkGrey', s=markersize/2.)
    # ax0.scatter(xdata3[:,0], xdata3[:,1], c='DarkGrey', s=markersize/2.)

    ax0.set_xlim((-20.,20.))
    ax0.set_ylim((-10.,40.))

    xdiff = np.zeros((int(n_samples*(n_samples - 1.)), 2))
    kk = 0
    for ii in range(n_samples):
        for jj in range(n_samples):
            if ii != jj:
                xdiff[kk] = xdata[ii] - xdata[jj]
                xdiff[kk] /= np.linalg.norm(xdiff[kk])
                kk += 1

    xangles = np.arccos(xdiff[:,0])
    ax1.hist(xangles, bins=100, range=(0.,np.pi))
    # ax1.set_ylim((-.1,100.1))

    if pshow:
        pl.show()



if __name__ == "__main__":
    plot_schema()
    # plot_panel()