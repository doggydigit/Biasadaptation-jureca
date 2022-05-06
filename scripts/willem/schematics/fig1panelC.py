import numpy as np

from datarep import paths
from datarep.matplotlibsettings import *


def plot_schema(axes=None):

    mu0 = np.array([3.,0.])

    m0 = np.array([[1.,6.], [6.,5.]])
    cov0 = np.dot(m0, m0.T)

    mu1 = np.array([-13.,6.])

    m1 = np.array([[1.,-5.], [-5.,2.]])
    cov1 = np.dot(m1, m1.T)

    mu2 = np.array([-3.,20.])

    m2 = np.array([[-2.,-1.], [-1.,7.]])
    cov2 = np.dot(m2, m2.T)

    mu3 = np.array([12.,22.])

    m3 = np.array([[2.,3.], [3.,-1.]])
    cov3 = np.dot(m3, m3.T)

    xdata0 = np.random.multivariate_normal(mu0, cov0, size=200)
    xdata1 = np.random.multivariate_normal(mu1, cov1, size=200)
    xdata2 = np.random.multivariate_normal(mu2, cov2, size=200)
    xdata3 = np.random.multivariate_normal(mu3, cov3, size=200)

    if axes is None:
        pl.figure(figsize=(7,2.5))
        ax0 = pl.subplot(121)
        ax1 = pl.subplot(122, projection='3d')
        pshow = True
    else:
        ax0 = axes[0]
        ax1 = axes[1]
        pshow = False

    ax0 = noFrameAx(ax0)

    ax0.scatter(xdata0[:,0], xdata0[:,1], c='DarkGrey', s=markersize/2.)
    ax0.scatter(xdata1[:,0], xdata1[:,1], c='DarkGrey', s=markersize/2.)
    ax0.scatter(xdata2[:,0], xdata2[:,1], c='DarkGrey', s=markersize/2.)
    ax0.scatter(xdata3[:,0], xdata3[:,1], c='DarkGrey', s=markersize/2.)

    b0 = [[-18.,35.], [-4.,4.], [4.,12.], [20.,9.]]
    b0 = [np.array(p_) for p_ in b0]
    s0 = 1.

    b1 = [[-17.,-12.], [2.,8.], [4.,35.]]
    b1 = [np.array(p_) for p_ in b1]
    s1 = -1

    bs = [b0, b1]
    ss = [s0, s1]
    # cs = [colours[0], colours[4]]
    csn = ['darkmagenta', 'darkblue']

    for jj, (b_, s_, c_) in enumerate(zip(bs, ss, csn)):

        prop = dict(arrowstyle="<|-,head_width=0.2,head_length=0.4", ec=c_, fc=c_, shrinkA=0, shrinkB=0)

        for ii, (p0, p1) in enumerate(zip(b_[:-1], b_[1:])):
            if ii == 0:
                ax0.plot([p0[0], p1[0]], [p0[1], p1[1]], ls='--', c=c_, lw=lwidth, label=r"Task %d"%(jj+1))
            else:
                ax0.plot([p0[0], p1[0]], [p0[1], p1[1]], ls='--', c=c_, lw=lwidth)

            w0 = (p0 + p1) / 2.
            wvec = p1 - p0
            w_ = s_ * np.array([-wvec[1], wvec[0]])
            w_ /= np.linalg.norm(w_)
            w_ *= 6.

            # ax0.annotate("", xy=w0, xytext=w0+w_, arrowprops=prop)
            w1 = w0 + w_
            arrow = mpatches.FancyArrowPatch(w0, w1,
                            lw=lwidth, arrowstyle="-|>", color=c_,
                            shrinkA=0, shrinkB=0, mutation_scale=10, zorder=10000)
            ax0.add_artist(arrow)
            txt = ax0.annotate(r"$\mathbf{w}_{\perp}$", xy=w0, xytext=w0+w_, fontsize=labelsize*1.2)
            txt.set_path_effects([patheffects.withStroke(foreground="w", linewidth=2)])

    myLegend(ax0, loc='upper left', bbox_to_anchor=(0.7,0.3), fontsize=labelsize*1.2, handlelength=.6, handletextpad=.2, add_frame=False)

    ax0.set_ylim((-20.,40.))
    ax0.set_xlim((-35.,25.))
    ax0.set_aspect(1.)

    # transform for 3d data
    transf = lambda x: 2.5*x[:,0] + 3.*x[:,1]

    ax1.scatter(xdata0[:,0], xdata0[:,1], transf(xdata0), c='DarkGrey', s=markersize/8., zorder=-10000, alpha=1)
    ax1.scatter(xdata1[:,0], xdata1[:,1], transf(xdata1), c='DarkGrey', s=markersize/8., zorder=-10000, alpha=1)
    ax1.scatter(xdata2[:,0], xdata2[:,1], transf(xdata2), c='DarkGrey', s=markersize/8., zorder=-10000, alpha=1)
    ax1.scatter(xdata3[:,0], xdata3[:,1], transf(xdata3), c='DarkGrey', s=markersize/8., zorder=-10000, alpha=1)

    b2 = [[-35., -30.], [35.,-30.], [35.,40.], [-35.,40.], [-35.,-30.]]
    b2 = [np.array(p_) for p_ in b2]
    for p0, p1 in zip(b2[:-1], b2[1:]):
            z0, z1 = transf(p0[None,:])[0], transf(p1[None,:])[0]
            ax1.plot([p0[0], p1[0]], [p0[1], p1[1]], [z0, z1], c='k', lw=.6*lwidth, zorder=9999)

    w_0 = [[17.,26.], [-19.,13.], [0., -10.]]
    w_1 = [[10., 4.], [-4., 30.], [2.,  15. ]]
    p1_bad = [[10., 4., 400.], [-4.,30.,350.], [2.,15.,-350.]]

    for ii, (p0, p1, p1b) in enumerate(zip(w_0, w_1, p1_bad)):
        z0, z1 = transf(np.array([p0]))[0], transf(np.array([p1]))[0]

        p0_ = np.array(p0 + [z0])
        p1_ = np.array(p1 + [z1])

        arrow = Arrow3D([p0_[0], p1_[0]], [p0_[1], p1_[1]], [p0_[2], p1_[2]],
                        lw=lwidth, arrowstyle="-|>", color="b",
                        shrinkA=0, shrinkB=0, mutation_scale=10, zorder=10000)
        ax1.add_artist(arrow)

        zo = 9998 if ii == 2 else 10000

        arrow = Arrow3D([p0_[0], p1b[0]], [p0_[1], p1b[1]], [p0_[2], p1b[2]],
                        lw=lwidth, arrowstyle="-|>", color="r",
                        shrinkA=0, shrinkB=0, mutation_scale=10, zorder=zo)
        ax1.add_artist(arrow)
        if ii == 0:
            txt1 = ax1.text(*p1b, r"$\mathbf{w}_{\perp}$", fontsize=labelsize*1.2, ha='left', va='bottom', color='r')
            txt2 = ax1.text(*p1_, r"$\mathbf{w}_{\perp}$", fontsize=labelsize*1.2, ha='left', va='top', color='b')
            txt1.set_path_effects([patheffects.withStroke(foreground="w", linewidth=2)])
            txt2.set_path_effects([patheffects.withStroke(foreground="w", linewidth=2)])



    ax1.set_zlim((-400., 600.))
    ax1.view_init(10., -73.)

    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])

    if pshow:
        pl.show()


def plot_panel():

    xcoords = getXCoords([0.05, .8, 0.05, 1.2, 0.05])

    fig = pl.figure('W_perp schema', figsize=(7,3))

    gs0 = GridSpec(1,1)
    gs0.update(top=0.95, bottom=0.05, left=xcoords[0], right=xcoords[1])
    gs1 = GridSpec(1,1)
    gs1.update(top=0.99, bottom=0.01, left=xcoords[2], right=xcoords[3])

    ax0 = pl.subplot(gs0[0,0])
    ax1 = pl.subplot(gs1[0,0], projection='3d')

    plot_schema(axes=[ax0, ax1])

    pl.savefig(paths.fig_path + "biasadapatation_fig1C.svg", transparent=True)

    pl.show()


if __name__ == "__main__":
    # plot_schema()
    plot_panel()