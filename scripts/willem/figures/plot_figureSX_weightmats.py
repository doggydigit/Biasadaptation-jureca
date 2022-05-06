import numpy as np

import os
import pickle
import sys
sys.path.append('..')

from datarep.matplotlibsettings import *
import datarep.paths as paths

from biasadaptation.utils import utils

import figtools, helperfuncs


GRID_DICT={10: [2,5],
           25: [5,5],
           50: [5,10],
           100: [10,10],
           250: [10,25],
           500: [20,25],
           }


def plot_weightmats(nh, algo, axes, axt, ncol):
    axt = noFrameAx(axt)
    axt.text(0.5, 0.5, figtools.LABELS[algo][0], fontsize=labelsize*1.2, ha='center', va='center')

    w_mat = helperfuncs.get_weight_matrix_in(nh, algo, dataset="EMNIST").T

    for ii, w_vec in enumerate(w_mat):
        jj = ii%ncol
        kk = ii//ncol

        ax = noFrameAx(axes[jj][kk])
        ax.imshow(utils.to_image_mnist(w_vec).T)



def plot_figure(nh=100):
    nrow, ncol = GRID_DICT[nh]

    pl.figure("weightmats", figsize=(14,14))

    xcoo = getXCoords([0.1, 1.0, 0.1, 1.0, 0.1])
    ycoo = getYCoords([0.1, 1.0, 0.001,0.1,0.1, 1.0, 0.001,0.1,0.05])

    # PMD matrix
    gst = GridSpec(1,1)
    gst.update(top=ycoo[-1], bottom=ycoo[-2], left=xcoo[0], right=xcoo[1], hspace=0.1, wspace=0.1)
    gs0 = GridSpec(nrow, ncol)
    gs0.update(top=ycoo[-3], bottom=ycoo[-4], left=xcoo[0], right=xcoo[1], hspace=0.1, wspace=0.1)

    axt = pl.subplot(gst[0,0])
    axes = [[pl.subplot(gs0[ii,jj]) for jj in range(ncol)] for ii in range(nrow)]
    plot_weightmats(nh, "pmdd", axes, axt, ncol)

    # PCA matrix
    gst = GridSpec(1,1)
    gst.update(top=ycoo[-1], bottom=ycoo[-2], left=xcoo[2], right=xcoo[3], hspace=0.1, wspace=0.1)
    gs0 = GridSpec(nrow, ncol)
    gs0.update(top=ycoo[-3], bottom=ycoo[-4], left=xcoo[2], right=xcoo[3], hspace=0.1, wspace=0.1)

    axt = pl.subplot(gst[0,0])
    axes = [[pl.subplot(gs0[ii,jj]) for jj in range(ncol)] for ii in range(nrow)]
    plot_weightmats(nh, "pca", axes, axt, ncol)

    # SCD matrix
    gst = GridSpec(1,1)
    gst.update(top=ycoo[-5], bottom=ycoo[-6], left=xcoo[0], right=xcoo[1], hspace=0.1, wspace=0.1)
    gs0 = GridSpec(nrow, ncol)
    gs0.update(top=ycoo[-7], bottom=ycoo[-8], left=xcoo[0], right=xcoo[1], hspace=0.1, wspace=0.1)

    axt = pl.subplot(gst[0,0])
    axes = [[pl.subplot(gs0[ii,jj]) for jj in range(ncol)] for ii in range(nrow)]
    plot_weightmats(nh, "scd", axes, axt, ncol)

    # SC matrix
    gst = GridSpec(1,1)
    gst.update(top=ycoo[-5], bottom=ycoo[-6], left=xcoo[2], right=xcoo[3], hspace=0.1, wspace=0.1)
    gs0 = GridSpec(nrow, ncol)
    gs0.update(top=ycoo[-7], bottom=ycoo[-8], left=xcoo[2], right=xcoo[3], hspace=0.1, wspace=0.1)

    axt = pl.subplot(gst[0,0])
    axes = [[pl.subplot(gs0[ii,jj]) for jj in range(ncol)] for ii in range(nrow)]
    plot_weightmats(nh, "sc", axes, axt, ncol)

    pl.savefig(paths.fig_path + "biasadaptation_v3_figSX.svg", transparent=True)

    pl.show()


if __name__ == "__main__":
    plot_figure()