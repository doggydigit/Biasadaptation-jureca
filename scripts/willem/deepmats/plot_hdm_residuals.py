import numpy as np

import os

from datarep.matplotlibsettings import *

NHS = [10, 25, 50, 100, 250, 500]
NH = 100
DATAPATH = "/Users/wybo/Data/weight_matrices/"
DATASET = "EMNIST"
ALGO1 = "pmdd"
ALGO2 = "scdfs"
PVAR = "err"


def plot_residuals(ax, dname, plot_variable="err"):
    try:
        data = np.load(os.path.join(DATAPATH, dname))

        if plot_variable == "err":
            pdata = data["errs"]
        elif plot_variable == "res":
            pdata = data["res"]
        else:
            raise IOError("unknown plot variable")

        ax.plot(np.arange(len(pdata)), pdata, "bD--", ms=markersize/2, lw=lwidth)

    except FileNotFoundError as e:
        print(e)


pl.figure("%s %s-%s"%(PVAR, ALGO1, ALGO2), figsize=(18,10))
gs = GridSpec(3, len(NHS))
gs.update(top=0.95, bottom=0.15, left=0.1, right=0.95, hspace=0.3, wspace=0.4)


for ii, nh in enumerate(NHS):
    ax = pl.subplot(gs[0,ii])

    ax.set_title("nh1 = %d, nh2 = %d"%(NH, nh), fontsize=legendsize)
    plot_residuals(ax,
                   "%s_ha_algos12=%s-%s_nh12=%d-%d_extra_data.npz"%(DATASET, ALGO1, ALGO2, NH, nh),
                   plot_variable=PVAR)

    ax = pl.subplot(gs[1,ii])
    ax.set_title("nh1 = %d, nh2 = %d"%(nh, NH), fontsize=legendsize)
    plot_residuals(ax,
                   "%s_ha_algos12=%s-%s_nh12=%d-%d_extra_data.npz"%(DATASET, ALGO1, ALGO2, nh, NH),
                   plot_variable=PVAR)

    ax = pl.subplot(gs[2,ii])
    ax.set_title("nh1 = %d, nh2 = %d"%(nh, nh), fontsize=legendsize)
    plot_residuals(ax,
                   "%s_ha_algos12=%s-%s_nh12=%d-%d_extra_data.npz"%(DATASET, ALGO1, ALGO2, nh, nh),
                   plot_variable=PVAR)

pl.show()