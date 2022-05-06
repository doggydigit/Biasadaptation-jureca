import numpy as np

import os
import pickle

from datarep.matplotlibsettings import *
import datarep.paths as paths


NHS = [10, 25, 50, 100, 250, 500]
METHODS = [["rpw", "br", "code"], ["sc", "pca", "rp"], ["bpo", "scd", "pmdd"]]
DATAPATH = "/Users/wybo/Data/results/biasopt/"
DATASET = "EMNIST"
TASKTYPE = "1vall"
READOUT = "tanh"


LABELS = {"rpw": ["FT", "None", "NA", r"$W_1$, $\mathbf{b}_1$,"+ "\n$\mathbf{w}_o$, $b_o$", "gradient descent \nsingle task"],
          "br": ["MTRU", r"$W_1$, $\mathbf{b}_1$", "gradient descent \nmulti-task", r"$\mathbf{w}_o$, $b_o$", "gradient descent\nsingle task"],
          "code": ["SCRU", r"$W_1$", "sparse dict code", r"$\mathbf{w}_o$, $b_o$", "gradient descent\nsingle task"],
          "sc": ["SDB", r"$W_1$, $\mathbf{w}_o$", r"$W_1$ sparse dict " + "\n$\mathbf{w}_o$ uniform", r"$\mathbf{b}_1$, $b_o$", "gradient descent\nsingle task"],
          "pca": ["PCA", r"$W_1$, $\mathbf{w}_o$", r"$W_1$ PCA" + "\n$\mathbf{w}_o$ uniform", r"$\mathbf{b}_1$, $b_o$", "gradient descent\nsingle task"],
          "rp": ["RP", r"$W_1$, $\mathbf{w}_o$", r"$W_1$ random" + "\n$\mathbf{w}_o$ uniform", r"$\mathbf{b}_1$, $b_o$", "gradient descent\nsingle task"],
          "scd": ["SDDB", r"$W_1$, $\mathbf{w}_o$", r"$W_1$ sparse dict $\Delta \mathbf{x}$" + "\n$\mathbf{w}_o$ uniform", r"$\mathbf{b}_1$, $b_o$", "gradient descent\nsingle task"],
          "pmdd": ["PMD", r"$W_1$, $\mathbf{w}_o$", r"$W_1$ " + "penalized matrix \n" + r"      decomposition $\Delta \mathbf{x}$" + "\n$\mathbf{w}_o$ uniform", r"$\mathbf{b}_1$, $b_o$", "gradient descent\nsingle task"],
          "bpo": ["MTB", r"$W_1$, $\mathbf{w}_o$", "gradient descent \nmulti-task", r"$\mathbf{b}_1$, $b_o$", "gradient descent\nsingle task"],
         }


def perfAx(ax, add_xticklabels=True, add_xlabel=True, add_yticklabels=True, add_ylabel=True):
    ax = myAx(ax)
    rect = Rectangle((0., 95.), len(NHS), 5., color='DarkGrey', alpha=.2)
    ax.add_patch(rect)
    ax.axhline(90, ls='--', lw=lwidth/3., c='DarkGrey')
    ax.axhline(80, ls='--', lw=lwidth/3., c='DarkGrey')
    ax.axhline(70, ls='--', lw=lwidth/3., c='DarkGrey')
    ax.axhline(60, ls='--', lw=lwidth/3., c='DarkGrey')

    ax.set_ylim((50., 100.))
    ax.set_xlim((0., len(NHS)))
    ax.set_yticks([50.,60., 70.,80.,90.,100.])
    ax.set_xticks(np.arange(len(NHS))+.5)


    if add_xlabel:
        ax.set_xlabel("no. of hidden units", fontsize=labelsize)

    if add_xticklabels:
        ax.set_xticklabels(NHS, rotation=60)
    else:
        ax.set_xticklabels([])

    if add_ylabel:
        ax.set_ylabel(r'% correct', fontsize=labelsize)

    if add_yticklabels:
        ax.set_yticklabels(["50", "", "70", "", "90", ""])
    else:
        ax.set_yticklabels([])

    return ax


def plot_data(axes=None):

    if axes is None:
        pl.figure()
        gs = pl.GridSpec(3,3)
        axes = [[pl.subplot(gs[jj,ii]) for ii in range(3)] for jj in range(3)]
        pshow = True
    else:
        pshow = False

    kk = 0
    lines, handles = [], []
    for jj, algos in enumerate(METHODS):
        for ii, algo in enumerate(algos):

            perfs_avg = []
            perfs_std = []
            for n_h in NHS:
                print(os.path.join(DATAPATH, "biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(DATASET, algo, n_h, TASKTYPE, READOUT)))

                if algo == 'code':
                    fname = os.path.join(DATAPATH, "codeopt_%s_algow-c=%s-%s_nh=%d_%s__ro=%s.p"%(DATASET, 'sc', 'sc', n_h, TASKTYPE, READOUT))
                else:
                    fname = os.path.join(DATAPATH, "biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(DATASET, algo, n_h, TASKTYPE, READOUT))

                try:
                    with open(fname, 'rb') as file:
                        reslist = pickle.load(file)

                    perf_avg = np.mean([res['perf']['test']['all'] for res in reslist])
                    perf_std = np.std([res['perf']['test']['all'] for res in reslist])

                except FileNotFoundError:
                    perf_avg = np.nan
                    perf_std = np.nan

                perfs_avg.append(perf_avg)
                perfs_std.append(perf_std)

            ax = perfAx(axes[ii][jj],
                        add_xticklabels=ii==len(algos)-1,
                        add_yticklabels=jj==0,
                        add_xlabel=ii==len(algos)-1 and jj==1,
                        add_ylabel=jj==0 and ii==1)

            ax.errorbar(np.arange(len(NHS))+0.5, perfs_avg, yerr=perfs_std,
                ls='--', marker=mfs[kk%len(mfs)], c=colours[kk%len(colours)], lw=lwidth*.8, ms=markersize*.8)

            handle = mlines.Line2D([], [],
                        marker=mfs[kk%len(mfs)], c=colours[kk%len(colours)], lw=lwidth*.8, ms=markersize*.8,
                        label=LABELS[algo])
            handles.append(handle)

            kk += 1

    if pshow:
        pl.show()


def plot_table(ax=None, fontsize=11):

    if ax is None:
        pl.figure()
        ax = noFrameAx(pl.gca())
        pshow = True
    else:
        pshow = False

    yborders = 0

    yborders = np.arange(11.)
    yborders[1:] += .5
    yborders[-1] += .5

    xborders = np.arange(6.)
    xborders[0] = .5
    xborders[2] -= .5
    xborders[3] -= .6
    xborders[5] -= .2

    yvals = (yborders[1:] + yborders[:-1]) / 2.
    xvals = (xborders[1:] + xborders[:-1]) / 2.

    ax.set_ylim((np.min(yborders), np.max(yborders)))
    ax.set_xlim((np.min(xborders), np.max(xborders)))

    ax.axvline(xborders[1], c='DarkGrey', lw=lwidth/2.)
    ax.axvline(xborders[2], c='k', lw=lwidth/2.)
    ax.axvline(xborders[3], c='DarkGrey', lw=lwidth/4.)
    ax.axvline(xborders[4], c='k', lw=lwidth/2.)
    ax.axvline(xborders[5], c='k', lw=lwidth/4.)
    # ax.axvline(xborders[5], c='k', lw=lwidth/2.)

    for yborder in yborders[:-1]:
        ax.axhline(yborder, c='k', lw=lwidth/2.)


    text_xloc = lambda ii : xvals[ii] - (xborders[ii+1] - xborders[ii])*.475

    ax.text(text_xloc(2), yvals[-1], "Params \nshared \nacross tasks", fontsize=fontsize, ha="left", va="center")
    ax.text(text_xloc(3), yvals[-1], "Training \nmethod", fontsize=fontsize, ha="left", va="center")
    ax.text(text_xloc(4), yvals[-1], "Task-\nspecific \nparams", fontsize=fontsize, ha="left", va="center")
    # ax.text(text_xloc(4), yvals[-1], "Training \nmethod", fontsize=fontsize, ha="left", va="center")


    algos = [algo for methods in METHODS for algo in methods]

    for kk in range(9):
        ax.plot([xvals[0]-.15,xvals[0]+.15], [yvals[-2-kk], yvals[-2-kk]],
            ls='--', marker=mfs[kk%len(mfs)], c=colours[kk%len(colours)], lw=lwidth, ms=markersize)

        ax.text(text_xloc(1), yvals[-2-kk], LABELS[algos[kk]][0], fontsize=fontsize, ha="left", va="center")
        ax.text(text_xloc(2), yvals[-2-kk], LABELS[algos[kk]][1], fontsize=fontsize, ha="left", va="center")
        ax.text(text_xloc(3), yvals[-2-kk], LABELS[algos[kk]][2], fontsize=fontsize, ha="left", va="center")
        ax.text(text_xloc(4), yvals[-2-kk], LABELS[algos[kk]][3], fontsize=fontsize, ha="left", va="center")

    if pshow:
        pl.show()


def plot_figure():
    pl.figure('perfs 1hl', figsize=(14,7))

    gs = GridSpec(3,3)
    gs.update(top=0.98, bottom=0.12, left=0.07, right=0.43, hspace=0.1, wspace=0.1)
    axes = [[pl.subplot(gs[jj,ii]) for ii in range(3)] for jj in range(3)]

    gst = GridSpec(1,1)
    gst.update(top=0.98, bottom=0.02, left=0.45, right=0.99)
    axt = noFrameAx(pl.subplot(gst[0,0]))

    plot_data(axes)
    plot_table(axt, fontsize=labelsize)

    pl.savefig(paths.fig_path + "biasadaptation_fig2.svg", transparent=True)

    pl.show()


if __name__ == "__main__":
    # plot_data()
    # plot_table()
    plot_figure()
