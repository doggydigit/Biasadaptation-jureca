import numpy as np

import os
import pickle

from datarep.matplotlibsettings import *
import datarep.paths as paths


NHS = [10, 25, 50, 100, 250, 500]
NSAMPLES = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500]
METHODS = ["pmdd", "bpo", "rp"]
LABELS = ["PMD", "MTB", "RP"]

DATAPATH = "/Users/wybo/Data/results/biasopt/"
DATASET = "EMNIST"
TASKTYPE = "1vall"
READOUT = "tanh"
PIND = [8,7,6]


def perfAx(ax, add_xticklabels=True, add_xlabel=True, add_yticklabels=True, add_ylabel=True):
    ax = myAx(ax)
    rect = Rectangle((0., 95.), len(NSAMPLES), 5., color='DarkGrey', alpha=.2)
    ax.add_patch(rect)
    ax.axhline(90, ls='--', lw=lwidth/3., c='DarkGrey')
    ax.axhline(80, ls='--', lw=lwidth/3., c='DarkGrey')
    ax.axhline(70, ls='--', lw=lwidth/3., c='DarkGrey')
    ax.axhline(60, ls='--', lw=lwidth/3., c='DarkGrey')

    ax.set_ylim((50., 100.))
    ax.set_xlim((0., len(NSAMPLES)))
    ax.set_yticks([50.,60., 70.,80.,90.,100.])
    ax.set_xticks(np.arange(len(NSAMPLES))+.5)

    if add_xlabel:
        ax.set_xlabel("no. of datapoints", fontsize=labelsize)

    if add_xticklabels:
        ax.set_xticklabels(NSAMPLES, rotation=60, fontsize=12)
    else:
        ax.set_xticklabels([])

    if add_ylabel:
        ax.set_ylabel(r'% correct', fontsize=labelsize)

    if add_yticklabels:
        ax.set_yticklabels(["50", "", "70", "", "90", ""], fontsize=ticksize)
    else:
        ax.set_yticklabels([])

    return ax


def plot_data(axes=None, eps=.05):

    if axes is None:
        pl.figure(figsize=(14,3))
        gs = GridSpec(1,len(NHS))
        axes = [pl.subplot(gs[0,ii]) for ii in range(len(NHS))]
        pshow = True
    else:
        pshow = False


    handles = []
    for ii, n_h in enumerate(NHS):


        ax = perfAx(axes[ii],
                    add_xticklabels=True,
                    add_yticklabels=ii==0,
                    add_xlabel=ii==2,
                    add_ylabel=ii==0)

        for jj, algo in enumerate(METHODS):

            perfs_avg_bias = []
            perfs_avg_full = []
            perfs_std_bias = []
            perfs_std_full = []
            for n_sample in NSAMPLES:
                fname = os.path.join(DATAPATH, "fewopt_1hl_%s_%s%d_%s_nsample=%d_ro=%s.p"%(DATASET, algo, n_h, TASKTYPE, n_sample, READOUT))
                print(fname)

                try:
                    with open(fname, 'rb') as file:
                        reslist_bias = pickle.load(file)
                        reslist_full = pickle.load(file)
                    # print([res['perf']['test'] for res in reslist])
                    perf_avg_bias = np.mean([res['perf']['test']['all'] for res in reslist_bias])
                    perf_avg_full = np.mean([res['perf']['test']['all'] for res in reslist_full])
                    # perf_avg_bias = np.median([res['perf']['test']['all'] for res in reslist_bias])
                    # perf_avg_full = np.median([res['perf']['test']['all'] for res in reslist_full])
                    perf_std_bias = np.std([res['perf']['test']['all'] for res in reslist_bias])
                    perf_std_full = np.std([res['perf']['test']['all'] for res in reslist_full])
                except FileNotFoundError:
                    perf_avg_bias = np.nan
                    perf_avg_full = np.nan
                    perf_std_bias = np.nan
                    perf_std_full = np.nan

                perfs_avg_bias.append(perf_avg_bias)
                perfs_avg_full.append(perf_avg_full)
                perfs_std_bias.append(perf_std_bias)
                perfs_std_full.append(perf_std_full)


            kk = PIND[jj]
            ax.errorbar(np.arange(len(NSAMPLES))+0.5+jj*eps, perfs_avg_bias, yerr=perfs_std_bias,
                ls='--', marker=mfs[kk%len(mfs)], c=colours[kk%len(colours)], lw=lwidth*.8, ms=markersize*.8)
            if ii == 0:
                handle = mlines.Line2D([], [],
                            marker=mfs[kk%len(mfs)], c=colours[kk%len(colours)], lw=lwidth*.8, ms=markersize*.8,
                            label=LABELS[jj])
                handles.append(handle)

            if jj == 0:
                kk = 0
                ax.errorbar(np.arange(len(NSAMPLES))+0.5-eps, perfs_avg_full, yerr=perfs_std_full,
                    ls='--', marker=mfs[kk%len(mfs)], c=colours[kk%len(colours)], lw=lwidth*.8, ms=markersize*.8, zorder=-1)
                if ii == 0:
                    handle = mlines.Line2D([], [],
                                marker=mfs[kk%len(mfs)], c=colours[kk%len(colours)], lw=lwidth*.8, ms=markersize*.8,
                                label="FT")
                    handles.append(handle)


    myLegend(ax, handles=handles, loc='center left', bbox_to_anchor=(.92,0.5), fontsize=labelsize, handlelength=2., numpoints=2, labelspacing=2.)


    if pshow: pl.show()


def plot_figure():
    pl.figure("fewopt", figsize=(14,3.5))
    gs0 = GridSpec(1,6)
    gs0.update(top=0.95, bottom=0.4, left=0.05, right=0.90, hspace=0.1, wspace=0.1)
    gs1 = GridSpec(1,1)
    gs1.update(top=0.19, bottom=0.18, left=0.05, right=0.90, hspace=0.1, wspace=0.1)

    axes = [pl.subplot(gs0[0,ii]) for ii in range(6)]

    plot_data(axes)

    axa = pl.subplot(gs1[0,0])

    axa.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    axa.spines['right'].set_color('none')
    axa.spines['left'].set_color('none')
    # ax.draw_frame = False

    axa.set_yticks([])
    xdata = (np.arange(6)+.5) / 6.
    axa.set_xlim((0.,1.))
    axa.set_xticks(xdata)
    axa.set_xticklabels([str(n_h) for n_h in NHS])
    axa.set_xlabel("no. of hidden units", fontsize=labelsize)

    pl.savefig(paths.fig_path + "biasadaptation_fig3.svg", transparent=True)

    pl.show()


if __name__ == "__main__":
    # plot_data()
    plot_figure()

