import numpy as np

import os
import pickle

from datarep.matplotlibsettings import *
import datarep.paths as paths


NHS = [10, 25, 50, 100, 250, 500]
NH = 100
ALGO_12C = [["pmdd", "pmd", "lstsq"],
            ["pmdd", "bpo", "na"],
           ]
DATAPATH = "/Users/wybo/Data/results/biasopt/"
DATASET = "EMNIST"
TASKTYPE = "1vall"
READOUT = "tanh"
PIND = [8]



def perfAx(ax, add_xticklabels=True, xlabel=None, add_yticklabels=True, add_ylabel=True):
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

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=labelsize)

    if add_xticklabels:
        ax.set_xticklabels(NHS, rotation=60, fontsize=12)
    else:
        ax.set_xticklabels([])

    if add_ylabel:
        ax.set_ylabel(r'% correct', fontsize=labelsize)

    if add_yticklabels:
        ax.set_yticklabels(["50", "", "70", "", "90", ""], fontsize=ticksize)
    else:
        ax.set_yticklabels([])

    return ax


def load_perf(namestring):
    try:
        with open(os.path.join(DATAPATH, namestring), 'rb') as file:
            reslist = pickle.load(file)
        perf_avg = np.mean([res['perf']['test']['all'] for res in reslist])
        perf_std = np.std( [res['perf']['test']['all'] for res in reslist])
    except (FileNotFoundError, TypeError):
        perf = np.nan
        perf = np.nan

    return perf_avg, perf_std


def plot_data(axes=None, eps=.02):

    if axes is None:
        pl.figure(figsize=(10,7))
        gs = GridSpec(2,3)
        axes = [[pl.subplot(gs[jj,ii]) for ii in range(3)] for jj in range(2)]
        pshow = True
    else:
        pshow = False

    for ii in range(2):
        algo_12c = '-'.join(ALGO_12C[ii])
        algo_1 = ALGO_12C[ii][0]

        perfs_nhx_avg, perfs_nhx_std = [], []
        perfs_xnh_avg, perfs_xnh_std = [], []
        perfs_xx_avg,  perfs_xx_std  = [], []
        perfs_1hl_avg, perfs_1hl_std = [], []
        for n_h in NHS:
            namestring_nhx = "deepopt_2hl_%s_algo12c=%s_nh12=%d-%d_%s_ro=%s.p"%(DATASET, algo_12c, NH, n_h, TASKTYPE, READOUT)
            p_avg, p_std = load_perf(namestring_nhx)
            perfs_nhx_avg.append(p_avg)
            perfs_nhx_std.append(p_std)

            namestring_xnh = "deepopt_2hl_%s_algo12c=%s_nh12=%d-%d_%s_ro=%s.p"%(DATASET, algo_12c, n_h, NH, TASKTYPE, READOUT)
            p_avg, p_std = load_perf(namestring_xnh)
            perfs_xnh_avg.append(p_avg)
            perfs_xnh_std.append(p_std)

            namestring_xx = "deepopt_2hl_%s_algo12c=%s_nh12=%d-%d_%s_ro=%s.p"%(DATASET, algo_12c, n_h, n_h, TASKTYPE, READOUT)
            p_avg, p_std = load_perf(namestring_xx)
            perfs_xx_avg.append(p_avg)
            perfs_xx_std.append(p_std)

            namestring_1hl = "biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(DATASET, algo_1, n_h, TASKTYPE, READOUT)
            p_avg, p_std = load_perf(namestring_1hl)
            perfs_1hl_avg.append(p_avg)
            perfs_1hl_std.append(p_std)


        print(algo_12c)
        print('nhx:', perfs_nhx_avg)
        print('xnh:', perfs_xnh_avg)
        print('xx :', perfs_xx_avg)
        print('1hl:', perfs_1hl_avg)

        namestring_1hl = "biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(DATASET, algo_1, NH, TASKTYPE, READOUT)
        perf_1hl = load_perf(namestring_1hl)[0]

        # case xnh
        xlabel = None if ii == 0 else "no. of hidden units \n1st layer"
        ax = perfAx(axes[ii][0],
                    add_xticklabels=ii==1,
                    add_yticklabels=True,
                    xlabel=xlabel,
                    add_ylabel=True)

        if ii == 0:
            ax.set_title('no. of hidden units \n2nd layer = 100', fontsize=labelsize)

        ax.errorbar(np.arange(len(NHS))+0.5-eps, perfs_1hl_avg, yerr=perfs_1hl_std,
            ls='--', marker='o', c='k', lw=lwidth*.8, ms=markersize*.8)

        ax.errorbar(np.arange(len(NHS))+0.5+eps, perfs_xnh_avg, yerr=perfs_xnh_std,
            ls='--', marker=mfs[PIND[0]%len(mfs)], c=colours[PIND[0]%len(colours)], lw=lwidth, ms=markersize)

        # case nhx
        xlabel = None if ii == 0 else "no. of hidden units \n2nd layer"
        ax = perfAx(axes[ii][1],
                    add_xticklabels=ii==1,
                    add_yticklabels=False,
                    xlabel=xlabel,
                    add_ylabel=False)

        if ii == 0:
            ax.set_title('no. of hidden units \n1st layer = 100', fontsize=labelsize)

        ax.axhline(perf_1hl,
            ls='--', c='k', lw=lwidth*.8, ms=markersize*.8)

        ax.errorbar(np.arange(len(NHS))+0.5, perfs_nhx_avg, yerr=perfs_nhx_std,
            ls='--', marker=mfs[PIND[0]%len(mfs)], c=colours[PIND[0]%len(colours)], lw=lwidth, ms=markersize)

        # case xx
        xlabel = None if ii == 0 else "no. of hidden units \n1st & 2nd layers"
        ax = perfAx(axes[ii][2],
                    add_xticklabels=ii==1,
                    add_yticklabels=False,
                    xlabel=xlabel,
                    add_ylabel=False)

        if ii == 0:
            ax.set_title('no. of hidden units \n1st layer = 2nd layer', fontsize=labelsize)

        ax.errorbar(np.arange(len(NHS))+0.5+eps, perfs_1hl_avg, yerr=perfs_1hl_std,
            ls='--', marker='o', c='k', lw=lwidth*.8, ms=markersize*.8)

        ax.errorbar(np.arange(len(NHS))+0.5-eps, perfs_xx_avg, yerr=perfs_xx_std,
            ls='--', marker=mfs[PIND[0]%len(mfs)], c=colours[PIND[0]%len(colours)], lw=lwidth, ms=markersize)

        handle1 = mlines.Line2D([], [],
                    marker='o', c='k',  lw=lwidth*.8, ms=markersize*.8,
                    label="1 layer PMD")
        if ii == 0:
            handle2 = mlines.Line2D([], [],
                        marker=mfs[PIND[0]%len(mfs)], c=colours[PIND[0]%len(colours)], lw=lwidth*.8, ms=markersize*.8,
                        label="2 layers PMD")
        else:
            handle2 = mlines.Line2D([], [],
                        marker=mfs[PIND[0]%len(mfs)], c=colours[PIND[0]%len(colours)], lw=lwidth*.8, ms=markersize*.8,
                        label="1st layer PMD, \n 2nd layer MTB")
        handles = [handle1, handle2]

        myLegend(ax, handles=handles, loc='center left', bbox_to_anchor=(.92,0.5),
                     fontsize=labelsize, handlelength=2., numpoints=2, labelspacing=1.8)

    if pshow: pl.show()


def plot_figure():
    pl.figure("deepopt", figsize=(10,6.5))
    gs = GridSpec(2,3)
    gs.update(top=0.9, bottom=0.17, left=0.07, right=0.75, hspace=0.1, wspace=0.1)
    axes = [[pl.subplot(gs[jj,ii]) for ii in range(3)] for jj in range(2)]

    plot_data(axes)

    pl.savefig(paths.fig_path + "biasadaptation_fig4.svg", transparent=True)
    pl.show()


if __name__ == "__main__":
    # plot_data()
    plot_figure()

