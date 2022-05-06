import numpy as np

import os
import pickle

from datarep.matplotlibsettings import *
import datarep.paths as paths

import figtools


NHS = [100]
NSAMPLES = [1, 3, 5, 10, 30, 50, 100, 300, 500]
METHODS = ["pmdd", "bpo", "br"]
LABELS = ["PMD", "MTB", "MTRU"]

DATAPATH = "/Users/wybo/Data/results/biasopt/"
DATASET = "EMNIST"
TASKTYPE = "1vall"
READOUT = "tanh"
PIND = [8,7,6]


def perfAx(ax, add_xticklabels=True, add_xlabel=True, add_yticklabels=True, add_ylabel=True):
    ax = myAx(ax)
    # rect = Rectangle((0., 95.), len(NSAMPLES), 5., color='DarkGrey', alpha=.2, zorder=-1000)
    # ax.add_patch(rect)
    ax.axhline(100, ls='--', lw=lwidth/3., c='DarkGrey', zorder=-1000)
    ax.axhline(90, ls='--', lw=lwidth/3., c='DarkGrey', zorder=-1001)
    ax.axhline(80, ls='--', lw=lwidth/3., c='DarkGrey', zorder=-1002)
    ax.axhline(70, ls='--', lw=lwidth/3., c='DarkGrey', zorder=-1003)
    ax.axhline(60, ls='--', lw=lwidth/3., c='DarkGrey', zorder=-1004)

    ax.axhline(95, ls=':', lw=lwidth/3., c='DarkGrey', zorder=-1005)
    ax.axhline(85, ls=':', lw=lwidth/3., c='DarkGrey', zorder=-1006)
    ax.axhline(75, ls=':', lw=lwidth/3., c='DarkGrey', zorder=-1007)
    ax.axhline(65, ls=':', lw=lwidth/3., c='DarkGrey', zorder=-1008)

    ax.set_ylim((50., 100.))
    ax.set_xlim((0., len(NSAMPLES)))
    ax.set_yticks([50.,60., 70.,80.,90.,100.])
    ax.set_xticks(np.arange(len(NSAMPLES))+.5)

    if add_xlabel:
        ax.set_xlabel("no. of datapoints", fontsize=labelsize)

    if add_xticklabels:
        ax.set_xticklabels(NSAMPLES, rotation=60, fontsize=labelsize)
    else:
        ax.set_xticklabels([])

    if add_ylabel:
        ax.set_ylabel(r'test perf (%)', fontsize=labelsize)

    if add_yticklabels:
        ax.set_yticklabels(["50", "", "70", "", "90", ""], fontsize=labelsize)
    else:
        ax.set_yticklabels([])

    return ax


def plot_data(axes=None, eps=.05, algos=['pmdd', 'rpw'], ccs=[colours[0], colours[2]], set_xticklabels=True, set_xlabel=True, legend_loc='top'):

    if axes is None:
        pl.figure(figsize=(14,3))
        gs = GridSpec(1,len(NHS))
        axes = [pl.subplot(gs[0,ii]) for ii in range(len(NHS))]
        pshow = True
    else:
        pshow = False

    xvals = np.arange(len(NSAMPLES))+0.5
    xwidth = 0.3

    handles = []
    for ii, n_h in enumerate(NHS):

        ax = perfAx(axes[ii],
                    add_xticklabels=set_xticklabels,
                    add_yticklabels=ii==0,
                    add_xlabel=((ii==int(len(NHS)/2)) and set_xlabel),
                    add_ylabel=ii==0)

        for jj, algo in enumerate(algos):

            perfs_avg = []
            perfs_std = []
            for n_sample in NSAMPLES:

                if algo == "code":
                    f_name = os.path.join(DATAPATH, "fewcodeopt_%s_algow-c=%s_nh=%d_%s_nsample=%d_ro=%s.p"%(DATASET, "sc-sc", n_h, TASKTYPE, n_sample, READOUT))
                    try:
                        with open(f_name, 'rb') as file:
                            reslist = pickle.load(file)
                        perf_avg = np.mean([res['perf']['test']['all'] for res in reslist])
                        perf_std = np.std([res['perf']['test']['all'] for res in reslist])
                    except (FileNotFoundError, pickle.UnpicklingError) as e:
                        perf_avg = np.nan

                else:
                    if algo == 'rpw':
                        fname = os.path.join(DATAPATH, "fewopt_1hl_%s_%s%d_%s_nsample=%d_ro=%s.p"%(DATASET, "pmdd", n_h, TASKTYPE, n_sample, READOUT))
                        load_full = True
                    else:
                        fname = os.path.join(DATAPATH, "fewopt_1hl_%s_%s%d_%s_nsample=%d_ro=%s.p"%(DATASET, algo, n_h, TASKTYPE, n_sample, READOUT))
                        load_full = False

                    try:
                        with open(fname, 'rb') as file:
                            reslist_bias = pickle.load(file)
                            reslist_full = pickle.load(file)
                        reslist = reslist_full if load_full else reslist_bias
                        perf_avg = np.mean([res['perf']['test']['all'] for res in reslist])
                        perf_std = np.std([res['perf']['test']['all'] for res in reslist])
                    except FileNotFoundError:
                        perf_avg = np.nan
                        perf_std = np.nan

                perfs_avg.append(perf_avg)
                perfs_std.append(perf_std)


            # kk = PIND[jj]
            # ax.errorbar(np.arange(len(NSAMPLES))+0.5+jj*eps, perfs_avg, yerr=perfs_std,
            #     ls='--', marker=mfs[kk%len(mfs)], c=colours[kk%len(colours)], lw=lwidth*.8, ms=markersize*.8)


            ax.bar(xvals+(jj-1)*xwidth, perfs_avg,
                   yerr=perfs_std, width=xwidth, align='edge',
                   color=figtools.COLOURS[algo], ecolor='k', edgecolor='k',
                   linewidth=.4*lwidth, label=figtools.LABELS[algo][0])

    if legend_loc == 'left':
        myLegend(ax, loc='center left', bbox_to_anchor=(.96,0.5), fontsize=labelsize, handlelength=2., numpoints=2, labelspacing=2.)
    elif legend_loc == 'top':
        myLegend(ax, loc='lower center', bbox_to_anchor=(0.5,0.9), fontsize=labelsize, handlelength=.6,
                    numpoints=2, labelspacing=1.0, columnspacing=1.0, handletextpad=.5, ncol=2, add_frame=False)


    if pshow: pl.show()


def plot_figure():
    pl.figure("fewopt", figsize=(12,4.))
    gs0 = GridSpec(1,3)
    gs0.update(top=0.95, bottom=0.45, left=0.1, right=0.85, hspace=0.1, wspace=0.1)
    gs1 = GridSpec(1,1)
    gs1.update(top=0.2, bottom=0.19, left=0.1, right=0.85, hspace=0.1, wspace=0.1)

    plot_data([pl.subplot(gs0[0,ii]) for ii in range(3)], algos=['pmdd', 'rpw'], ccs=[colours[0], colours[4]], set_xticklabels=True, set_xlabel=True)
    # plot_data([pl.subplot(gs0[1,ii]) for ii in range(5)], algos=['pmdd', 'code'], ccs=[colours[0], colours[1]], set_xticklabels=False, set_xlabel=False)
    # plot_data([pl.subplot(gs0[1,ii]) for ii in range(5)], algos=['pmdd', 'br'], ccs=[colours[0], colours[3]], set_xticklabels=True, set_xlabel=True)

    axa = pl.subplot(gs1[0,0])

    axa.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    axa.spines['right'].set_color('none')
    axa.spines['left'].set_color('none')
    # ax.draw_frame = False

    axa.set_yticks([])
    xdata = (np.arange(len(NHS))+.5) / len(NHS)
    axa.set_xlim((0.,1.))
    axa.set_xticks(xdata)
    axa.set_xticklabels(["[%d]"%n_h for n_h in NHS], fontsize=labelsize)
    axa.set_xlabel("no. of hidden units", fontsize=labelsize)

    pl.savefig(paths.fig_path + "biasadaptation_v2_fig5.svg", transparent=True)

    pl.show()


def plot_figure_v3():
    pl.figure("fewopt", figsize=(14,3.5))

    labels = getAnnotations(3)

    gs0 = GridSpec(1,3)
    gs0.update(top=0.80, bottom=0.25, left=0.1, right=0.95, hspace=0.3, wspace=0.5)
    # gs1 = GridSpec(1,1)
    # gs1.update(top=0.2, bottom=0.19, left=0.1, right=0.85, hspace=0.1, wspace=0.1)
    ax0 = pl.subplot(gs0[0,0])
    ax1 = pl.subplot(gs0[0,1])
    ax2 = pl.subplot(gs0[0,2])

    ax0.add_artist(labels[0])
    ax1.add_artist(labels[1])
    ax2.add_artist(labels[2])

    plot_data([ax0], algos=['pmdd', 'rpw'], set_xticklabels=True, set_xlabel=True)
    plot_data([ax1], algos=['pmdd', 'br'], set_xticklabels=True, set_xlabel=True)
    plot_data([ax2], algos=['pmdd', 'code'], set_xticklabels=True, set_xlabel=True)


    pl.savefig(paths.fig_path + "biasadaptation_v3_fig5.svg", transparent=True)

    pl.show()


def plot_suppl_panels(algos):
    pl.figure("fewopt", figsize=(12,4.))
    gs0 = GridSpec(1,3)
    gs0.update(top=0.95, bottom=0.45, left=0.1, right=0.85, hspace=0.1, wspace=0.1)
    gs1 = GridSpec(1,1)
    gs1.update(top=0.2, bottom=0.19, left=0.1, right=0.85, hspace=0.1, wspace=0.1)

    plot_data([pl.subplot(gs0[0,ii]) for ii in range(3)], algos=algos, ccs=[figtools.COLOURS[algos[0]], figtools.COLOURS[algos[1]]], set_xticklabels=True, set_xlabel=True)
    # plot_data([pl.subplot(gs0[1,ii]) for ii in range(5)], algos=['pmdd', 'code'], ccs=[colours[0], colours[1]], set_xticklabels=False, set_xlabel=False)
    # plot_data([pl.subplot(gs0[1,ii]) for ii in range(5)], algos=['pmdd', 'br'], ccs=[colours[0], colours[3]], set_xticklabels=True, set_xlabel=True)

    axa = pl.subplot(gs1[0,0])

    axa.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    axa.spines['right'].set_color('none')
    axa.spines['left'].set_color('none')
    # ax.draw_frame = False

    axa.set_yticks([])
    xdata = (np.arange(len(NHS))+.5) / len(NHS)
    axa.set_xlim((0.,1.))
    axa.set_xticks(xdata)
    axa.set_xticklabels(["[%d]"%n_h for n_h in NHS], fontsize=labelsize)
    axa.set_xlabel("no. of hidden units", fontsize=labelsize)

    pl.savefig(paths.fig_path + "biasadaptation_v2_panel_suppl_%s.svg"%"<->".join(algos), transparent=True)

    pl.show()


if __name__ == "__main__":
    # plot_data()
    plot_figure_v3()
    # plot_suppl_panels(["pmdd", "code"])
    # plot_suppl_panels(["pmdd", "br"])
    # plot_suppl_panels(["pmdd", "bpo"])
    # plot_suppl_panels(["pmdd", "rp"])
