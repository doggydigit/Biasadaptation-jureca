import numpy as np

import os
import pickle

from datarep.matplotlibsettings import *
import datarep.paths as paths
import figtools


NH = 100
METHODS = ["rpw", "pmdd"]
DATAPATH = "/Users/wybo/Data/results/biasopt/"
DATASET = "EMNIST"
TASKTYPE = "1vall"
READOUT = "tanh"
SUFFIX = ""
NSAMPLE = 30


def perfAx(ax, add_yticklabels=True, add_ylabel=True):
    ax = myAx(ax)
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
    ax.set_yticks([50.,60., 70.,80.,90.,100.])


    if add_ylabel:
        ax.set_ylabel(r'test perf (%)', fontsize=labelsize)

    if add_yticklabels:
        ax.set_yticklabels(["50", "", "70", "", "90", ""])
    else:
        ax.set_yticklabels([])

    return ax


def read_file(fname):
    try:
        with open(fname, 'rb') as file:
            reslist = pickle.load(file)
        # print([res['perf']['test'] for res in reslist])
        perf_avg = np.mean([res['perf']['test']['all'] for res in reslist])
        perf_std = np.std([res['perf']['test']['all'] for res in reslist])
    except FileNotFoundError:
        perf_avg = np.nan
        perf_std = np.nan

    return perf_avg, perf_std

def read_file_few(fname):
    try:
        with open(fname, 'rb') as file:
            reslist_bias = pickle.load(file)
            reslist_full = pickle.load(file)
        # print([res['perf']['test'] for res in reslist])
        perf_bias_avg = np.mean([res['perf']['test']['all'] for res in reslist_bias])
        perf_bias_std = np.std([res['perf']['test']['all'] for res in reslist_bias])
        perf_full_avg = np.mean([res['perf']['test']['all'] for res in reslist_full])
        perf_full_std = np.std([res['perf']['test']['all'] for res in reslist_full])
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        perf_bias_avg = np.nan
        perf_full_std = np.nan

    return perf_bias_avg, perf_bias_std, perf_full_avg, perf_full_std




fname_full = os.path.join(DATAPATH, "biasopt_1hl_%s_%s%d_%s_ro=%s%s.p"%(DATASET, METHODS[0], NH, TASKTYPE, READOUT, SUFFIX))
fname_pmdd = os.path.join(DATAPATH, "biasopt_1hl_%s_%s%d_%s_ro=%s%s.p"%(DATASET, METHODS[1], NH, TASKTYPE, READOUT, SUFFIX))

fname_few = os.path.join(DATAPATH, "fewopt_1hl_%s_%s%d_%s_nsample=%d_ro=%s.p"%(DATASET, METHODS[1], NH, TASKTYPE, NSAMPLE, READOUT))

perf_full = read_file(fname_full)
perf_pmdd = read_file(fname_pmdd)
perfs_few = read_file_few(fname_few)


perfs_avg = {}
perfs_std = {}

perfs_avg['full'] = [perf_full[0], perfs_few[2]]
perfs_avg['pmdd'] = [perf_pmdd[0], perfs_few[0]]
perfs_std['full'] = [perf_full[1], perfs_few[3]]
perfs_std['pmdd'] = [perf_pmdd[1], perfs_few[1]]


pl.figure(figsize=(2.5,3.5))
gs = GridSpec(1,1)
gs.update(top=0.95, bottom=0.4, left=0.3, right=0.95, hspace=0.1, wspace=0.1)

ax = perfAx(pl.subplot(gs[0,0]))

xpos = [.25, .75]
xwidth = .125

ax.set_xticks(xpos)
ax.set_xticklabels(['all \nsamples', '%d \nsamples'%NSAMPLE], rotation=60)

ax.set_xlim([xpos[0]-2.*xwidth, xpos[1]+2*xwidth])

ax.bar(xpos, perfs_avg['full'],
       yerr=perfs_std['full'], width=-xwidth, align='edge',
       color=figtools.COLOURS["rpw"], ecolor='k', edgecolor='k',# hatch="///",
       linewidth=.4*lwidth, label=figtools.LABELS["rpw"][0][0])

ax.bar(xpos, perfs_avg['pmdd'],
       yerr=perfs_std['pmdd'], width=xwidth, align='edge',
       color=figtools.COLOURS["pmdd"], ecolor='k', edgecolor='k',# hatch="///",
       linewidth=.4*lwidth, label=figtools.LABELS["pmdd"][0][0])

ax.legend(loc=0)


pl.show()

