import numpy as np

import os
import pickle

from datarep.matplotlibsettings import *
import datarep.paths as paths
import figtools


NH = 100
ALGO = "pmdd"

WEIGHTDATASETS = [["K49"], ["EMNIST"]]
M1DATASETS= [["K49"], ["K49"]]
P1DATASETS = [["K49"], ["K49"]]

DATAPATH = "/Users/wybo/Data/results/biasopt/"
DATASET = "EMNIST"
TASKTYPE = "1vall"
READOUT = "tanh"
SUFFIX=''


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


pl.figure(figsize=(2.,3.5))
gs = GridSpec(1,1)
gs.update(top=0.95, bottom=0.4, left=0.4, right=0.95, hspace=0.1, wspace=0.1)
xlabels = []
xpos = [0.25,0.75]
xwidth = 0.25

algo = ALGO

for ii, (weightdatasets, m1datasets, p1datasets) in enumerate(zip(WEIGHTDATASETS, M1DATASETS, P1DATASETS)):

    datasetstr = "_".join(["-".join(weightdatasets),
                           "-".join(m1datasets),
                           "-".join(p1datasets)
                           ])
    fname = os.path.join(DATAPATH, "transferopt_1hl_%s_%s%d_%s_ro=%s%s.p"%(datasetstr, algo, NH, TASKTYPE, READOUT, SUFFIX))

    try:
        with open(fname, 'rb') as file:
            reslist = pickle.load(file)
        perf_avg = np.mean([res['perf']['test']['all'] for res in reslist])
        perf_std = np.std([res['perf']['test']['all'] for res in reslist])
    except FileNotFoundError:
        perf_avg = np.nan
        perf_std = np.nan

    ax = perfAx(pl.subplot(gs[0,0]))

    ax.bar(xpos[ii], perf_avg,
           yerr=perf_std, width=xwidth, align='center',
           color=figtools.COLOURS["pmdd"], ecolor='k', edgecolor='k',# hatch="///",
           linewidth=.4*lwidth, label=figtools.LABELS["pmdd"][0][0])

    xlabels.append(r"%s $\rightarrow$"%(weightdatasets[0])+"\n%s"%(p1datasets[0]))

ax.set_xlim([xpos[0]-xwidth, xpos[1]+xwidth])
ax.set_xticks(xpos)
ax.set_xticklabels(xlabels, rotation=60, fontsize=14)

pl.show()

