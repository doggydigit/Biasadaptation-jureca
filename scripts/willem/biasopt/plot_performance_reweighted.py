import numpy as np

import os
import pickle

from datarep.matplotlibsettings import *
import datarep.paths as paths


NHS = [10, 25, 50, 100, 250, 500]
METHODS = ["scd", "pmdd"]
DATAPATH = "/Users/wybo/Data/results/biasopt/"
DATASET = "EMNIST"
TASKTYPE = "1vall"
READOUT = "tanh"
SUFFIX=''


pl.figure('Perf %s ro=%s'%(TASKTYPE, READOUT), figsize=(5,10))
gs = GridSpec(2, 1)
gs.update(top=0.90, bottom=0.15, left=0.15, right=0.95, hspace=1.1, wspace=0.4)

def load_perf(fname):
    try:
        with open(fname, 'rb') as file:
            reslist = pickle.load(file)
        # print([res['perf']['test'] for res in reslist])
        perf = np.mean([res['perf']['test']['all'] for res in reslist])
    except FileNotFoundError:
        perf = np.nan
    return perf

for ii, algo in enumerate(METHODS):

    perfs1 = []
    perfs2 = []
    for n_h in NHS:

        fname1 = os.path.join(DATAPATH, "biasopt_1hl_%s_%s%d_%s_ro=%s%s.p"%(DATASET, algo, n_h, TASKTYPE, READOUT, SUFFIX))
        fname2 = os.path.join(DATAPATH, "biasopt_1hl_%s_reweighted_%s%d_%s_ro=%s%s.p"%(DATASET, algo, n_h, TASKTYPE, READOUT, SUFFIX))

        perfs1.append(load_perf(fname1))
        perfs2.append(load_perf(fname2))

    print('>>>', algo, "<<<")
    print(perfs1)
    print(perfs2)

    ax = myAx(pl.subplot(gs[ii, 0]))
    ax.set_title(algo, fontsize=labelsize)

    rect = Rectangle((0., 95.), len(NHS), 5., color='DarkGrey', alpha=.3)
    ax.add_patch(rect)
    ax.axhline(90, ls='--', lw=lwidth/2., c='k')
    ax.axhline(80, ls='--', lw=lwidth/2., c='k')

    ax.plot(np.arange(len(NHS))+0.5, perfs1,
        ls='--', marker=mfs[ii%len(mfs)], c=colours[ii%len(colours)], lw=lwidth, ms=markersize, label="normal")
    ax.plot(np.arange(len(NHS))+0.5, perfs2,
        ls='--', marker=mfs[ii%len(mfs)], c="DarkGrey", lw=lwidth, ms=markersize, label="reweighted")

    ax.set_ylim((50., 100.))
    ax.set_xlim((0., len(NHS)))
    ax.set_xticks(np.arange(len(NHS))+.5)
    ax.set_xticklabels(NHS, rotation=60)
    ax.set_xlabel(r'$n_h$')
    # ax.set_ylabel(r'frac correct')
    myLegend(ax, loc=0)


figname = os.path.join(paths.fig_path, 'biasopt/res_biasopt_reweighted_%s_%s_ro=%s%s'%(DATASET, TASKTYPE, READOUT, SUFFIX))
pl.savefig(figname + '.png', transparent=True)
pl.savefig(figname + '.svg', transparent=True)

pl.show()