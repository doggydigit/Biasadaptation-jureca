import numpy as np

import os
import pickle

from datarep.matplotlibsettings import *
import datarep.paths as paths


NHS = [10, 25, 50, 100, 250, 500]
ALGOW = 'sc'
ALGOC = ['sc', 'lstsq']
DATAPATH = "/Users/wybo/Data/results/biasopt/"
DATASET = "EMNIST"
TASKTYPE = "1vall"
READOUT = ["tanh", "hardtanh", "linear", "sigmoid1", "sigmoid5", "sigmoid10"]


pl.figure('Perf %s'%(TASKTYPE), figsize=(9,6))
gs = GridSpec(2, 3)
gs.update(top=0.90, bottom=0.15, left=0.15, right=0.95, hspace=1.1, wspace=0.4)


for ii, readout in enumerate(READOUT):

    ax = pl.subplot(gs[ii//3, ii%3])
    ax.set_title(readout)

    rect = Rectangle((0., 95.), len(NHS), 5., color='DarkGrey', alpha=.3)
    ax.add_patch(rect)
    ax.axhline(90, ls='--', lw=lwidth/2., c='k')
    ax.axhline(80, ls='--', lw=lwidth/2., c='k')

    for jj, algo_c in enumerate(ALGOC):
        perfs = []
        for n_h in NHS:
            f_name = os.path.join(DATAPATH, "codeopt_%s_algow-c=%s-%s_nh=%d_%s__ro=%s.p"%(DATASET, ALGOW, algo_c, n_h, TASKTYPE, readout))
            print(f_name)

            try:
                with open(f_name, 'rb') as file:
                    reslist = pickle.load(file)
                perf = np.mean([res['perf']['test']['all'] for res in reslist])
            except FileNotFoundError:
                perf = np.nan

            perfs.append(perf)

        print(perfs)


        ax.plot(np.arange(len(NHS))+0.5, perfs,
            ls='--', marker=mfs[jj%len(mfs)], c=colours[jj%len(colours)], lw=lwidth, ms=markersize, label=algo_c)

        ax.set_ylim((50., 100.))
        ax.set_xlim((0., len(NHS)))
        ax.set_xticks(np.arange(len(NHS))+.5)
        ax.set_xticklabels(NHS, rotation=60)
        ax.set_xlabel(r'$n_h$')
        if ii == len(READOUT)-1:
            myLegend(ax, loc=0)
        # ax.set_ylabel(r'frac correct')


figname = os.path.join(paths.fig_path, 'biasopt/res_codeopt_%s_%s'%(DATASET, TASKTYPE))
pl.savefig(figname + '.png', transparent=True)
pl.savefig(figname + '.svg', transparent=True)


pl.show()