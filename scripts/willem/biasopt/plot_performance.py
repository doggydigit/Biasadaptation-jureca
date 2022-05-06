import numpy as np

import os
import pickle

from datarep.matplotlibsettings import *
import datarep.paths as paths


NHS = [10, 25, 50, 100, 250, 500, 1000, 2500]
# METHODS = ["pca", "ica", "rp", "rpw", "mr", "mrb", "br", "brb", "pmd", "pmdd", "sc", "scd", "bp", "bpo", "code", "sm"]
METHODS = ["rp", "rpw",  "pmdd", "scd", "rpsym", "rpwsym", "pmddsym", "scdsym"]
DATAPATH = "/Users/wybo/Data/results/biasopt/"
DATASET = "EMNIST"#"CIFAR100"#"EMNIST"
TASKTYPE = "1vall"
READOUT = "tanh"
SUFFIX=''


pl.figure('Perf %s ro=%s'%(TASKTYPE, READOUT), figsize=(14,4*int(len(METHODS)/4)+1))
gs = GridSpec(int(len(METHODS)/4)+1, 4)
gs.update(top=0.90, bottom=0.15, left=0.15, right=0.95, hspace=1.1, wspace=0.4)


for ii, algo in enumerate(METHODS):

    perfs = []
    for n_h in NHS:

        if algo == 'code':
            fname = os.path.join(DATAPATH, "codeopt_%s_algow-c=%s-%s_nh=%d_%s__ro=%s.p"%(DATASET, 'sc', 'sc', n_h, TASKTYPE, READOUT))
        else:
            fname = os.path.join(DATAPATH, "biasopt_1hl_%s_%s%d_%s_ro=%s%s.p"%(DATASET, algo, n_h, TASKTYPE, READOUT, SUFFIX))
        print(fname)

        try:
            with open(fname, 'rb') as file:
                reslist = pickle.load(file)
            # print([res['perf']['test'] for res in reslist])
            perf = np.mean([res['perf']['test']['all'] for res in reslist])
        except FileNotFoundError:
            perf = np.nan

        perfs.append(perf)

    print(perfs)

    ax = myAx(pl.subplot(gs[ii//4, ii%4]))
    ax.set_title(algo, fontsize=labelsize)

    rect = Rectangle((0., 95.), len(NHS), 5., color='DarkGrey', alpha=.3)
    ax.add_patch(rect)
    ax.axhline(90, ls='--', lw=lwidth/2., c='k')
    ax.axhline(80, ls='--', lw=lwidth/2., c='k')

    ax.plot(np.arange(len(NHS))+0.5, perfs,
        ls='--', marker=mfs[ii%len(mfs)], c=colours[ii%len(colours)], lw=lwidth, ms=markersize)

    ax.set_ylim((50., 100.))
    ax.set_xlim((0., len(NHS)))
    ax.set_xticks(np.arange(len(NHS))+.5)
    ax.set_xticklabels(NHS, rotation=60)
    ax.set_xlabel(r'$n_h$')
    # ax.set_ylabel(r'frac correct')


figname = os.path.join(paths.fig_path, 'biasopt/res_biasopt_%s_%s_ro=%s%s'%(DATASET, TASKTYPE, READOUT, SUFFIX))
pl.savefig(figname + '.png', transparent=True)
pl.savefig(figname + '.svg', transparent=True)

pl.show()