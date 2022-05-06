import numpy as np

import os
import pickle

from datarep.matplotlibsettings import *
import datarep.paths as paths



NHS = [10, 25, 50, 100, 250, 500]
NSAMPLES = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500]
ALGO = "rp"
DATAPATH = "/Users/wybo/Data/results/biasopt/"
DATASET = "EMNIST"
TASKTYPE = "1vall"
READOUT = "tanh"
ALGOWC = ""


pl.figure('Perf %s %s'%(ALGO, TASKTYPE), figsize=(14,8))
gs = GridSpec(2, 3)
gs.update(top=0.90, bottom=0.15, left=0.15, right=0.95, hspace=1.1, wspace=0.4)


for ii, n_h in enumerate(NHS):

    perfs_bias = []
    perfs_full = []
    perfs_fewc = []
    for n_sample in NSAMPLES:
        print(os.path.join(DATAPATH, "fewopt_1hl_%s_%s%d_%s_nsample=%d_ro=%s.p"%(DATASET, ALGO, n_h, TASKTYPE, n_sample, READOUT)))

        try:
            with open(os.path.join(DATAPATH, "fewopt_1hl_%s_%s%d_%s_nsample=%d_ro=%s.p"%(DATASET, ALGO, n_h, TASKTYPE, n_sample, READOUT)), 'rb') as file:
                reslist_bias = pickle.load(file)
                reslist_full = pickle.load(file)
            # print([res['perf']['test'] for res in reslist])
            perf_bias = np.mean([res['perf']['test']['all'] for res in reslist_bias])
            perf_full = np.mean([res['perf']['test']['all'] for res in reslist_full])
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            perf_bias = np.nan
            perf_full = np.nan

        perfs_bias.append(perf_bias)
        perfs_full.append(perf_full)


        f_name_few = os.path.join(DATAPATH,
                                  "fewcodeopt_%s_algow-c=%s_nh=%d_%s_nsample=%d_ro=%s.p"%(DATASET, ALGOWC, n_h, TASKTYPE, n_sample, READOUT))

        try:
            with open(f_name_few, 'rb') as file:
                reslist_fewc = pickle.load(file)
            perf_fewc = np.mean([res['perf']['test']['all'] for res in reslist_fewc])
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            perf_fewc = np.nan

        perfs_fewc.append(perf_fewc)

    print(perfs_bias)
    print(perfs_full)
    print(perfs_fewc)

    ax = myAx(pl.subplot(gs[ii//3, ii%3]))
    ax.set_title(r''+'$n_h = %d$'%n_h, fontsize=labelsize)

    rect = Rectangle((0., 95.), len(NSAMPLES), 5., color='DarkGrey', alpha=.3)
    ax.add_patch(rect)
    ax.axhline(90, ls='--', lw=lwidth/2., c='k')
    ax.axhline(80, ls='--', lw=lwidth/2., c='k')

    ax.plot(np.arange(len(NSAMPLES))+0.5, perfs_full,
        ls='--', marker=mfs[ii%len(mfs)], c='DarkGrey', lw=lwidth, ms=markersize)
    ax.plot(np.arange(len(NSAMPLES))+0.5, perfs_bias,
        ls='--', marker=mfs[ii%len(mfs)], c=colours[ii%len(colours)], lw=lwidth, ms=markersize)
    ax.plot(np.arange(len(NSAMPLES))+0.5, perfs_fewc,
        ls='--', marker=mfs[ii%len(mfs)], c='k', lw=lwidth, ms=markersize)

    ax.set_ylim((50., 100.))
    ax.set_xlim((0., len(NSAMPLES)))
    ax.set_xticks(np.arange(len(NSAMPLES))+.5)
    ax.set_xticklabels(NSAMPLES, rotation=60)
    ax.set_xlabel(r'$n_{sample}$')
    # ax.set_ylabel(r'frac correct')


figname = os.path.join(paths.fig_path, 'biasopt/res_fewopt_%s_%s_%s_ro=%s'%(DATASET, TASKTYPE, ALGO, READOUT))
pl.savefig(figname + '.png', transparent=True)
pl.savefig(figname + '.svg', transparent=True)


pl.show()