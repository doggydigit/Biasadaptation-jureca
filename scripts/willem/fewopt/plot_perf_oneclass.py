import numpy as np

import os
import pickle
import sys
sys.path.append('..')

import optim, helperfuncs, testperf

from datarep.matplotlibsettings import *
import datarep.paths as paths


NHS = [10, 25, 50, 100, 250, 500]
NSAMPLES = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500]
ALGO = "pmdd"
DATAPATH = "/Users/wybo/Data/results/results_bias_opt/biasopt/"
DATASET = "EMNIST"
TASKTYPE = "1vall"
READOUT = "linear"
TARGET = -1


pl.figure('Perf %s %s'%(ALGO, TASKTYPE), figsize=(14,8))
gs = GridSpec(2, 3)
gs.update(top=0.90, bottom=0.15, left=0.15, right=0.95, hspace=1.1, wspace=0.4)


def increase_nsample(task):
    for key in task[-1]['EMNIST']:
        task[-1]['EMNIST'][key] = 1000000
    return task


for ii, n_h in enumerate(NHS):

    perfs_bias_m, perfs_bias_p = [], []
    perfs_full_m, perfs_full_p = [], []

    for n_sample in NSAMPLES:
        print(os.path.join(DATAPATH, "fewopt_1hl_%s_%s%d_%s_nsample=%d_ro=%s.p"%(DATASET, ALGO, n_h, TASKTYPE, n_sample, READOUT)))

        try:
            with open(os.path.join(DATAPATH, "fewopt_1hl_%s_%s%d_%s_nsample=%d_ro=%s.p"%(DATASET, ALGO, n_h, TASKTYPE, n_sample, READOUT)), 'rb') as file:
                reslist_bias = pickle.load(file)
                reslist_full = pickle.load(file)

            print('\n----- bias -----')
            p_bias_m = []
            p_bias_p = []
            for res in reslist_bias:
                task = increase_nsample(res['task'])
                print(res['task'])
                print('\n--> test accuracy stored: %.3f'%res['perf']['test'])
                perfs = testperf.test_single_class(res['ws'], res['bs'], task, DATASET, READOUT, targets=[-1, 1])
                p_bias_m.append(perfs[-1])
                p_bias_p.append(perfs[1])

            print('\n----- full -----')
            p_full_m = []
            p_full_p = []
            for res in reslist_full:
                task = increase_nsample(res['task'])
                print('\n--> test accuracy stored: %.3f'%res['perf']['test'])
                perfs = testperf.test_single_class(res['ws'], res['bs'], task, DATASET, READOUT, targets=[-1, 1])
                p_full_m.append(perfs[-1])
                p_full_p.append(perfs[1])

            perf_bias_m = np.mean(p_bias_m)
            perf_full_m = np.mean(p_full_m)
            perf_bias_p = np.mean(p_bias_p)
            perf_full_p = np.mean(p_full_p)

        except FileNotFoundError:
            perf_bias_m = np.nan
            perf_bias_p = np.nan
            perf_full_m = np.nan
            perf_full_p = np.nan

        perfs_bias_m.append(perf_bias_m)
        perfs_bias_p.append(perf_bias_p)
        perfs_full_m.append(perf_full_m)
        perfs_full_p.append(perf_full_p)


    ax = myAx(pl.subplot(gs[ii//3, ii%3]))
    ax.set_title(r''+'$n_h = %d$'%n_h, fontsize=labelsize)

    rect = Rectangle((0., 95.), len(NSAMPLES), 5., color='DarkGrey', alpha=.3)
    ax.add_patch(rect)
    ax.axhline(90, ls='--', lw=lwidth/2., c='k')
    ax.axhline(80, ls='--', lw=lwidth/2., c='k')

    # target -1
    ax.plot(np.arange(len(NSAMPLES))+0.5, perfs_full_m,
        ls='--', marker='_', c='DarkGrey', lw=lwidth, ms=3*markersize)
    ax.plot(np.arange(len(NSAMPLES))+0.5, perfs_bias_m,
        ls='--', marker='_', c=colours[ii%len(colours)], lw=lwidth, ms=3*markersize)

    # target +1
    ax.plot(np.arange(len(NSAMPLES))+0.5, perfs_full_p,
        ls='--', marker='+', c='DarkGrey', lw=lwidth, ms=3*markersize)
    ax.plot(np.arange(len(NSAMPLES))+0.5, perfs_bias_p,
        ls='--', marker='+', c=colours[ii%len(colours)], lw=lwidth, ms=3*markersize)

    ax.set_ylim((-1., 100.))
    ax.set_xlim((0., len(NSAMPLES)))
    ax.set_xticks(np.arange(len(NSAMPLES))+.5)
    ax.set_xticklabels(NSAMPLES, rotation=60)
    ax.set_xlabel(r'$n_{sample}$')
    # ax.set_ylabel(r'frac correct')


figname = os.path.join(paths.fig_path, 'biasopt/res_fewopt_%s_%s_pertarget_%s_ro=%s'%(DATASET, TASKTYPE, ALGO, READOUT))
pl.savefig(figname + '.png', transparent=True)
pl.savefig(figname + '.svg', transparent=True)


pl.show()