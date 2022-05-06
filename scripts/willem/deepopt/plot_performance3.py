import numpy as np

import os
import pickle

from datarep.matplotlibsettings import *
import datarep.paths as paths


NHS = [10, 25, 50, 100, 250, 500]
NH = 100
# ALGO_12C = [["scd", "sc", "lstsq"], ["scd", "sc", "sc"],
#             ["pmdd", "pmd", "lstsq"], ["pmdd", "pmd", "sc"],
#             ["scd", "bp", "na"], ["scd", "bpo", "na"],
#             ["pmdd", "bp", "na"], ["pmdd", "bpo", "na"],
#            ]
ALGO_123C = [
            ["pmdd", "pmd", "pmd", "lstsq"], ["pmdd", "pmd", "pmd", "sc"],
            # ["scd", "sc", "sc", "lstsq"], ["scd", "sc", "sc", "sc"],
            ["pmdd", "pmd", "pmd", "lstsq"], ["pmdd", "pmd", "rp", "na"],
            ["pmdd", "rp", "rp", "na"], ["pmdd", "pmd", "pmd", "lstsq"],
            ["pmdd", "rp", "rp", "na"], ["pmdd", "pmd", "rp", "na"],
           ]
DATAPATH = "/Users/wybo/Data/results/biasopt/"
DATASET = "EMNIST"
TASKTYPE = "1vall"
READOUT = "tanh"


pl.figure('Perf %s ro=%s'%(TASKTYPE, READOUT), figsize=(14,9))
gs = GridSpec(4, 3)
gs.update(top=0.85, bottom=0.15, left=0.3, right=0.7, hspace=1.1, wspace=0.4)
gs0 = GridSpec(4, 1)
gs0.update(top=0.85, bottom=0.15, left=0.05, right=0.2, hspace=1.1, wspace=0.4)
gs1 = GridSpec(1, 3)
gs1.update(top=0.95, bottom=0.9, left=0.3, right=0.7, hspace=1.1, wspace=0.4)


ax = noFrameAx(pl.subplot(gs0[0,0]))
ax.text(.5,.5, '->'.join(ALGO_123C[0][0:3]), fontsize=labelsize, rotation=45, ha='center', va='center')
ax = noFrameAx(pl.subplot(gs0[1,0]))
ax.text(.5,.5, '->'.join(ALGO_123C[2][0:3]), fontsize=labelsize, rotation=45, ha='center', va='center')
ax = noFrameAx(pl.subplot(gs0[2,0]))
ax.text(.5,.5, '->'.join(ALGO_123C[4][0:3]), fontsize=labelsize, rotation=45, ha='center', va='center')
# ax = noFrameAx(pl.subplot(gs0[3,0]))
# ax.text(.5,.5,'->'.join(ALGO_123C[0][0:3]), fontsize=labelsize, rotation=45, ha='center', va='center')
ax = noFrameAx(pl.subplot(gs1[0,0]))
ax.text(.5,.5, r''+'$n_{h1} = x, n_{h2} = x, n_{h3} = %d$'%NH, fontsize=labelsize, ha='center', va='center', rotation=30)
ax = noFrameAx(pl.subplot(gs1[0,1]))
ax.text(.5,.5, r''+'$n_{h1} = %d, n_{h2} = %d, n_{h3} = x$'%(NH, NH), fontsize=labelsize, ha='center', va='center', rotation=30)
ax = noFrameAx(pl.subplot(gs1[0,2]))
ax.text(.5,.5, r''+'$n_{h1} = x, n_{h2} = x, n_{h3} = x$', fontsize=labelsize, ha='center', va='center', rotation=30)


def load_perf(namestring):
    try:
        with open(os.path.join(DATAPATH, namestring), 'rb') as file:
            reslist = pickle.load(file)
        perf = np.mean([res['perf']['test']['all'] for res in reslist])
    except (FileNotFoundError, TypeError):
        perf = np.nan

    return perf


for ii in range(4):

    algos_all = [
                 '-'.join(ALGO_123C[2*ii]),
                 '-'.join(ALGO_123C[2*ii+1]),
                ][::-1]
    algos_12c = [
                 '-'.join(ALGO_123C[2*ii][0:2] + ALGO_123C[2*ii][3:]),
                 '-'.join(ALGO_123C[2*ii+1][0:2] + ALGO_123C[2*ii+1][3:])
                ][::-1]

    perfs_nhx_c = {}
    perfs_xnh_c = {}
    perfs_xx_c = {}
    for jj, (algo_12c, algo_123c) in enumerate(zip(algos_12c, algos_all)):

        perfs_nhx = []
        perfs_xnh = []
        perfs_xx  = []
        perfs_2hl = []
        for n_h in NHS:
            namestring_nhx = "deepopt_3hl_%s_algo123c=%s_nh12=%d-%d-%d_%s_ro=%s.p"%(DATASET, algo_123c, NH, NH, n_h, TASKTYPE, READOUT)
            print(namestring_nhx)
            perfs_nhx.append(load_perf(namestring_nhx))

            namestring_xnh = "deepopt_3hl_%s_algo123c=%s_nh12=%d-%d-%d_%s_ro=%s.p"%(DATASET, algo_123c, n_h, n_h, NH, TASKTYPE, READOUT)
            perfs_xnh.append(load_perf(namestring_xnh))

            namestring_xx = "deepopt_3hl_%s_algo123c=%s_nh12=%d-%d-%d_%s_ro=%s.p"%(DATASET, algo_123c, n_h, n_h, n_h, TASKTYPE, READOUT)
            perfs_xx.append(load_perf(namestring_xx))

            namestring_2hl = "deepopt_2hl_%s_algo12c=%s_nh12=%d-%d_%s_ro=%s.p"%(DATASET, algo_12c, n_h, n_h, TASKTYPE, READOUT)
            # print(namestring_2hl)
            perfs_2hl.append(load_perf(namestring_2hl))

        perfs_nhx_c[algo_123c] = perfs_nhx
        perfs_xnh_c[algo_123c] = perfs_xnh
        perfs_xx_c[algo_123c] = perfs_xx

        print(algo_123c)
        print('nhnhx:', perfs_nhx)
        print('xxnh:', perfs_xnh)
        print('xxx :', perfs_xx)
        print('2hl:', perfs_2hl)


    namestring_2hl = "deepopt_2hl_%s_algo12c=%s_nh12=%d-%d_%s_ro=%s.p"%(DATASET, algo_12c, NH, NH, TASKTYPE, READOUT)
    perf_2hl = load_perf(namestring_2hl)


    ax = myAx(pl.subplot(gs[ii, 0]))

    rect = Rectangle((0., 95.), len(NHS), 5., color='DarkGrey', alpha=.3)
    ax.add_patch(rect)
    ax.axhline(90, ls='--', lw=lwidth/2., c='k')
    ax.axhline(80, ls='--', lw=lwidth/2., c='k')

    ax.plot(np.arange(len(NHS))+0.5, perfs_2hl,
        ls='--', marker="o", c=colours[2], lw=lwidth, ms=markersize, label='2hl')

    for jj, algo_123c in enumerate(algos_all):
        ax.plot(np.arange(len(NHS))+0.5, perfs_xnh_c[algo_123c],
            ls='--', marker="s", c=colours[jj], lw=lwidth, ms=markersize, label=algo_123c)

    ax.set_ylim((75., 100.))
    ax.set_xlim((0., len(NHS)))
    ax.set_xticks(np.arange(len(NHS))+.5)
    ax.set_xticklabels(NHS, rotation=60)
    ax.set_xlabel(r'$n_h$')


    ax = myAx(pl.subplot(gs[ii, 1]))

    rect = Rectangle((0., 95.), len(NHS), 5., color='DarkGrey', alpha=.3)
    ax.add_patch(rect)
    ax.axhline(90, ls='--', lw=lwidth/2., c='k')
    ax.axhline(80, ls='--', lw=lwidth/2., c='k')

    ax.axhline(perf_2hl, c=colours[2], lw=lwidth*1.6, label='2hl')

    for jj, algo_123c in enumerate(algos_all):
        ax.plot(np.arange(len(NHS))+0.5, perfs_nhx_c[algo_123c],
            ls='--', marker="s", c=colours[jj], lw=lwidth, ms=markersize, label=algo_123c)

    ax.set_ylim((75., 100.))
    ax.set_xlim((0., len(NHS)))
    ax.set_xticks(np.arange(len(NHS))+.5)
    ax.set_xticklabels(NHS, rotation=60)
    ax.set_xlabel(r'$n_h$')


    ax = myAx(pl.subplot(gs[ii, 2]))

    rect = Rectangle((0., 95.), len(NHS), 5., color='DarkGrey', alpha=.3)
    ax.add_patch(rect)
    ax.axhline(90, ls='--', lw=lwidth/2., c='k')
    ax.axhline(80, ls='--', lw=lwidth/2., c='k')

    ax.plot(np.arange(len(NHS))+0.5, perfs_2hl,
        ls='--', marker="o", c=colours[2], lw=lwidth, ms=markersize, label='2hl')

    for jj, algo_123c in enumerate(algos_all):
        ax.plot(np.arange(len(NHS))+0.5, perfs_xx_c[algo_123c],
            ls='--', marker="s", c=colours[jj], lw=lwidth, ms=markersize, label=algo_123c)

    ax.set_ylim((75., 100.))
    ax.set_xlim((0., len(NHS)))
    ax.set_xticks(np.arange(len(NHS))+.5)
    ax.set_xticklabels(NHS, rotation=60)
    ax.set_xlabel(r'$n_h$')

    myLegend(ax, loc='center left', bbox_to_anchor=(1.0,0.5))


figname = os.path.join(paths.fig_path, 'biasopt/res_deepopt_%s_%s_ro=%s'%(DATASET, TASKTYPE, READOUT))
pl.savefig(figname + '.png', transparent=True)
pl.savefig(figname + '.svg', transparent=True)

pl.show()