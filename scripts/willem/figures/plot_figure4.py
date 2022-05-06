import numpy as np

import os
import pickle

from datarep.matplotlibsettings import *
import datarep.paths as paths

import figtools


NHS = [25, 50, 100, 250, 500]
NH = 100

ALGOS_1HL = ["pmdd", "code"]
PIND = [0,1]
ALGOS_SC = ["sc", "scd"]
PIND_SC = [2,4]
ALGOS_W = ["pca", "rp"]
PIND_W = [3,7]

ALGO_1 = "pmdd"
ALGOS = [["pmdd", "pmd", "lstsq"],
         ["pmdd", "rp", "na"],
         ["rp", "rp", "na"],
        ]
PINDS = [0,1,2]

NH_3HL = 100
ALGOS_3HL = [
             ["pmdd", "rp", "na"],
             ["rp", "rp", "na"],
            ]


DATAPATH = "/Users/wybo/Data/results/biasopt/"
DATASET = "EMNIST"
TASKTYPE = "1vall"
READOUT = "tanh"



def load_perf(namestring):
    try:
        with open(os.path.join(DATAPATH, namestring), 'rb') as file:
            reslist = pickle.load(file)
        perf_avg = np.mean([res['perf']['test']['all'] for res in reslist])
        perf_std = np.std( [res['perf']['test']['all'] for res in reslist])
    except (FileNotFoundError, TypeError):
        perf_avg = np.nan
        perf_std = np.nan

    return perf_avg, perf_std



def plot_data_1hl(ptype, ax=None, add_yticklabels=True, add_ylabel=True, add_xlabel=True):

    if ptype == "1HL":
        algos = ALGOS_1HL
        pind = PIND
    elif ptype == "SC":
        algos = ALGOS_SC
        pind = PIND_SC
    elif ptype == "W":
        algos = ALGOS_W
        pind = PIND_W
    else:
        raise IOError("ptype not recognized")

    if ax is None:
        pl.figure()
        gs = pl.GridSpec(1,1)
        ax = pl.subplot(gs[0,0])
        pshow = True
    else:
        pshow = False

    kk = 0
    lines, handles = [], []
    xlabel = "no. of hidden units" if add_xlabel else None

    ax = figtools.perfAx(ax,
                ylim=(70., 100.),
                skip_yt=False,
                add_xticklabels=True,
                add_yticklabels=add_yticklabels,
                xlabel=xlabel,
                add_ylabel=add_ylabel,
                nhs=[[nh] for nh in NHS])
    xvals = np.arange(len(NHS))+0.5
    xwidth = .35

    for ii, algo in enumerate(algos):

        perfs_avg = []
        perfs_std = []
        for n_h in NHS:
            print(os.path.join(DATAPATH, "biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(DATASET, algo, n_h, TASKTYPE, READOUT)))

            if algo == 'code':
                fname = os.path.join(DATAPATH, "codeopt_%s_algow-c=%s-%s_nh=%d_%s__ro=%s.p"%(DATASET, 'sc', 'sc', n_h, TASKTYPE, READOUT))
            else:
                fname = os.path.join(DATAPATH, "biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(DATASET, algo, n_h, TASKTYPE, READOUT))

            try:
                with open(fname, 'rb') as file:
                    reslist = pickle.load(file)

                perf_avg = np.mean([res['perf']['test']['all'] for res in reslist])
                perf_std = np.std([res['perf']['test']['all'] for res in reslist])

            except FileNotFoundError:
                perf_avg = np.nan
                perf_std = np.nan

            perfs_avg.append(perf_avg)
            perfs_std.append(perf_std)

        # ax.errorbar(xvals, perfs_avg, yerr=perfs_std,
        #     ls='--', marker=mfs[kk%len(mfs)], c=colours[kk%len(colours)], lw=lwidth*.8, ms=markersize*.8)

        ax.bar(xvals+(ii-1)*xwidth, perfs_avg,
               yerr=perfs_std, width=xwidth, align='edge',
               color=colours[pind[ii]%len(colours)], ecolor='k', edgecolor='k',
               linewidth=.4*lwidth, label=figtools.LABELS[algo][0])

    myLegend(ax, loc='lower center', add_frame=False, bbox_to_anchor=(.5,0.98),
                 fontsize=labelsize, handlelength=.6, numpoints=2, ncol=2, labelspacing=1., handletextpad=.2, columnspacing=.8)

    if pshow:
        pl.show()


def plot_data_2hl(ax, eps=.02):

    # if ptype == "PMD":
    #     algos = ALGO_PMD
    #     pind = PIND_PMD
    #     pstr = r''+'PMD \n2 layers'
    # elif ptype == "BPO":
    #     algos = ALGO_BPO
    #     pind = PIND_BPO
    #     pstr = r''+'1st layer PMD \n2nd layer MT'
    # else:
    #     raise IOError("invalid ptype")

    if ax is None:
        pl.figure(figsize=(8,4))
        gs = GridSpec(1,1)
        ax = pl.subplot(gs[0,0])
        pshow = True
    else:
        pshow = False

    xvals = np.arange(len(NHS))+0.5
    xwidth = .25

    perfd_nhx_avg = {}
    perfd_xnh_avg = {}
    perfd_xx_avg = {}
    perfd_nhx_std = {}
    perfd_xnh_std = {}
    perfd_xx_std = {}

    algo_list = ['-'.join(algos) for algos in ALGOS]
    algo_color = []

    for algos in ALGOS:

        algo_12c = '-'.join(algos)
        algo_1 = ALGO_1
        algo_color.append(algos[1])

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

        perfd_nhx_avg[algo_12c] = perfs_nhx_avg
        perfd_xnh_avg[algo_12c] = perfs_xnh_avg
        perfd_xx_avg[algo_12c] = perfs_xx_avg

        perfd_nhx_std[algo_12c] = perfs_nhx_std
        perfd_xnh_std[algo_12c] = perfs_xnh_std
        perfd_xx_std[algo_12c] = perfs_xx_std

        perfd_nhx_avg[algo_1] = perfs_1hl_avg
        perfd_xnh_avg[algo_1] = perfs_1hl_avg
        perfd_xx_avg[algo_1] = perfs_1hl_avg

        perfd_nhx_std[algo_1] = perfs_1hl_std
        perfd_xnh_std[algo_1] = perfs_1hl_std
        perfd_xx_std[algo_1] = perfs_1hl_std

    algo_list = [algo_1] + algo_list


    print(algo_12c)
    print('nhx:', perfs_nhx_avg)
    print('xnh:', perfs_xnh_avg)
    print('xx :', perfs_xx_avg)
    print('1hl:', perfs_1hl_avg)

    namestring_1hl = "biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(DATASET, algo_1, NH, TASKTYPE, READOUT)
    perf_1hl = load_perf(namestring_1hl)[0]

    # case xx
    xlabel = "no. of hidden units"
    ax = figtools.perfAx(ax,
                ylim=(70., 100.),
                skip_yt=False,
                add_xticklabels=True,
                add_yticklabels=True,
                xlabel=xlabel,
                add_ylabel=True,
                nhs=[[nh, nh] for nh in NHS])

    print(algo_list)

    b1 = ax.bar(xvals-xwidth, perfd_xx_avg[algo_list[1]], yerr=perfd_xx_std[algo_list[1]],
           width=xwidth, align='center',
           color=figtools.COLOURS['pmdd'], ecolor='k', edgecolor='k',
           linewidth=.4*lwidth,
           label=r''+'$\Delta$PMD \n2 layers')

    b2 = ax.bar(xvals, perfd_xx_avg[algo_list[2]], yerr=perfd_xx_std[algo_list[2]],
           width=xwidth, align='center',
           color=figtools.COLOURS['rp'], ecolor='k', edgecolor='k',
           linewidth=.4*lwidth,
           label=r''+'1st layer $\Delta$PMD \n2nd layer RP')


    b3 = ax.bar(xvals+xwidth, perfd_xx_avg[algo_list[3]], yerr=perfd_xx_std[algo_list[3]],
           width=xwidth, align='center',
           color="White", ecolor='k', edgecolor='k',
           linewidth=.4*lwidth,
           label=r''+'RP \n2 layers')

    x0 = np.array([pl.getp(item, 'x') for item in b1])
    # x0 = x1 + np.array([pl.getp(item, 'width') for item in b1])
    x1 = x0 + 3*np.array([pl.getp(item, 'width') for item in b2])

    ax.hlines(perfd_xx_avg[algo_list[0]], x0, x1, colors='r', lw=lwidth*1.6, label=r''+'$\Delta$PMD \n1 layer')

    # # case xnh
    # xlabel = "no. of hidden units"
    # ax = figtools.perfAx(axes[1],
    #             ylim=(70., 100.),
    #             skip_yt=False,
    #             add_xticklabels=True,
    #             add_yticklabels=False,
    #             xlabel=None,
    #             add_ylabel=False,
    #             nhs=[[nh, NH] for nh in NHS])

    # b1 = ax.bar(xvals, perfd_xnh_avg[algo_list[1]], yerr=perfd_xx_std[algo_list[1]],
    #        width=-xwidth, align='edge',
    #        color=figtools.COLOURS['pmdd'], ecolor='k', edgecolor='k',
    #        linewidth=.4*lwidth, label=r''+'PMD \n2 layers')

    # b2 = ax.bar(xvals, perfd_xnh_avg[algo_list[2]], yerr=perfd_xx_std[algo_list[2]],
    #        width=xwidth, align='edge',
    #        color=figtools.COLOURS['rp'], ecolor='k', edgecolor='k',
    #        linewidth=.4*lwidth, label=r''+'1st layer PMD \n2nd layer RP')

    # x1 = np.array([pl.getp(item, 'x') for item in b1])
    # x0 = x1 + np.array([pl.getp(item, 'width') for item in b1])
    # x1 = x1 + np.array([pl.getp(item, 'width') for item in b2])

    # ax.hlines(perfd_xx_avg[algo_list[0]], x0, x1, colors='r', lw=lwidth*1.6, label=r''+'PMD \n1 layer')

    # myLegend(ax, loc='center left', add_frame=False, bbox_to_anchor=(.95,0.5),
    #              fontsize=labelsize, handlelength=.6, numpoints=2, ncol=1, labelspacing=1., handletextpad=.2, columnspacing=.8)


    # myLegend(ax, loc='lower center', add_frame=False, bbox_to_anchor=(.5,0.98),
    #              fontsize=labelsize, handlelength=.6, numpoints=2, ncol=2, labelspacing=1., handletextpad=.2, columnspacing=.8)
    myLegend(ax, loc='center left', add_frame=False, bbox_to_anchor=(.95,0.5),
                 fontsize=labelsize, handlelength=.6, numpoints=2, ncol=1, labelspacing=1., handletextpad=.2, columnspacing=.8)

    if pshow:
        pl.tight_layout()
        pl.show()


def plot_data_3hl(ax=None, eps=.02):

    if ax is None:
        pl.figure(figsize=(4,4))
        ax = pl.gca()
        pshow = True
    else:
        pshow = False

    xvals = np.arange(len(NHS))+0.5
    xwidth = .3

    perfd_xxx_avg = {}
    perfd_xxx_std = {}
    perfd_2hl_avg = {}
    perfd_2hl_std = {}

    for algos in ALGOS_3HL:

        algo_123c = '-'.join(algos)
        algo_1  = '-'.join(algos[0:2] + algos[3:])

        print(algo_123c)
        print(algo_12c)

        perfs_xxx_avg, perfs_xxx_std = [], []
        perfs_2hl_avg, perfs_2hl_std = [], []
        for n_h in NHS:
            namestring_xxx = "deepopt_3hl_%s_algo123c=%s_nh12=%d-%d-%d_%s_ro=%s.p"%(DATASET, algo_123c, n_h, n_h, n_h, TASKTYPE, READOUT)
            p_avg, p_std = load_perf(namestring_xxx)
            perfs_xxx_avg.append(p_avg)
            perfs_xxx_std.append(p_std)

            namestring_2hl = "deepopt_2hl_%s_algo12c=%s_nh12=%d-%d_%s_ro=%s.p"%(DATASET, algo_12c, n_h, n_h, TASKTYPE, READOUT)
            p_avg, p_std = load_perf(namestring_2hl)
            perfs_2hl_avg.append(p_avg)
            perfs_2hl_std.append(p_std)

        perfd_xxx_avg[algo_123c] = perfs_xxx_avg
        perfd_2hl_avg[algo_12c]  = perfs_2hl_avg

        perfd_xxx_std[algo_123c] = perfs_xxx_std
        perfd_2hl_std[algo_12c]  = perfs_2hl_std

    print('xxx:', perfd_xxx_avg)
    print('2hl:', perfd_2hl_avg)

    # case xx
    xlabel = "no. of hidden units"
    ax = figtools.perfAx(ax,
                ylim=(70., 100.),
                skip_yt=False,
                add_xticklabels=True,
                add_yticklabels=True,
                xlabel=xlabel,
                add_ylabel=True,
                nhs=[[n_h, n_h, n_h] for n_h in NHS])

    b1 = ax.bar(xvals, perfd_xxx_avg['pmdd-pmd-pmd-lstsq'], yerr=perfd_xxx_std['pmdd-pmd-pmd-lstsq'],
           width=-xwidth, align='edge',
           color=figtools.COLOURS['pmdd'], ecolor='k', edgecolor='k',
           hatch='',
           linewidth=.4*lwidth,
           label=r''+'3 layers PMD')

    b2 = ax.bar(xvals, perfd_xxx_avg['pmdd-pmd-rp-na'], yerr=perfd_xxx_std['pmdd-pmd-rp-na'],
    # b2 = ax.bar(xvals, perfd_xxx_avg['pmdd-pmd-rp-na'], yerr=perfd_xxx_std['pmdd-pmd-rp-na'],
           width=xwidth, align='edge',
           color=figtools.COLOURS['rp'], ecolor='k', edgecolor='k',
           hatch='',
           linewidth=.4*lwidth,
           label=r''+'1st & 2nd layers PMD\n 3rd layer RP')

    x1 = np.array([pl.getp(item, 'x') for item in b1])
    x0 = x1 + np.array([pl.getp(item, 'width') for item in b1])
    x2 = x1 + np.array([pl.getp(item, 'width') for item in b2])

    ax.hlines(perfd_2hl_avg['pmdd-pmd-lstsq'], x0, x1, colors='r', lw=lwidth*1.6)
    ax.hlines(perfd_2hl_avg['pmdd-pmd-lstsq'], x1, x2, colors='r', lw=lwidth*1.6, label=r''+'PMD \n2 layers')

    myLegend(ax, loc='center left', add_frame=False, bbox_to_anchor=(.95,0.5),
                 fontsize=labelsize, handlelength=1.0, numpoints=2, ncol=1, labelspacing=1., handletextpad=.2, columnspacing=.8)

    if pshow:
        pl.tight_layout()
        pl.show()


def plot_data_2hl_(ax=None, eps=.02):

    if ax is None:
        pl.figure(figsize=(4,4))
        ax = pl.gca()
        pshow = True
    else:
        pshow = False

    xvals = np.arange(len(NHS))+0.5
    xwidth = .3

    perfd_xx_avg = {}
    perfd_xx_std = {}
    perfd_1hl_avg = {}
    perfd_1hl_std = {}

    for algos in ALGOS_3HL:

        algo_12c = '-'.join(algos)
        algo_1  = algos[0]

        print(algo_12c)
        print(algo_1)

        perfs_xx_avg, perfs_xx_std = [], []
        perfs_1hl_avg, perfs_1hl_std = [], []
        for n_h in NHS:
            namestring_xxx = "deepopt_2hl_%s_algo12c=%s_nh12=%d-%d_%s_ro=%s.p"%(DATASET, algo_12c, n_h, n_h, TASKTYPE, READOUT)
            p_avg, p_std = load_perf(namestring_xxx)
            perfs_xx_avg.append(p_avg)
            perfs_xx_std.append(p_std)

            namestring_2hl = "biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(DATASET, algo_1, n_h, TASKTYPE, READOUT)
            p_avg, p_std = load_perf(namestring_2hl)
            perfs_1hl_avg.append(p_avg)
            perfs_1hl_std.append(p_std)

        perfd_xx_avg[algo_12c] = perfs_xx_avg
        perfd_1hl_avg[algo_1]  = perfs_1hl_avg

        perfd_xx_std[algo_12c] = perfs_xx_std
        perfd_1hl_std[algo_1]  = perfs_1hl_std

    print('xxx:', perfd_xx_avg)
    print('2hl:', perfd_1hl_avg)

    # case xx
    xlabel = "no. of hidden units"
    ax = figtools.perfAx(ax,
                ylim=(70., 100.),
                skip_yt=False,
                add_xticklabels=True,
                add_yticklabels=True,
                xlabel=xlabel,
                add_ylabel=True,
                nhs=[[n_h, n_h] for n_h in NHS])

    b1 = ax.bar(xvals, perfd_xx_avg['pmdd-rp-na'], yerr=perfd_xx_std['pmdd-rp-na'],
           width=-xwidth, align='edge',
           color=figtools.COLOURS['rp'], ecolor='k', edgecolor='k',
           hatch='',
           linewidth=.4*lwidth,
           label=r''+'1st layer PMD \n2nd layer RP')

    b2 = ax.bar(xvals, perfd_xx_avg['rp-rp-na'], yerr=perfd_xx_std['rp-rp-na'],
           width=xwidth, align='edge',
           color="White", ecolor='k', edgecolor='k',
           hatch='',
           linewidth=.4*lwidth,
           label=r''+'RP 2 layers')

    x1 = np.array([pl.getp(item, 'x') for item in b1])
    x0 = x1 + np.array([pl.getp(item, 'width') for item in b1])
    x2 = x1 + np.array([pl.getp(item, 'width') for item in b2])

    ax.hlines(perfd_1hl_avg['pmdd'], x0, x1, colors='r', lw=lwidth*1.6)
    ax.hlines(perfd_1hl_avg['pmdd'], x1, x2, colors='r', lw=lwidth*1.6, label=r''+'PMD \n1 layer')

    myLegend(ax, loc='center left', add_frame=False, bbox_to_anchor=(.95,0.5),
                 fontsize=labelsize, handlelength=.6, numpoints=2, ncol=1, labelspacing=1., handletextpad=.2, columnspacing=.8)

    if pshow:
        pl.tight_layout()
        pl.show()


def plot_figure():

    xcoo1 = getXCoords([.15, .5, .6, .5, .6, .5, .45])
    ycoo = getYCoords([.15,.5,0.25,.5,.15])
    print(ycoo)

    pl.figure("semi-supervised", figsize=(14,6))
    labels = getAnnotations(5)

    # annotations
    gs0_ = GridSpec(1,1)
    gs0_.update(top=ycoo[-1], bottom=ycoo[-2], left=xcoo[0], right=xcoo[1], hspace=0.1, wspace=0.1)
    gs1_ = GridSpec(1,1)
    gs1_.update(top=ycoo[-1], bottom=ycoo[-2], left=xcoo[2], right=xcoo[3], hspace=0.1, wspace=0.05)
    ax0_, ax1_ = noFrameAx(pl.subplot(gs0_[0,0])), noFrameAx(pl.subplot(gs1_[0,0]))
    ax0_.add_artist(labels[0])
    ax1_.add_artist(labels[1])

    gs0 = GridSpec(1,1)
    gs0.update(top=ycoo[-3], bottom=ycoo[-4], left=xcoo[0], right=xcoo[1], hspace=0.1, wspace=0.1)
    gs1 = GridSpec(1,1)
    gs1.update(top=ycoo[-3], bottom=ycoo[-4], left=xcoo[2], right=xcoo[3], hspace=0.1, wspace=0.05)

    ax = pl.subplot(gs0[0,0])
    axes = [pl.subplot(gs1[0,ii]) for ii in range(2)]

    plot_data_1hl("PMD", ax)
    plot_data_2hl("PMD", axes)

    pl.savefig(paths.fig_path + "biasadaptation_v2_fig4.svg", transparent=True)
    pl.show()


def plot_figure_v2():

    xcoo1 = getXCoords([.25, .5, .15, .5, .15, .5, .1])
    xcoo2 = getXCoords([.25, .6, .1, .6, .6])
    ycoo = getYCoords([.35,.5,0.4,.5,.2])
    print(ycoo)

    pl.figure("semi-supervised", figsize=(7,7))
    labels = getAnnotations(5, size_factor=1.2)

    # annotations
    # gs0_ = GridSpec(1,1)
    # gs0_.update(top=ycoo[-1], bottom=ycoo[-2], left=xcoo[0], right=xcoo[1], hspace=0.1, wspace=0.1)
    # gs1_ = GridSpec(1,1)
    # gs1_.update(top=ycoo[-1], bottom=ycoo[-2], left=xcoo[2], right=xcoo[3], hspace=0.1, wspace=0.05)
    # ax0_, ax1_ = noFrameAx(pl.subplot(gs0_[0,0])), noFrameAx(pl.subplot(gs1_[0,0]))
    # ax0_.add_artist(labels[0])
    # ax1_.add_artist(labels[1])

    # 1 hidden layer
    gs0 = GridSpec(1,1)
    gs0.update(top=ycoo[-1], bottom=ycoo[-2], left=xcoo1[0], right=xcoo1[1], hspace=0.1, wspace=0.1)
    gs1 = GridSpec(1,1)
    gs1.update(top=ycoo[-1], bottom=ycoo[-2], left=xcoo1[2], right=xcoo1[3], hspace=0.1, wspace=0.05)
    gs2 = GridSpec(1,1)
    gs2.update(top=ycoo[-1], bottom=ycoo[-2], left=xcoo1[4], right=xcoo1[5], hspace=0.1, wspace=0.05)

    ax0 = pl.subplot(gs0[0,0])
    ax1 = pl.subplot(gs1[0,0])
    ax2 = pl.subplot(gs2[0,0])

    ax0.add_artist(labels[0])
    # ax1.add_artist(labels[1])
    # ax2.add_artist(labels[2])

    # plot_data_1hl("1HL", ax0, add_yticklabels=True, add_ylabel=True, add_xlabel=False)
    # plot_data_1hl("SC", ax1, add_yticklabels=False, add_ylabel=False, add_xlabel=True)
    # plot_data_1hl("W", ax2, add_yticklabels=False, add_ylabel=False, add_xlabel=False)

    # 2 hidden layers
    gs0 = GridSpec(1,1)
    gs0.update(top=ycoo[-3], bottom=ycoo[-4], left=xcoo2[0], right=xcoo2[1], hspace=0.1, wspace=0.1)
    gs1 = GridSpec(1,1)
    gs1.update(top=ycoo[-3], bottom=ycoo[-4], left=xcoo2[2], right=xcoo2[3], hspace=0.1, wspace=0.05)
    # gs2 = GridSpec(1,1)
    # gs2.update(top=ycoo[-3], bottom=ycoo[-4], left=xcoo2[4], right=xcoo2[5], hspace=0.1, wspace=0.05)
    # gs3 = GridSpec(1,1)
    # gs3.update(top=ycoo[-3], bottom=ycoo[-4], left=xcoo2[6], right=xcoo2[7], hspace=0.1, wspace=0.05)

    axes0 = [pl.subplot(gs0[0,0]), pl.subplot(gs1[0,0])]
    # axes1 = [pl.subplot(gs2[0,0]), pl.subplot(gs3[0,0])]

    axes0[0].add_artist(labels[1])
    # axes1[0].add_artist(labels[4])

    plot_data_2hl(axes0)
    # plot_data_2hl("BPO", axes1)


    pl.savefig(paths.fig_path + "biasadaptation_v2_fig4.svg", transparent=True)
    pl.show()

def plot_figure_v3():

    xcoo1 = getXCoords([.25, .45, .35, .45, .35, .45, .35, .6, .6])
    # xcoo2 = getXCoords([.25, .5, .1, .5, .8,.5,.7])
    ycoo = getYCoords([1.0,1.5,.5])
    print(ycoo)

    pl.figure("semi-supervised", figsize=(14,4.0))
    labels = getAnnotations(6, size_factor=1.2)

    # 1 hidden layer
    gs0 = GridSpec(1,1)
    gs0.update(top=ycoo[-1], bottom=ycoo[-2], left=xcoo1[0], right=xcoo1[1], hspace=0.1, wspace=0.1)
    gs1 = GridSpec(1,1)
    gs1.update(top=ycoo[-1], bottom=ycoo[-2], left=xcoo1[2], right=xcoo1[3], hspace=0.1, wspace=0.05)
    gs2 = GridSpec(1,1)
    gs2.update(top=ycoo[-1], bottom=ycoo[-2], left=xcoo1[4], right=xcoo1[5], hspace=0.1, wspace=0.05)

    ax0 = pl.subplot(gs0[0,0])
    ax1 = pl.subplot(gs1[0,0])
    ax2 = pl.subplot(gs2[0,0])

    ax0.add_artist(labels[0])
    ax1.add_artist(labels[1])
    ax2.add_artist(labels[2])

    plot_data_1hl("1HL", ax0, add_yticklabels=True, add_ylabel=True, add_xlabel=True)
    plot_data_1hl("SC", ax1, add_yticklabels=True, add_ylabel=True, add_xlabel=True)
    plot_data_1hl("W", ax2, add_yticklabels=True, add_ylabel=True, add_xlabel=True)

    # 2 hidden layers
    gs0 = GridSpec(1,1)
    gs0.update(top=ycoo[-1], bottom=ycoo[-2], left=xcoo1[6], right=xcoo1[7], hspace=0.1, wspace=0.1)
    # gs1 = GridSpec(1,1)
    # gs1.update(top=ycoo[-3], bottom=ycoo[-4], left=xcoo2[2], right=xcoo2[3], hspace=0.1, wspace=0.05)
    # gs2 = GridSpec(1,1)
    # gs2.update(top=ycoo[-3], bottom=ycoo[-4], left=xcoo2[4], right=xcoo2[5], hspace=0.1, wspace=0.05)

    ax0 = pl.subplot(gs0[0,0])
    ax0.add_artist(labels[3])

    plot_data_2hl(ax0)

    # ax = pl.subplot(gs2[0,0])
    # ax.add_artist(labels[4])
    # # plot_data_3hl(ax)
    # plot_data_2hl_(ax)

    pl.savefig(paths.fig_path + "biasadaptation_v3_fig4.svg", transparent=True)
    pl.show()


if __name__ == "__main__":
    # plot_data_1hl("SC")
    # plot_data_2hl("BPO")
    # plot_data_3hl()
    plot_figure_v3()



