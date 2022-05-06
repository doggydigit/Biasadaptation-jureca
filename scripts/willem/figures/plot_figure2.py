import numpy as np
import torch

import os
import pickle, pickle5
import sys

# from biasadaptation.biasfit.specificbiasfit import ReLuFit
from biasadaptation.biasfit.biasfit import ReLuFit
from biasadaptation.utils import utils

from datarep.matplotlibsettings import *
import datarep.paths as paths

sys.path.append('..')
import figtools
import helperfuncs

# matthias scripts imports
sys.path.append('../../matthias/')
from tasks_2d_helper import task_2d_label


NHS = [25, 100, 500]
DATAFILE = "/Users/wybo/Data/results/biasopt/Fig2C.pickle"

TRAIN_MULTI  = ["train_sg_full", "test_train_bmr_full", "polish_b_full"]
TRAIN_TRANSF = ["transfer_bmr_l1o", "transfer_b_l1o_b_w"]


def load_data():
    with open(DATAFILE, 'rb') as f:
        res = pickle5.load(f)
    print(list(res.keys()))
    return res


def get_arch(nh, nl):
    return [nh for _ in range(nl)]


def get_archstr(nh, nl):
    ls = ['%d'%nh for _ in range(nl)]
    return '[' + ', '.join(ls) + ']'


def plot_data(ax0, add_legend=True):
    # ax0, ax1 = axes
    res = load_data()

    # ax0.set_title("Multi-task", fontsize=labelsize)
    # ax1.set_title("Transfer", fontsize=labelsize)

    ax0 = figtools.perfAx(ax0,
                ylim=(85., 100.),
                skip_yt=False,
                add_xticklabels=True,
                add_yticklabels=True,
                xlabel="no. of hidden units",
                add_ylabel=True,
                nhs=[get_arch(nh, 1) for nh in NHS] + [get_arch(nh, 3) for nh in NHS])
    # ax1 = figtools.perfAx(ax1,
    #             ylim=(70., 100.),
    #             skip_yt=False,
    #             add_xticklabels=True,
    #             add_yticklabels=False,
    #             xlabel="",
    #             add_ylabel=False,
    #             nhs=[get_arch(nh, 1) for nh in NHS] + [get_arch(nh, 3) for nh in NHS])
    xvals = np.arange(2*len(NHS))+0.5
    xwidth = .15

    # fully trained
    perfs_avg = [res[TRAIN_MULTI[0]][get_archstr(nh, 1)]['mean'] for nh in NHS] + \
                [res[TRAIN_MULTI[0]][get_archstr(nh, 3)]['mean'] for nh in NHS]
    perfs_std = [res[TRAIN_MULTI[0]][get_archstr(nh, 1)]['std']  for nh in NHS] + \
                [res[TRAIN_MULTI[0]][get_archstr(nh, 3)]['std']  for nh in NHS]

    print(perfs_avg)

    b_ft = ax0.bar(xvals-2*xwidth, perfs_avg,
           yerr=perfs_std, width=xwidth, align='center',
           color=figtools.COLOURS["rpw"], ecolor='k', edgecolor='k',
           linewidth=.4*lwidth, label=figtools.LABELS['rpw'][0])

    # multi-task binary readout
    perfs_avg = [res[TRAIN_MULTI[1]][get_archstr(nh, 1)]['mean'] for nh in NHS] + \
                [res[TRAIN_MULTI[1]][get_archstr(nh, 3)]['mean'] for nh in NHS]
    perfs_std = [res[TRAIN_MULTI[1]][get_archstr(nh, 1)]['std']  for nh in NHS] + \
                [res[TRAIN_MULTI[1]][get_archstr(nh, 3)]['std']  for nh in NHS]

    b_mr = ax0.bar(xvals-xwidth, perfs_avg,
           yerr=perfs_std, width=xwidth, align='center',
           color=figtools.COLOURS["br"], ecolor='k', edgecolor='k',
           linewidth=.4*lwidth, label=figtools.LABELS['br'][0])



    # transfer binary readout
    perfs_avg = [res[TRAIN_TRANSF[0]][get_archstr(nh, 1)]['mean'] for nh in NHS] + \
                [res[TRAIN_TRANSF[0]][get_archstr(nh, 3)]['mean'] for nh in NHS]
    perfs_std = [res[TRAIN_TRANSF[0]][get_archstr(nh, 1)]['std']  for nh in NHS] + \
                [res[TRAIN_TRANSF[0]][get_archstr(nh, 3)]['std']  for nh in NHS]

    ax0.bar(xvals, perfs_avg,
           yerr=perfs_std, width=-xwidth, align='center',
           color=figtools.COLOURS["br"], ecolor='k', edgecolor='k',# hatch="///",
           linewidth=.4*lwidth, label=figtools.LABELS["br"][0][0])


    # multi-task bias-adaptation
    perfs_avg = [res[TRAIN_MULTI[2]][get_archstr(nh, 1)]['mean'] for nh in NHS] + \
                [res[TRAIN_MULTI[2]][get_archstr(nh, 3)]['mean'] for nh in NHS]
    perfs_std = [res[TRAIN_MULTI[2]][get_archstr(nh, 1)]['std']  for nh in NHS] + \
                [res[TRAIN_MULTI[2]][get_archstr(nh, 3)]['std']  for nh in NHS]

    b_ba = ax0.bar(xvals+xwidth, perfs_avg,
           yerr=perfs_std, width=xwidth, align='center',
           color=figtools.COLOURS["bpo"], ecolor='k', edgecolor='k',
           linewidth=.4*lwidth, label=figtools.LABELS['bpo'][0])


    # transfer bias-adaptation
    perfs_avg = [res[TRAIN_TRANSF[1]][get_archstr(nh, 1)]['mean'] for nh in NHS] + \
                [res[TRAIN_TRANSF[1]][get_archstr(nh, 3)]['mean'] for nh in NHS]
    perfs_std = [res[TRAIN_TRANSF[1]][get_archstr(nh, 1)]['std']  for nh in NHS] + \
                [res[TRAIN_TRANSF[1]][get_archstr(nh, 3)]['std']  for nh in NHS]

    ax0.bar(xvals+2*xwidth, perfs_avg,
           yerr=perfs_std, width=xwidth, align='center',
           color=figtools.COLOURS["bpo"], ecolor='k', edgecolor='k',# hatch="///",
           linewidth=.4*lwidth, label=figtools.LABELS["bpo"][0][0])

    # handles, labels = ax.get_legend_handles_labels()
    # create handle for transfer
    # b_tr = Rectangle((0,0),1,1, facecolor="White", edgecolor='k', label="Transfer")
    b_tr = Rectangle((0,0),1,1, facecolor="White", edgecolor='k', hatch='///', label="Transfer")
    # handles.append(handle)
    # labels.append("Transfer")

    if add_legend:
        myLegend(ax0, loc='upper left', add_frame=False, bbox_to_anchor=(.95,1.1), handles=[b_ft, b_mr, b_ba, b_tr],
                 fontsize=labelsize, handlelength=.6, numpoints=2, ncol=1, labelspacing=1.0, handletextpad=.2, columnspacing=.8)


def plot_2d_tasks(axes, model=None):
    nr_tasks = len(axes)
    tasks = [30, 42, 40, 46, 34, 22, 6, 26, 27, 29, 2, 13]
    assert len(tasks) == nr_tasks

    datamin, datamax, inputsize, datasize = 0., 1., 2, 2048
    aa = torch.linspace(datamin, datamax, 100)
    input_x, input_y = torch.meshgrid(aa, aa)
    input_x = torch.reshape(input_x, (10000,))
    input_y = torch.reshape(input_y, (10000,))
    input_data = torch.stack((input_x, input_y), 1)

    for ii, task in enumerate(tasks):
        if model:
            data_labels = model.forward(input_data, task)
            g = data_labels > 0.
            g = g.squeeze()
            data_labels = task_2d_label(task, input_data)
            truth = data_labels > 0.5
            errors = torch.logical_xor(truth, g)
        else:
            data_labels = task_2d_label(task, input_data)
            g = data_labels > 0.5
        r = [not i for i in g]
        a = axes[ii]
        a.plot(input_data[g, 0], input_data[g, 1], 's', color='orange', markersize=0.9)
        a.plot(input_data[r, 0], input_data[r, 1], 's', color="deepskyblue",  markersize=0.9)
        if model:
            a.plot(input_data[errors, 0], input_data[errors, 1], 's', color="k", markersize=0.1)
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.xaxis.set_ticks_position('none')
        a.yaxis.set_ticks_position('none')
        a.axis('off')


def plot_2d_panel(axes=None):
    if axes is None:
        pl.figure()
        gs = pl.GridSpec(3,4)
        axes = [pl.subplot(gs[ii,jj]) for ii in range(3) for jj in range(4)]
        pshow = True
    else:
        pshow = False

    net = [50, 50, 50, 50]
    seed = 2
    with open(os.path.join(paths.data_path, "weight_matrices/EMNIST_tsai/biaslearner_{}_seed_{}.pickle".format(net, seed)), 'rb') as file:
        ws, bs = pickle.load(file)
    plot_2d_tasks(axes, model=ReLuFit(ws, bs, readout="tanh"))

    if pshow:
        pl.show()


def plot_task_schema(axes=None, task_idx=1, seed_t=12):

    if axes is None:

        pl.figure('-1')
        gs0 = GridSpec(4,4)

        axes_top = []
        for kk in range(16):
            ii, jj = kk%4, kk//4
            axes_top.append(noFrameAx(pl.subplot(gs0[ii,jj])))

        pl.figure('+1')
        gs0 = GridSpec(4,4)

        axes_bottom = []
        for kk in range(16):
            ii, jj = kk%4, kk//4
            axes_bottom.append(noFrameAx(pl.subplot(gs0[ii,jj])))

        pl.figure('net')
        axn0 = noFrameAx(pl.gca())

        pshow = True

    else:
        ax_text = axes[0]

        axes_top = axes[1]['top']
        axes_bottom = axes[1]['bottom']

        pshow = False

    ax_text.text(.5,.5, r'Task $%s$'%str(task_idx), fontsize=labelsize, ha='center', va='center')

    # train and test datasets
    source_train = helperfuncs.get_dataset('EMNIST', train=True,  rotate=True, path='~/Data/')
    source_test = helperfuncs.get_dataset('EMNIST', train=False,  rotate=True, path='~/Data/')

    task = helperfuncs.sample_binary_tasks_(1, dataset='EMNIST', task_type='1vall', seed=seed_t)[0]

    data_loaders = helperfuncs.construct_knm_dataloader_triplet(
                            'EMNIST', task, copy.deepcopy(task),
                            source_train, source_test,
                            1000, 100, 100)

    (xdata, xlabel), (xtask, xtarget) = next(iter(data_loaders[0]))

    idx0 = torch.where(xtarget < 0)[0]
    idx1 = torch.where(xtarget > 0)[0]

    for kk, ax in enumerate(axes_top[:-1]):
        ax.imshow(utils.to_image_mnist(xdata[idx0[kk]].numpy()))
    axes_top[-1].text(.5,.5, r'...', fontsize=labelsize, ha='center', va='center')


    for kk, ax in enumerate(axes_bottom[:-1]):
        ax.imshow(utils.to_image_mnist(xdata[idx1[kk]].numpy()))
    axes_bottom[-1].text(.5,.5, r'...', fontsize=labelsize, ha='center', va='center')

def get_axes(gs0, gst):
    axt = noFrameAx(pl.subplot(gst[0,0]))

    axtext = noFrameAx(pl.subplot(gs0[4,:]))
    axtext.text(.5,.5, r'vs', fontsize=labelsize, ha='center', va='center')

    axes = {'top': [], 'bottom': []}
    for kk in range(16):
        ii, jj = kk%4, kk//4

        axes['top'].append(noFrameAx(pl.subplot(gs0[ii,jj])))
        axes['bottom'].append(noFrameAx(pl.subplot(gs0[ii+5,jj])))

    return axt, axes



def plot_figure():

    xcoo = getXCoords([.25, .6, .1, .2, .4, .6, .6])
    ycoo = getYCoords([.5,.5,0.5,.5,.2])
    print(ycoo)

    pl.figure("semi-supervised", figsize=(7,7.5))
    labels = getAnnotations(2, size_factor=1.2)

    # 1 hidden layer
    gs0 = GridSpec(1,1)
    gs0.update(top=ycoo[-1], bottom=ycoo[-2], left=xcoo[0], right=xcoo[1], hspace=0.1, wspace=0.1)
    gs1 = GridSpec(1,1)
    gs1.update(top=ycoo[-1], bottom=ycoo[-2], left=xcoo[2], right=xcoo[3], hspace=0.1, wspace=0.05)

    ax0 = pl.subplot(gs0[0,0])
    ax1 = pl.subplot(gs1[0,0])
    ax0.add_artist(labels[0])

    plot_data([ax0, ax1], 1, add_legend=True)

    # 2 hidden layers
    gs0 = GridSpec(1,1)
    gs0.update(top=ycoo[-3], bottom=ycoo[-4], left=xcoo[0], right=xcoo[1], hspace=0.1, wspace=0.1)
    gs1 = GridSpec(1,1)
    gs1.update(top=ycoo[-3], bottom=ycoo[-4], left=xcoo[2], right=xcoo[3], hspace=0.1, wspace=0.05)

    ax0 = pl.subplot(gs0[0,0])
    ax1 = pl.subplot(gs1[0,0])

    ax0.add_artist(labels[1])

    plot_data([ax0, ax1], 3, add_legend=False)

    pl.savefig(paths.fig_path + "biasadaptation_v2_fig2.svg", transparent=True)
    pl.show()


def plot_figure_v3():

    xcoo = getXCoords([.10, 1.2, .2, .4,.04,0.05,0.04,.4, 0.35, 1.0, .6])

    pl.figure("supervised", figsize=(14,4.5))
    labels = getAnnotations(3, size_factor=1.2)

    # 2D tasks
    gs = GridSpec(3,4)
    gs.update(top=0.85, bottom=0.15, left=xcoo[0], right=xcoo[1], hspace=0.1, wspace=0.1)

    axes = [pl.subplot(gs[ii,jj]) for ii in range(3) for jj in range(4)]
    axes[0].add_artist(labels[0])

    # plot_2d_panel(axes)

    # plot EMNIST tasks
    gst0 = GridSpec(1,1)
    gst0.update(top=0.95, bottom=0.9, left=xcoo[2], right=xcoo[3], hspace=0.01, wspace=0.01)
    gs0 = GridSpec(9,4)
    gs0.update(top=0.85, bottom=0.15, left=xcoo[2], right=xcoo[3], hspace=0.01, wspace=0.01)
    axes = get_axes(gs0, gst0)
    axes[1]['top'][0].add_artist(labels[1])


    plot_task_schema(axes, seed_t=11, task_idx=1)

    gs = GridSpec(1,1)
    gs.update(top=0.85, bottom=0.15, left=xcoo[4], right=xcoo[5], hspace=0.01, wspace=0.01)
    ax = noFrameAx(pl.subplot(gs[0,0]))
    ax.text(.5,.5, r'...', fontsize=labelsize, ha='center', va='center')

    gst0 = GridSpec(1,1)
    gst0.update(top=0.95, bottom=0.9, left=xcoo[6], right=xcoo[7], hspace=0.01, wspace=0.01)
    gs0 = GridSpec(9,4)
    gs0.update(top=0.85, bottom=0.15, left=xcoo[6], right=xcoo[7], hspace=0.01, wspace=0.01)
    axes = get_axes(gs0, gst0)

    plot_task_schema(axes, seed_t=12, task_idx="N")

    # 1 hidden layer
    gs_ = GridSpec(1,2)
    gs_.update(top=0.85, bottom=0.45, left=xcoo[8], right=xcoo[9], hspace=0.1, wspace=0.1)

    ax0 = pl.subplot(gs_[0,:])
    # ax1 = pl.subplot(gs_[0,1])
    ax0.add_artist(labels[2])

    plot_data(ax0, add_legend=True)

    pl.savefig(paths.fig_path + "biasadaptation_v3_fig2.svg", transparent=True)
    pl.savefig(paths.fig_path + "biasadaptation_v3_fig2.pdf", transparent=True)
    pl.savefig(paths.fig_path + "biasadaptation_v3_fig2.png", transparent=True)
    pl.show()


if __name__ == "__main__":
    # plot_figure()
    plot_figure_v3()
    # plot_task_schema()


