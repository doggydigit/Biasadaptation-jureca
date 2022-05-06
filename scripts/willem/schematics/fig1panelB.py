import numpy as np
import torch

from datarep import paths
from datarep.matplotlibsettings import *

import random
import sys
sys.path.append('..')

from biasadaptation.utils import utils
import helperfuncs


def plot_schema(axes=None, task_idx=1, seed_t=12, seed_w=18):

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
        axn0 = axes[2]

        pshow = False

    ax_text.text(.5,.5, r'Task %d'%task_idx, fontsize=labelsize, ha='center', va='center')

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

    seed_b = random.randint(0, 1000000)
    np.random.seed(seed_b)
    xp0 = np.zeros(10)
    yp0 = np.arange(10, dtype=float)
    yp0 -= np.mean(yp0)
    cp0 = np.random.rand(10)

    xp1 = np.ones(1)
    yp1 = np.arange(1)
    cp1 = np.random.rand(1)

    cmap = cm.get_cmap('Blues')

    axn0.scatter(xp0, yp0, c=cp0, s=100, cmap=cmap, edgecolors='k')
    axn0.scatter(xp1, yp1, c=cmap(cp1), s=100, edgecolors='k')

    # draw arrows
    np.random.seed(seed_w)
    ymin, ymax = np.min(yp0), np.max(yp0)
    y0s = np.random.rand(20) * (ymax - ymin) + ymin
    y1s = np.concatenate((np.random.choice(yp0, 10),
                          np.random.choice(yp0, 10, replace=False)))
    cs = np.random.rand(20)
    cmap = cm.get_cmap('Reds')

    for kk, (y0, y1, c_) in enumerate(zip(y0s, y1s, cs)):
        prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.4", ec=cmap(c_), fc=cmap(c_), shrinkA=0, shrinkB=0)
        axn0.annotate("", xytext=(-1.,y0), xy=(-.2,y1), arrowprops=prop)

    dys = np.linspace(-.8, .8, len(yp0))
    for y0, dy in zip(yp0, dys):
        prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.4", ec=cmap(.5), fc=cmap(.5), shrinkA=0, shrinkB=0)
        axn0.annotate("", xytext=(.2,y0), xy=(.8,yp1[0]+dy), arrowprops=prop)



    axn0.set_xlim((-1,1.2))

    if pshow:
        pl.show()


def get_axes(gs0, gs1, gst):
    axt = noFrameAx(pl.subplot(gst[0,0]))

    axtext = noFrameAx(pl.subplot(gs0[4,:]))
    axtext.text(.5,.5, r'vs', fontsize=labelsize, ha='center', va='center')

    axes = {'top': [], 'bottom': []}
    for kk in range(16):
        ii, jj = kk%4, kk//4

        axes['top'].append(noFrameAx(pl.subplot(gs0[ii,jj])))
        axes['bottom'].append(noFrameAx(pl.subplot(gs0[ii+5,jj])))

    axl1 = noFrameAx(pl.subplot(gs1[0,0]))

    return axt, axes, axl1


def plot_panel():

    xcoords = getXCoords([0.05, 0.3, 0.05, 0.5, 0.2, 0.3, 0.05, 0.5, 0.1, 0.1, 0.1])
    ycoords = [0.1, 0.85]

    fig = pl.figure('Bias schema', figsize=(6,2.5))

    gst0 = GridSpec(1,1)
    gst0.update(top=0.95, bottom=0.9, left=xcoords[0], right=xcoords[1], hspace=0.01, wspace=0.01)
    gs0 = GridSpec(9,4)
    gs0.update(top=ycoords[-1], bottom=ycoords[-2], left=xcoords[0], right=xcoords[1], hspace=0.01, wspace=0.01)
    gs1 = GridSpec(1,1)
    gs1.update(top=ycoords[-1], bottom=ycoords[-2], left=xcoords[2], right=xcoords[3], hspace=0.01, wspace=0.01)

    gst0_ = GridSpec(1,1)
    gst0_.update(top=0.95, bottom=0.9, left=xcoords[4], right=xcoords[5], hspace=0.01, wspace=0.01)
    gs0_ = GridSpec(9,4)
    gs0_.update(top=ycoords[-1], bottom=ycoords[-2], left=xcoords[4], right=xcoords[5], hspace=0.01, wspace=0.01)
    gs1_ = GridSpec(1,1)
    gs1_.update(top=ycoords[-1], bottom=ycoords[-2], left=xcoords[6], right=xcoords[7], hspace=0.01, wspace=0.01)

    gs = GridSpec(1,1)
    gs.update(top=ycoords[-1], bottom=ycoords[0], left=xcoords[8], right=xcoords[9], hspace=0.01, wspace=0.01)

    seed_w = 112
    plot_schema(axes=get_axes(gs0, gs1, gst0), task_idx=1, seed_t=11, seed_w=seed_w)
    plot_schema(axes=get_axes(gs0_, gs1_, gst0_), task_idx=2, seed_t=14, seed_w=seed_w)

    ax = noFrameAx(pl.subplot(gs[0,0]))
    ax.text(.5,.5, r'...', fontsize=labelsize, ha='center', va='center')

    pl.savefig(paths.fig_path + "biasadapatation_fig1B.svg", transparent=True)


    pl.show()

if __name__ == "__main__":
    # plot_schema()
    plot_panel()