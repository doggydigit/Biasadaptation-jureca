import numpy as np
import torch

from datarep import paths
from datarep.matplotlibsettings import *

import random
import sys
sys.path.append('..')

from biasadaptation.utils import utils
import helperfuncs


def plot_schema(axes=None, task_idx=1, seed_t=12, seed_w=18,
                n_layers=1, n_neuron=10, n_arrow=20, n_sample=100000):
    assert n_arrow >= n_neuron

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

    task = helperfuncs.sample_binary_tasks_(1, nsample=n_sample, dataset='EMNIST', task_type='1vall', seed=seed_t)[0]

    data_loaders = helperfuncs.construct_knm_dataloader_triplet(
                            'EMNIST', task, copy.deepcopy(task),
                            source_train, source_test,
                            1000, 100, 100)

    (xdata, xlabel), (xtask, xtarget) = next(iter(data_loaders[0]))

    idx0 = torch.where(xtarget < 0)[0]
    idx1 = torch.where(xtarget > 0)[0]

    # for kk, ax in enumerate(axes_top[:-1]):
    #     ax.imshow(utils.to_image_mnist(xdata[idx0[kk]].numpy()))
    # axes_top[-1].text(.5,.5, r'...', fontsize=labelsize, ha='center', va='center')


    # for kk, ax in enumerate(axes_bottom[:-1]):
    #     ax.imshow(utils.to_image_mnist(xdata[idx1[kk]].numpy()))
    # axes_bottom[-1].text(.5,.5, r'...', fontsize=labelsize, ha='center', va='center')


    # plot neurons
    xvals = np.linspace(0.,1.,n_layers+1)
    dxv = xvals[1] - xvals[0]

    seed_b = random.randint(0, 1000000)
    np.random.seed(seed_b)
    cmap = cm.get_cmap('Blues')

    for ll in range(n_layers):
        xp0 = xvals[ll]*np.ones(n_neuron)
        yp0 = np.arange(n_neuron, dtype=float)
        yp0 -= np.mean(yp0)
        cp0 = np.random.rand(n_neuron)

        axn0.scatter(xp0, yp0, c=cp0, s=130, cmap=cmap, edgecolors='k')

    xp1 = xvals[-1]*np.ones(1)
    yp1 = np.arange(1)
    cp1 = np.random.rand(1)

    axn0.scatter(xp1, yp1, c=cmap(cp1), s=130, edgecolors='k')

    # draw arrows
    np.random.seed(seed_w)
    ymin, ymax = np.min(yp0), np.max(yp0)

    for ll in range(n_layers):
        x0 = xvals[ll] - .8*dxv
        x1 = xvals[ll] - .2*dxv

        y0s = np.random.rand(n_arrow) * (ymax - ymin) + ymin
        y1s = np.concatenate((np.random.choice(yp0, n_arrow-n_neuron),
                              np.random.choice(yp0, n_neuron, replace=False)))
        cs = np.random.rand(n_arrow)
        cmap = cm.get_cmap('Oranges')

        for kk, (y0, y1, c_) in enumerate(zip(y0s, y1s, cs)):
            prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.4", ec=cmap(c_), fc=cmap(c_), lw=lwidth*1.3, shrinkA=0, shrinkB=0)
            axn0.annotate("", xytext=(x0,y0), xy=(x1,y1), arrowprops=prop)

    x0 = xvals[-1] - .8*dxv
    x1 = xvals[-1] - .2*dxv

    dys = np.linspace(-.8, .8, len(yp0))
    for y0, dy in zip(yp0, dys):
        prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.4", ec=cmap(.5), fc=cmap(.5), lw=lwidth*1.3, shrinkA=0, shrinkB=0)
        axn0.annotate("", xytext=(x0,y0), xy=(x1,yp1[0]+dy), arrowprops=prop)

    axn0.set_xlim((-dxv,1.+dxv*.2))
    ypd = yp0[1] - yp0[0]
    axn0.set_ylim((yp0[0]-ypd/2., yp0[-1]+ypd/2.))

    if pshow:
        pl.show()


def get_axes(gs0, gs1, gst, n_image=4):
    axt = noFrameAx(pl.subplot(gst[0,0]))

    axtext = noFrameAx(pl.subplot(gs0[4,:]))
    axtext.text(.5,.5, r'vs', fontsize=labelsize, ha='center', va='center')

    axes = {'top': [], 'bottom': []}
    for kk in range(n_image**2):
        ii, jj = kk%n_image, kk//n_image

        axes['top'].append(noFrameAx(pl.subplot(gs0[ii,jj])))
        axes['bottom'].append(noFrameAx(pl.subplot(gs0[ii+n_image+1,jj])))

    axl1 = noFrameAx(pl.subplot(gs1[0,0]))

    return axt, axes, axl1


def plot_panel(n_network=1, n_layers=3, n_image=3, n_neuron=6, n_arrow=10, n_sample=1,
               seed_w=112, seed_t=[34,3,19], suffix="fig5"):
    seeds_t = np.random.randint(10000, size=n_network)

    figsize_x = 7. + .2 * (n_layers - 1)
    dx = .5 + .2  * (n_layers - 1)

    xcoo = [0.05]
    for _ in range(n_network):
        xcoo.extend([0.3,0.05,dx,0.2])
    xcoo.extend([0.1, 0.1])

    xcoords = getXCoords(xcoo)
    ycoords = [0.05, 0.80]

    fig = pl.figure('Bias schema', figsize=(figsize_x*n_network,2.5))

    for nn in range(n_network):
        gst0 = GridSpec(1,1)
        gst0.update(top=0.91, bottom=0.85, left=xcoords[0+4*nn], right=xcoords[1+4*nn], hspace=0.01, wspace=0.01)
        gs0 = GridSpec(2*n_image+1,n_image)
        gs0.update(top=ycoords[-1], bottom=ycoords[-2], left=xcoords[0+4*nn], right=xcoords[1+4*nn], hspace=0.005, wspace=0.005)
        gs1 = GridSpec(1,1)
        gs1.update(top=ycoords[-1], bottom=ycoords[-2], left=xcoords[2+4*nn], right=xcoords[3+4*nn], hspace=0.01, wspace=0.01)

        plot_schema(axes=get_axes(gs0, gs1, gst0, n_image=n_image),
                    task_idx=nn+1, seed_t=seeds_t[nn], seed_w=seed_w,
                    n_layers=n_layers, n_neuron=n_neuron, n_arrow=n_arrow, n_sample=n_sample)

    # gst0_ = GridSpec(1,1)
    # gst0_.update(top=0.95, bottom=0.9, left=xcoords[4], right=xcoords[5], hspace=0.01, wspace=0.01)
    # gs0_ = GridSpec(2*n_image+1,n_image)
    # gs0_.update(top=ycoords[-1], bottom=ycoords[-2], left=xcoords[4], right=xcoords[5], hspace=0.01, wspace=0.01)
    # gs1_ = GridSpec(1,1)
    # gs1_.update(top=ycoords[-1], bottom=ycoords[-2], left=xcoords[6], right=xcoords[7], hspace=0.01, wspace=0.01)

    gs = GridSpec(1,1)
    gs.update(top=ycoords[-1], bottom=ycoords[0], left=xcoords[-3], right=xcoords[-2], hspace=0.01, wspace=0.01)

    # seed_w = 112
    # plot_schema(axes=get_axes(gs0_, gs1_, gst0_, n_image=n_image), task_idx=2, seed_t=14, seed_w=seed_w, n_neuron=n_neuron, n_arrow=n_arrow, n_sample=n_sample)

    ax = noFrameAx(pl.subplot(gs[0,0]))
    ax.text(.5,.5, r'...', fontsize=labelsize, ha='center', va='center')

    pl.savefig(paths.fig_path + "biasadapatation_v2_mini_%s.svg"%suffix, transparent=True)
    pl.show()



if __name__ == "__main__":
    # plot_schema()
    plot_panel()