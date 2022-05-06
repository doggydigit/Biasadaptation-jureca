import numpy as np

from datarep import paths
from datarep.matplotlibsettings import *

import pickle
import sys
sys.path.append('..')

from biasadaptation.biasfit import specificbiasfit
from biasadaptation.utils import utils, zerofinder, losses
import helperfuncs


# NHS = ["100"]
# ALGOS = ["pmdd"]

NHS = ["100", "100"]
ALGOS = ["pmdd", "rp", "na"]
# ALGOS = ["pmdd", "pmd", "lstsq"]

DATAPATH = "/Users/wybo/Data/results/biasopt/"
DATASET = "EMNIST"
TASKTYPE = "1vall"
READOUT = "tanh"
X_DIV_B = .5

def plot_zerocrossing(axes=None, task_idx=1, seed_t=116):

    if axes is None:
        pl.figure(figsize=(3.5,2.5))
        gs = GridSpec(4,4)
        axes = [[pl.subplot(gs[ii,jj]) for jj in range(4)] for ii in range(4)]
        pshow = True
    else:
        pshow = False

    if len(NHS) == 1:
        assert len(ALGOS) == 1
        fname = os.path.join(DATAPATH, "biasopt_1hl_%s_%s%s_%s_ro=%s.p"%(DATASET, ALGOS[0], NHS[0], TASKTYPE, READOUT))
    else:
        fname = os.path.join(DATAPATH, "deepopt_2hl_%s_algo12c=%s_nh12=%s_%s_ro=%s.p"%(DATASET, "-".join(ALGOS), "-".join(NHS), TASKTYPE, READOUT))

    with open(fname, 'rb') as file:
        reslist = pickle.load(file)

    # train and test datasets
    source_train = helperfuncs.get_dataset('EMNIST', train=True, x_div=X_DIV_B, path='~/Data/')
    source_test = helperfuncs.get_dataset('EMNIST', train=False,  x_div=X_DIV_B, path='~/Data/')

    task = helperfuncs.sample_binary_tasks_(1, dataset='EMNIST', task_type='1vall', seed=seed_t)[0]

    data_loaders = helperfuncs.construct_knm_dataloader_triplet(
                            'EMNIST', task, copy.deepcopy(task),
                            source_train, source_test,
                            1000, 100, 100)

    (xdata, xlabel), (xtask, xtarget) = next(iter(data_loaders[0]))

    # find the task
    found = False; kk = 0
    while not found:
        res = reslist[kk]
        found = next(iter(task[-1][DATASET].keys())) == next(iter(res['task'][-1][DATASET].keys()))
        kk += 1

    print(task)
    print(res['task'])

    ws = res['ws']
    bs = res['bs']
    ws_ = [w.T for w in ws]
    bs_ = [b[0] for b in bs]

    xdat0 = xdata[xtarget < 0].numpy()
    xdat1 = xdata[xtarget > 0].numpy()

    zf = zerofinder.ZeroFinder(ws_, bs_)
    rlnet = specificbiasfit.ReLuFit(ws, bs)

    xout0 = rlnet.forward(xdata[xtarget < 0], xtask[xtarget < 0])
    xout1 = rlnet.forward(xdata[xtarget > 0], xtask[xtarget > 0])
    xout = rlnet.forward(xdata, xtask)

    lb = losses.binary_loss(xout, xtarget)
    perf = (1.-float(lb))*100.
    print('Perf measured = %.3f, perf stored = %.3f \n'%(perf, res['perf']['test']['all']))

    ts = np.linspace(0., 1., 500)
    x0s = []
    xds = []
    wns = []

    kk = 0; ll = 0
    while kk < 4 and ll < min(len(xdat0), len(xdat1)):

        try:
            o0 = zf.tfunc(0., xdat0[ll], xdat1[ll])
            o1 = zf.tfunc(1., xdat0[ll], xdat1[ll])

            if o0 > 0. and o1 < 0.:
                raise ValueError

            x_ = zf.find_zero(xdat0[ll], xdat1[ll])
            wn = zf.find_affine_transform(x_)[0]

            x0s.append(x_)
            xds.append(xdat0[ll] - xdat1[ll])
            wns.append(wn)

            ax0 = noFrameAx(axes[kk][0])
            ax0.imshow(utils.to_image_mnist(xdat0[ll]).T)
            ax0.set_xticks([]); ax0.set_yticks([])

            ax1 = noFrameAx(axes[kk][1])
            ax1.imshow(utils.to_image_mnist(xdat1[ll]).T)
            ax1.set_xticks([]); ax1.set_yticks([])

            ax2 = myAx(axes[kk][2])
            ax2.axhline(0., ls='--', c='DarkGrey', lw=lwidth*.7)
            fs = np.array([zf.tfunc(t, xdat0[ll], xdat1[ll]) for t in ts])
            ax2.plot(ts, fs)
            ax2.set_xticks([]); ax2.set_yticks([0.])

            ax3 = noFrameAx(axes[kk][3])
            ax3.imshow(utils.to_image_mnist(wn).T)
            ax3.set_xticks([]); ax2.set_yticks([])

            con = mpatches.ConnectionPatch(xyA=(1.,0.5), xyB=(0.,0.5), coordsA='axes fraction', coordsB='axes fraction',
                      axesA=ax0, axesB=ax1,
                      arrowstyle="<|-|>", shrinkA=1., shrinkB=1., ec="r", fc="r")
            ax1.add_artist(con)

            if kk == 0:
                ax0.set_title(r"$\mathbf{x}_0(t=-1)$", fontsize=labelsize, rotation=30, pad=0., loc='left')
                ax1.set_title(r"$\mathbf{x}_1(t=+1)$", fontsize=labelsize, rotation=30, pad=0., loc='left')
                ax2.set_title(r"$y(\mathbf{x}_0 \rightarrow \mathbf{x}_1 )$", fontsize=labelsize, rotation=30, pad=0., loc='left')
                ax3.set_title(r"$\mathbf{w}_{\perp S}$", fontsize=labelsize, rotation=30, pad=0., loc='left')

            kk += 1
            ll += 1
        except ValueError:
            ll += 1

    if pshow:
        pl.show()


def plot_panel():

    xcoords = getXCoords([0.1, 1.0, 0.2, 1.0, 0.1, 0.05, 0.05])
    ycoords = [0.05, 0.68]


    fig = pl.figure('zerocrossing', figsize=(7,3))

    gst0 = GridSpec(1,1)
    gst0.update(top=0.99, bottom=0.90, left=xcoords[0], right=xcoords[0]+0.01)
    gst1 = GridSpec(1,1)
    gst1.update(top=0.99, bottom=0.90, left=xcoords[2], right=xcoords[2]+0.01)
    gst2 = GridSpec(1,1)
    gst2.update(top=ycoords[-1], bottom=ycoords[-2], left=xcoords[4], right=xcoords[5]+0.01)

    gs0 = GridSpec(4,4)
    gs0.update(top=ycoords[-1], bottom=ycoords[-2], left=xcoords[0], right=xcoords[1], hspace=0.05, wspace=0.2)
    gs1 = GridSpec(4,4)
    gs1.update(top=ycoords[-1], bottom=ycoords[-2], left=xcoords[2], right=xcoords[3], hspace=0.05, wspace=0.2)

    axt0 = noFrameAx(pl.subplot(gst0[0,0]))
    axt0.text(0.,0.5, "Task 1", ha="left", va="center", fontsize=labelsize)
    axt1 = noFrameAx(pl.subplot(gst1[0,0]))
    axt1.text(0.,0.5, "Task 2", ha="left", va="center", fontsize=labelsize)
    axt2 = noFrameAx(pl.subplot(gst2[0,0]))
    axt2.text(0.5,0.5, "...", ha="center", va="center", fontsize=labelsize)

    axes0 = [[pl.subplot(gs0[ii,jj]) for jj in range(4)] for ii in range(4)]
    axes1 = [[pl.subplot(gs1[ii,jj]) for jj in range(4)] for ii in range(4)]

    plot_zerocrossing(axes0, seed_t=11)
    # plot_zerocrossing(axes1, seed_t=14)
    plot_zerocrossing(axes1, seed_t=34)

    pl.savefig(paths.fig_path + "biasadapatation_fig1D.svg", transparent=True)

    pl.show()

if __name__ == "__main__":
    # plot_zerocrossing()
    plot_panel()