import numpy as np
import torch

import copy
import operator

from biasadaptation.biasfit import specificbiasfit
from biasadaptation.utils import losses

import helperfuncs

def test_single_class(ws, bs, gs, task, dataset, readout, targets=[-1, 1, 'all'], verbose=True):
    # construct the data sampler
    source_train = helperfuncs.get_dataset(dataset, train=True)
    source_test = helperfuncs.get_dataset(dataset, train=False)

    _, dl_test, _ = helperfuncs.construct_knm_dataloader_triplet(
                    dataset, task, copy.deepcopy(task),
                    source_train, source_test,
                    1, 1, 10000,
        )

    return test_single_class_from_dl(ws, bs, gs, dl_test, readout, targets=targets, verbose=verbose)


def test_single_class_from_dl(ws, bs, gs, dl_test, readout, targets=[-1, 1, 'all'], verbose=True):

    rlnet = specificbiasfit.ReLuFit(ws, bs, gs, readout=readout)

    # performance on test set
    (xdata, xlabel), (xtask, xtarget) = next(iter(dl_test))
    xdata = xdata.reshape(xdata.shape[0], np.prod(list(xdata.shape[1:])))

    perfs = {}
    for target in targets:
        if target == 'all':
            out = rlnet.forward(xdata, xtask)
            lb = losses.binary_loss(out, xtarget)
            if verbose: print('--> test accuracy both: %.3f'%((1.-float(lb))*100.))
            perf = (1.-float(lb))*100.

            perfs[target] = perf

        elif isinstance(target, int):

            xinds = torch.where(xtarget == target)[0]

            out = rlnet.forward(xdata[xinds], xtask[xinds])
            lb = losses.binary_loss(out, xtarget[xinds])
            if verbose: print('--> test accuracy %d: %.3f'%(target, (1.-float(lb))*100.))
            perf = (1.-float(lb))*100.

            perfs[target] = perf

        else:
            raise TypeError('Invalid input type in `targets`, should be \'all\' or int')

    return perfs