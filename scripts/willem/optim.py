import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as tfunc

import copy

from biasadaptation.biasfit import specificbiasfit
from biasadaptation.utils import losses, utils

import testperf


class LRLin:
    def __init__(self, n_max, lr0=0.002, lr1=0.0005):
        self.n_max = n_max
        self.lr0, self.lr1 = lr0, lr1

    def __call__(self, n):
        return self.lr0 + (self.lr1 - self.lr0) * n / self.n_max


def run_optim(ws, bs, sampler_triplet, gs=None,
                n_epoch=100, n_batch_per_epoch=20,
                b_idx=None, w_idx=[], g_idx=[],
                lr=None, readout="linear", dataset_norm=1.,
                return_g=False, test_perf=True,
                verbose=True):
    """
    Parameters
    ----------
    ws: list of torch.FloatTensor or np.ndarray
        The weight matrices of the network. Each matrix has shape (Nhi-1, Nhi).
        The last layer has 1 hidden unit, thus last weight matrix has shape
        (Nhk-1, 1)
    bs: list of torch.FloatTensor or np.ndarray
        The list of biasses for each layer. For each layer, the biasses have
        shape (Nt, Nhi)
    gs: list of torch.FloatTensor or np.ndarray
        The list of gains for each layer. For each layer, the biasses have
        shape (Nt, Nhi). If ``None``, gains are initialized to be 1
    b_idx: list of ints
        Indices of the layers whose biasses are trained
    w_idx: list of ints
        Indices of the layers whose weights are trained (`b_idx` and `w_idx`
        can not both be empty)
    g_idx: list of ints
        Indices of the layers whose gains are trained
    lr: float or callable
        The learning rate. Constant if it is a float, if it is a called give the
        learning rate as a function of epoch number
    readout: str ("linear", "tanh", "hardtanh" or "sigmoidX")
        the output nonlinearity. Also determines the error measure. If linear,
        `perceptron_loss` is used, for "tanh" or "hardtanh" `mse_loss` is used,
        and for "sigmoidX" - with X a number specifying the scale of the sigmoid -
        `ce_loss` is used.
    return_g: bool
        whether or not to return the optimized gain values
    test_perf: bool
        whether or not to measure test performance
    """
    if lr is None:
        lr = LRLin(n_epoch)

    if gs is None:
        gs = [np.ones_like(b) for b in bs]

    ws = copy.deepcopy(ws)
    bs = copy.deepcopy(bs)
    gs = copy.deepcopy(gs)
    ws_init = copy.deepcopy(ws)
    bs_init = copy.deepcopy(bs)
    gs_init = copy.deepcopy(gs)

    dl_train = sampler_triplet[0]
    dl_test = sampler_triplet[1]
    dl_validate = sampler_triplet[2]

    if readout == "linear":
        loss_func = losses.perceptron_loss
    elif readout == "hardtanh" or readout == "tanh":
        loss_func = losses.mse_loss
    elif "sigmoid" in readout:
        scale = float(readout[7:])
        loss_func = losses.ce_loss(scale)
        # linear readout for the network
        readout = "linear"
    else:
        raise NotImplementedError("The available readout activation functions are: \'linear\', \'tanh\', \'hardtanh\', \'sigmoid1\', \'sigmoid5\' and \'sigmoid10\'")

    rlnet = specificbiasfit.ReLuFit(ws, bs, gs, b_idx=b_idx, w_idx=w_idx, g_idx=g_idx, readout=readout)
    if verbose:
        print('\n------------------')
        print('Network structure:')
        for ii, (w,b) in enumerate(zip(rlnet.ws, rlnet.bs)):
            print('> Layer', ii+1, ' --> w_shape =', w.shape, ', b_shape =', b.shape)

    if callable(lr):
        optimizer = optim.Adam(rlnet.parameters(), lr=lr(0), betas=(0.9, 0.999))
    else:
        optimizer = optim.Adam(rlnet.parameters(), lr=lr, betas=(0.9, 0.999))

    perf = {'train': [], 'test': 0., 'loss': []}
    ws_opt, bs_opt, gs_opt = None, None, None
    perf_max = 0.

    for nn in range(n_epoch):

        if callable(lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr(nn)

        iter_train = enumerate(dl_train)
        condition = True
        while condition:
            try:
                idx, ((xdata, xlabel), (xtask, xtarget)) = next(iter_train)
            except StopIteration:
                condition = False
            condition *= idx < n_batch_per_epoch

            xdata = xdata.reshape(xdata.shape[0], np.prod(list(xdata.shape[1:])))

            # compute network output
            optimizer.zero_grad()
            out = rlnet.forward(xdata, xtask)

            # compute loss and gradient
            loss = loss_func(out, xtarget)
            loss.backward()

            # perform gradient step
            optimizer.step()

        # performance on train set
        (xdata, xlabel), (xtask, xtarget) = next(iter(dl_validate))
        xdata = xdata.reshape(xdata.shape[0], np.prod(list(xdata.shape[1:])))
        out = rlnet.forward(xdata, xtask)
        lb = losses.binary_loss(out, xtarget)
        if verbose: print('\n>>> epoch %d --> train accuracy : %.3f'%(nn, 1.-lb))
        perf['train'].append((1.-float(lb))*100.)
        perf['loss'].append(float(loss))

        # store the params of optimal configuration and
        # measure performance on test set
        if perf['train'][-1] > perf_max:
            perf_max = perf['train'][-1]

            ws_opt = [copy.deepcopy(w.detach().numpy()) for w in rlnet.ws]
            bs_opt = [copy.deepcopy(b.detach().numpy()) for b in rlnet.bs]
            gs_opt = [copy.deepcopy(g.detach().numpy()) for g in rlnet.gs]

            # performance on  test set
            if test_perf:
                perf['test'] = testperf.test_single_class_from_dl(rlnet.ws, rlnet.bs, rlnet.gs, dl_test, readout, targets=[-1, 1, 'all'], verbose=verbose)


    if verbose:
        print('')
        for ii, (w, w_init) in enumerate(zip(ws_opt, ws_init)):
            print('L%d -> |w_init - w_final| = %.5f'%(ii,np.linalg.norm(w - w_init)))
        for ii, (b, b_init) in enumerate(zip(bs_opt, bs_init)):
            print('L%d -> |b_init - b_final| = %.5f'%(ii,np.linalg.norm(b - b_init)))
        for ii, (g, g_init) in enumerate(zip(gs_opt, gs_init)):
            print('L%d -> |g_init - g_final| = %.5f'%(ii,np.linalg.norm(g - g_init)))

    if return_g:
        return ws_opt, bs_opt, gs_opt, perf
    else:
        return ws_opt, bs_opt, perf



