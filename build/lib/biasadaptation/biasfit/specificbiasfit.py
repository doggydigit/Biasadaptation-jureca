import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tfunc
import torch.optim as optim


class ReLuFit(nn.Module):
    """
    Optimize a ReLu-network with K hidden layers where biasses adapt to
    different binary classification tasks (Nt the number of tasks).

    Nhi the number of units in the i'th hidden layer (i=0,...,K-1).

    ReLu for first K-1 layers and identity for output layer.

    Attributes
    ----------
    ws: list of torch.FloatTensor or np.ndarray
        The weight matrices of the network. Each matrix has shape (Nhi-1, Nhi).
        The last layer has 1 hidden unit, thus last weight matrix has shape
        (Nhk-1, 1)
    bs: list of torch.FloatTensor or np.ndarray
        The list of biasses for each layer. For each layer, the biasses have
        shape (Nt, Nhi)
    b_idx: list of ints
        Indices of the layers whose biasses are trained
    g_idx: list of ints
        Indices of the layers whose gainsß are trained
    w_idx: list of ints
        Indices of the layers whose weights are trained (`b_idx` and `w_idx`
        can not both be empty)
    readout: str ("linear", "tanh" or "hardtanh")
        readout of nonlinearity of the network
    """
    def __init__(self, ws, bs, gs=None, w_idx=[], b_idx=None, g_idx=[], readout="linear"):
        super(ReLuFit, self).__init__()

        # assert dimensions are compatible
        for ii, (w, b) in enumerate(zip(ws, bs)):
            assert w.shape[1] == b.shape[1]
            if ii > 0:
                assert w.shape[0] == ws[ii-1].shape[1]

        # initialize the weights
        self.ws = [nn.Parameter(torch.FloatTensor(w)) if ii in w_idx else \
                   torch.FloatTensor(w) for ii, w in enumerate(ws)]

        # initialize the biasses
        if b_idx is None:
            b_idx = list(range(len(ws)))
        self.bs = [nn.Parameter(torch.FloatTensor(b)) if ii in b_idx else \
                   torch.FloatTensor(b) for ii, b in enumerate(bs)]

        # initialize the gains
        if gs is None:
            gs = [torch.ones_like(b) for b in bs]
        self.gs = [nn.Parameter(torch.FloatTensor(g)) if ii in g_idx else \
                   torch.FloatTensor(g) for ii, g in enumerate(gs)]

        # all parameters
        self.ps = nn.ParameterList([self.ws[idx] for idx in w_idx] + \
                                   [self.bs[idx] for idx in b_idx] + \
                                   [self.gs[idx] for idx in g_idx])

        # initialize the activation functions
        self.afs = [tfunc.relu for _ in range(len(self.ws)-1)]
        # print('READOUT =', readout)
        if readout == "linear":
            self.afs += [lambda x: x]
        elif readout == "tanh":
            self.afs += [torch.tanh]
        elif readout == "hardtanh":
            self.afs += [torch.nn.functional.hardtanh]
        else:
            raise NotImplementedError("The available readout activation functions are: \'linear\', \'tanh\', \'hardtanh\'")

    def forward(self, x, t):
        """
        compute the output of the network

        Parameters
        ----------
        x: torch.floatTensor (batch_size, input_dim)
            The input data points
        t: torch.LongTensor (batch_size)
            The task index for each input data point
        """
        o = torch.FloatTensor(x)
        t = torch.LongTensor(t)
        assert o.shape[0] == t.shape[0]

        for afunc, w, b, g in zip(self.afs, self.ws, self.bs, self.gs):
            o = afunc(g[t, :] * torch.mm(o, w) + b[t, :])

        return o.squeeze()








