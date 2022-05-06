from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tfunc


"""
Superseded by specificbiasfit.py
"""


class ReLuFit(nn.Module, ABC):
    """
    Optimize a ReLu-network with K hidden layers where gains and biasses adapt to
    different binary classification tasks (Nt the number of tasks).

    Nhi the number of units in the i'th hidden layer (i=0,...,K-1).

    ReLu for first K-1 layers and identity for output layer.

    Attributes
    ----------
    ws: list of torch.FloatTensor or np.ndarray
        The weight matrices of the network. Each matrix has shape (Nhi, Nhi-1).
        The last layer has 1 hidden unit, thus last weight matrix has shape
        (1, NhK-1)
    bs: list of torch.FloatTensor or np.ndarray
        The list of biasses for each layer. For each layer, the biasses have
        shape (Nhi, Nt)
    gs: list of torch.FloatTensor
        The gains of the hidden layers.
    """
    def __init__(self, ws, bs, gs=None, opt_w=False, readout="linear"):
        super(ReLuFit, self).__init__()
        # initialize the weights
        self.ws = [torch.FloatTensor(w) for w in ws]
        if opt_w:
            ws = [nn.Parameter(w) for w in self.ws]
            self.ws = nn.ParameterList(ws)
        # initialize the biasses
        bs = [nn.Parameter(torch.FloatTensor(b)) for b in bs]
        self.bs = nn.ParameterList(bs)
        # initialize the gains
        if gs is None:
            gs = [nn.Parameter(torch.ones_like(b)) for b in bs]
        else:
            gs = [nn.Parameter(torch.FloatTensor(g)) for g in gs]
        self.gs = nn.ParameterList(gs)

        # initialize the activation functions
        self.afs = [tfunc.relu for _ in range(len(self.ws)-1)]
        if readout == "linear":
            self.afs += [lambda x: x]
        elif readout == "tanh":
            self.afs += [torch.tanh]
        elif readout == "hardtanh":
            self.afs += [torch.nn.functional.hardtanh]
        else:
            raise NotImplementedError(
                "The available readout activation functions are: linear, tanh and hardtanh")

    def forward(self, x, t):
        """
        compute the output of the network

        Parameters
        ----------
        x: torch.floatTensor (input_dim, batch_size)
            The input data points
        t: torch.LongTensor (batch_size)
            The task index for each input data point
        """
        o = torch.FloatTensor(x)
        t = torch.LongTensor(t)

        for afunc, w, g, b in zip(self.afs, self.ws, self.gs, self.bs):
            o = afunc(g[t, :] * torch.mm(o, w) + b[t, :])

        return o
