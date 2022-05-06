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
    xs: list of torch.FloatTensor or np.ndarray
        The list of biasses for each layer. For each layer, the biasses have
        shape (Nt, Nhi)
    """
    def __init__(self, ws, xs, gs, opt_w=False, readout="linear"):
        super(ReLuFit, self).__init__()
        # initialize the weights
        self.ws = [torch.FloatTensor(w) for w in ws]
        if opt_w:
            ws = [nn.Parameter(w) for w in self.ws]
            self.ws = nn.ParameterList(ws)

        # initialize the biasses
        xs = [nn.Parameter(torch.FloatTensor(b)) for b in xs]
        self.xs = nn.ParameterList(xs)

        # initialize the gains
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
            raise NotImplementedError("The available readout activation functions are: linear, tanh and hardtanh")

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

        for afunc, w, b, g in zip(self.afs, self.ws, self.xs, self.gs):
            o = afunc(g[t, :] * (torch.mm(o, w) + b))

        return o

    def get_best_task(self, x):
        """
        Find the task that maximizes network output for a given input x.
        Parameters
        ----------
        x: input

        Returns ID of the task maximizing output and the value of output for that task
        -------

        """
        nr_tasks = self.bs[0].shape[0]
        predictions = [self.forward(x, t) for t in range(nr_tasks)]
        highest_prediction = max(predictions)
        return predictions.index(highest_prediction), highest_prediction
