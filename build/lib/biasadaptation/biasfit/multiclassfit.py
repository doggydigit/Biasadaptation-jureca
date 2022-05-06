import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tfunc
import torch.optim as optim


class MultiClass(nn.Module):
    """
    Multiclass classifier by applying Nc 1vall biasses

    Attributes
    ----------
    ws: list of torch.FloatTensor or np.ndarray
        The weight matrices of the network. Each matrix has shape (Nhi-1, Nhi).
        The last layer has 1 hidden unit, thus last weight matrix has shape
        (Nhk-1, 1)
    bs: list of torch.FloatTensor or np.ndarray
        The list of biasses for each layer. For each layer, the biasses have
        shape (Nc, Nhi)
    readout: str ("linear", "tanh" or "hardtanh")
        readout of nonlinearity of the network
    """
    def __init__(self, ws, bs, readout="linear"):
        super(MultiClass, self).__init__()

        self.n_class = bs[0].shape[0]

        # assert dimensions are compatible
        for ii, (w, b) in enumerate(zip(ws, bs)):
            assert b.shape[0] == bs[0].shape[0]
            assert w.shape[1] == b.shape[1]
            if ii > 0:
                assert w.shape[0] == ws[ii-1].shape[1]

        # initialize the weights
        self.ws = [torch.FloatTensor(w) for w in ws]
        # initialize the biasses
        self.bs = [torch.FloatTensor(b) for b in bs]

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

    def forward(self, x):
        """
        compute the output of the network

        Parameters
        ----------
        x: torch.floatTensor (batch_size, input_dim)
            The input data points
        """
        # extend x to (batch_size, n_class, input_dim)
        o = torch.FloatTensor(x)[:, None, :]
        o = o.repeat(1, self.n_class, 1)

        for afunc, w, b in zip(self.afs, self.ws, self.bs):
            o = afunc(torch.matmul(o, w) + b[None, :, :])

        return o.squeeze()








