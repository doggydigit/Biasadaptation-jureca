from abc import ABC
from pickle import HIGHEST_PROTOCOL
from pickle import dump as pickle_dump
from torch import FloatTensor, as_tensor, transpose, tanh, gather, sigmoid, add
from torch import stack as torch_stack
from torch import mm as torch_mm
from torch import squeeze as torch_squeeze
from torch.optim import Adam
from torch.nn import Module, Parameter, ParameterList
from torch.nn.functional import softmax, relu


class MRBiasFit(Module, ABC):
    def __init__(self, readout_weights, readout_biases, ws, bs, readout="softmax"):
        super(MRBiasFit, self).__init__()

        # initialize the weights and biases
        self.nr_hidden_layers = len(ws)
        self.ws = ParameterList([Parameter(FloatTensor(w)) for w in ws])
        self.bs = ParameterList([Parameter(FloatTensor(b)) for b in bs])
        self.rbs = ParameterList([Parameter(as_tensor(rb)) for rb in readout_biases])
        self.rs = ParameterList([Parameter(FloatTensor(r)) for r in readout_weights])
        self.readout = readout

        # initialize the activation functions
        if readout == "linear":
            self.af = lambda x: x
        elif readout == "tanh":
            self.af = tanh
        elif readout == "sigmoid":
            self.af = sigmoid
        elif readout == "softmax":
            self.af = softmax
        else:
            raise NotImplementedError("The available readout activation functions are: linear and softmax")

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
        o = FloatTensor(x)

        # Hidden layer neuron activations
        for w, b in zip(self.ws, self.bs):
            o = relu(torch_mm(o, w) + b[t, :])

        # Output neuron activations
        return self.af(torch_mm(o, self.rs[t]) + self.rbs[t])
