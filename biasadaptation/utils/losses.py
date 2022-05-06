# custom losses

import torch
import torch.nn.functional as tfunc


def perceptron_loss(y, yt):
    """
    Perceptron style loss for y compared to target

    Parameters
    ----------
    y: torch.Tensor
    yt: torch.Tensor
        Shape of `yt` has to match shape of `y`. `yt` should only contain
        ``-1`` and ``1``.
    """
    return torch.sum(tfunc.relu(-yt*y))


def binary_loss(y, yt):
    """
    Binary loss (fraction of misclassifications) for y compared to target

    Parameters
    ----------
    y: torch.Tensor
    yt: torch.Tensor
        Shape of `yt` has to match shape of `y`. `yt` should only contain
        ``-1`` and ``1``.
    """
    return torch.mean((torch.sign(-yt*y) + 1.) / 2.)


def squared_hinge_loss(y, yt):
    """
    Squared hinge loss max(0,1-t*y)**2 for y compared to target

    Parameters
    ----------
    y: torch.Tensor
    yt: torch.Tensor
        Shape of `yt` has to match shape of `y`. `yt` should only contain
        ``-1`` and ``1``.
    """
    return torch.sum(torch.max(torch.add(-yt*y, 1.), torch.FloatTensor([0.])) ** 2)


def mse_loss(y, yt):
    """
    Mean squared error for y compared to target

    Parameters
    ----------
    y: torch.Tensor
    yt: torch.Tensor
        Shape of `yt` has to match shape of `y`. `yt` should only contain
        ``-1`` and ``1``.
    """
    return torch.sum((y - yt) ** 2)


class ce_loss:
    """
    Cross entropy loss for y compared to target

    Parameters
    ----------
    y: torch.Tensor
    yt: torch.Tensor
        Shape of `yt` has to match shape of `y`. `yt` should only contain
        ``-1`` and ``1``.

    Attributes
    ----------
    scale: float
        scale parameter for loss function
    """
    def __init__(self, scale=1.):
        self.scale = scale

    def __call__(self, y, yt):
        yt = (yt + 1.) / 2.

        # print(y.shape)

        # ym = torch.FloatTensor([torch.zeros_like(y), -self.scale*y])
        # yp = torch.FloatTensor([torch.zeros_like(y),  self.scale*y])

        # print(ym.shape)

        lll = (       yt  * torch.logaddexp(torch.zeros_like(y), -self.scale*y) + \
                (1. - yt) * torch.logaddexp(torch.zeros_like(y),  self.scale*y) ) / self.scale


        # print(lll.shape)
        # print(torch.sum(lll).shape)



        # lll = (       yt  * torch.log(1. + torch.exp(-self.scale*y)) + \
        #         (1. - yt) * torch.log(1. + torch.exp( self.scale*y)) ) / self.scale

        return torch.sum(lll)


def get_loss_function(f_name):
    if f_name == "l1":
        return torch.nn.functional.l1_loss
    elif f_name == "mse":
        return mse_loss
    elif f_name == "squared_hinge":
        return squared_hinge_loss
    elif f_name == "perceptron":
        return perceptron_loss
    elif f_name == "ce":
        return ce_loss
    else:
        raise NotImplementedError("Desired loss function '{}' is not available. The implemented functions are: l1, "
                                  "mse, squared_hinge, perceptron, ce".format(f_name))
