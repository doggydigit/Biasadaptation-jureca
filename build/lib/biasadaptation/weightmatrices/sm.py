# subspace matching

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..utils import utils


class SubSpaceOpt(nn.Module):
    """
    Optimize the K-dimensional subspaced spanned by K vectors to minimize
    the orthogonal distance between subspace and datapoints

    Attributes
    ----------
    w: torch.tensor (N,K)
        matrix with basis vectors as columns (N is data space dimension and
        K the subspace dimension)
    """
    def __init__(self, w_init):
        super(SubSpaceOpt, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor(w_init))
        self.norm_w()


    def norm_w(self):
        """
        Ensure columns of `w` are vectors of unit norm
        """
        self.wn = self.w / torch.norm(self.w, dim=0, keepdim=True)


    def get_coords(self, x):
        """
        Compute coordinates of `x` in basis spanned by columns of `self.w`

        Parameters
        ----------
        x: torch.tensor (*,N)
            The input batch

        Returns
        -------
        torch.tensor (*,K)
            The coordinates of `x` in the normed basis spanned by `self.w`
        """
        b = torch.mm(x, self.wn).unsqueeze(-1)
        a = torch.mm(torch.t(self.wn), self.wn).unsqueeze(0)
        a_ = torch.cat(b.shape[0]*[a])

        return torch.solve(b, a_)[0][:,:,0]

    def project(self, x):
        """
        Compute projections of input vectors on a subspace

        Parameters
        ----------
        x: torch.tensor (*,N)
            the input batch

        Returns
        -------
        torch.tensor (*,N)
            the projection of `x` on the subspaced spanned by columns of `self.w`
        """
        x = torch.FloatTensor(x)

        self.norm_w()
        v = self.get_coords(x)

        return torch.mm(v, torch.t(self.wn))


def find_subspace(data_loader, n_h,
                  n_max_iter=2000, reg=None):
    """
    Match the ``n_h``-dimensional subspace to the distribution of difference
    vectors of the EMNIST dataset

    Parameters
    ----------
    data_loader: torch.dataloader
        the data loader object
    n_h: int
        the number of basis vectors that span the subspace
    n_max_iter: int
        maximum number of iterations
    reg: None or dict {str: float}
        parameters for regularization term. 'alpha' is the regularization strength
        and 'p' order of the matrix norm. If ``None``, no regularization is applied.

    Returns
    -------
    np.ndarray (n_h, input_dimension)
        The basis vectors that minimaze orthogonal distance between differences
        and subspace
    """
    # get input dimension
    batch = next(iter(data_loader))[0]
    batch_size = batch.shape[0]
    input_dim = batch.view(batch_size, -1).shape[1]

    # subspace optimizer
    w_init = np.random.randn(input_dim, n_h)
    sso = SubSpaceOpt(w_init)

    # stochastic gradien descent with MSEloss
    optimizer = optim.Adam(sso.parameters(), lr=0.005, betas=(0.9, 0.999))
    mse = nn.MSELoss()

    kk = 0
    while kk < n_max_iter:
        for ii, batch in enumerate(data_loader):
            optimizer.zero_grad()

            # compute difference vectors
            x_data = batch[0].float().view(batch[0].shape[0], -1)
            x_diff = utils.differences_torch(x_data, x_data.shape[0])

            # compute projection, loss, gradient
            x_ = sso.project(x_diff)
            loss = mse(x_, x_diff)
            if reg is not None:
                loss += reg['alpha'] * torch.norm(sso.w, p=reg['p'])
            loss.backward()
            # perform gradient step
            optimizer.step()

            print('\n>>> iter %d --> loss = %.8f <<<'%(kk+ii, loss))

            if kk+ii > n_max_iter:
                break
        kk += ii

    return sso.w.detach().numpy().T


def get_weightmatrix_sm(data_loader, n_h):
    # l1 regularizer
    reg = dict(alpha=.1, p=1)

    return find_subspace(data_loader, n_h, reg=reg)