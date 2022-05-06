# sparcification of subspace matching

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import cvxpy as cp
import diffcp
from cvxpylayers.torch import CvxpyLayer

from datarep.matplotlibsettings import *

from .sm import find_subspace


class SpaceOpt(nn.Module):
    """
    Given K weight vectors, optimize the basis of the space spanned by the weight
    vectors so that the coordinates of the data are as sparse as possible

    Attributes
    ----------
    w: torch.tensor or np.ndarray (K, N)
        matrix with basis vectors as columns (N is data space dimension and
        K the subspace dimension)
    """
    def __init__(self, w_mat):
        super(SpaceOpt, self).__init__()
        self.w = torch.FloatTensor(w_mat)

        # subspace rearrangement matrix
        k = self.w.shape[0]
        self.r = nn.Parameter(torch.eye(k))

        self.norm_w()
        self.construct_sparse_sol_layer()

    def norm_w(self):
        """
        Ensure rows of `w` are vectors of unit norm
        """
        w_new = torch.matmul(self.r, self.w)
        self.wn = w_new / torch.norm(w_new, dim=1, keepdim=True)

    def construct_sparse_sol_layer(self):
        """
        Construct the optimization to find the sparse coordinates as a
        differentiable pytorch layer
        """
        n_ = self.wn.shape[1]
        k_ = self.wn.shape[0]

        _a_mat = cp.Parameter((k_, 2*k_))
        _b_vec = cp.Parameter(k_)
        _x = cp.Variable(2*k_)

        obj = cp.Minimize(cp.sum(_x))
        cons = [_a_mat @ _x == _b_vec, _x >= 0.]
        prob = cp.Problem(obj, cons)

        self.layer = CvxpyLayer(prob, parameters=[_a_mat, _b_vec], variables=[_x])

    def get_coords(self, x):
        """
        Compute coordinates of `x` in basis spanned by rows of `self.w` with
        sparseness favouring solution

        Parameters
        ----------
        x: torch.tensor (*,N)
            The input batch

        Returns
        -------
        torch.tensor (*,K)
            The coordinates of `x` in the normed basis spanned by `self.w`
        """
        k_ = self.wn.shape[0]

        # matrices for equality constraint
        phi_mat = torch.matmul(self.wn, torch.t(self.wn))
        b_vec   = torch.matmul(self.wn, x.T).T

        # matrix for linear programming problem
        a_mat = torch.cat((phi_mat, -phi_mat), dim=1)

        # solve the linear programming problem
        z_sol = self.layer(a_mat, b_vec)[0]

        return z_sol[:,:k_] - z_sol[:,k_:]

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
            the projection of `x` on the subspace spanned by the rows of `self.w`
        torch.tensor (*,K)
            the coordinates of `x` in the subspace spanned by the rows of `self.w`
        """
        x = torch.FloatTensor(x)

        self.norm_w()
        v = self.get_coords(x)

        return torch.mm(v, self.wn), v


class OptimStep:
    def __init__(self, optimizer, space_obt):
        self.optimizer = optimizer
        self.so = space_obt

    def __call__(self, x_batch):
        try:
            self.optimizer.zero_grad()
            # compute projection of x on subspace
            x_, a_ = self.so.project(x_batch)
            # compute loss as l1 coordinate norm
            loss = torch.norm(a_, p=1)
            loss.backward()
            # perform gradient step
            self.optimizer.step()

            return loss, x_, a_

        except diffcp.cone_program.SolverError as err:
            print(err)

            return None, None, None


def construct_basis(data_loader, n_h, n_max_iter=10000):
    """
    1) Match the ``n_h``-dimensional subspace to the distribution of difference
    vectors of the EMNIST dataset

    2) Construct the basis so that the data-points can be expressed in it with
    the highest degree of sparsity

    Parameters
    ----------
    data_loader: torch.dataloader
        the data loader object
    n_h: int
        the number of basis vectors that span the subspace
    n_max_iter: int
        maximum number of iterations

    Returns
    -------
    np.ndarray (n_h, input_dimension)
        The basis vectors that minimaze orthogonal distance between differences
        and subspace
    """
    # get input dimension
    batch = next(iter(data_loader))
    input_dim = batch[0].view(batch_size, -1).shape[1]

    if n_h < input_dim:
        # unregularized subspace
        w_fix = find_subspace(data_loader, n_h)
    else:
        # normalized random weight vectors
        w_fix = np.random.randn(input_dim, n_h)
        w_fix /= np.linalg.norm(w_fix, axis=1)[:,None]

    so = SpaceOpt(w_fix)

    # stochastic gradien descent with MSEloss
    optimizer = optim.Adam(sso.parameters(), lr=0.005, betas=(0.9, 0.999))
    opt_step = OptimStep(optimizer, loss_func, so)

    kk = 0
    while kk < n_max_iter:
        for ii, batch in enumerate(data_loader):

            # compute difference vectors
            x_data = batch[0].float().view(batch_size, -1)
            x_diff = utils.differences_torch(x_data, x_data.shape[0])

            loss = opt_step(x_diff)

            print('\n>>> iter %d --> loss ='%kk+ii, loss)

            if kk+ii > n_max_iter:
                break
        kk += ii

    so.norm_w()

    return sso.wn.detach().numpy()


def get_weightmatrix_ssm(data_loader, n_h):
    # l1 regularizer
    reg = dict(alpha=.1, p=1)

    return find_subspace(data_loader, n_h, reg=reg)


def test_hiddenopt(n_iter=400, n_hidden=3, n_point=100):
    # input weight matrix
    # theta_bounds = np.linspace(0., 2*np.pi, n_hidden+1)
    # theta_arr = np.random.rand(n_hidden) * (theta_bounds[1:] - theta_bounds[:-1]) + theta_bounds[:-1]
    theta_arr = np.pi * np.arange(n_hidden, dtype=float) / float(n_hidden)
    w_inp = np.array([np.cos(theta_arr), np.sin(theta_arr)]).T
    # initial output weights
    w_init = np.random.randn(1,n_hidden)

    # data points
    theta = 2.*np.pi * np.arange(n_point, dtype=float) / float(n_point)
    x_data = np.array([np.cos(theta), np.sin(theta)]).T

    y_data = np.dot(x_data, w_inp.T)

    so = SpaceOpt(w_init)
    # optimizer = optim.SGD(so.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(so.parameters(), lr=0.005, betas=(0.9, 0.999))
    loss_func = Loss(lambda_1=1., lambda_2=0.01)
    opt_step = OptimStep(optimizer, loss_func, so)

    loss_list = []
    for ni in range(n_iter):
        print('--- %d ---'%ni)
        idx = np.random.choice(np.arange(n_point, dtype=int), 10, replace=False)
        loss = opt_step(y_data[idx,:])

        if loss is not None:
            loss_list.append(loss)

    so.norm_w()

    print('--- initial ---')
    print(w_init)

    print('--- final ---')
    w_final = so.wn.detach().numpy()
    # w_final = np.array([[-1. if ii%2 == 0 else 1 for ii in range(n_hidden)]])
    # w_final = np.ones((1,n_hidden))
    print(w_final)

    import matplotlib.pyplot as pl

    pl.figure()
    pl.plot(np.arange(len(loss_list)), loss_list)
    pl.show()

    # ipair = [(ii,jj) for ii in range(n_hidden) for jj in range(ii+1,n_hidden)]
    # idx = np.random.choice(np.arange(len(ipair), dtype=int), 20, replace=False)

    # for ix in idx:
    #     i0, i1 = ipair[ix]
    #     pl.figure("(%d,%d)"%(i0, i1))
    #     ax = pl.gca()
    #     ax.plot(y_data[:,i0], y_data[:,i1], c='DarkGrey', marker='o', ls='')
    #     ax.plot([0.,w_final[0,i0]], [0.,w_final[0,i1]], 'r')
    #     ax.set_aspect('equal')
    #     pl.show()

    pl.figure()
    ax = pl.gca(projection='3d')
    ax.plot(y_data[:,0], y_data[:,1], y_data[:,2], c='DarkGrey', marker='o', ls='')
    ax.plot([0.,w_final[0,0]], [0.,w_final[0,1]], [0.,w_final[0,2]], 'r')
    # ax.set_aspect('equal')
    pl.show()