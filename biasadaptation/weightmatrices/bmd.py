import numpy as np
import scipy.linalg as sla
import torch
import torch.nn as nn
import torch.optim as optim

import copy
import warnings

from ..utils import utils


def normnorm(v1, v2):
    return np.linalg.norm(v1 / np.linalg.norm(v1) - v2 / np.linalg.norm(v2))


class Decomposition(nn.Module):
    def __init__(self, C, O, W):
        """
        Parameters
        ----------
        C: np.ndarray of float (k, n_h1)
        O: np.ndarray of float (k, n_h2)
        W: np.ndarray of float (n_h2, n_h1)
        """
        super(Decomposition, self).__init__()

        self.C = torch.FloatTensor(C)
        self.O = torch.FloatTensor(O)

        self.W = nn.Parameter(torch.FloatTensor(W))

        # create (k, n_h1, n_h1) tensor diagonal in last two dims
        self.S = (np.abs(self.C) > 0.).float()

    def cost(self):
        WN = self.W / torch.norm(self.W, dim=1, keepdim=True)
        OWN  = torch.matmul(self.O, WN)
        OWNP = torch.einsum('ki,ki->k', self.C, OWN)

        return -torch.mean(OWNP)


def minimize_w_cost(C, O, W, n_iter=100, verbose=False):
    """
    Given a data matrix `C` and a binary matrix `O`, compute a new basis vector
    matrix that minimizes `|C - OW|`

    Parameters
    ----------
    C: np.ndarray of float (k, d)
    O: np.ndarray of float (k, n)
    W: np.ndarray of float (n, d)
    """
    W_copy = copy.deepcopy(W)
    decomp = Decomposition(C, O, W)

    optimizer = optim.Adam(decomp.parameters(), lr=0.01, betas=(0.9, 0.999))

    for nn in range(n_iter):
        optimizer.zero_grad()
        loss = decomp.cost()
        loss.backward()
        # perform gradient step
        optimizer.step()

        if verbose: print("> iter %d --> loss = %.8f"%(nn, float(loss)))

    W_ = copy.deepcopy(decomp.W.detach().numpy())

    print('>>> W optimization')
    print('    final loss = %.8f'%(float(loss)))
    print('    |W_orig - W_new| = %.8f'%np.linalg.norm(W_copy - W_))

    sds = []
    for ii, o_vec in enumerate(O):
        c_vec = C[ii,:]
        v0 = o_vec @ W_
        sdot = np.dot(v0/np.linalg.norm(v0), c_vec/np.linalg.norm(c_vec))
        sds.append(sdot)

    print("    Average score = %.5f"%np.nanmean(sds))

    return W_


def binary_match(c_vec, W, verbose=False):
    """
    Find binary combination of basis vectors in `W` that closely matches `c_vec`

    c_vec: np.array of float
        vector of `k` elements
    W: np.ndarray of float
        matrix of `(p,k)` elements
    """

    v0 = np.zeros(W.shape[1])
    i0s = list(range(W.shape[0]))
    i1s = []
    s0, s1 = -0.1, 0.
    kk = 0
    sdot = 0.

    while s1 > s0 and kk < W.shape[0]:
        if verbose: print('\n--> iter %d, current score = %.5f'%(kk, s0))
        kk += 1

        W__ = np.array([W[ii] for ii in i0s])
        W_ = v0[None,:] + W__

        scores = np.dot(W_, c_vec) / (np.linalg.norm(W_, axis=1) * np.linalg.norm(c_vec))
        imax = np.argmax(scores)

        s0 = s1
        s1 = scores[imax]

        if s1 > s0:
            i0 = i0s.pop(imax)
            i1s.append(i0)

            v0 = np.sum(np.array([W[ii] for ii in i1s]), axis=0)

            sdot = np.dot(v0/np.linalg.norm(v0), c_vec/np.linalg.norm(c_vec))
            sdiff = normnorm(v0, c_vec)

            if verbose: print("new comp = %d, new score = %.5f, score diff = %.5f"%(i0, sdot, sdiff))


    return v0, i1s, sdot


def minimize_o_cost(C, W):
    """
    Parameters
    ----------
    C: np.ndarray of float (k, d)
        input data, `d` is input dimension, `k` data points
    W: np.ndarray of float (n, d)
        current weights, `n` is hidden units and `d` is input dimension

    Returns
    -------
    O: np.ndarray of float (k, n)
        binary coordinates of `C` in basis `W`
    """
    O = np.zeros((C.shape[0], W.shape[0]))

    # np.argmin(np.linalg.norm(C[:,None,:] - W[None,:,:], axis=2), axis=1)

    print(">>> O optimization")

    idx0, idx1 = [], []
    sds = []
    for ii, c_vec in enumerate(C):
        v0, i1s, sdot = binary_match(c_vec, W, verbose=False)
        idx0.extend([ii for _ in i1s])
        idx1.extend(i1s)

        sds.append(sdot)

    print("    Average score = %.5f"%np.nanmean(sds))

    O[(idx0, idx1)] = 1.

    return O


def find_binary_decomposition(C, nh):
    """
    Function implementing binary matrix decomposition from ... (infeasible
    for reasonable number of hidden neurons)
    """
    nc = nh - 1

    ## STEP 1: subtract column mean
    p = np.mean(C, axis=1)
    P = C - p[:,None]

    ## step 2:
    U, S, V = sla.svd(C)
    U_ = U[:nc,:nc]

    ## STEP 3: construct auxiliory matrices
    Z = U[:,:nc] @ sla.inv(U_)
    # all vertices on the unit hypercube
    B = ((np.arange(2**nc)[:,None] & (1 << np.arange(nc))) > 0).astype(float).T

    ## STEP 4: construct trial solutions
    T_ = Z @ (B - p[:nc, None]) + p[:, None]
    T = (T_ > 0.5).astype(float)

    ## STEP 5: vertices closest to Aff(C)
    E = np.linalg.norm(T - T_, axis=0)
    idx = np.argsort(E)

    return T[idx,:]


def find_weight_and_mask(O, W, C, n_h, n_iter=5):
    """
    Given a data matrix `C`, minimize the problem `|C - OW|^2` with `O` a binary
    matrix

    Parameters
    ----------
    C: np.ndarray of float (k, d)
    O: np.ndarray of float (k, n)
    W: np.ndarray of float (n, d)
    """

    for nn in range(n_iter):
        print('\n---- iter %d ----'%nn)

        W = minimize_w_cost(C, O, W)
        O = minimize_o_cost(C, W)

    return O, W


def get_weightmatrix_bmd(data_matrix, n_h):
    """
    Run PMD on the input matrix

    Parameters
    ----------
    data_matrix: array like (k, d)
        The input data, `k` number of points, `d` input dimension
    n_h: int
        number of hidden units
    """
    d = data_matrix.shape[1]

    W0 = np.random.randn(n_h, d)
    W0 /= np.linalg.norm(W0, axis=1)[:,None]
    O0 = minimize_o_cost(data_matrix, W0)

    O, W = find_weight_and_mask(O0, W0, data_matrix, n_h, n_iter=100)

    return W


def get_weightmatrix_bmdd(data_matrix, n_h):
    """
    Run PMD on differences of the input matrix

    Parameters
    ----------
    data_matrix: array like (k, d)
        The input data, `k` number of points, `d` input dimension
    n_h: int
        number of hidden units
    """
    diff_matrix = utils.differences_numpy(data_matrix, data_matrix.shape[0])

    return get_weightmatrix_bmd(diff_matrix, n_h)







