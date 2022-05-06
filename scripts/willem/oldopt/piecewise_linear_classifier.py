import numpy as np
import torch

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans

import optim, hiddenmatrix
from biasadaptation.utils import samplers, losses, utils


# Function which detects the Tomeklinks
# @param: X The datapoints e.g.: [f1, f2, ... ,fn]
# @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
# @return: 1-D arrays with the indices of the TomekLinks and others
def detect_tomek_links(X,y):
    nonlinks = []
    neigh = NearestNeighbors(n_neighbors=2, algorithm='kd_tree')
    neigh.fit(X)

    # k2 stores the first nearest neighbour
    k2 = neigh.kneighbors(X)[1]

    # k_tomek stores the ones where the labels conflict.
    k_tomek = k2[y != y[k2[:,1]]]

    # This is for getting the positions
    tomekList = np.unique(np.concatenate([k_tomek[:,0],k_tomek[:,1]]))
    index = np.arange(0,len(X))
    nonlinks = set(index) - set(tomekList)
    nonlinks = list(nonlinks)
    return np.asarray(nonlinks), np.array(tomekList)


def intersect(X0, X1, w, b):
    """
    find the links that are intersected by a given hyperplane
    """
    wX0, wX1 = np.dot(w[:,None], X0), np.dot(w[:,None], X1)
    ts = - (wX1 + b) / (wX0 - wX1)

    return np.where(np.logical_and(ts >= 0, ts <= 1))[0].tolist()


def initial_hyperplane(X0, X1):
    """
    Find an initial hyperplane among all hyperplanes defined as going perpendicularly through
    the centers of the Tomek links as the one that intersects the largets number
    of links

    Parameters
    ----------
    X0, X1: np.ndarray (n_links, n_features)
        The data points defining Tomek links (i.e. a link is `(X0[ii], X1[ii])`)
    """
    nn = len(X0)
    assert nn == X1.shape[0]

    # find the hyperplane that intersects the maximal number of linear segments
    idx_ = 0
    for ii, (x0, x1) in enumerate(zip(X0, X1)):
        # the hyperplane perpendicular to the link
        w = (x0 - x1) / np.linalg.norm(x0 - x1)
        b = - np.dot(w, (x0 + x1) / 2.)
        # the indices of intersected links
        idx_intersect = intersect(X0, X1, w, b)

        if len(idx_intersect) > n_:
            idx_ = idx_
            w_ = w
            b_ = b

    return w_, b_, idx_


class Hyperplane(nn.Module):
    def __init__(self, w, m):
        self.w = nn.Parameter(torch.FloatTensor(w))
        self.m = torch.FloatTensor(m)

    def forward(self, x):
        v = torch.mm((x - self.m[None,:]), self.w[:,None])

        return v


def minimize_hyperplane(X0, X1, w, m, k_iter=100):
    """
    Optimize the hyperplane with normal vector `w` and going through the point
    `m` to intersect as many Tomek links as possible

    Parameters
    ----------
    X0, X1: np.ndarray (n_links, n_features)
        The data points defining Tomek links (i.e. a link is `(X0[ii], X1[ii])`)
    w: np.array (n_features,)
        The normal vector of the iniital hyperplane
    m: np.array (n_features,)
        The point through which the hyperplane passes
    """
    hp = Hyperplane(w, m)
    optimizer = optim.Adam(hp.parameters(), lr=0.005, betas=(0.9, 0.999))

    centers = torch.FloatTensor((X0 + X1) / 2.)

    for kk in range(k_iter):
        optimizer.zero_grad()
        out = hp.forward(centers)

        loss = torch.sum(v**2)
        loss.backward()

        optimizer.step()

    w_ = hp.w.detach().numpy()
    b_ = - np.dot(w, m)

    return w_, b_


def find_maximal_hyperplane(X0, X1):
    """
    Find the hyperplane that intersects the largest number of Tomek links

    Parameters
    ----------
    X0, X1: np.ndarray (n_links, n_features)
        The data points defining Tomek links (i.e. a link is `(X0[ii], X1[ii])`)
    """
    nn = X0.shape[0]

    w_init, b_init, idx = initial_hyperplane(X0, X1)

    k0 = len(idx) - 1
    idx_ = copy.deepcopy(idx)

    while len(idx) > k0
        for ind in set(range(nn)) - set(idx_):
            m = (X0[ind,:] + X1[ind,:]) / 2.

            idx_aux = idx + [ind]
            w_, b_ = minimize_hyperplane(X0[idx_aux,:], X1[idx_aux,:], w, m)

            idx__ = interset(X0, X1, w, b)

            if len(idx__) > len(idx):
                idx = idx__
                w = w_
                b = b_
                k0 = len(idx_)

    return w, b, idx


def calc_decision_boundary(X0, X1):
    """
    Compute the piecewise linear decision boundary based on the tomek links

    Parameters
    ----------
    X0, X1: np.ndarray (n_links, n_features)
        The data points defining Tomek links (i.e. a link is `(X0[ii], X1[ii])`)

    Returns
    -------
    decision boundary: dict {'hp': [], 'links': []}
        'hp' contains a list whose entries are the tuples (w, b) defining a hyperplane
        'links' contains a list whose entries are the tomex links associated to this hyperplane
    """
    nn = X0.shape[0]
    idx_ = list(range(nn))

    decision_boundary = {'hp': [], 'links': []}
    while nn > 0:
        w, b, idx = find_maximal_hyperplane(X0[idx_,:], X1[idx_,:])
        idx = idx_[idx]

        decision_boundary['hp'].append({'w': w, 'b': b})
        decision_boundary['links'].append((X0[idx,:], X1[idx,:]))

        idx_ = list(set(idx_) - set(idx))

    return decision_boundary






