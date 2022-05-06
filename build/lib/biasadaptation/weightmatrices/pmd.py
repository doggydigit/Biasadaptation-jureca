import numpy as np

import copy
import warnings

from ..utils import utils

"""
Penalized matrix decomposition algorithm from Witten et al., 2009
"""


def norm_l1(x):
    return np.sum(np.abs(x))


class SS:
    def __init__(self, x, l1):
        self.x = x

        # auxiliary arrays
        self.x_abs = np.abs(x)
        self.x_sgn = np.sign(x)

        # max l1 norm
        self.l1 = l1

    def eval(self, thr):
        y = self.x_abs - thr
        rly = y * (y > 0).astype(float)
        return self.x_sgn * rly

    def __call__(self, eps=1e-4, max_iter=100):
        thr0 = 0.
        thr1 = np.max(self.x_abs)
        thr_mid = (thr0 + thr1) / 2.

        kk = 0
        while kk < max_iter and (thr1 - thr0) / thr_mid > eps:
            thr_mid = (thr0 + thr1) / 2.

            y = self.eval(thr_mid)
            y /= np.linalg.norm(y)

            l1_y = np.sum(np.abs(y))
            if l1_y > self.l1:
                thr0 = thr_mid
            else:
                thr1 = thr_mid

            thr_mid = (thr0 + thr1) / 2.

            kk += 1

        return y #/ np.linalg.norm(y)


class PenalizedMatrixDecomposition:
    def __init__(self, X, c1=None, c2=None, make_copy=True):
        self.X_ = X
        if make_copy:
            self.X_orig = copy.deepcopy(X)
        else:
            self.X_orig = None

        n, p = X.shape
        self.c1 = c1 if c1 is not None else np.sqrt(n)
        self.c2 = c2 if c2 is not None else np.sqrt(p)

        self.us = []
        self.vs = []
        self.ds = []

    def calc_component(self, eps=1e-8, max_iter=100):
        X = self.X_
        n, p = X.shape

        # step 1: initialize v randomly with unit norm
        v = np.random.randn(p)
        v /= np.linalg.norm(v)
        # dummy initialization for u
        u = np.random.randn(n)
        u /= np.linalg.norm(u)

        # step 2: iterate until convergence
        udiff, vdiff = 10.*eps, 10.*eps
        kk = 0

        while udiff > eps and vdiff > eps and kk < max_iter:
            sfunc = SS(np.dot(X, v), self.c1)
            u_= sfunc()

            sfunc = SS(np.dot(X.T, u_), self.c2)
            v_ = sfunc()

            udiff = np.linalg.norm(u - u_)
            vdiff = np.linalg.norm(v - v_)

            u = u_
            v = v_

            kk += 1

        # step 3: compute d
        d = np.dot(u, np.dot(X, v))

        self.X_ -= d * np.dot(u[:,None], v[None,:])

        self.us.append(u)
        self.vs.append(v)
        self.ds.append(d)

    def calc_error(self):

        V = np.vstack(self.vs)
        U = np.vstack(self.us).T
        D = np.diag(self.ds)

        X_ = np.dot(U, np.dot(D, V))

        try:
            return np.linalg.norm(self.X_orig - X_)
        except AttributeError:
            warnings.warn('No copy of the original matrix stored, ' + \
                          'cannot evaluate reconstruction error')
            return np.nan

    def __call__(self, k):
        """
        k: the number of components of the decomposition
        """
        for cc in range(k):
            self.calc_component()

            if self.X_orig is None:
                print('Iter %d')
            else:
                err = self.calc_error()
                print('Iter %d --> |X - X_| = %.8f'%(cc, err))

        V = np.vstack(self.vs)
        U = np.vstack(self.us).T
        D = np.diag(self.ds)

        return D, U, V


def calc_svd(X, k):
    u, s, vh = np.linalg.svd(X, full_matrices=False)

    X_ = np.dot(u[:, :k] * s[None, :k], vh[:k,:])

    err = np.linalg.norm(X - X_)

    print('SVD %d --> |X - X_| = %.8f'%(k, err))


def get_weightmatrix_pmd(data_matrix, n_h, with_svd=False):
    """
    Run PMD on the input matrix

    Parameters
    ----------
    data_matrix: array_like
        the input data
    n_h: int
        number of neurons in hidden layer
    """
    n, p = data_matrix.shape

    pmd = PenalizedMatrixDecomposition(data_matrix,
                                       c1=0.5*np.sqrt(n), c2=0.3*np.sqrt(p),
                                       make_copy=True)
    D, U, V = pmd(n_h)

    if with_svd:
        # svd for reference
        pmd.calc_svd(pmd.X_orig, nh)

    return V


def get_weightmatrix_pmdd(data_matrix, n_h, with_svd=False):
    """
    Run PMD on differences of the input matrix

    Parameters
    ----------
    data_matrix: array_like
        the input data
    n_h: int
        number of neurons in hidden layer
    """
    diff_matrix = utils.differences_numpy(data_matrix, data_matrix.shape[0])

    return get_weightmatrix_pmd(diff_matrix, n_h, with_svd=with_svd)


def test_pmd():
    X = np.random.randn(1000,500)

    calc_svd(X, 10)

    pmd = PenalizedMatrixDecomposition(X, c1=5, c2=4)

    D, U, V = pmd(10)

    print(D.shape)
    print(U.shape)
    print(V.shape)

    print(np.diag(D))
    print(np.sum((np.abs(U) > 0.).astype(int), axis=1))
    print(np.sum((np.abs(V) > 0.).astype(int), axis=1))


    X_ = np.dot(U, np.dot(D, V))

    print(np.sum((np.abs(X_) > 0.).astype(int), axis=1))


def test_norm_constraints():
    X = np.random.randn(2,2)

    calc_svd(X, 2)

    pmd = PenalizedMatrixDecomposition(X, c1=1, c2=1)

    D, U, V = pmd(2)

    print(D.shape)
    print(U.shape)
    print(V.shape)

    print(np.diag(D))
    print(np.sum((np.abs(U) > 0.).astype(int), axis=1))
    print(np.sum((np.abs(V) > 0.).astype(int), axis=1))


    X_ = np.dot(U, np.dot(D, V))

    print(np.sum((np.abs(X_) > 0.).astype(int), axis=1))


if __name__ == "__main__":
    test_pmd()
    # test_norm_constraints()

