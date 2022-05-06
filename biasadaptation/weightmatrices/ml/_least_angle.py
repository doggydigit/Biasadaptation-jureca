import numpy as np

from sklearn import linear_model

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def sparse_encode_featureselect(X, W,
                  n_nonzero_coefs=None, alpha=0.,
                  max_iter=1000, positive=False, verbose=0):
    """
    Sparse coding
    Each row of the result is the solution to a sparse coding problem.
    The goal is to find a sparse array `code` such that::
        X ~= code * dictionary

    Only selects the dictionary features that correspond to non-zero components
    in the data sample

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.
    W : ndarray of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows for meaningful
        output.
    n_nonzero_coefs: int
        number of non-zero coefficients, if given, uses LARS algorithm
    alpha: float
        L1 regularization constant for Lasso (default algorithm if
        `n_nonzero_coefs` is not specified)
    max_iter: int
        maximum number of iterations if using LARS.

    Returns
    -------
    C : ndarray of shape (n_samples, n_components)
        The code matrix
    """
    if len(X.shape) == 1:
        X = X[None,:]
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_components = W.shape[0]

    # n_nonzero_coefs takes precedence if defined
    if n_nonzero_coefs != None:
        alpha = 0.
        max_iter = n_nonzero_coefs

    C = np.empty((n_samples, n_components))

    # print(alpha, max_iter)

    for k in range(n_samples):
        bidx = np.abs(X[k,:]) > 1e-10 # only select nonzero targets for the fit

        # alphas, active, code, n_iter = linear_model.lars_path(
        #             W.T[bidx,:], X.T[bidx, k],
        #             Gram=None, Xy=None, copy_X=True,
        #             alpha_min=alpha, method="lar",
        #             verbose=verbose, max_iter=max_iter,
        #             return_path=False,
        #             return_n_iter=True, positive=positive, eps=1e-7)


        alphas, codes, dual_gaps, n_iter = linear_model.enet_path(
                W.T[bidx,:], X.T[bidx,k],
                l1_ratio=1., n_alphas=5,
                return_n_iter=True, max_iter=1000
                )

        n_active = np.sum(np.abs(codes) > 1e-16, axis=0)
        idx = np.where(n_active > n_nonzero_coefs)[0]
        if len(idx) > 0:
            C[k] = codes[:,idx[0]]
        else:
            C[k] = codes[:,-1]

        # print(code)

        # code, _, _, _ = np.linalg.lstsq(W.T[bidx,:], X.T[bidx,k])

        # print(code[np.abs(code) > 1e-10])
        # print(code)

        # C[k] = code

    # print(C)

    return C

# @ignore_warnings(category=ConvergenceWarning)
def sparse_encode_no_featureselect(X, W,
                  n_nonzero_coefs=None, alpha=0.,
                  max_iter=1000, positive=False, verbose=0):
    """
    Sparse coding
    Each row of the result is the solution to a sparse coding problem.
    The goal is to find a sparse array `code` such that::
        X ~= code * dictionary

    Only selects the dictionary features that correspond to non-zero components
    in the data sample

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.
    W : ndarray of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows for meaningful
        output.
    n_nonzero_coefs: int
        number of non-zero coefficients, if given, uses LARS algorithm
    alpha: float
        L1 regularization constant for Lasso (default algorithm if
        `n_nonzero_coefs` is not specified)
    max_iter: int
        maximum number of iterations if using LARS.

    Returns
    -------
    C : ndarray of shape (n_samples, n_components)
        The code matrix
    """
    if len(X.shape) == 1:
        X = X[None,:]
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_components = W.shape[0]

    # n_nonzero_coefs takes precedence if defined
    if n_nonzero_coefs != None:
        alpha = 0.
        max_iter = n_nonzero_coefs

    C = np.empty((n_samples, n_components))

    for k in range(n_samples):

        # alphas, active, code, n_iter = linear_model.lars_path(
        #             W.T, X.T[:, k],
        #             Gram=None, Xy=None, copy_X=True,
        #             alpha_min=alpha, method="lar",
        #             verbose=verbose, max_iter=max_iter,
        #             return_path=False,
        #             return_n_iter=True, positive=positive)



        alphas, codes, dual_gaps = linear_model.enet_path(
                W.T, X.T[:,k],
                l1_ratio=.5, n_alphas=5
                )

        n_active = np.sum(np.abs(codes) > 1e-16, axis=0)
        idx = np.where(n_active > n_nonzero_coefs)[0]
        if len(idx) > 0:
            C[k] = codes[:,idx[0]]
        else:
            C[k] = codes[:,-1]

        # code, _, _, _ = np.linalg.lstsq(W.T, X.T[:,k])

        # print(code[np.abs(code) > 1e-10])

        # C[k] = code

    # print(C)

    return C
