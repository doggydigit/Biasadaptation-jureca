import numpy as np
from scipy import linalg

import copy


def _update_dict(dictionary, Y, code, verbose=False, return_r2=False,
                 random_state=None, positive=False):
    """Update the dense dictionary factor in place.

    Parameters
    ----------
    dictionary : ndarray of shape (n_features, n_components)
        Value of the dictionary at the previous iteration.

    Y : ndarray of shape (n_features, n_samples)
        Data matrix.

    code : ndarray of shape (n_components, n_samples)
        Sparse coding of the data against which to optimize the dictionary.

    verbose: bool, default=False
        Degree of output the procedure will print.

    return_r2 : bool, default=False
        Whether to compute and return the residual sum of squares corresponding
        to the computed solution.

    random_state : int, RandomState instance or None, default=None
        Used for randomly initializing the dictionary. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    positive : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    Returns
    -------
    dictionary : ndarray of shape (n_features, n_components)
        Updated dictionary.
    """
    n_components = len(code)
    n_features = Y.shape[0]
    # random_state = check_random_state(random_state)
    # Get BLAS functions
    gemm, = linalg.get_blas_funcs(('gemm',), (dictionary, code, Y))
    ger, = linalg.get_blas_funcs(('ger',), (dictionary, code))
    nrm2, = linalg.get_blas_funcs(('nrm2',), (dictionary,))
    # Residuals, computed with BLAS for speed and efficiency
    # R <- -1.0 * U * V^T + 1.0 * Y
    # Outputs R as Fortran array for efficiency
    R = gemm(-1.0, dictionary, code, 1.0, Y)
    for k in range(n_components):
        # R <- 1.0 * U_k * V_k^T + R
        R = ger(1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)
        dictionary[:, k] = np.dot(R, code[k, :])
        if positive:
            np.clip(dictionary[:, k], 0, None, out=dictionary[:, k])
        # Scale k'th atom
        # (U_k * U_k) ** 0.5
        atom_norm = nrm2(dictionary[:, k])
        if atom_norm < 1e-10:
            pass
            # if verbose == 1:
            #     sys.stdout.write("+")
            #     sys.stdout.flush()
            # elif verbose:
            #     print("Adding new random atom")
            # dictionary[:, k] = random_state.randn(n_features)
            # if positive:
            #     np.clip(dictionary[:, k], 0, None, out=dictionary[:, k])
            # # Setting corresponding coefs to 0
            # code[k, :] = 0.0
            # # (U_k * U_k) ** 0.5
            # atom_norm = nrm2(dictionary[:, k])
            # dictionary[:, k] /= atom_norm
        else:
            dictionary[:, k] /= atom_norm
            # R <- -1.0 * U_k * V_k^T + R
            R = ger(-1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)
    if return_r2:
        R = nrm2(R) ** 2.0
        return dictionary, R
    return dictionary


def _update_dict_featureselect(W, X, C, verbose=False, return_r2=False,
                 random_state=None, positive=False, mode=1):
    """
    W: dictionary (n_features, n_components)
    X: data (n_features, n_samples)
    C: code (n_components, n_samples)
    """
    n_samples = X.shape[1]
    n_components = W.shape[1]
    n_features = X.shape[0]

    D = (np.abs(X) > 1e-16).astype(int)

    # print(D)

    G = np.dot(X, C.T)

    # Ct = np.array([np.dot(C[:,i:i+1], C[:,i:i+1].T) for i in range(n_samples)])
    # E = np.sum(Ct, axis=0)
    # #?
    E = np.dot(C, C.T)



    # CC = np.zeros((n_features, n_components, n_components))
    # for i in range(n_samples):
    #     CC += D[:,i][:,None,None] * C[:,i][None,:,None] * C[:,i][None,None,:]
    # CC = np.einsum('ji,ki,li->jkl', D, C, C)
    if mode == 3:
        # CD = np.diagonal(CC, axis1=1, axis2=2)
        pass
    else:
        CD = np.zeros_like(W)
        for i in range(n_samples):
            CD += C[:,i:i+1].T**2 * D[:,i:i+1]


    # --> changes during the iteration
    if mode == 2:
        M = np.zeros_like(W)
        for i in range(n_samples):
            M += (D[:,i] * np.dot(W, C[:,i]))[:, None] * C[:,i][None,:]

    if mode == 3:
        # werkt
        # M = np.einsum('ji,jk,ki,li->jl', D, W, C, C)

        # werkt
        DWC = D * (W @ C)
        M = DWC @ C.T

        # werkt
        # WC = W @ C
        # M = np.einsum('ji,ji,li->jl', D, WC, C)


    for k in range(n_components):

        if mode == 3:
            cd = np.sum(C[k:k+1,:]**2 * D, axis=1)
            b_ = np.abs(cd) > 1e-16
        else:
            b_ = np.abs(CD[:,k]) > 1e-16


        # if k < 2:
        #     print("\n> %d"%k)
        #     print(b_)
        #     print(((G[b_,k] - M[b_,k]) / CD[b_, k])[:10])

        if mode == 1:
            # --> changes during the iteration
            M = np.zeros_like(W)
            for i in range(n_samples):
                M += (D[:,i] * np.dot(W, C[:,i]))[:, None] * C[:,i][None,:]

        if mode in [2,3]:
            dW = - W[:,k].copy()

        if mode == 3:
            # W[b_, k] += (G[b_,k] - M[b_,k]) / cd[b_]
            W[b_,k] *= cd[b_]
            W[b_,k] += (G[b_,k] - M[b_,k])
        else:
            W[b_, k] += (G[b_,k] - M[b_,k]) / CD[b_, k]



        atom_norm = np.linalg.norm(W[:,k])
        W[:,k] /= atom_norm

        if mode == 2:
            dW += W[:,k]

            for i in range(n_samples):
                M += (D[:,i] * dW * C[k,i])[:, None] * C[:,i:i+1].T

        if mode == 3:
            dW += W[:,k]

            CkC = C[k:k+1,:] * C
            CkC = D[b_,:] @ CkC.T
            # cc = np.einsum('ji,li->jkl', D, )

            # for i in range(n_samples):
            #     M += (D[:,i] * dW * C[k,i])[:, None] * C[:,i:i+1].T

            M[b_,:] += dW[b_, None] * CkC

    # C *= atom_norms[:,None]

    return W, C


def _update_dict_no_featureselect_tryout(W, X, C, mode=1):
    """
    W: dictionary (n_features, n_components)
    X: data (n_features, n_samples)
    C: code (n_components, n_samples)
    mode: int
        1: W normalized, C renormalized, recreate all matrices each iteration
        2: W normalized, C renormalized, update all matrices
        3: W normalized
        4: no normalization
        residuals for 1,2,4 should be the same, residual for 3 should be same as
        sklearn
    """
    # print(W.shape, ',', X.shape, ',', C.shape)
    n_samples = X.shape[1]
    n_components = W.shape[1]

    # auxiliary matrices
    G = np.dot(X, C.T)
    E = C @ C.T

    # changes during iteration since W changes
    WC = W @ C
    M = W @ E

    # Ct = np.array([np.dot(C[:,i:i+1], C[:,i:i+1].T) for i in range(n_samples)])
    # E = np.sum(Ct, axis=0)
    # #?
    # E = np.dot(C, C.T)


    C_ = C.copy()

    atom_norms = np.ones(n_components)
    for k in range(n_components):

        if mode == 1:
            G = np.dot(X, C_.T)
            E = C_ @ C_.T
            M = W @ E

        cd = np.sum(C[k,:]**2)

        if mode in [3,4]:
            dM = - W[:,k].copy()

        if mode == 2:
            dM = - W[:,k:k+1] @ E[k:k+1,:]

        W[:, k] += (G[:,k] - M[:,k]) / cd

        atom_norms[k] = np.linalg.norm(W[:,k])

        if mode in [1,2,3]:
            W[:,k] /= atom_norms[k]

        if mode ==1:
            C_[k,:] *= atom_norms[k]


        if mode == 2:
            G[:,k] *= atom_norms[k]
            E[:,k] *= atom_norms[k]
            E[k,:] *= atom_norms[k]

            # update M
            dM += W[:,k:k+1] @ E[k:k+1,:]
            M += dM

        if mode in [3,4]:
            # update M
            dM += W[:,k]
            dM = dM[:,None] @ E[k:k+1,:]
            M += dM

    if mode in [1,2]:
        C *= atom_norms[:,None]


    return W, C


def _update_dict_like_sklearn(W, X, C):
    """
    Dictionary update as in sklearn

    W: dictionary (n_features, n_components) --> overwritten
    X: data (n_features, n_samples)
    C: code (n_components, n_samples) --> overwritten
    """
    n_samples = X.shape[1]
    n_components = W.shape[1]

    # auxiliary matrices
    G = np.dot(X, C.T)
    E = C @ C.T

    # changes during iteration since W changes
    M = W @ E

    for k in range(n_components):

        cd = np.sum(C[k,:]**2)

        # difference in M is (W[:,k]_post - W[:,k]_pre) @ E[k,:]
        # we already store W[:,k]_pre
        dM = - W[:,k].copy()

        # update of the dictionary atom
        W[:, k] += (G[:,k] - M[:,k]) / cd

        atom_norm = np.linalg.norm(W[:,k])
        W[:,k] /= atom_norm

        # update M
        dM += W[:,k]
        dM = dM[:,None] @ E[k:k+1,:]
        M += dM

    return W, C


def _update_dict_normalized(W, X, C):
    """
    Dictionary update while rescaling code items

    W: dictionary (n_features, n_components) --> overwritten
    X: data (n_features, n_samples)
    C: code (n_components, n_samples) --> overwritten
    """
    # print(W.shape, ',', X.shape, ',', C.shape)
    n_samples = X.shape[1]
    n_components = W.shape[1]

    # auxiliary matrices
    G = np.dot(X, C.T)
    E = C @ C.T

    # changes during iteration since W changes
    M = W @ E

    atom_norms = np.ones(n_components)
    for k in range(n_components):

        cd = np.sum(C[k,:]**2)

        # difference in M is (W[:,k]_post @ E[k,:]_post - W[:,k]_pre @ E[k,:]_pre)
        # we already store W[:,k]_pre @ E[k,:]_pre
        dM = - W[:,k:k+1] @ E[k:k+1,:]

        # update the dictionary atom
        W[:, k] += (G[:,k] - M[:,k]) / cd

        atom_norms[k] = np.linalg.norm(W[:,k])
        W[:,k] /= atom_norms[k]

        # rescale the code items so that code_k * atom_k remains unchanged under
        # division of atom_k by atom_norm k
        G[:,k] *= atom_norms[k]
        E[:,k] *= atom_norms[k]
        E[k,:] *= atom_norms[k]

        # update M
        dM += W[:,k:k+1] @ E[k:k+1,:]
        M += dM

    # compute the rescaled code items
    C *= atom_norms[:,None]

    return W, C

