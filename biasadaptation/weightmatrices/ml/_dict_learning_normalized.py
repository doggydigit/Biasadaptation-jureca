import numpy as np
from scipy import linalg

# from sklearn.decomposition._dict_learning import *
from sklearn.decomposition._dict_learning import _check_positive_coding, _update_dict, sparse_encode
from sklearn.utils import check_random_state

import time


def _update_dict_normalized(W, X, C, verbose=False, return_r2=False,
                 random_state=None, positive=False):
    """
    Dictionary update while rescaling code items

    W: ndarray of shape (n_features, n_components) --> overwritten
        Value of the dictionary at the previous iteration.

    X: ndarray of shape (n_features, n_samples)
        Data matrix.

    C: ndarray of shape (n_components, n_samples) --> overwritten
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
    n_samples = X.shape[1]
    n_components = W.shape[1]
    random_state = check_random_state(random_state)

    # auxiliary matrices
    G = np.dot(X, C.T)
    E = C @ C.T

    # changes during iteration since W changes
    M = W @ E

    nc = 0
    for k in range(n_components):

        cd = np.sum(C[k,:]**2)
        if cd > 1e-16:
            nc += 1
            # print(cd)
            # print(G[:,k] - M[:,k])

            # difference in M is (W[:,k]_post @ E[k,:]_post - W[:,k]_pre @ E[k,:]_pre)
            # we already store W[:,k]_pre @ E[k,:]_pre
            dM = - W[:,k:k+1] @ E[k:k+1,:]

            # update the dictionary atom
            W[:, k] += (G[:,k] - M[:,k]) / cd
            if positive:
                np.clip(W[:, k], 0, None, out=W[:, k])
            atom_norm = np.linalg.norm(W[:,k])

            if atom_norm < 1e-10:
                if verbose == 1:
                    sys.stdout.write("+")
                    sys.stdout.flush()
                elif verbose:
                    print("Adding new random atom")
                W[:, k] = random_state.randn(n_features)
                if positive:
                    np.clip(W[:, k], 0, None, out=W[:, k])
                # Setting corresponding code elements to 0
                G[:,k] = 0.0
                E[:,k] = 0.0
                E[k,:] = 0.0
                C[k,:] = 0.0

                atom_norm = np.linalg.norm(W[:, k])
                W[:, k] /= atom_norm

            else:
                W[:,k] /= atom_norm

                # rescale the code items so that code_k * atom_k remains unchanged under
                # division of atom_k by atom_norm k
                G[:,k] *= atom_norm
                E[:,k] *= atom_norm
                E[k,:] *= atom_norm
                C[k,:] *= atom_norm

            # update M
            dM += W[:,k:k+1] @ E[k:k+1,:]
            M += dM

    print(nc)

    if return_r2:
        R = X - W @ C
        R = np.linalg.norm(R)

        return W, R

    return W



def calc_residual(X, W, C, with_featureselect=True):
    """
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.
    W : ndarray of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows for meaningful
        output.
    C : ndarray of shape (n_samples, n_components)
        The code matrix
    """
    n_samples = X.shape[0]
    residual = np.sum(np.linalg.norm(X - C @ W, axis=1))

    return residual

def dict_learning_normalized(X, n_components, *, alpha, max_iter=100, tol=1e-8,
                  method='lars', n_jobs=None, dict_init=None, code_init=None,
                  callback=None, verbose=False, random_state=None,
                  return_n_iter=False, positive_dict=False,
                  positive_code=False, method_max_iter=1000,
                  evaluate_residual=False, X_test=None,
                  dict_method='original'):
    """Solves a dictionary learning matrix factorization problem.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_2^2 + alpha * || U ||_1
                     (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.

    n_components : int
        Number of dictionary atoms to extract.

    alpha : int
        Sparsity controlling parameter.

    max_iter : int, default=100
        Maximum number of iterations to perform.

    tol : float, default=1e-8
        Tolerance for the stopping condition.

    method : {'lars', 'cd'}, default='lars'
        The method used:

        * `'lars'`: uses the least angle regression method to solve the lasso
           problem (`linear_model.lars_path`);
        * `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). Lars will be faster if
          the estimated components are sparse.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial value for the dictionary for warm restart scenarios. Only used
        if `code_init` and `dict_init` are not None.

    code_init : ndarray of shape (n_samples, n_components), default=None
        Initial value for the sparse code for warm restart scenarios. Only used
        if `code_init` and `dict_init` are not None.

    callback : callable, default=None
        Callable that gets invoked every five iterations

    verbose : bool, default=False
        To control the verbosity of the procedure.

    random_state : int, RandomState instance or None, default=None
        Used for randomly initializing the dictionary. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    method_max_iter : int, default=1000
        Maximum number of iterations to perform.

    evaluate_residual : bool
        Evaluate the residual after deriving a new code, will use `X_test` if
        provided. Otherwise, uses `X`.

    X_test: ndarray of shape (n_samples, n_features)
        Test array to evaluate the residual

    dict_method : bool
        the original sklearn method or the normalized method

        .. versionadded:: 0.22

    Returns
    -------
    code : ndarray of shape (n_samples, n_components)
        The sparse code factor in the matrix factorization.

    dictionary : ndarray of shape (n_components, n_features),
        The dictionary factor in the matrix factorization.

    errors : array
        Vector of errors at each iteration.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to True.

    See Also
    --------
    dict_learning_online
    DictionaryLearning
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA
    """
    if method not in ('lars', 'cd'):
        raise ValueError('Coding method %r not supported as a fit algorithm.'
                         % method)

    _check_positive_coding(method, positive_code)

    method = 'lasso_' + method

    t0 = time.time()
    # Avoid integer division problems
    alpha = float(alpha)
    random_state = check_random_state(random_state)

    # Init the code and the dictionary with SVD of Y
    if code_init is not None and dict_init is not None:
        code = np.array(code_init, order='F')
        # Don't copy V, it will happen below
        dictionary = dict_init
    else:
        code, S, dictionary = linalg.svd(X, full_matrices=False)
        dictionary = S[:, np.newaxis] * dictionary
    r = len(dictionary)
    if n_components <= r:  # True even if n_components=None
        code = code[:, :n_components]
        dictionary = dictionary[:n_components, :]
    else:
        code = np.c_[code, np.zeros((len(code), n_components - r))]
        dictionary = np.r_[dictionary,
                           np.zeros((n_components - r, dictionary.shape[1]))]

    # Fortran-order dict, as we are going to access its row vectors
    dictionary = np.array(dictionary, order='F')

    residuals = 0

    errors = []
    current_cost = np.nan

    if verbose == 1:
        print('[dict_learning]', end=' ')

    # If max_iter is 0, number of iterations returned should be zero
    ii = -1

    if evaluate_residual:
        residuals = []

    for ii in range(max_iter):
        dt = (time.time() - t0)
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        elif verbose:
            print("Iteration % 3i "
                  "(elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)"
                  % (ii, dt, dt / 60, current_cost))

        # Update code
        code = sparse_encode(X, dictionary, algorithm=method, alpha=alpha,
                             init=code, n_jobs=n_jobs, positive=positive_code,
                             max_iter=method_max_iter, verbose=verbose)

        if evaluate_residual:
            if X_test is None:
                res = calc_residual(X, dictionary, code)
            else:
                code_test = sparse_encode(X_test, dictionary,
                             algorithm=method, alpha=alpha,
                             init=code, n_jobs=n_jobs, positive=positive_code,
                             max_iter=method_max_iter, verbose=verbose)
                res = calc_residual(X, dictionary, code_test)

            residuals.append(res)
            # if verbose > 1:
            print("> residual (new code %d) = %.8f"%(ii, res))

        # Update dictionary
        if dict_method == 'original':
            dictionary, residual2 = _update_dict(dictionary.T, X.T, code.T,
                                                 verbose=verbose, return_r2=True,
                                                 random_state=random_state,
                                                 positive=positive_dict)
        elif dict_method == 'normalized':
            dictionary, residual2 = _update_dict_normalized(dictionary.T, X.T, code.T,
                                                 verbose=verbose, return_r2=True,
                                                 random_state=random_state,
                                                 positive=positive_dict)
        else:
            raise IOError("dict method should be \'original\' or \'normalized\'")

        dictionary = dictionary.T

        # Cost function
        current_cost = 0.5 * residual2 + alpha * np.sum(np.abs(code))
        errors.append(current_cost)

        if ii > 0:
            dE = errors[-2] - errors[-1]
            # assert(dE >= -tol * errors[-1])
            if dE < tol * errors[-1]:
                if verbose == 1:
                    # A line return
                    print("")
                elif verbose:
                    print("--- Convergence reached after %d iterations" % ii)
                break
        if ii % 5 == 0 and callback is not None:
            callback(locals())

    if return_n_iter:
        return code, dictionary, errors, ii + 1
    elif evaluate_residual:
        return code, dictionary, errors, residuals
    else:
        return code, dictionary, errors