import numpy as np

from dictlearning import dict_learning, sparse_encoder


def get_weightmatrix_scfs(data_matrix, n_h, n_max_iter=200, with_featureselect=True, frac_nonzero=.2,
                          return_all=False):
    """
    Parameters
    ----------
    data_matrix: ndarray (n_datapoints, input_dim)
        the data matrix
    n_h: int
        the number of dictionary vectors
    n_max_iter: int
        maximum number of iterations

    Returns
    -------
    np.ndarray (n_h, input_dim)
        The weight matrix
    """
    dict_method = 'featureselect' if with_featureselect else 'original'

    # initial weight matrix
    n_i = data_matrix.shape[1]
    W_init = np.random.randn(n_i, n_h)
    W_init /= np.linalg.norm(W_init, axis=1)[:,None]

    # initial code
    n_nonzero_coefs = int(np.round(frac_nonzero * n_h))
    if with_featureselect:
        C_init = sparse_encoder.sparse_encode_featureselect(data_matrix, W_init.T,
                                        n_nonzero_coefs=n_nonzero_coefs)
    else:
        C_init = sparse_encoder.sparse_encode(data_matrix, W_init.T,
                                        n_nonzero_coefs=n_nonzero_coefs)

    C, W, errs, res = \
            dict_learning.dict_learning(data_matrix, n_h,
                            dict_method=dict_method,
                            alpha=.1, max_iter=n_max_iter, frac_nonzero=frac_nonzero,
                            evaluate_residual=True, res_with_fs=True,
                            dict_init=W_init.T, code_init=C_init.copy())

    if return_all:
        return W, C, errs, res
    else:
        return W