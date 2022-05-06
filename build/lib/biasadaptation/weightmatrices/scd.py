# Sparse Coding of difference matrix

import time
from tqdm import tqdm

import numpy as np
from sklearn.decomposition import dict_learning_online
from sklearn.preprocessing import normalize

from ..utils import utils


def get_sparsity(code):
    return np.sum(code == 0.) / code.size


def get_sc_trafo_matrix(data_matrix, n_h, getsparsity=True):
    diff_matrix = utils.differences_numpy(data_matrix, data_matrix.shape[0])

    dictout = dict_learning_online(diff_matrix, n_components=n_h, alpha=1, n_iter=10000,
                            return_code=getsparsity, dict_init=None, callback=None,
                            batch_size=50, verbose=True, shuffle=True,
                            n_jobs=None, method='cd', iter_offset=0,
                            random_state=None, return_inner_stats=False,
                            inner_stats=None, return_n_iter=False,
                            positive_dict=False, positive_code=True,
                            method_max_iter=1000)

    if getsparsity:
        code = dictout[0]
        print("sparsity for "+str(n_h)+" hidden neurons: "+str(get_sparsity(code)))
        dictionary = dictout[1]
    else:
        dictionary = dictout
    return dictionary


def get_weightmatrix_scd(data_matrix, n_h, getsparsity=True):
    print("creating weigth matrix for SC for "+str(n_h)+" hidden neurons...")
    w_mat = get_sc_trafo_matrix(data_matrix, n_h, getsparsity=getsparsity)
    w_mat = normalize(w_mat)
    return w_mat
