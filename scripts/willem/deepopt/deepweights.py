import torch
import numpy as np
import sklearn.decomposition as skdec

import random
import argparse
import pickle
import os

import sys
sys.path.append('..')

# own package
from biasadaptation.utils import utils
from biasadaptation.weightmatrices import pmd

import optim, helperfuncs

'''
example usage

python3 deepweights.py --nhidden1 25 --nhidden2 25 --algo1 scd --algo2 sc --algoc sc --ndata 100 --enrich True
'''


# read command line args and kwargs
parser = argparse.ArgumentParser()
parser.add_argument("--nhidden1", nargs="*", type=int, help="number of hidden neurons", default=[10, 25, 50, 100, 250, 500])
parser.add_argument("--nhidden2", nargs="*", type=int, help="number of hidden neurons", default=[10, 25, 50, 100, 250, 500, 1000])
parser.add_argument("--ndata", type=int, help="number of data points", default=100000)

parser.add_argument("--algo1", type=str, help="methods to be applied to create weight matrix", default='scd')
parser.add_argument("--algo2", type=str, help="methods to be applied to create weight matrix", default='sc')
parser.add_argument("--algoc", type=str, help="methods to be applied to create weight matrix", default='sc')
parser.add_argument("--enrich", type=bool, help="whether to enrich coordinate matrix", default=False)

parser.add_argument("--path", type=str, help="path to which to save the file", default="")
parser.add_argument("--datapath", type=str, help="path to which the data sets are saved", default="./datasets")
parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")
args = parser.parse_args()


def get_data_matrix(nperbatch):
    data = helperfuncs.get_dataset(args.dataset, x_div='data', train=True)

    data_loader = torch.utils.data.DataLoader(
                         dataset=data,
                         batch_size=nperbatch,
                         shuffle=False,
                         drop_last=True)

    data_matrix, _ = next(iter(data_loader))
    diff_matrix = utils.differences_numpy(data_matrix.numpy(), data_matrix.shape[0])

    # print(diff_matrix.nbytes*1e-9, 'GB')

    return diff_matrix


def get_coordinates(algo, X_data, W_in, enrich=False):
    """
    Get the coordinates of the datapoints in X_data in the basis spanned by the
    rows of W_in

    Parameters
    ----------
    algo: 'lstsq' or 'sc'
        The algorithm used to estimate the coordinates
    X_data: np.ndarray (shape=(k, n))
        k datapoints with n features
    W_in: np.ndarray(nh, n)
        the nh vectors that consitute the basis of the coordinates
    enrich: bool
        whether to enrich the coordinate matrix with random elements

    Returns
    -------
    np.ndarray (shape=(k,nh))
        The coordinates
    """
    if algo == 'lstsq':
        C_oo = np.linalg.lstsq(W_in.T, X_data.T, rcond=None)[0].T
    elif algo == 'sc':
        C_oo = skdec.sparse_encode(X_data, W_in, algorithm='omp', alpha=0.1)
    else:
        raise ValueError('[algo] should be \'lstsq\' or \'sc\'')

    if enrich:
        eps = .1
        v_max = 20.
        ndraws = 10
        k, nh = C_oo.shape

        drand = eps + (v_max - eps) * np.random.rand(k*ndraws, nh)
        c_arrs = [coo[None,:] * drand[ndraws*ii:ndraws*(ii+1),:] for ii, coo in enumerate(C_oo)]

        C_oo = np.concatenate(c_arrs, axis=0)

    return C_oo


def construct_matrix_sc(n_h1, n_h2, algo_1, algo_2, algo_c):
    """
    Use sparse coding to construct the second weight matrix

    Parameters
    ----------
    n_h1: int
        number of units in second hidden layer
    h_h2: int
        number of units in first hidden layer
    algo_1: 'scd' or 'pmdd'
        algorithm for the first hidden layer weight matrix
    algo_c: 'sc' or 'lstsq'
        algorithm to compute the coordinates
    """
    X_data = get_data_matrix(args.ndata)

    W_in = helperfuncs.get_weight_matrix_in(n_h1, algo_1, dataset=args.dataset).T
    C_oo = get_coordinates(algo_c, X_data, W_in, enrich=args.enrich)

    if algo_2 == 'sc':

        print('\n--> coordinate SC')
        C_, W_out, n_iter = skdec.dict_learning_online(C_oo, n_components=n_h2, alpha=0.1, n_iter=100,
                            return_code=True, dict_init=None, callback=None,
                            batch_size=3, verbose=True, shuffle=True,
                            n_jobs=None, method='cd', iter_offset=0,
                            random_state=None, return_inner_stats=False,
                            inner_stats=None, return_n_iter=True,
                            positive_dict=False, positive_code=True,
                            method_max_iter=1000)
        print('    done, %d iterations\n'%n_iter)

    elif algo_2 == 'pmd':

        print('\n--> coordinate PMD')
        n, p = C_oo.shape
        pmd_ = pmd.PenalizedMatrixDecomposition(C_oo, c1=0.5*np.sqrt(n), c2=0.3*np.sqrt(p))
        _, _, W_out = pmd_(n_h2)
        print('    done\n')

    specifier = '_enriched' if args.enrich else ''
    namestring = '%s%s_algos12C=%s-%s-%s_nh12=%d-%d.p'%(args.dataset, specifier, algo_1, algo_2, algo_c, n_h1, n_h2)
    print(namestring)
    with open(os.path.join(args.path, namestring), 'wb') as f:
        pickle.dump(W_out, f)

    print('\n--- n_h1 = %d, n_h2 = %d, W_out.shape ='%(n_h1, n_h2), W_out.shape, '\n')


for n_h1 in args.nhidden1:
    for n_h2 in args.nhidden2:
        construct_matrix_sc(n_h1, n_h2, args.algo1, args.algo2, args.algoc)
