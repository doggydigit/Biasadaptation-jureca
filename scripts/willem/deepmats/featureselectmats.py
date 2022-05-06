import numpy as np
import torch
from sklearn import linear_model, decomposition

import argparse
import sys
import pickle
import copy

sys.path.append('..')
import helperfuncs

from biasadaptation.weightmatrices.ml import _least_angle, _dict_learning, _dict_learning_normalized
from biasadaptation.utils import utils, countfuncs

from datarep import paths

from dictlearning import dict_learning_normalized, dict_learning_featureselect, sparse_encoder

"""
example usage:

python l1opt_testing.py --nhidden1 250 --nhidden2 50 --algo1 pmdd
"""


parser = argparse.ArgumentParser()
parser.add_argument("--nhidden1", type=int, help="number of hidden neurons in first layer", default=100)
parser.add_argument("--nhidden2", type=int, help="number of hidden neurons in second layer", default=25)

parser.add_argument("--algo1", type=str, help="algorithm for input weight matrix", default='scd')

parser.add_argument("--ndata", type=int, help="number of data points", default=500)
parser.add_argument("--nperbatch", type=int, help="number of data points", default=10)

parser.add_argument("--datasetpath", type=str, help="path to which to save the file", default=None)
parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")
parser.add_argument("--weightpath", type=str, help="path to which to save the file", default="/Users/wybo/Data/weight_matrices/")

args = parser.parse_args()

args = parser.parse_args()
algo1 = args.algo1
algo2 = "fs"
n_h1 = args.nhidden1
n_h2 = args.nhidden2

# get the dataset
ds = helperfuncs.get_dataset_with_options(algo1, args.dataset, args.datasetpath)


dname = "%s_ha_algos12=%s-%s_nh12=%d-%d"%(args.dataset, algo1, algo2, n_h1, n_h2)

# load the 1hl data file
with open(paths.result_path + "biasopt/biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(args.dataset, algo1, n_h1, "1vall", "tanh"), 'rb') as file:
    reslist = pickle.load(file)
w_in = torch.FloatTensor(reslist[0]['ws'][0])
b_in = torch.cat([torch.FloatTensor(res['bs'][0]) for res in reslist], dim=0)


def construct_data_matrix(use_diff=True):
    """
    Construct data matrix as within-task difference vectors between hidden
    activities
    """
    dl = torch.utils.data.DataLoader(ds, batch_size=args.nperbatch, shuffle=True)

    data_list = []
    data_count = 0

    while data_count < args.ndata:
        for tt in range(ntask):
            try:
                xdata, _ = next(iter_dl)
            except (StopIteration, NameError) as e:
                iter_dl = iter(dl)
                xdata, _ = next(iter_dl)

            # compute hidden acitivities
            xhid = xdata @ w_in + b_in[tt:tt+1,:]
            xhid[xhid < 0.] = 0. # relu

            # compute differences
            if use_diff:
                xhid = utils.differences_torch(xhid, xhid.shape[0])

            data_list.append(xhid)
            data_count += xhid.shape[0]

    data_matrix = torch.cat(data_list, dim=0)
    print(data_matrix.shape)
    return data_matrix.numpy()


X = construct_data_matrix()
W = np.random.randn(n_h1, n_h2)
W /= np.linalg.norm(W, axis=1)[:,None]

Cfs = sparse_encoder.sparse_encode_featureselect(X, W.T, alpha=.2, n_nonzero_coefs=int(n_h2/10))

print(">>> dict learning standard")
C_orig, W_orig, errs_orig, res_orig = dict_learning_featureselect.dict_learning_featureselect(X, n_h2, dict_method='original',
                                                            alpha=1., max_iter=30, evaluate_residual=True, res_with_fs=True, dict_init=W.T, code_init=Cfs.copy())
print(">>> dict learning featureselect")
C_fs, W_fs, errs_fs, res_fs = dict_learning_featureselect.dict_learning_featureselect(X, n_h2, dict_method='featureselect',
                                                            alpha=1., max_iter=100, evaluate_residual=True, res_with_fs=True, dict_init=W.T, code_init=Cfs.copy())


def save_weight_matrix(path_name, file_name, W):
    import os
    np.save(os.path.join(path_name, file_name), W)




