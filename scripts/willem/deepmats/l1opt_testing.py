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

from dictlearning import dict_learning, dict_updater, sparse_encoder

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

args = parser.parse_args()

args = parser.parse_args()
algo1 = args.algo1
n_h1 = args.nhidden1
n_h2 = args.nhidden2

# get the dataset
ds = helperfuncs.get_dataset_with_options(algo1, args.dataset, args.datasetpath)

# load the 1hl data file
with open(paths.result_path + "biasopt/biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(args.dataset, algo1, n_h1, "1vall", "tanh"), 'rb') as file:
    reslist = pickle.load(file)
w_in = torch.FloatTensor(reslist[0]['ws'][0])
b_in = torch.cat([torch.FloatTensor(res['bs'][0]) for res in reslist], dim=0)

ntask = b_in.shape[0]

print(b_in.shape)

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

Cse = decomposition.sparse_encode(X, W.T)
# C = _least_angle.sparse_encode_no_featureselect(X, W.T, alpha=.2, n_nonzero_coefs=int(n_h2/10))
Cfs = _least_angle.sparse_encode_featureselect(X, W.T, alpha=.2, n_nonzero_coefs=int(n_h2/10))

print(Cfs.shape)

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

    residual = 0.
    if with_featureselect:
        for k in range(n_samples):
            bidx = np.abs(X[k,:]) > 1e-10 # only select nonzero targets
            res = np.linalg.norm(X[k,bidx] - C[k,:] @ W[:,bidx])

            # print(C[k,np.abs(C[k,:]) > 1e-10])
            # print(res)
            residual += res

    else:
        # for k in range(n_samples):
        #     residual += np.linalg.norm(X[k,:] - C[k,:] @ W)

        residual = np.sum(np.linalg.norm(X - C @ W, axis=1))

        # print("resres:", residual, residual_)

    return residual


# print("\n(i) Residuals with feature select")
# print("    > C without feature select:", calc_residual(X, W.T, C))
# print("    > C with feature select:   ", calc_residual(X, W.T, Cfs))

# print("(ii) Residuals without feature select")
# print("    > C without feature select:", calc_residual(X, W.T, C, with_featureselect=False))
# print("    > C with feature select:   ", calc_residual(X, W.T, Cfs, with_featureselect=False))

fs_flag = False

print("\n(i) Residual before dict update")
print("    > C Lasso:             ", calc_residual(X, W.T, Cse, with_featureselect=fs_flag))
print("    > C with featureselect:", calc_residual(X, W.T, Cfs, with_featureselect=fs_flag))

Wskl = _dict_learning._update_dict(W.copy(), X.T, Cse.T)
Wn1, Cn1 = _dict_learning._update_dict_like_sklearn(W.copy(), X.T, Cse.T.copy())
Cn1 = Cn1.T
# Wn2, Cn2 = _dict_learning._update_dict_normalized(W.copy(), X.T, Cse.T.copy())
# Wn3, Cn3 = _dict_learning._update_dict_no_featureselect_tryout(W.copy(), X.T, Cse.T.copy(), mode=4)

Wnfs1, Cnfs1 = _dict_learning._update_dict_featureselect(W.copy(), X.T, Cfs.T.copy(), mode=3)
Cnfs1 = Cnfs1.T
Wnfs2 = dict_updater._update_dict_featureselect(W.copy(), X.T, Cfs.T.copy())
Cnfs2 = Cfs.copy()

# import timeit
# print("\n> sklearn->update_dict:")
# print(timeit.timeit("_dict_learning._update_dict(W.copy(), X.T, Cse.T.copy())", number=10, globals=globals()))
# print("\n> _update_dict_like_sklearn:")
# print(timeit.timeit("dict_learning_normalized._update_dict2(W.copy(), X.T, Cse.T.copy())", number=10, globals=globals()))
# # print("\n> _update_dict_normalized:")
# # print(timeit.timeit("_dict_learning._update_dict_normalized(W.copy(), X.T, Cse.T.copy())", number=10, globals=globals()))
# # print("\n> _update_dilct_tryout:")
# # print(timeit.timeit("_dict_learning._update_dict_no_featureselect_tryout(W.copy(), X.T, Cse.T.copy(), mode=4)", number=10, globals=globals()))
# print("\n> _update_dict_featureselect orig:")
# print(timeit.timeit("_dict_learning._update_dict_featureselect(W.copy(), X.T, Cfs.T.copy(), mode=3)", number=10, globals=globals()))
# print("\n> _update_dict_featureselect new:")
# print(timeit.timeit("dict_learning_featureselect._update_dict_featureselect(W.copy(), X.T, Cfs.T.copy())", number=10, globals=globals()))

# # print(Wn - W)

print("\n(ii) Residuals after updating dictionary")
print("    > W sklearn:               ", calc_residual(X, Wskl.T, Cse, with_featureselect=fs_flag))
print("    > W like sklearn:          ", calc_residual(X, Wn1.T, Cn1, with_featureselect=fs_flag))
# print("    > W true gradient:         ", calc_residual(X, Wn2.T, Cn2, with_featureselect=fs_flag))
# print("    > W tryout:                ", calc_residual(X, Wn3.T, Cn3, with_featureselect=fs_flag))
print("    > W with feature select:   ", calc_residual(X, Wnfs1.T, Cfs, with_featureselect=fs_flag))
print("    > W with feature select:   ", calc_residual(X, Wnfs2.T, Cnfs2, with_featureselect=fs_flag))


# # 2nd iteration
# Cskl = decomposition.sparse_encode(X, Wskl.T)
# Cnorm = decomposition.sparse_encode(X, Wn2.T)

# print("\n(iii) Residual second iteration")
# print("    > C sklearn:   ", calc_residual(X, Wskl.T, Cskl, with_featureselect=fs_flag))
# print("    > C custom:    ", calc_residual(X, Wn2.T, Cnorm, with_featureselect=fs_flag))
# # print("    > C custom:    ", calc_residual(X, Wn2.T, Cn2, with_featureselect=fs_flag))


# # iteratively learn dictionary
# Wnorm = Wn2
# for kk in range(10):
#     # compute new dict
#     Wskl = _dict_learning._update_dict(Wskl.copy(), X.T, Cskl.T.copy())
#     Wnorm, Cnorm = _dict_learning._update_dict_normalized(Wnorm.copy(), X.T, Cnorm.T.copy())

#     # compute new code
#     Cskl  = decomposition.sparse_encode(X, Wskl.T)
#     Cnorm = decomposition.sparse_encode(X, Wnorm.T)


#     print("\n(iter %d) Residual "%(kk+3))
#     r_skl  = calc_residual(X, Wskl.T, Cskl, with_featureselect=fs_flag)
#     r_norm = calc_residual(X, Wnorm.T, Cnorm, with_featureselect=fs_flag)

#     print("    > Residual sklearn:   ", r_skl)
#     print("    > Resdiual normalized:", r_norm)
#     print("    > Delta:", r_skl - r_norm)

# print(">>> dict learning sklearn")
# decomposition.dict_learning(X, n_h2, alpha=1., max_iter=10)#, dict_init=W.T, code_init=Cse.copy())
print(">>> dict learning like sklearn/LARS")
# # _dict_learning_normalized.dict_learning_normalized(X, n_h2, dict_method='original', alpha=1., max_iter=10, evaluate_residual=True)#, dict_init=W.T, code_init=Cse.T.copy())
print(">>> dict learning LASSO")
dict_learning.dict_learning(X, n_h2, dict_method='original', alpha=1., max_iter=30, evaluate_residual=True, res_with_fs=True, dict_init=W.T, code_init=Cfs.copy())
print(">>> dict learning featureselect")
dict_learning.dict_learning(X, n_h2, dict_method='featureselect', alpha=1., max_iter=100, evaluate_residual=True, res_with_fs=True, dict_init=W.T, code_init=Cfs.copy())
