import numpy as np
import torch

import argparse
import pickle
import copy
import sys
import os

sys.path.append('..')
import optim, helperfuncs#, countfuncs

from datarep.matplotlibsettings import *
from datarep import paths

from biasadaptation.biasfit import specificbiasfit
from biasadaptation.weightmatrices import bmd
from biasadaptation.utils import utils, countfuncs
from biasadaptation.utils import k_task_n_class_m_dataset_data as knm


"""
axample usage

python hiddendiffmat.py --algo1 pmdd --algo2 pmdd --nhidden1 100 --nhidden2 250
"""

parser = argparse.ArgumentParser()
parser.add_argument("--nhidden1", type=int, help="number of hidden neurons in first layer", default=25)
parser.add_argument("--nhidden2", type=int, help="number of hidden neurons in second layer", default=25)

parser.add_argument("--algo1", type=str, help="algorithm for input weight matrix", default='scd')
parser.add_argument("--algo2", type=str, help="algorithm for input weight matrix", default='scd')

parser.add_argument("--ndata", type=int, help="number of data points", default=470)
parser.add_argument("--nperbatch", type=int, help="number of data points", default=10)
parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")
parser.add_argument("--weightpath", type=str, help="path to which to save the file", default="/Users/wybo/Data/weight_matrices/")
parser.add_argument("--datasetpath", type=str, help="path to which to save the file", default=None)

args = parser.parse_args()
algo1 = args.algo1
algo2 = args.algo2
n_h1 = args.nhidden1
n_h2 = args.nhidden2

dname = "%s_ha_algos12=%s-%s_nh12=%d-%d"%(args.dataset, algo1, algo2, n_h1, n_h2)

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


def save_weight_matrix(path_name, file_name, W):
    np.save(os.path.join(path_name, file_name), W)

def save_extra_data(path_name, file_name, **kwargs):
    file_name_ = file_name + '_extra_data'
    np.savez(os.path.join(path_name, file_name_), **kwargs)


# weight matrix creation with different methods
if 'pca' == algo2:
    from biasadaptation.weightmatrices import pca
    print("creating weight matrix using PCA (%d -> %d)"%(n_h1, n_h2))
    # deprecated, should take data matrix
    data_matrix = construct_data_matrix(use_diff=False)
    if n_h2 <= nh1: # Number of requested components <= input dimensionality
        W_pca = pca.get_weightmatrix_pca(data_loader, n_h2)
        save_weight_matrix(args.weightpath, dname, W_pca)

if 'ica' == algo2:
    from  biasadaptation.weightmatrices import ica
    print("creating weight matrix using ICA (%d -> %d)"%(n_h1, n_h2))
    data_matrix = construct_data_matrix(use_diff=False)
    if n_h2 <= n_h1: # Number of requested components <= input dimensionality
        W_ica = ica.get_weightmatrix_ica(data_matrix, n_h2)
        save_weight_matrix(args.weightpath, dname, W_pca)

if 'sc' == algo2:
    from biasadaptation.weightmatrices import sc
    print("creating weight matrix using SC (%d -> %d)"%(n_h1, n_h2))
    data_matrix = construct_data_matrix(use_diff=False)
    W_sc = sc.get_weightmatrix_sc(data_matrix, n_h2, getsparsity=False)
    save_weight_matrix(args.weightpath, dname, W_pca)

if 'scd' == algo2:
    from biasadaptation.weightmatrices import sc
    print("creating weight matrix using SC of the input differences (%d -> %d)"%(n_h1, n_h2))
    data_matrix = construct_data_matrix(use_diff=True)
    W_scd = sc.get_weightmatrix_sc(data_matrix, n_h2, getsparsity=False)
    save_weight_matrix(args.weightpath, dname, W_scd)

if 'pmd' == algo2:
    from biasadaptation.weightmatrices import pmd
    print("creating weight matrix using PMD (%d -> %d)"%(n_h1, n_h2))
    data_matrix = construct_data_matrix(use_diff=False)
    W_pmd = pmd.get_weightmatrix_pmd(data_matrix, n_h2)
    save_weight_matrix(args.weightpath, dname, W_pmd)

if 'pmdd' == algo2:
    from biasadaptation.weightmatrices import pmd
    print("creating weight matrix using PMD of input differences (%d -> %d)"%(n_h1, n_h2))
    data_matrix = construct_data_matrix(use_diff=True)
    W_pmdd = pmd.get_weightmatrix_pmd(data_matrix, n_h2)
    save_weight_matrix(args.weightpath, dname, W_pmdd)

if 'scdfs' == algo2:
    from biasadaptation.weightmatrices import scfs
    print("creating weight matrix using SC with featureselect of input differences (%d -> %d)"%(n_h1, n_h2))
    data_matrix = construct_data_matrix(use_diff=True)
    W_scdfs, C_scdfs, errs_scdfs, res_scdfs = scfs.get_weightmatrix_scfs(data_matrix, n_h2, with_featureselect=True, return_all=True)
    # W_scdfs = scfs.get_weightmatrix_scfs(data_matrix, n_h2, with_featureselect=False)
    save_weight_matrix(args.weightpath, dname, W_scdfs)
    save_extra_data(args.weightpath, dname, C=C_scdfs, errs=np.array(errs_scdfs), res=np.array(res_scdfs))

if 'scdnfs' == algo2:
    from biasadaptation.weightmatrices import scfs
    print("creating weight matrix using SC without featureselect of input differences (%d -> %d)"%(n_h1, n_h2))
    data_matrix = construct_data_matrix(use_diff=True)
    W_scdnfs, C_scdnfs, errs_scdnfs, res_scdnfs = scfs.get_weightmatrix_scfs(data_matrix, n_h2, with_featureselect=False, return_all=True)
    save_weight_matrix(args.weightpath, dname, W_scdnfs)
    save_extra_data(args.weightpath, dname, C=C_scdnfs, errs=np.array(errs_scdnfs), res=np.array(res_scdnfs))






