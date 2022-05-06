"""
optimizations for one hidden layer
"""
import numpy as np
import torch

import argparse
import pickle
import copy
import os
import random

import sys
sys.path.append('..')

import optim, helperfuncs, countfuncs
from biasadaptation.utils import utils
from datarep.matplotlibsettings import *

"""
example usage

python3 usagewmats.py --algo1 scd  --nhidden1 50

"""

# some global variables
TASKTYPE = "1vall"
READOUT = "tanh"
SUFFIX=''
torch.set_num_threads(1)

# read command line args and kwargs
parser = argparse.ArgumentParser()
parser.add_argument("--nhidden1", type=int, help="number of hidden neurons", default=25)
parser.add_argument("--nhidden2", type=int, help="number of hidden neurons", default=25)
parser.add_argument("--algo1", type=str, help="methods to be applied to create weight matrix", default='scd')
parser.add_argument("--algo2", type=str, help="methods to be applied to create weight matrix", default='')
parser.add_argument("--algoc", type=str, help="methods to be applied to create weight matrix", default='')

parser.add_argument("--nperbatch", type=int, help="number of datapoints per batch", default=1000)
parser.add_argument("--ndiffmat", type=int, help="number of datapoints per batch", default=1000)

parser.add_argument("--recount", type=bool, help="number of datapoints per batch", default=False)
parser.add_argument("--save", type=bool, help="whether to save results or not", default=True)

parser.add_argument("--respath", type=str, help="path from which to load the trained network", default="/Users/wybo/Data/results/biasopt/")
parser.add_argument("--weightpath", type=str, help="path from which to load the trained network", default="/Users/wybo/Data/weight_matrices/")
parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")
parser.add_argument("--datasetpath", type=str, help="path where dataset is downloaded", default='/Users/wybo/Data/')

args = parser.parse_args()
n_h1 = args.nhidden1
n_h2 = args.nhidden2
algo1 = args.algo1
algo2 = args.algo2
algoc = args.algoc

# source dataset
data_source = helperfuncs.get_dataset_with_options(algo1, args.dataset, args.datasetpath)
data_loader = torch.utils.data.DataLoader(data_source, batch_size=args.nperbatch)

# save the weights
def save_weight_matrix(path_name, file_name, W):
    np.save(os.path.join(path_name, file_name), W)

if len(algo2) == 0:
    n_h = n_h1
    algo12c = algo1
    nhstr = str(n_h)

    fname_res = "biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(args.dataset, algo1, n_h1, TASKTYPE, READOUT)
    fname_count = "count_1hl_%s_%s%d_%s_ro=%s.p"%(args.dataset, algo1, n_h1, TASKTYPE, READOUT)
    fname_res = os.path.join(args.respath, fname_res)
    fname_count = os.path.join(args.respath, fname_count)

    w_mat = torch.FloatTensor(helperfuncs.get_weight_matrix_in(n_h1, algo1, dataset=args.dataset))
    count = countfuncs.maybe_count(fname_count, fname_res, data_loader, dataset=args.dataset, recompute=args.recount, hl=1)
    data_matrix = countfuncs.differences_weighted(data_loader, w_mat, count, n_diff=args.ndiffmat, n_batch=args.nperbatch, verbose=True)
else:
    n_h = n_h2
    algo12c = "-".join([algo1, algo2, algoc])
    nhstr = "_".join([str(n_h1), str(n_h2)])

    fname_res = "deepopt_2hl_%s_algo12c=%s_nh12=%d-%d_%s_ro=%s.p"%(args.dataset, algo12c, n_h1, n_h2, TASKTYPE, READOUT)
    fname_count = "count_2hl_%s_algo12c=%s_nh12=%d-%d_%s_ro=%s.p"%(args.dataset, algo12c, n_h1, n_h2, TASKTYPE, READOUT)
    fname_res = os.path.join(args.respath, fname_res)
    fname_count = os.path.join(args.respath, fname_count)

    w_mat1 = torch.FloatTensor(helperfuncs.get_weight_matrix_in(n_h1, algo1, dataset=args.dataset))
    w_mat2 = torch.FloatTensor(helperfuncs.get_weight_matrix_hidden(n_h1, n_h2, algo1, algo2, algoc, dataset=args.dataset))
    count = countfuncs.maybe_count(fname_count, fname_res, data_loader, dataset=args.dataset, recompute=args.recount, hl=2)
    data_matrix = countfuncs.coordinates_weighted(data_loader, w_mat1, w_mat2, count, algoc, n_diff=args.ndiffmat, n_batch=args.nperbatch, verbose=True)

data_matrix = data_matrix.numpy()
print(data_matrix.shape)

if algo1 == 'scd':
    from biasadaptation.weightmatrices import sc
    W_scd = sc.get_weightmatrix_scd(data_matrix, n_h, getsparsity=False)
    if args.save:
        save_weight_matrix(args.weightpath, args.dataset+'_reweighted_%s%s'%(algo12c, nhstr), W_scd)

elif algo1 == 'pmdd':
    from biasadaptation.weightmatrices import pmd
    W_pmdd = pmd.get_weightmatrix_pmdd(data_matrix, n_h)
    if args.save:
        save_weight_matrix(args.weightpath, args.dataset+'_reweighted_%s%s'%(algo12c, nhstr), W_pmdd)

