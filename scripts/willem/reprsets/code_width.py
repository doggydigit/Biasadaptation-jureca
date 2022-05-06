"""
optimizations for one hidden layer
"""
import numpy as np
import torch
import torch.nn.functional as tfunc

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

python code_width.py --algo1 scd --nhidden1 50

"""

# some global variables
TASKTYPE = "1vall"
READOUT = "tanh"
SUFFIX=''
NTASK = 47
torch.set_num_threads(1)

# read command line args and kwargs
parser = argparse.ArgumentParser()
parser.add_argument("--nhidden1", type=int, help="number of hidden neurons", default=25)
parser.add_argument("--algo1", type=str, help="methods to be applied to create weight matrix", default='scd')

parser.add_argument("--nperbatch", type=int, help="number of datapoints per batch", default=200)

parser.add_argument("--save", type=bool, help="whether to save results or not", default=False)
parser.add_argument("--respath", type=str, help="path from which to load the trained network", default="/Users/wybo/Data/results/biasopt/")
parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")
parser.add_argument("--datasetpath", type=str, help="path where dataset is downloaded", default='/Users/wybo/Data/')

args = parser.parse_args()

dataset = args.dataset
n_h1 = args.nhidden1
algo1 = args.algo1

# source dataset
data_source = helperfuncs.get_dataset_with_options(algo1, args.dataset, args.datasetpath)
data_loader = torch.utils.data.DataLoader(data_source, batch_size=args.nperbatch)

namestring = "biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(args.dataset, algo1, n_h1, TASKTYPE, READOUT)


def load_results(fname):
    # load the relevant data from the results file
    with open(fname, 'rb') as file:
        reslist = pickle.load(file)
    ws_list = [res['ws'] for res in reslist]
    bs_list = [res['bs'] for res in reslist]
    task_list = [list(res['task'][-1][dataset].keys())[0] for res in reslist]

    # rearrange to task indices
    ws_dict, bs_dict = countfuncs.rearrange_to_task_ind(task_list, ws_list, bs_list)

    return ws_dict, bs_dict


xdata = next(iter(data_loader))[0]
xdiff = utils.differences_torch(xdata, args.nperbatch)

ws_dict, bs_dict = load_results(os.path.join(args.respath, namestring))
w_mat = ws_dict[0][0].T
coo = countfuncs.get_coordinates("sc", xdiff.numpy(), w_mat)

perturbation_norms = np.arange(0., 100., 5)
error_norms = []
for pert_norm in perturbation_norms:
    dcoo = np.random.randn(*coo.shape)
    dcoo *= pert_norm / np.linalg.norm(dcoo, axis=1)[:,None]

    coo_ = coo + dcoo

    print(coo_.shape, w_mat.shape)

    err = np.linalg.norm(xdiff - coo_ @ w_mat)

    error_norms.append(err)

pl.figure(algo1)
ax = pl.gca()

ax.plot(perturbation_norms, error_norms, 'bD--')

pl.show()







