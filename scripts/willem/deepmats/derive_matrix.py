import numpy as np
import torch

import argparse
import pickle
import copy
import sys

sys.path.append('..')
import optim, helperfuncs#, countfuncs

from datarep.matplotlibsettings import *
from datarep import paths

from biasadaptation.biasfit import specificbiasfit
from biasadaptation.weightmatrices.ml import coom
from biasadaptation.utils import utils, countfuncs
from biasadaptation.utils import k_task_n_class_m_dataset_data as knm

"""
axample usage

python3 derive_matrix.py --nhidden1 25 --nhidden2 25 --algo1 pmdd --algoc lstsq
"""

parser = argparse.ArgumentParser()
parser.add_argument("--nhidden1", type=int, help="number of hidden neurons in first layer", default=25)
parser.add_argument("--nhidden2", type=int, help="number of hidden neurons in second layer", default=25)

parser.add_argument("--algo1", type=str, help="algorithm for input weight matrix", default='scd')
parser.add_argument("--algo2", type=str, help="algorithm for hidden weight matrix", default='sc')
parser.add_argument("--algoc", type=str, help="algorithm for coordinate transform", default='sc')
parser.add_argument("--taskid", type=int, help="task id", default=0)

parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")

args = parser.parse_args()

n_h1 = args.nhidden1
n_h2 = args.nhidden2
algo_1 = args.algo1
algo_2 = args.algo2
algo_c = args.algoc

# load weights and biasses
# w_in  = helperfuncs.get_weight_matrix_in(n_h1, algo_1, dataset=args.dataset)
with open(paths.result_path + "biasopt/biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(args.dataset, algo_1, n_h1, "1vall", "tanh"), 'rb') as file:
        reslist = pickle.load(file)
w_in = torch.FloatTensor(reslist[0]['ws'][0]).T
b_in = torch.cat([torch.FloatTensor(res['bs'][0]) for res in reslist], dim=0)

# load data
ds = helperfuncs.get_dataset_with_options(algo_1, args.dataset, None)
dl = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=True)


coom.find_matrix(dl, n_h2, w_in, b_in, n_max_iter=1000)


