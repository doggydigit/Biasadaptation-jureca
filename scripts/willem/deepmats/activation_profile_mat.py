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
from biasadaptation.weightmatrices import bmd
from biasadaptation.utils import utils, countfuncs
from biasadaptation.utils import k_task_n_class_m_dataset_data as knm

"""
axample usage

python activation_profile_mat.py --algo1 pmdd --nhidden1 100
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
# with open(paths.result_path + "biasopt/biasopt_1hl_%s_reweighted_%s%d_%s_ro=%s.p"%(args.dataset, algo_1, n_h1, "1vall", "tanh"), 'rb') as file:
        reslist = pickle.load(file)
w_in = torch.FloatTensor(reslist[0]['ws'][0])
b_in = torch.cat([torch.FloatTensor(res['bs'][0]) for res in reslist], dim=0)

print(w_in.shape)
print("\n> coherence:")
print((w_in.T @ w_in) - torch.eye(w_in.shape[1]))
print(torch.max(torch.abs((w_in.T @ w_in) - torch.eye(w_in.shape[1]))))


# load data
ds = helperfuncs.get_dataset_with_options(algo_1, args.dataset, None)
dl = torch.utils.data.DataLoader(ds, batch_size=1000, shuffle=True)
xdata, _ = next(iter(dl))
# compute differences and midpoints
xdiff, xmidp = utils.differences_and_midpoints_torch(xdata, 1000)



# compute activation profile
# print(w_in.shape)
activation_profiles_midp = countfuncs.get_activation_profile(xmidp, w_in, b_in)
activation_profiles_data = countfuncs.get_activation_profile(xdata, w_in, b_in)


def plot_profile(act_profiles, pname="", **kwargs):


    tot_profile = torch.zeros(n_h1)
    for act_profile in act_profiles:
        full_profile = torch.sum(act_profile, dim=1)
        tot_profile += full_profile

        # pl.figure(pname)
        # ax = pl.subplot(121)
        # ax.imshow(b_in.T)
        # ax = pl.subplot(122)
        # ax.imshow(act_profile)
        # pl.show()

        # pl.plot(np.arange(n_h1), full_profile)

    # print(tot_profile)

    # pl.figure(pname)
    tot_profile /= act_profiles.shape[0]
    pl.plot(np.arange(n_h1), tot_profile, **kwargs)

plot_profile(activation_profiles_midp, pname="midpoints", ls='-', c='k', lw=1.3)
plot_profile(activation_profiles_data, pname="datapoints", ls='--', c='b', lw=2.6)

# pl.show()


