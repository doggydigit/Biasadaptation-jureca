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
from datarep.matplotlibsettings import *

"""
example usage

python3 plot_usage_hists.py --algo1 scd --algo2 sc --algoc sc --nhidden1 50 --nhidden2 50

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

parser.add_argument("--nperbatch", type=int, help="number of datapoints per batch", default=100)

parser.add_argument("--save", type=bool, help="whether to save results or not", default=False)
parser.add_argument("--respath", type=str, help="path from which to load the trained network", default="/Users/wybo/Data/results/biasopt/")
parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")
parser.add_argument("--datasetpath", type=str, help="path where dataset is downloaded", default='/Users/wybo/Data/')

args = parser.parse_args()

n_h1 = args.nhidden1
n_h2 = args.nhidden2
algo1 = args.algo1
algo2 = args.algo2
algoc = args.algoc
algo12c = "-".join([algo1, algo2, algoc])

# source dataset
data_source = helperfuncs.get_dataset_with_options(algo1, args.dataset, args.datasetpath)
data_loader = torch.utils.data.DataLoader(data_source, batch_size=args.nperbatch)


namestring_1hl = "biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(args.dataset, algo1, n_h1, TASKTYPE, READOUT)
ws_dict_1hl, bs_dict_1hl, counts_dict_1hl = countfuncs.count_usage(os.path.join(args.respath, namestring_1hl), data_loader, args.dataset, verbose=True)

if len(algo2) > 0:
    namestring_2hl = "deepopt_2hl_%s_algo12c=%s_nh12=%d-%d_%s_ro=%s.p"%(args.dataset, algo12c, n_h1, n_h2, TASKTYPE, READOUT)
    ws_dict_2hl, bs_dict_2hl, counts_dict_2hl = countfuncs.count_usage(os.path.join(args.respath, namestring_2hl), data_loader, args.dataset, verbose=True)


# plot for first layer
pl.figure('counts l1', figsize=(18,9.5))
gs = GridSpec(12,8)

tot_count_1hl = torch.zeros(n_h1)
tot_count_2hl = torch.zeros(n_h1)
corr_same, corr_diff = 0., 0.
for kk in range(47):
    jj = kk%8
    ii = kk//8

    count_1hl = counts_dict_1hl[kk][0]
    tot_count_1hl += count_1hl

    ax = pl.subplot(gs[2*ii,jj])
    ax.set_title(str(kk))
    ax.bar(np.arange(len(count_1hl)), count_1hl, width=1., color=colours[0])

    ax.set_xticks([])
    ax.set_yticks([])


    if len(algo2) > 0:
        count_2hl = counts_dict_2hl[kk][0]
        tot_count_2hl += count_2hl

        ax = pl.subplot(gs[2*ii+1,jj])
        ax.bar(np.arange(len(count_2hl)), count_2hl, width=1., color=colours[1])

        ax.set_xticks([])
        ax.set_yticks([])

        # compute correlation between same and between random combs
        corr_same += np.corrcoef(count_1hl, count_2hl)[0,1]
        ll = np.random.choice(list(set(range(47)) - {kk}), 1)[0]
        corr_diff += np.corrcoef(count_1hl, counts_dict_2hl[ll][0])[0,1]


print('>>> correlation analysis')
print('    C(same task) =', corr_same/47)
print('    C(diff task) =', corr_diff/47)

ax = pl.subplot(gs[10,7])
ax.set_title('total')
ax.bar(np.arange(len(tot_count_1hl)), tot_count_1hl, width=1., color=colours[2])

ax.set_xticks([])
ax.set_yticks([])

if len(algo2) > 0:
    ax = pl.subplot(gs[11,7])
    ax.bar(np.arange(len(tot_count_2hl)), tot_count_2hl, width=1., color=colours[3])

    ax.set_xticks([])
    ax.set_yticks([])


# plot for second layer
if len(algo2) > 0:
    pl.figure('counts l2', figsize=(18,9.5))
    gs = GridSpec(6,8)

    tot_count_2hl = torch.zeros(n_h2)
    for kk in range(47):
        jj = kk%8
        ii = kk//8

        count_2hl = counts_dict_2hl[kk][1]
        tot_count_2hl += count_2hl

        ax = pl.subplot(gs[ii,jj])
        ax.set_title(str(kk))
        ax.bar(np.arange(len(count_2hl)), count_2hl, width=1., color=colours[1])

        ax.set_xticks([])
        ax.set_yticks([])


    ax = pl.subplot(gs[5,7])
    ax.set_title('total')
    ax.bar(np.arange(len(tot_count_2hl)), tot_count_2hl, width=1., color=colours[3])

    ax.set_xticks([])
    ax.set_yticks([])

if args.save:
    fname_1hl = "count_1hl_%s_%s%d_%s_ro=%s.p"%(args.dataset, algo1, n_h1, TASKTYPE, READOUT)
    with open(os.path.join(args.respath, fname_1hl), 'wb') as file:
        pickle.dump(counts_dict_1hl, file)

    fname_2hl = "count_2hl_%s_algo12c=%s_nh12=%d-%d_%s_ro=%s.p"%(args.dataset, algo12c, n_h1, n_h2, TASKTYPE, READOUT)
    with open(os.path.join(args.respath, fname_2hl), 'wb') as file:
        pickle.dump(counts_dict_2hl, file)

pl.tight_layout()
pl.show()

