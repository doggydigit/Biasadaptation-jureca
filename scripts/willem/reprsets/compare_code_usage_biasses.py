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
from datarep.matplotlibsettings import *

"""
example usage

python compare_code_usage_biasses.py --algo1 scd --nhidden1 50

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
parser.add_argument("--nhidden2", type=int, help="number of hidden neurons", default=25)
parser.add_argument("--algo1", type=str, help="methods to be applied to create weight matrix", default='scd')
parser.add_argument("--algo2", type=str, help="methods to be applied to create weight matrix", default='')
parser.add_argument("--algoc", type=str, help="methods to be applied to create weight matrix", default='')

parser.add_argument("--nperbatch", type=int, help="number of datapoints per batch", default=200)

parser.add_argument("--save", type=bool, help="whether to save results or not", default=False)
parser.add_argument("--respath", type=str, help="path from which to load the trained network", default="/Users/wybo/Data/results/biasopt/")
parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")
parser.add_argument("--datasetpath", type=str, help="path where dataset is downloaded", default='/Users/wybo/Data/')

args = parser.parse_args()

dataset = args.dataset
n_h1 = args.nhidden1
n_h2 = args.nhidden2
algo1 = args.algo1
algo2 = args.algo2
algoc = args.algoc
algo12c = "-".join([algo1, algo2, algoc])

# source dataset
data_source = helperfuncs.get_dataset_with_options(algo1, args.dataset, args.datasetpath)
data_loader = torch.utils.data.DataLoader(data_source, batch_size=args.nperbatch)

namestring = "biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(args.dataset, algo1, n_h1, TASKTYPE, READOUT)
# ws_dict_1hl, bs_dict_1hl, counts_dict_1hl = countfuncs.count_usage(os.path.join(args.respath, namestring_1hl), data_loader, args.dataset, verbose=True)

if len(algo2) > 0:
    namestring = "deepopt_2hl_%s_algo12c=%s_nh12=%d-%d_%s_ro=%s.p"%(args.dataset, algo12c, n_h1, n_h2, TASKTYPE, READOUT)
    # ws_dict_2hl, bs_dict_2hl, counts_dict_2hl = countfuncs.count_usage(os.path.join(args.respath, namestring_2hl), data_loader, args.dataset, verbose=True)


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


def differences_with_data(x, n_diff):
    """
    Sample differences between datapoints

    Parameters
    ----------
    x: torch.tensor (batch_size, input_dim)
        the data points
    n_diff: int
        the number of differences

    Returns
    -------
    torch.tensor (n_diff, input_dim)
        Random sample of the difference vectors
    """

    id1 = np.random.randint(x.shape[0], size=n_diff)
    id2 = np.random.randint(x.shape[0], size=n_diff)

    idx = np.where(np.not_equal(id1, id2))

    id1 = torch.LongTensor(id1[idx])
    id2 = torch.LongTensor(id2[idx])

    return x[id1,:] - x[id2,:], x[id1], x[id2]


def count_usage(ws, bs, xdata):
    ws = [torch.FloatTensor(w) for w in ws]
    bs = [torch.FloatTensor(b) for b in bs]

    # to store the usage count, same structure as `bs`
    counts = [torch.zeros(b.shape[1], dtype=int) for b in bs]

    o = xdata
    # compute forward pass
    for w, b, count in zip(ws, bs, counts):
        o = tfunc.relu(torch.mm(o, w) + b[0, :])

        count += torch.sum((o > 1e-10).int(), 0)

    return counts


def get_task_differences_with_data(data_loader):
    (xdata, xlabel), (xtask, xtarget) = next(iter(data_loader))
    xdat0 = xdata[xtarget < 0]
    xdat1 = xdata[xtarget > 0]

    xlen = min(len(xdat0), len(xdat1))

    xdiff = xdat0[:xlen] - xdat1[:xlen]

    return xdiff, xdat0[:xlen], xdat1[:xlen]


def get_tasks(nsample=1000000):
    class_idx = np.arange(NTASK)

    return [{-1: {'EMNIST': {c_: nsample}}, 1: {'EMNIST': [cc for cc in class_idx if cc != c_]}} for c_ in class_idx]


def check_feasibility(coo, xce, w_mat, nsample=3):
    """
    coo: np.array
        sparse coordinates of difference vectors
    xce: np.array
        difference vector centroids
    w_mat: np.array
        weight matrix
    """
    coo = coo[:nsample]
    xce = xce[:nsample]

    xw =  xce @ w_mat

    for ii in range(n_h1):
        xw1 = xw[coo[:,ii] > 1e-10, ii]
        xw0 = xw[coo[:,ii] < 1e-10, ii]

        try:
            xmin = torch.min(xw1)
        except RuntimeError:
            xmin = 1e10
        try:
            xmax = torch.max(xw0)
        except RuntimeError:
            xmax = -1e10

        # print(xw1)
        # print(xw0)

        # print("Unit %d feasible? %s (max_i_notin_S = %.5f, min_i_in_S = %.5f"%(ii, xmin > xmax, xmax, xmin) )





xdata = next(iter(data_loader))[0]
xdiff, xdata0, xdata1 = differences_with_data(xdata, args.nperbatch)

tasks = get_tasks()
ws_dict, bs_dict = load_results(os.path.join(args.respath, namestring))

# for ii in range(len(xdiff)):

#     coo = countfuncs.get_coordinates("sc", xdiff.numpy(), ws_dict[0][0].T)
#     ccount = np.sum(coo, axis=0)


corrtot_count_bias = 0.
corrtot_count_coords = 0.
corrtot_coords_bias = 0.
for kk in range(NTASK):
    jj = kk%8
    ll = kk//8

    data_loaders = helperfuncs.construct_knm_dataloader_triplet(
                            'EMNIST', tasks[kk], copy.deepcopy(tasks[kk]),
                            data_source, data_source,
                            args.nperbatch, 100, 100)
    xdiff, xdata0, xdata1 = get_task_differences_with_data(data_loaders[0])

    coo = countfuncs.get_coordinates("sc", xdiff.numpy(), ws_dict[0][0].T)
    ccount = np.sum(coo, axis=0)


    coo0 = countfuncs.get_coordinates("sc", xdiff.numpy(), ws_dict[0][0].T)
    coo1 = countfuncs.get_coordinates("sc", xdiff.numpy(), ws_dict[0][0].T)
    coo2 = countfuncs.get_coordinates("sc", xdiff.numpy(), ws_dict[0][0].T)

    print('\n>>> codes')
    print(coo0[0])
    print(coo1[0])
    print(coo2[0])

    check_feasibility(coo, (xdata0+xdata1)/2., ws_dict[kk][0])

    counts0 = count_usage(ws_dict[kk], bs_dict[kk], xdata0)
    counts1 = count_usage(ws_dict[kk], bs_dict[kk], xdata1)

    corr_count_bias = np.corrcoef(counts0[0]+counts1[0], bs_dict[kk][0][0])[0,1]
    corr_count_coords = np.corrcoef(counts0[0]+counts1[0], ccount)[0,1]
    corr_coords_bias = np.corrcoef(ccount, bs_dict[kk][0][0])[0,1]

    corrtot_count_bias += np.abs(corr_count_bias)
    corrtot_count_coords += np.abs(corr_count_coords)
    corrtot_coords_bias += np.abs(corr_coords_bias)

    print('\n>>> Task %d'%kk)
    print('C count <-> bias   = %.4f'%corr_count_bias)
    print('C count <-> coords = %.4f'%corr_count_coords)
    print('C bias <-> coords  = %.4f'%corr_coords_bias)

    # pl.figure("xdat %d"%kk, figsize=(10,10))
    # gs = GridSpec(5,5)
    # for mm in range(5):
    #     for nn in range(5):
    #         oo = mm*5+nn
    #         ax = pl.subplot(gs[mm,nn])
    #         ax.bar(np.arange(n_h1), bs_dict[kk][0][0], width=1., color=colours[3])
    #         ax.bar(np.arange(n_h1), coo[oo], width=1., color=colours[4])


    # pl.figure(str(kk), figsize=(5,10))
    # ax0 = pl.subplot(311)
    # ax0.bar(np.arange(n_h1), ccount, width=1., color=colours[0])

    # ax1 = pl.subplot(312)
    # ax1.bar(np.arange(n_h1), counts0[0], width=1., color=colours[1])
    # ax1.bar(np.arange(n_h1), counts1[0], bottom=counts0[0], width=1., color=colours[2])

    # ax2 = pl.subplot(313)
    # ax2.bar(np.arange(n_h1), bs_dict[kk][0][0], width=1., color=colours[3])

    # pl.tight_layout()
    # pl.show()


print('\n>>> Average')
print('C count <-> bias   = %.4f'%(corrtot_count_bias/NTASK))
print('C count <-> coords = %.4f'%(corrtot_count_coords/NTASK))
print('C bias <-> coords  = %.4f'%(corrtot_coords_bias/NTASK))

# # plot for first layer
# pl.figure('counts l1', figsize=(18,9.5))
# gs = GridSpec(12,8)

# for kk in range(47):
#     jj = kk%8
#     ii = kk//8

#     count_1hl = counts_dict_1hl[kk][0]
#     tot_count_1hl += count_1hl

#     ax = pl.subplot(gs[2*ii,jj])
#     ax.set_title(str(kk))
#     ax.bar(np.arange(len(count_1hl)), count_1hl, width=1., color=colours[0])

#     ax.set_xticks([])
#     ax.set_yticks([])


#     if len(algo2) > 0:
#         count_2hl = counts_dict_2hl[kk][0]
#         tot_count_2hl += count_2hl

#         ax = pl.subplot(gs[2*ii+1,jj])
#         ax.bar(np.arange(len(count_2hl)), count_2hl, width=1., color=colours[1])

#         ax.set_xticks([])
#         ax.set_yticks([])

#         # compute correlation between same and between random combs
#         corr_same += np.corrcoef(count_1hl, count_2hl)[0,1]
#         ll = np.random.choice(list(set(range(47)) - {kk}), 1)[0]
#         corr_diff += np.corrcoef(count_1hl, counts_dict_2hl[ll][0])[0,1]


# print('>>> correlation analysis')
# print('    C(same task) =', corr_same/47)
# print('    C(diff task) =', corr_diff/47)

# ax = pl.subplot(gs[10,7])
# ax.set_title('total')
# ax.bar(np.arange(len(tot_count_1hl)), tot_count_1hl, width=1., color=colours[2])

# ax.set_xticks([])
# ax.set_yticks([])

# if len(algo2) > 0:
#     ax = pl.subplot(gs[11,7])
#     ax.bar(np.arange(len(tot_count_2hl)), tot_count_2hl, width=1., color=colours[3])

#     ax.set_xticks([])
#     ax.set_yticks([])


# # plot for second layer
# if len(algo2) > 0:
#     pl.figure('counts l2', figsize=(18,9.5))
#     gs = GridSpec(6,8)

#     tot_count_2hl = torch.zeros(n_h2)
#     for kk in range(47):
#         jj = kk%8
#         ii = kk//8

#         count_2hl = counts_dict_2hl[kk][1]
#         tot_count_2hl += count_2hl

#         ax = pl.subplot(gs[ii,jj])
#         ax.set_title(str(kk))
#         ax.bar(np.arange(len(count_2hl)), count_2hl, width=1., color=colours[1])

#         ax.set_xticks([])
#         ax.set_yticks([])


#     ax = pl.subplot(gs[5,7])
#     ax.set_title('total')
#     ax.bar(np.arange(len(tot_count_2hl)), tot_count_2hl, width=1., color=colours[3])

#     ax.set_xticks([])
#     ax.set_yticks([])

# if args.save:
#     fname_1hl = "count_1hl_%s_%s%d_%s_ro=%s.p"%(args.dataset, algo1, n_h1, TASKTYPE, READOUT)
#     with open(os.path.join(args.respath, fname_1hl), 'wb') as file:
#         pickle.dump(counts_dict_1hl, file)

#     fname_2hl = "count_2hl_%s_algo12c=%s_nh12=%d-%d_%s_ro=%s.p"%(args.dataset, algo12c, n_h1, n_h2, TASKTYPE, READOUT)
#     with open(os.path.join(args.respath, fname_2hl), 'wb') as file:
#         pickle.dump(counts_dict_2hl, file)

# pl.tight_layout()
# pl.show()

