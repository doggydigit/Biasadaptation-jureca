import numpy as np
import torch

import argparse
import pickle
import copy
import sys

sys.path.append('..')
import optim, helperfuncs

from datarep.matplotlibsettings import *
from datarep import paths

from biasadaptation.biasfit import specificbiasfit
from biasadaptation.weightmatrices import bmd
from biasadaptation.utils import utils
from biasadaptation.utils import k_task_n_class_m_dataset_data as knm

"""
axample usage

python3 analysis.py --nhidden1 25 --nhidden2 25 --algo1 scd --algo2 sc --algoc sc
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

w_in  = helperfuncs.get_weight_matrix_in(n_h1, algo_1, dataset=args.dataset)
w_hid = helperfuncs.get_weight_matrix_hidden(n_h1, n_h2, algo_1, algo_2, algo_c, dataset=args.dataset, task={-1: {'EMNIST': {args.taskid : 0}}})


def plot_matrices():

    pl.figure(figsize=(14,8))

    n_row = 5
    n_col = 5
    n_plot = 5*5

    gs0 = GridSpec(n_row, n_col)
    gs0.update(top=0.95, bottom=0.05, left=0.05, right=0.45, hspace=1.1, wspace=0.4)
    gs1 = GridSpec(1,1)
    gs1.update(top=0.95, bottom=0.05, left=0.55, right=0.95, hspace=1.1, wspace=0.4)

    idx_in = np.random.choice(np.arange(args.nhidden1), n_plot, replace=False)
    idx_hid = np.random.choice(np.arange(args.nhidden2), n_plot, replace=False)

    for ii in range(n_plot):
        jj = ii // n_row
        kk = ii % n_row

        ax = noFrameAx(pl.subplot(gs0[jj,kk]))
        ax.imshow(utils.to_image_mnist(w_in[:,idx_in[ii]]))

    ax = noFrameAx(pl.subplot(gs1[0,0]))
    # ax.imshow(w_hid[:,idx_hid])
    ax.imshow(w_hid)

    pl.show()


def plot_mats_1d():
    pl.figure()
    ax = pl.gca()

    for ii in range(n_h2):
        pl.plot(np.arange(n_h1), w_hid[:,ii], c='k', alpha=0.1)

    pl.show()


def plot_combined_rfs():

    pl.figure(figsize=(14,8))

    n_row = 5
    n_col = 5
    n_plot = 5*5

    gs0 = GridSpec(n_row, n_col)
    gs0.update(top=0.95, bottom=0.05, left=0.05, right=0.45, hspace=1.1, wspace=0.4)
    gs1 = GridSpec(n_row, n_col)
    gs1.update(top=0.95, bottom=0.05, left=0.55, right=0.95, hspace=1.1, wspace=0.4)

    idx_in = np.random.choice(np.arange(args.nhidden1), n_plot, replace=False)
    idx_hid = np.random.choice(np.arange(args.nhidden2), n_plot, replace=False)

    for ii in range(n_plot):
        jj = ii // n_row
        kk = ii % n_row

        ax = noFrameAx(pl.subplot(gs0[jj,kk]))
        ax.imshow(utils.to_image_mnist(w_in[:,idx_in[ii]]))

    w_rf = np.dot(w_in, w_hid)

    for ii in range(n_plot):
        jj = ii // n_row
        kk = ii % n_row

        ax = noFrameAx(pl.subplot(gs1[jj,kk]))
        ax.imshow(utils.to_image_mnist(w_rf[:,idx_hid[ii]]))

    # ax.imshow(w_hid[:,idx_hid])
    # ax.imshow(w_hid)

    pl.show()



def compare_mats_1d():
    ws_hid_bp = [helperfuncs.get_weight_matrix_hidden(n_h1, n_h2, algo_1, 'bp', 'na', dataset=args.dataset, task={-1: {'EMNIST': {t_id : 0}}})
                 for t_id in range(47)]


    # load data
    ds = helperfuncs.get_dataset(args.dataset)
    dl = torch.utils.data.DataLoader(ds, batch_size=1000, shuffle=True)
    xdata, _ = next(iter(dl))
    xdata = xdata.reshape(xdata.shape[0], np.prod(list(xdata.shape[1:])))


    with open(paths.result_path + "biasopt/biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(args.dataset, algo_1, n_h1, "1vall", "tanh"), 'rb') as file:
        reslist = pickle.load(file)

    ahs = []
    abins = []
    for res in reslist:
        task = res['task']
        w_ = torch.FloatTensor(res['ws'][0])
        b_ = torch.FloatTensor(res['bs'][0])

        a1 = torch.nn.functional.relu(torch.mm(xdata, w_) + b_)
        a1_ = a1.detach().numpy()
        ab_ = (a1_ > 1e-9).astype(float)

        a1m = np.mean(a1_, axis=0)
        ahs.append(a1m)
        abm = np.mean(ab_, axis=0)
        abins.append(abm)


    pl.figure()
    ax = pl.gca()

    for ii in range(47):
        w_hid_ = ws_hid_bp[ii]
        ax.plot(np.arange(n_h1), np.mean(w_hid_, axis=1), c='k', alpha=.2)
        # ax.plot(np.arange(n_h1), w_hid_[:,np.random.randint(47)], c='k', alpha=.2)

        # ax.plot(np.arange(n_h1), ahs[ii], c='r', alpha=.2)

    ax.plot(np.arange(n_h1), np.mean(ahs, axis=0), c='r')
    ax.plot(np.arange(n_h1), np.mean(abins, axis=0), c='b')
    # ax.plot(np.arange(n_h1), np.mean(w_hid, axis=1)*10, c='r')

    pl.show()


def matching_pursuit(dataset='EMNIST'):
    # n_hs = [25, 25, 25, 25]
    n_hs = [100, 25]
    # n_hs = [50, 25]

    xdata = xdata.reshape(xdata.shape[0], np.prod(list(xdata.shape[1:]))).numpy()
    DX = utils.differences_torch(xdata.T)

    NN = 10000

    namestring = '_'.join([str(nh) for nh in n_hs])
    with open(paths.data_path + 'weight_matrices/%s_weight_mats_pmd_nh=%s.p'%(args.dataset, namestring), 'rb') as f:
        Ws = pickle.load(f)
        Cs = pickle.load(f)
        DX = pickle.load(f)

    nh = n_hs[-1]
    Bs = [((np.abs(C) > 1e-6)).astype(float) for C in Cs]

    print('\n-------\nBegin Testing BMD')
    # U, D, V = sla.svd(Cs[0][:100,:], full_matrices=False)
    # U = U[:,:nh]
    # D = D[:nh]
    # V = V[:nh,:]
    # U = U @ np.diag(D)
    print('i')
    C_ = np.linalg.lstsq(Ws[0].T, DX.T, rcond=None)[0].T
    print('ii')

    VV = np.random.randn(n_hs[1],n_hs[0])
    # VV = np.eye(nh)
    O0 = bmd.minimize_o_cost(DX[:NN], Ws[0])

    O, W = bmd.find_weight_and_mask(np.ones((NN, nh)), VV, C_[:NN,:], nh, n_iter=100)
    # O, W = bmd.find_weight_and_mask(np.ones((NN, nh)), VV, Cs[0][:NN,:], 25, n_iter=100)
    # O, W = bmd.find_weight_and_mask(np.ones((NN, nh)), VV, O0, 25, n_iter=100)

    # namestring = '_'.join(['100', '50'])
    with  open(paths.data_path + 'weight_matrices/%s_weight_mats_pbmd_nh=%s.p'%(dataset, namestring), 'wb') as f:
        Ws_ = [Ws[0], W]
        pickle.dump(Ws_, f)

    V = np.dot(np.dot(O, W) * Bs[0][:NN,:], Ws[0])


    V_ = ((np.ones((NN, nh)) @ VV) * O0) @ Ws[0]
    # V_ = ((np.ones((NN, nh)) @ VV) * Bs[0][:NN,:]) @ Ws[0]
    print('End Testing BMD\n-------\n')

    # DX_1L_ = np.dot(Cs[0], Ws[0])
    DX_1L_ = np.dot(C_, Ws[0])
    DX_2L_ = np.dot(np.dot(Cs[1], Ws[1]), Ws[0])
    DX_3L_ = np.dot(np.dot(np.dot(Cs[2], Ws[2]), Ws[1]), Ws[0])
    DX_4L_ = np.dot(np.dot(np.dot(np.dot(Cs[3], Ws[3]), Ws[2]), Ws[1]), Ws[0])

    Bs = [B / np.linalg.norm(B, axis=1)[:, None] * np.linalg.norm(C, axis=1)[:, None] for B, C in zip(Bs, Cs)]

    DX_1L = np.dot(Bs[0], Ws[0])

    # inds = np.random.choice(np.arange(DX.shape[0]), size=5, replace=False)
    # inds = [1,3,5,7,9]
    inds = [0,2,4,6,8]

    pl.figure('DX', figsize=(10,5))
    gs0 = GridSpec(5,5)
    gs0.update(top=0.95, bottom=0.05, left=0.05, right=0.475, hspace=0.05, wspace=0.05)
    gs1 = GridSpec(5,5)
    gs1.update(top=0.95, bottom=0.05, left=0.525, right=0.95, hspace=0.05, wspace=0.05)

    for kk, ind in enumerate(inds):
        print("\n--> Image %d <--"%ind)

        # original
        axa = pl.subplot(gs0[kk,0])
        axb = pl.subplot(gs1[kk,0])

        pl_im(axa, DX[ind])
        pl_im(axb, DX[ind])

        # 1st layer
        print("-- layer 1 --")
        axa = pl.subplot(gs0[kk,1])
        axb = pl.subplot(gs1[kk,1])

        v1, idx, score = bmd.binary_match(DX[ind], w_s, verbose=False)
        vn = v1

        pl_im(axa, vn)
        pl_im(axb, DX_1L_[ind])

        print('components         : %s'%(', '.join([str(ii) for ii in np.sort(idx)])))
        print('coordinates (c > 0): %s'%(', '.join([str(ii) for ii in np.where(Cs[0][ind] > 0.)[0]])))

        print('diff components : %.6f'%normnorm(vn, DX[ind]))
        print('diff coordinates: %.6f'%normnorm(DX_1L_[ind], DX[ind]))

        # 2nd layer
        print("\n-- layer 2 --")
        axa = pl.subplot(gs0[kk,2])
        axb = pl.subplot(gs1[kk,2])

        v1, idx, score = bmd.binary_match(Cs[0][ind], Ws[1], verbose=False)

        idx0 = np.where(Cs[0][ind] > 0.)[0]
        vn = np.dot(v1[idx0], Ws[0][idx0,:])

        pl_im(axa, V[ind])
        pl_im(axb, DX_2L_[ind])

        print('components         : %s'%(', '.join([str(ii) for ii in np.sort(idx)])))
        print('coordinates (c > 0): %s'%(', '.join([str(ii) for ii in np.where(Cs[1][ind] > 0.)[0]])))

        print('diff components : %.6f'%normnorm(V[ind], DX[ind]))
        print('diff components2: %.6f'%normnorm(vn, DX[ind]))
        # print('diff components2: %.6f'%normnorm(V_[ind], DX[ind]))
        print('diff coordinates: %.6f'%normnorm(DX_2L_[ind], DX[ind]))

        print("----------------")



if __name__ == "__main__":
    # plot_matrices()
    # plot_combined_rfs()
    # plot_mats_1d()
    compare_mats_1d()

