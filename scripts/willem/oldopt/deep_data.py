
import numpy as np
import torch
import torchvision.datasets as tdatasets
import torchvision.transforms as ttransforms
import torch.utils.data as tdata
from tqdm import tqdm

import os
import copy
import pickle
from subprocess import call, run

from datarep import paths
from datarep.matplotlibsettings import *

from sklearn.neighbors import NearestNeighbors

import optim, hiddenmatrix
from biasadaptation.utils import samplers, losses, utils
from biasadaptation.weightmatrices import scd


def sample_binary_task(dataset='EMNIST'):
    classes = np.random.choice(np.arange(samplers.N_CLASSES[dataset]), 2, replace=False)
    return [(cc,) for cc in classes]


def load_input_weight_matrix(n_h, algo, dataset='EMNIST'):
    """
    Parameters
    ----------
    n_h: int
        number of hidden units
    algo: str ('pca', 'ica', 'rp', 'rg', 'sc', 'scd', 'sm')
        name of the algorithm
    """
    path_name = os.path.join(paths.data_path, 'weight_matrices/', '%s_%s%d.npy'%(dataset, algo, n_h))
    w_mat = np.load(path_name)

    return w_mat


def load_hidden_weight_matrix(n_h2, n_h1, algo, dataset='EMNIST'):
    f_name = os.path.join(paths.data_path,
                               'weight_matrices/',
                               '%s_%s_nh2=%d_nh1=%d.npy'%(dataset, algo, n_h2, n_h1))
    w_mat = np.load(f_name)

    return w_mat


def get_output_weight(n_h):
    w_mat = np.random.randn(1,n_h)
    w_mat /= np.linalg.norm(w_mat)

    return w_mat


def get_coordinate_matrix(n_h, algo, dataset='EMNIST'):
    path_name = os.path.join(paths.data_path, 'weight_matrices/', '%s_%s%d.npy'%(dataset, algo, n_h))
    w_mat = np.load(path_name)

    transforms = ttransforms.Compose([ttransforms.ToTensor(), lambda x: x/samplers.DATA_NORMS[dataset]])
    data_set = tdatasets.EMNIST(paths.tool_path, train=True, download=True, split="bymerge", transform=transforms)
    data_loader = tdata.DataLoader(data_set, batch_size=100000)

    coords = []

    for (d, t) in tqdm(data_loader):
        s = d.shape
        x = d.numpy().reshape(s[0], s[-1]*s[-2])

        b = np.dot(w_mat, x.T)
        a = np.dot(w_mat, w_mat.T)
        print(a.shape, b.shape)
        coo = np.linalg.solve(a, b)

        coords.append(coo.T)

    return np.vstack(coords)


def calc_weight_matrix(n_h2, n_h1, dataset='EMNIST', store_mat=True):
    data_matrix = get_coordinate_matrix(n_h1, algo='sm')
    w_mat = scd.get_sc_trafo_matrix(data_matrix, n_h2)

    if store_mat:
        f_name = os.path.join(paths.data_path,
                               'weight_matrices/',
                               '%s_scd_nh2=%d_nh1=%d.npy'%(dataset, n_h2, n_h1))
        np.save(f_name, w_mat)

    return w_mat


def plot_receptive_fields(n_h2, n_h1, algo='sm', dataset='EMNIST'):
    wm0 = load_input_weight_matrix(n_h1, algo=algo, dataset=dataset)
    wm1 = load_hidden_weight_matrix(n_h2, n_h1, algo='scd', dataset=dataset)

    pl.figure('receptive_fields', figsize=(8,8))
    gs = GridSpec(5,5)
    gs.update(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.05, wspace=0.05)

    w_fields  = np.dot(wm1, wm0)

    for kk, w_vec in enumerate(w_fields):
        ii, jj = kk//5, kk%5

        ax = pl.subplot(gs[ii,jj])
        ax.imshow(utils.to_image_mnist(w_vec))
        ax.set_xticks([])
        ax.set_yticks([])

    pl.show()


def load_optimization_data(n_h, algo):
    f_name = paths.data_path + 'bias_storage_%s_nh=%d.p'%(algo, n_h)

    with open(f_name, 'rb') as file:
        bias_storage = pickle.load(file)

    return bias_storage


def run_optimizations(n_h2, n_h1, n_task=15, algo='sm',
                      n_per_batch=100, n_per_epoch=20, n_epoch=200):
    storage_1l = load_optimization_data(n_h1, algo)

    # construct the data sampler
    sampler_pair = samplers.SamplerPair('EMNIST', nb_factor_test=10,
                        n_per_batch=n_per_batch, n_per_epoch=n_per_epoch)
    n_inp = sampler_pair.get_input_dim()

    storage_2l = []
    perfs_2l = []
    perfs_1l = []
    for nt, resdict in enumerate(storage_1l):
        task = resdict['task']
        bs_ = resdict['b_final']
        perfs_1l.append(resdict['perf'])

        print('\nTask', nt, ', Classifying classes ', task)
        sampler_pair.set_tasks(task, target_type='perceptron')

        # construct the initial weights and biases
        ws = [load_input_weight_matrix(n_h1, algo),
              load_hidden_weight_matrix(n_h2, n_h1, 'scd'),
              get_output_weight(n_h2)]
        bs = [np.random.randn(n_h1, 1) / (10.*n_h1),
              np.random.randn(n_h2, 1) / (10.*n_h2),
              np.random.randn(1, 1) / 10.]
        # bs = [np.random.rand(n_h1, 1) / (10.*n_h1),
        #       np.random.rand(n_h2, 1) / (10.*n_h2),
        #       np.random.rand(1, 1) / 10.]
        # bs = [bs_[0],
        #       np.random.randn(n_h2, 1) / (10.*n_h2),
        #       np.random.randn(1, 1) / 10.]

        lr = optim.LRLin(n_epoch, lr0=0.01, lr1=0.0005)
        w_final, b_final, perf = optim.run_optim(ws, bs, sampler_pair, n_epoch=n_epoch, w_idx=[], b_idx=[0,1,2], lr=lr)

        print('\n-> Final perf 1 hidden layer: %.2f'%(perfs_1l[-1][-1]))

        storage_2l.append({'w_final': w_final, 'b_final': b_final, 'perf': perf, 'task': task})

        perfs_2l.append(np.array(perf))

    print('\n>>> average performance per epoch -- 1 hidden layer:')
    print(np.mean(perfs_1l, 0)[:-10])

    print('\n>>> average performance per epoch -- 2 hidden layers:')
    print(np.mean(perfs_2l, 0)[:-10])

    with open(paths.data_path + 'deep_fit_storage_%s_nh2=%d_nh1=%d.p'%(algo, n_h2, n_h1), 'wb') as f:
        pickle.dump(storage_2l, f)



if __name__ == "__main__":
    # calc_coordinate_matrix(25, 'sm')
    # calc_weight_matrix(25, 25)
    # plot_receptive_fields(25, 25)
    run_optimizations(25, 25)
