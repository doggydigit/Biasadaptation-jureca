import numpy as np
import torch

import os
import copy
import pickle
from subprocess import call, run

from datarep import paths
from datarep.matplotlibsettings import *

from sklearn.neighbors import NearestNeighbors

# import optim, hiddenmatrix
from biasadaptation.utils import samplers, losses, utils



def sample_binary_task(dataset='EMNIST'):
    classes = np.random.choice(np.arange(samplers.N_CLASSES[dataset]), 2, replace=False)
    return [(cc,) for cc in classes]
    # return [(classes[0], classes[1])]


def get_weight_matrix(n_h, algo, dataset='EMNIST'):
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

def get_hidden_weight(n_h2, n_h1):
    w_mat = np.random.randn(n_h2, n_h1)
    w_mat /= np.linalg.norm(w_mat, axis=1)[:,None]

    return w_mat

def get_output_weight(n_h):
    w_vec = np.random.randn(1,n_h)
    w_vec /= np.linalg.norm(w_vec)

    return w_vec


def run_optimizations(n_h2, n_h1, n_task=15, algo='rp',
                      n_per_batch=100, n_per_epoch=20, n_epoch=50):
    # construct the data sampler
    sampler_pair = samplers.SamplerPair('EMNIST', nb_factor_test=10,
                        n_per_batch=n_per_batch, n_per_epoch=n_per_epoch)
    n_inp = sampler_pair.get_input_dim()

    storage = []
    perfs = []
    for nt in range(n_task):
        task = sample_binary_task()
        print('\nTask', nt, ', Classifying classes ', task)
        sampler_pair.set_tasks(task, target_type='perceptron')

        # construct the initial weights and biases
        ws = [get_weight_matrix(n_h1, algo), get_hidden_weight(n_h2, n_h1), get_output_weight(n_h2)]
        bs = [np.random.randn(n_h1, 1) / (10.*n_h1), np.random.randn(n_h2, 1) / (10.*n_h2), np.random.randn(1, 1) / 10.]
        # ws = [get_weight_matrix(n_h1, algo), get_output_weight(n_h2)]
        # bs = [np.random.randn(n_h1, 1) / (10.*n_h1), np.random.randn(1, 1) / 10.]

        w_final, b_final, perf = optim.run_optim(ws, bs, sampler_pair, n_epoch=n_epoch, w_idx=[1,2], b_idx=[0,1,2], lr=0.001)

        storage.append({'w_final': w_final, 'b_final': b_final, 'perf': perf, 'task': task})

        perfs.append(np.array(perf))

    print('\n>>> average performance per epoch:')
    print(np.mean(perfs, 0))

    with open(paths.data_path + 'expansion_fit_storage_%s_nh=%d.p'%(algo, n_h1), 'wb') as f:
        pickle.dump(storage, f)


def load_optimization_data(n_h, algo, reduced=False):
    f_name = paths.data_path + 'expansion_fit_storage_%s_nh=%d.p'%(algo, n_h)
    if reduced:
        f_name = paths.data_path + 'expansion_fit_reduced_storage_%s_nh=%d.p'%(algo, n_h)

    with open(f_name, 'rb') as file:
        storage = pickle.load(file)

    return storage


def run_optimizations_shrink(n_h2, n_h1, n_task=15, algo='rp',
                             n_per_batch=100, n_per_epoch=20, n_epoch=5):
    storage = load_optimization_data(n_h1, algo)

    # construct the data sampler
    sampler_pair = samplers.SamplerPair('EMNIST', nb_factor_test=10,
                        n_per_batch=n_per_batch, n_per_epoch=n_per_epoch)
    n_inp = sampler_pair.get_input_dim()

    n_unit = np.arange(n_h1, n_h2, 25)
    n_step = int(n_h2 / 25) - 1

    new_storage = []
    perfs = []
    for nt, resdict in enumerate(storage):
        task = resdict['task']
        ws = resdict['w_final']
        bs = resdict['b_final']
        perf_ = []

        print('\nTask', nt, ', Classifying classes ', task)
        sampler_pair.set_tasks(task, target_type='perceptron')


        for ss in range(n_step):
            # construct the initial weights and biases
            idx = np.argsort(bs[1][:,0])[25:]
            ws[1] = ws[1][idx,:]
            bs[1] = bs[1][idx,:]
            ws[2] = ws[2][:,idx]

            print('\n> nHidden = ', len(idx))

            ws, bs, perf = optim.run_optim(ws, bs, sampler_pair, n_epoch=n_epoch, w_idx=[1,2], b_idx=[0,1,2], lr=0.001)

            perf_.extend(perf_)

        new_storage.append({'w_final': ws, 'b_final': bs, 'perf': perf, 'task': task})

        perfs.append(np.array(perf))

    print('\n>>> average performance per epoch:')
    print(np.mean(perfs, 0))

    with open(paths.data_path + 'expansion_fit_reduced_storage_%s_nh=%d.p'%(algo, n_h1), 'wb') as f:
        pickle.dump(new_storage, f)



def check_combinations(n_h2, n_h1, n_sample=1000):
    w_hid = get_hidden_weight(n_h2, n_h1)
    w_out = get_output_weight(n_h2)

    pl.figure(figsize=(12,12))
    gs = GridSpec(5,5)
    gs.update(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.05, wspace=0.05)

    for cc in range(n_h1):

        rand_arr = np.random.randint(2, size=(n_sample, n_h2))
        x_arr = np.zeros(n_sample)

        for ii, r_arr in enumerate(rand_arr):
            idx = np.where(r_arr)[0]
            x_arr[ii] = np.dot(w_out[:,idx], w_hid[idx,cc])[0]

        ax = pl.subplot(gs[cc%5, cc//5])
        ax.hist(x_arr, bins=100)

    pl.show()


def relu(x):
    x_ = np.array([x,np.zeros_like(x)])
    return np.max(x_, axis=0)


def check_network_coeff(n_h1, algo='sm', n_sample=990, reduced=True):
    storage = load_optimization_data(n_h1, algo, reduced=reduced)

    sampler = samplers.NTaskSampler('EMNIST',
                        n_per_batch=n_sample, n_per_epoch=1)
    n_inp = sampler.get_input_dim()

    for resdict in storage:
        sampler.set_tasks(resdict['task'])
        xdata, xtask, xtarget = next(iter(sampler))

        ws = resdict['w_final']
        bs = resdict['b_final']

        a1 = relu(np.dot(ws[0], xdata.T) + bs[0])
        a2 = relu(np.dot(ws[1], a1) + bs[1])

        pl.figure(figsize=(12,12))
        gs = GridSpec(5,5)
        gs.update(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.05, wspace=0.05)

        vmin, vmax = 0., 0.
        axes = []

        for cc in range(n_h1):
            x_arr = np.zeros(n_sample)
            print('\nbasis coef:')
            for ii,a in enumerate(a2.T):
                idx = np.where(a > 0)[0]
                # print('>', idx)
                x_arr[ii] = np.dot(ws[2][0,idx], ws[1][idx, cc])


            ax = pl.subplot(gs[cc%5, cc//5])
            axes.append(ax)
            ax.hist(x_arr, bins=100)

            vmin = min(vmin, np.min(x_arr))
            vmax = max(vmax, np.max(x_arr))

        for ax in axes: ax.set_xlim((vmin, vmax))

        pl.figure(figsize=(24,12))
        gs = GridSpec(5,10)
        gs.update(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.05, wspace=0.05)

        for ii, (a1_, a2_) in enumerate(zip(a1.T, a2.T)):
            if ii < 50:
                kk = ii % 5
                ll = ii // 5

                idx1 = np.where(a1_ > 0.)[0]
                idx2 = np.where(a2_ > 0.)[0]

                w_vec = np.dot(np.dot(ws[2][:,idx2], ws[1][idx2,:][:,idx1]), ws[0][idx1,:])

                ax = pl.subplot(gs[kk,ll])
                ax.imshow(utils.to_image_mnist(w_vec))

                ax.set_xticks([])
                ax.set_yticks([])

        pl.show()


def plot_receptive_fields(n_h1, algo='sm', reduced=True):
    storage = load_optimization_data(n_h1, algo, reduced=reduced)

    sampler = samplers.NTaskSampler('EMNIST',
                        n_per_batch=25, n_per_epoch=1)
    n_inp = sampler.get_input_dim()

    for ll, resdict in enumerate(storage):
        ws = resdict['w_final']


        sampler.set_tasks(resdict['task'])
        xdata, xtask, xtarget = next(iter(sampler))

        pl.figure('Task %d = '%ll + str(resdict['task']) + ' | perf = %.2f'%resdict['perf'][-1], figsize=(16,8))
        gs_ = GridSpec(5,5)
        gs_.update(top=0.95, bottom=0.05, left=0.05, right=0.46, hspace=0.05, wspace=0.05)
        gs = GridSpec(5,5)
        gs.update(top=0.95, bottom=0.05, left=0.54, right=0.95, hspace=0.05, wspace=0.05)

        w_fields  = np.dot(ws[1], ws[0])

        for kk, w_vec in enumerate(w_fields):
            ii, jj = kk//5, kk%5

            ax = pl.subplot(gs[ii,jj])
            ax.imshow(utils.to_image_mnist(w_vec))
            ax.set_xticks([])
            ax.set_yticks([])


        for kk, (x_vec, x_target) in enumerate(zip(xdata, xtarget)):
            ii, jj = kk//5, kk%5

            ax = pl.subplot(gs_[ii,jj])
            ax.set_title('target = %d'%x_target)
            ax.imshow(utils.to_image_mnist(x_vec))
            ax.set_xticks([])
            ax.set_yticks([])



        pl.show()


def plot_hidden_weights(n_h1, algo='sm', reduced=True):
    storage = load_optimization_data(n_h1, algo, reduced=reduced)

    pl.figure('w1-2', figsize=(12,8))
    gs = GridSpec(3,5)
    gs.update(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.05, wspace=0.05)

    for kk, resdict in enumerate(storage):
        ws = resdict['w_final']
        # wmin, wmax = np.min(ws[1]), np.max(ws[1])
        print(ws[1].shape)

        ii, jj = kk//5, kk%5

        ax = pl.subplot(gs[ii,jj])
        ax.imshow(ws[1])

        ax.set_xticks([])
        ax.set_yticks([])

    pl.show()



def optimize_basis(n_h1, algo='sm', n_sample=990, reduced=True):
    storage = load_optimization_data(n_h1, algo, reduced=reduced)

    sampler = samplers.NTaskSampler('EMNIST',
                        n_per_batch=n_sample, n_per_epoch=1)
    n_inp = sampler.get_input_dim()

    for resdict in storage:
        sampler.set_tasks(resdict['task'])
        xdata, xtask, xtarget = next(iter(sampler))

        ws = resdict['w_final']
        bs = resdict['b_final']

        a1 = relu(np.dot(ws[0], xdata.T) + bs[0])
        a2 = relu(np.dot(ws[1], a1) + bs[1])

        for ii, (a1_, a2_) in enumerate(zip(a1.T, a2.T)):
            idx1 = np.where(a1_ > 0)[0]
            idx2 = np.where(a2_ > 0)[0]
            wvals = np.dot(ws[2][0,idx2], ws[1][idx2,:])
            print(wvals.shape)

            print(idx1)
            print(np.where(wvals > 0)[0])





if __name__ == "__main__":
    # run_optimizations(1000, 25, algo='rp', n_epoch=200)
    # run_optimizations_shrink(1000, 25, algo='rp', n_epoch=5)
    # check_combinations(25, 25)
    # check_network_coeff(25, algo='sm')
    # plot_hidden_weights(25, algo='sm')
    plot_receptive_fields(25, algo='scd')
    # optimize_basis(25)