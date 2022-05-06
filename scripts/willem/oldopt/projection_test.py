import numpy as np
import torch

import os
import copy
import pickle
from subprocess import call, run

from datarep import paths
from datarep.matplotlibsettings import *

from sklearn.neighbors import NearestNeighbors

import optim, hiddenmatrix
from biasadaptation.utils import samplers, losses, utils


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

def get_input_weight(n_h, n_i):
    w_mat = np.random.randn(n_h, n_i)
    w_mat /= np.linalg.norm(w_mat, axis=1)[:,None]

    return w_mat

def get_output_weight(n_h):
    w_vec = np.random.randn(1,n_h)
    w_vec /= np.linalg.norm(w_vec)

    return w_vec


def sample_binary_task(dataset='EMNIST'):
    classes = np.random.choice(np.arange(samplers.N_CLASSES[dataset]), 2, replace=False)
    # return [(cc,) for cc in classes]
    return [(classes[0], classes[1])]


def run_optimizations(n_h, n_task=15, algo='rp',
                      n_per_batch=100, n_per_epoch=20, n_epoch=50):
    # construct the data sampler
    sampler_pair = samplers.SamplerPair_('EMNIST', nb_factor_test=10,
                        n_per_batch=n_per_batch, n_per_epoch=n_per_epoch)
    n_inp = sampler_pair.get_input_dim()

    storage = []
    perfs = []
    for nt in range(n_task):
        task = sample_binary_task()
        print('\nClassifying classes ', task)
        sampler_pair.set_tasks(task)#, target_type='perceptron')

        # construct the initial weights and biases
        ws = [get_input_weight(n_h, n_inp), get_output_weight(n_h)]
        bs = [np.random.randn(n_h, 1) / (10.*n_h), np.random.randn(1, 1) / 10.]

        w_final, b_final, perf = optim.run_optim(ws, bs, sampler_pair, n_epoch=n_epoch, w_idx=[0,1], b_idx=[0,1])

        storage.append({'w_final': w_final, 'b_final': b_final, 'perf': perf, 'task': task})

        perfs.append(np.array(perf))

    print('\n>>> average performance per epoch:')
    print(np.mean(perfs, 0))

    with open(paths.data_path + 'weight_bias_storage_%s_nh=%d.p'%(algo, n_h), 'wb') as f:
        pickle.dump(storage, f)


def load_optimization_data(n_h, algo):
    f_name = paths.data_path + 'weight_bias_storage_%s_nh=%d.p'%(algo, n_h)

    with open(f_name, 'rb') as file:
        storage = pickle.load(file)

    return storage

def compute_average_diff(task, dataset='EMNIST'):
    # construct the data sampler
    sampler = samplers.NTaskSampler(dataset, n_per_batch=2000, n_per_epoch=1)
    # task = sample_binary_task()
    sampler.set_tasks(task, target_type='perceptron')

    xdata, xtask, xtarget = next(iter(sampler))
    xdata   = xdata.numpy()
    xtarget = xtarget.numpy()

    idx0 = np.where(xtarget == -1)[0]
    idx1 = np.where(xtarget == 1)[0]

    idx0_ = np.random.choice(idx0, size=2000)
    idx1_ = np.random.choice(idx1, size=2000)

    xdiff_avg = np.mean(xdata[idx0_] - xdata[idx1_], axis=0)

    return xdiff_avg


def get_subspaces(n_h, dataset='EMNIST'):
    storage = load_optimization_data(n_h, 'sm')
    # storage = load_optimization_data(n_h, 'rp')

    w_algo = get_weight_matrix(n_h, 'sm', dataset=dataset)
    # construct the data sampler
    sampler = samplers.NTaskSampler(dataset, n_per_batch=50, n_per_epoch=1)
    # sampler = samplers.NBinarySampler(dataset, n_per_batch=50, n_per_epoch=1)
    n_inp = sampler.get_input_dim()


    for resdict in storage:
        # ws = resdict['w_final']
        bs = resdict['b_final']
        task = resdict['task']

        print('\nChecking classes ', task, ', perf =', np.mean(resdict['perf'][-20:]))
        # sampler.set_tasks(task)#, target_type='perceptron')
        sampler.set_tasks(task, target_type='index')

        xdata, xtask, xtarget = next(iter(sampler))
        xtarget = (xtarget + 1.)/2.

        # a_hid = np.dot(ws[0], xdata.T) + bs[0]
        a_hid = np.dot(w_algo, xdata.T) + bs[0]

        pl.figure('task = ' +str(task)+ ', perf =' +str(np.mean(resdict['perf'][-20:])), figsize=(8,8))
        gs1 = GridSpec(10,10)
        gs1.update(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.05, wspace=0.05)

        for kk, ah, xt, xd in zip(range(sampler.n_per_batch), a_hid.T, xtarget, xdata):
            idx = np.where(ah > 0)[0]

            # wvec = np.dot(ws[1][:,idx], ws[0][idx,:])
            wvec = np.dot(np.ones(idx.shape)/np.sqrt(n_h), w_algo[idx,:])


            ii = kk//5
            jj = kk%5

            ax1 = pl.subplot(gs1[2*jj,ii])
            ax1.set_title(str(int(xt)))
            ax1.imshow(utils.to_image_mnist(xd))

            ax2 = pl.subplot(gs1[2*jj+1,ii])
            # ax2.set_title(str(int(xt)))
            ax2.imshow(utils.to_image_mnist(wvec))

            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])

        pl.show()


def get_coords(vecs, basis):
    """
    get the coordinates of `vecs` in the given basis
    """
    bb = np.dot(basis, basis.T)
    vv = np.dot(basis, vecs)

    coords = np.linalg.solve(bb, vv)

    return coords


def project_diffs(n_h, dataset='EMNIST'):
    storage = load_optimization_data(n_h, 'rp')

    w_algo = get_weight_matrix(n_h, 'sm', dataset=dataset)

    sampler_pair = samplers.SamplerPair('EMNIST', nb_factor_test=10,
                        n_per_batch=100, n_per_epoch=20)

    for resdict in storage:
        task = resdict['task']
        task_ = [(task[0][0],), (task[0][0],)]
        w_final = resdict['w_final']


        pl.figure('orig', figsize=(10,5))
        gs1 = GridSpec(5,5)
        gs1.update(top=0.95, bottom=0.05, left=0.05, right=0.475, hspace=0.05, wspace=0.05)
        gs2 = GridSpec(5,5)
        gs2.update(top=0.95, bottom=0.05, left=0.525, right=0.95, hspace=0.05, wspace=0.05)

        for kk in range(n_h):
            ii = kk//5
            jj = kk%5

            ax1 = pl.subplot(gs1[ii,jj])
            ax1.imshow(utils.to_image_mnist(w_final[0][kk,:]))
            # ax1.imshow(utils.to_image_mnist(w_algo[kk,:]))

            ax1.set_xticks([])
            ax1.set_yticks([])

        xdiff_avg = compute_average_diff([(task[0][0],), (task[0][1],)])

        ax2 = pl.subplot(gs2[:,:])
        ax2.imshow(utils.to_image_mnist(xdiff_avg))

        ax2.set_xticks([])
        ax2.set_yticks([])

        coords = get_coords(w_final[0].T, w_algo)
        w_reconstr = np.dot(coords.T, w_algo)

        pl.figure('reconstr', figsize=(10,5))
        gs1 = GridSpec(5,5)
        gs1.update(top=0.95, bottom=0.05, left=0.05, right=0.475, hspace=0.05, wspace=0.05)
        gs2 = GridSpec(5,5)
        gs2.update(top=0.95, bottom=0.05, left=0.525, right=0.95, hspace=0.05, wspace=0.05)

        for kk in range(n_h):
            ii = kk//5
            jj = kk%5

            ax1 = pl.subplot(gs1[ii,jj])
            ax1.imshow(utils.to_image_mnist(w_reconstr[kk,:]))
            # ax1.imshow(utils.to_image_mnist(w_algo[kk,:]))

            ax1.set_xticks([])
            ax1.set_yticks([])

        coords = get_coords(xdiff_avg.T, w_algo)
        print(np.array([coords]).shape)
        xdiff_reconstr = np.dot(coords.T, w_algo)

        ax2 = pl.subplot(gs2[:,:])
        ax2.imshow(utils.to_image_mnist(xdiff_reconstr))

        ax2.set_xticks([])
        ax2.set_yticks([])

        ws = [w_algo, np.array([coords])]
        bs = [2.*np.ones((n_h, 1)), np.array([[-5.]])]
        # bs = [np.random.rand(n_h, 1) , np.random.rand(1, 1)]

        sampler_pair.set_tasks(task_)
        w_final, b_final, perf = optim.run_optim(ws, bs, sampler_pair, n_epoch=20, w_idx=[], b_idx=[1])

        print(b_final)


        pl.show()


def project_diffs_2(n_h, dataset='EMNIST'):
    storage = load_optimization_data(n_h, 'sm')
    # storage = load_optimization_data(n_h, 'rp')

    w_algo = get_weight_matrix(n_h, 'sm', dataset=dataset)
    # construct the data sampler
    sampler = samplers.NTaskSampler(dataset, n_per_batch=50, n_per_epoch=1)
    # sampler = samplers.NBinarySampler(dataset, n_per_batch=50, n_per_epoch=1)
    n_inp = sampler.get_input_dim()


    for resdict in storage:
        # ws = resdict['w_final']
        bs = resdict['b_final']
        task = resdict['task']

        print('\nChecking classes ', task, ', perf =', np.mean(resdict['perf'][-20:]))
        # sampler.set_tasks(task)#, target_type='perceptron')
        sampler.set_tasks(task, target_type='index')

        xdata, xtask, xtarget = next(iter(sampler))
        xtarget = (xtarget + 1.)/2.

        # a_hid = np.dot(ws[0], xdata.T) + bs[0]
        a_hid = np.dot(w_algo, xdata.T) + bs[0]

        pl.figure('task = ' +str(task)+ ', perf =' +str(np.mean(resdict['perf'][-20:])), figsize=(8,8))
        gs1 = GridSpec(10,10)
        gs1.update(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.05, wspace=0.05)

        xdiff = utils.differences_numpy(xdata, sampler.n_per_batch)

        for kk, xd in enumerate(xdiff):

            coords = get_coords(xd, w_algo)
            xd_reconstr = np.dot(coords.T, w_algo)

            ii = kk//5
            jj = kk%5

            ax1 = pl.subplot(gs1[2*jj,ii])
            ax1.imshow(utils.to_image_mnist(xd))

            ax2 = pl.subplot(gs1[2*jj+1,ii])
            ax2.imshow(utils.to_image_mnist(xd_reconstr))

            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])

        pl.show()

        # for ah, xt in zip(a_hid.T, xtarget):
        #     print(xt, ' ---> ', idx)




if __name__ == "__main__":
    # run_optimizations(25)
    # compute_average_diff([(20,),(12,)])
    # get_subspaces(25)
    project_diffs(25)
    # project_diffs_2(25)

