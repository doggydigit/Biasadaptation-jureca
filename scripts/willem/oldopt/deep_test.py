import numpy as np
import torch

import os
import copy
import pickle

from datarep import paths
from datarep.matplotlibsettings import *

from sklearn.neighbors import NearestNeighbors

import optim, hiddenmatrix
from biasadaptation.utils import samplers, losses


def sample_binary_task(dataset='EMNIST'):
    classes = np.random.choice(np.arange(samplers.N_CLASSES[dataset]), 2, replace=False)
    return [(cc,) for cc in classes]


def load_input_weight(n_h, algo, dataset='EMNIST'):
    """
    Parameters
    ----------
    n_h: int
        number of hidden units
    algo: str ('pca', 'ica', 'rp', 'rg', 'sc', 'scd', 'sm')
        name of the algorithm
    """
    path_name = os.path.join(paths.data_path, 'weight_matrices/', '%s_%s%d.npy'%(dataset, algo, n_h))

    return np.load(path_name)


def load_hidden_weight(n_h1, n_h2, algo, dataset='EMNIST'):
    """
    Parameters
    ----------
    n_h: int
        number of hidden units
    algo: str ('pca', 'ica', 'rp', 'rg', 'sc', 'scd', 'sm')
        name of the algorithm
    """
    path_name = os.path.join(paths.data_path, 'weight_matrices/', 'hidden_neighb_%s_%s_nh1=%d_nh2=%d.npy'%(dataset, algo, n_h1, n_h2))

    return np.load(path_name)


def get_output_weight(n_h):
    w_vec = np.ones((1,n_h))
    w_vec /= np.linalg.norm(w_vec)

    return w_vec


def run_optimizations(n_h, n_task=15,
                     opt_g=False, algo='random',
                     n_per_batch=100, n_per_epoch=20, n_epoch=200,
                     old_flag=False):
    # construct the data sampler
    if old_flag:
        sampler_pair = samplers.SamplerPair_('EMNIST', nb_factor_test=10,
                            n_per_batch=n_per_batch, n_per_epoch=n_per_epoch)
        so = '_old'
    else:
        sampler_pair = samplers.SamplerPair('EMNIST', nb_factor_test=10,
                            n_per_batch=n_per_batch, n_per_epoch=n_per_epoch)
        so = ''
    n_inp = sampler_pair.get_input_dim()

    bias_storage = []
    perfs = []
    for nt in range(n_task):
        task = sample_binary_task()
        print('optimizing task ', task)
        if old_flag:
            tasks = [(task[0], task[1])]
            sampler_pair.set_nbinary_tasks(tasks)
        else:
            sampler_pair.set_tasks(task, target_type='perceptron')

        # construct the initial weights and biases
        ws = [load_input_weight(n_h, algo), get_output_weight(n_h)]
        bs = [np.random.randn(n_h, 1) / (10.*n_h), np.random.randn(1, 1) / 10.]

        w_final, b_final, perf = optim.run_optim(ws, bs, sampler_pair, opt_w=False, n_epoch=n_epoch, opt_g=False)
        # w_final, b_final, perf = optim.run_optim(ws, bs, sampler_pair, opt_w=False, n_epoch=n_epoch, opt_g=True)

        bias_storage.append({'b_final': b_final, 'perf': perf, 'task': task})

        perfs.append(np.array(perf))

    ss = ' old sampler' if old_flag else ' new sampler'
    print('\n>>> average performance per epoch %s:'%ss)
    print(np.mean(perfs, 0))

    with open(paths.data_path + 'bias_storage_%s_nh=%d%s.p'%(algo, n_h, so), 'wb') as f:
        pickle.dump(bias_storage, f)


def load_optimization_data(n_h, algo):
    f_name = paths.data_path + 'bias_storage_%s_nh=%d.p'%(algo, n_h)

    with open(f_name, 'rb') as file:
        bias_storage = pickle.load(file)

    # print(bias_storage)
    tasks = [resdict['task'] for resdict in bias_storage]
    perfs = [resdict['perf'][-1] for resdict in bias_storage]

    biasses_h = [resdict['b_final'][0] for resdict in bias_storage]
    biasses_o = [resdict['b_final'][1] for resdict in bias_storage]

    # print(tasks)
    # print(biasses)

    return tasks, biasses_h, biasses_o, perfs


def run_2l_optimizations(n_h1, n_h2, algo='random',
                     n_per_batch=100, n_per_epoch=20, n_epoch=200):
    # construct the data sampler
    sampler_pair = samplers.SamplerPair('EMNIST', nb_factor_test=10,
                        n_per_batch=n_per_batch, n_per_epoch=n_per_epoch)
    n_inp = sampler_pair.get_input_dim()

    tasks, biasses_h, biasses_o, perfs = load_optimization_data(n_h1, algo)

    bias_storage = []
    for task, bias_h, bias_o, perf_1l in zip(tasks, biasses_h, biasses_o, perfs):
        print('output bias =', bias_o)
        # task = sample_binary_task()
        sampler_pair.set_tasks(task, target_type='perceptron')

        # construct the initial weights and biases
        # w_out = np.array([1. for _ in range(n_h2-2)] + [1., -1])
        # w_out /= np.linalg.norm(w_out)
        w_hid = np.random.randn(n_h2, n_h1)
        w_hid /= np.linalg.norm(w_hid, axis=1)[:,None]

        ws = [load_input_weight(n_h1, algo), load_hidden_weight(n_h1, n_h2, algo), get_output_weight(n_h2)]
        # ws = [load_input_weight(n_h1, algo), w_hid, get_output_weight(n_h2)]


        # ws = [load_input_weight(n_h1, algo), get_output_weight(n_h1)]
        # w_hid = np.ones((n_h1, n_h2))
        # w_hid /= np.linalg.norm(w_hid, axis=1)[:,None]

        bs = [bias_h, np.random.randn(n_h2, 1) / (10.*n_h2), np.random.randn(1, 1) / 10.]
        # bs = [bias_h, np.concatenate((-.1*np.ones((n_h2-2,1)), bias_o, -bias_o), axis=0),
        #               np.zeros((1,1))]
        # bs = [bias_h, bias_o]
        for w in ws: print(w.shape)
        for b in bs: print(b.shape)

        w_final, b_final, perf = optim.run_optim(ws, bs, sampler_pair, n_epoch=n_epoch,
                                                 b_idx=[0,1,2], w_idx=[])
        # w_final, b_final, perf = optim.run_optim(w_, b_, sampler_pair, opt_w=False, n_epoch=n_epoch, opt_g=False, b_idx=[0,1,2])
        # w_final, b_final, perf = optim.run_optim(ws, bs, sampler_pair, opt_w=False, n_epoch=n_epoch, opt_g=True)

        print('')
        print('--> max w_0 diff:', np.max(np.abs(w_final[0] - ws[0])))
        print('--> max w_1 diff:', np.max(np.abs(w_final[1] - ws[1])))
        print('--> max w_2 diff:', np.max(np.abs(w_final[2] - ws[2])))

        bias_storage.append({'b_final': b_final, 'perf': perf, 'task': task})

        print('\n>>> Perf 1 hidden:')
        print(perf_1l)
        print('\n>>> Perf 2:')
        print(np.mean(perf[-20:]))

        pl.figure('w_hid loaded', figsize=(10,5))
        gs1 = GridSpec(5,5)
        gs1.update(top=0.95, bottom=0.05, left=0.05, right=0.475, hspace=0.05, wspace=0.05)
        gs2 = GridSpec(5,5)
        gs2.update(top=0.95, bottom=0.05, left=0.525, right=0.95, hspace=0.05, wspace=0.05)

        for kk in range(n_h2):
            ii = kk//5
            jj = kk%5

            ax1 = pl.subplot(gs1[ii,jj])
            ax2 = pl.subplot(gs2[ii,jj])

            w_i = np.reshape(ws[1][kk,:], (5,5))
            w_f = np.reshape(w_final[1][kk,:], (5,5))
            ax1.imshow(w_i)
            ax2.imshow(w_f)

            kk += 1

            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])

        pl.show()


    with open(paths.data_path + 'bias_storage_%s_nh1=%d_nh2=%d.p'%(algo, n_h1, n_h2), 'wb') as f:
        pickle.dump(bias_storage, f)


def check_hidden_distr(n_h1, algo='random', dataset='EMNIST'):
    load_input_weight(n_h1, algo)
    tasks, biasses_h, biasses_o, perfs = load_optimization_data(n_h1, algo)

    # print(tasks)
    # print(biasses)

    hdl = hiddenmatrix.HiddenDataLoader(n_h1, tasks, biasses_h, biasses_o, perfs, algo, dataset=dataset, batch_size=1000, n_per_epoch=10)

    act = np.zeros(n_h1)

    vecs = []

    for htup in hdl:
        for hdat, odat, target, task, bias_h, bias_o, perf in zip(*htup):
            v0 = []

            bias_ = bias_h.numpy()
            hdat_ = hdat.numpy()
            odat_ = odat.numpy()
            target_ = target.numpy()

            f_arr_ = np.concatenate([np.where(hd > 1e-4)[0] for hd in hdat_])
            f_arr0 = np.concatenate([np.where(hd > 1e-4)[0] for tg, hd in zip(target, hdat_) if tg < .5])
            f_arr1 = np.concatenate([np.where(hd > 1e-4)[0] for tg, hd in zip(target, hdat_) if tg > .5])

            n_act_, n_count_ = np.unique(f_arr_, return_counts=True)
            n_act0, n_count0 = np.unique(f_arr0, return_counts=True)
            n_act1, n_count1 = np.unique(f_arr1, return_counts=True)

            print('>>> h task %s'%str(task))
            print('bias:')
            print(np.where(bias_ > 0.)[0][:,None])
            # print('hidden:')
            # # print([np.where(hd > 1e-4)[0][:,None] for hd in hdat_])
            # for tg, hd in zip(target, hdat_):
            #     print('T%d ---'%tg, np.where(hd > 1e-4)[0])

            # print(dict(zip(n_act, n_count)))

            dims_ = np.argsort(n_count_)[::-1][:3]
            act_ = [n_act_[dims_[0]], n_act_[dims_[1]], n_act_[dims_[2]]]
            dims0 = np.argsort(n_count0)[::-1][:3]
            act0 = [n_act0[dims0[0]], n_act0[dims0[1]], n_act0[dims0[2]]]
            dims1 = np.argsort(n_count1)[::-1][:3]
            act1 = [n_act1[dims1[0]], n_act1[dims1[1]], n_act1[dims1[2]]]

            print('common hidden activations:')
            print('-> class 0:', act0)
            print('-> class 1:', act1)

            for na, nc in zip(n_act_, n_count_):
                act[na] += nc

            # pl.figure('perf = '+str(perf))
            # ax = pl.gca(projection='3d')
            # pl.figure('avg diff vec')
            # axd = pl.gca()

            idx_c = np.where(np.heaviside(odat_, 0).astype(int) == target_)[0]
            idx_w = np.where(np.heaviside(odat_, 0).astype(int) != target_)[0]

            # ax.scatter(hdat_[idx_c,act_[0]], hdat_[idx_c,act_[1]], hdat_[idx_c,act_[2]], c=target[idx_c], marker='o', cmap=pl.get_cmap('jet'))
            # ax.scatter(hdat_[idx_w,act_[0]], hdat_[idx_w,act_[1]], hdat_[idx_w,act_[2]], c=target[idx_w], marker='s', cmap=pl.get_cmap('jet'))

            # indices of datapoints in respective classes in respective classes
            idx_c1 = np.where(target > .5)[0]
            idx_c2 = np.where(target <= .5)[0]

            # inidices of correctly classified datapoints
            idx_c1c = np.intersect1d(idx_c, idx_c1)
            idx_c2c = np.intersect1d(idx_c, idx_c2)

            # inidices of wrongly classified datapoints
            idx_c1w = np.intersect1d(idx_w, idx_c1)
            idx_c2w = np.intersect1d(idx_w, idx_c2)

            nn_c1 = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
            nn_c2 = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')

            nn_c1.fit(hdat_[idx_c1c,:])
            nn_c2.fit(hdat_[idx_c2c,:])

            try:
                # find nearest neighbors of wrongly classified examples
                idx_nn1 = nn_c1.kneighbors(hdat_[idx_c2w], return_distance=False)
                idx_nn2 = nn_c2.kneighbors(hdat_[idx_c1w], return_distance=False)

                # print(idx_nn1.shape)
                # print(idx_c2w.shape)

                # draw lines between incorrectly classified points and nearest neighbours
                for idx_s, idx_c2 in zip(idx_nn1, idx_c2w):
                    for idx in idx_s:
                        idx_ = idx_c1c[idx]
                        # print(target[idx_], target[idx_c2])
                        # ax.plot([hdat_[idx_,act_[0]], hdat_[idx_c2,act_[0]]],
                        #         [hdat_[idx_,act_[1]], hdat_[idx_c2,act_[1]]],
                        #         [hdat_[idx_,act_[2]], hdat_[idx_c2,act_[2]]], c='r')

                        v0.append(hdat[idx_] - hdat[idx_c2])

                for idx_s, idx_c1 in zip(idx_nn2, idx_c1w):
                    for idx in idx_s:
                        idx_ = idx_c2c[idx]
                        # print(target[idx_], target[idx_c1])
                        # ax.plot([hdat_[idx_,act_[0]], hdat_[idx_c1,act_[0]]],
                        #         [hdat_[idx_,act_[1]], hdat_[idx_c1,act_[1]]],
                        #         [hdat_[idx_,act_[2]], hdat_[idx_c1,act_[2]]], c='b')

                        v0.append(hdat[idx_] - hdat[idx_c1])

                v0 = np.array([vt.numpy() for vt in v0])
                vavg = np.mean(v0, 0)
                print('---', v0.shape)
                vecs.append(v0)

            except ValueError:
                print('No wrongly classified examples in this batch')


            # print(v0)

            # axd.bar(np.arange(len(vavg)), vavg, width=1.)

            # pl.show()

    vecs = np.concatenate(vecs, axis=0)
    vecs = np.concatenate((vecs, np.ones_like(vecs)), axis=0)
    vecs /= np.linalg.norm(vecs, axis=1)[:,None]


    from biasadaptation.weightmatrices import sc
    n_h2 = 25
    w_hid = sc.get_weightmatrix_sc(vecs, n_h2)
    # print('---',w_hid.shape)

    # w_hid = np.concatenate((w_hid, np.ones((1,n_h1)), -np.ones((1,n_h1))), axis=0)
    # print('!!!', w_hid.shape)

    f_name = paths.data_path + 'weight_matrices/' + 'hidden_neighb_%s_%s_nh1=%d_nh2=%d.npy'%(dataset, algo, n_h1, n_h2)
    np.save(f_name, w_hid)

    pl.figure('w_hid', figsize=(6,6))
    gs = GridSpec(5,5)

    kk = 0
    for ii in range(5):
        for jj in range(5):
            ax = pl.subplot(gs[ii,jj])

            w_plot = np.reshape(w_hid[kk,:], (5,5))
            ax.imshow(w_plot)

            kk += 1

            ax.set_xticks([])
            ax.set_yticks([])

    pl.show()




if __name__ == "__main__":
    # run_optimizations(25, n_task=20, algo='sm', old_flag=True)

    run_2l_optimizations(25, 25, algo='sm')

    # check_hidden_distr(25, algo='sm')
