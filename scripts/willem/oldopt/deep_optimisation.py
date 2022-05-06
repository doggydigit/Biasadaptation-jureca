import numpy as np
import scipy.linalg as sla
import torch
import torchvision.datasets as tdatasets
import torchvision.transforms as ttransforms
import torch.utils.data as tdata
from tqdm import tqdm
from sklearn.decomposition import MiniBatchDictionaryLearning, dict_learning_online

import copy
import pickle

from datarep import paths
from datarep.matplotlibsettings import *

import penalized_matrix_decomposition as pmd
import binary_matrix_decomposition as bmd
from biasadaptation.utils import samplers, losses, utils
from biasadaptation.biasfit import specificbiasfit
import optim, zerofinder


def sample_binary_task(dataset='EMNIST', task_type='1v1'):
    if task_type == '1v1':
        classes = np.random.choice(np.arange(samplers.N_CLASSES[dataset]), 2, replace=False)
        return [(cc,) for cc in classes]
    elif task_type == '1vall':
        class_idx = np.arange(samplers.N_CLASSES[dataset])
        c1 = np.random.choice(class_idx, 1, replace=False)[0]
        return [(c1,)] + [tuple([cc for cc in class_idx if cc != c1])]
    else:
        raise Exception('Invalid `task_type`')


def load_weights(n_hs, n_l, dataset='EMNIST', algo='pbmd'):
    if algo == 'pmd':
        namestring = '_'.join([str(n_h) for n_h in n_hs])
        with open(paths.data_path + 'weight_matrices/%s_weight_mats_pmd_nh=%s.p'%(dataset, namestring), 'rb') as f:
            Ws = pickle.load(f)
            Cs = pickle.load(f)
            DX = pickle.load(f)

            Ws = Ws[:2]

    elif algo == 'pbmd':
        namestring = '_'.join([str(n_h) for n_h in n_hs])
        with open(paths.data_path + 'weight_matrices/%s_weight_mats_pbmd_nh=%s.p'%(dataset, namestring), 'rb') as f:
            Ws = pickle.load(f)

    elif algo == 'scd':
        namestring = '_'.join([str(n_h) for n_h in n_hs])

        with open(paths.data_path + 'weight_matrices/%s_weight_mats_scd_nh=%s.p'%(dataset, namestring), 'rb') as f:
            Ws = pickle.load(f)

        # path_name = os.path.join(paths.data_path, 'weight_matrices/', '%s_scd%d.npy'%(dataset, n_h))
        # w0 = np.load(path_name)

        # path_name = os.path.join(paths.data_path, 'weight_matrices/', 'hidden_%s_scd_nh1=%d_nh2=%d.npy'%(dataset, n_h, n_h))
        # w1 = np.load(path_name)

        # Ws = [w0, w1]

    elif algo == 'sm':
        path_name = os.path.join(paths.data_path, 'weight_matrices/', '%s_sm%d.npy'%(dataset, n_h))
        w0 = np.load(path_name)

        path_name = os.path.join(paths.data_path, 'weight_matrices/', 'hidden_neighb_%s_sm_nh1=%d_nh2=%d.npy'%(dataset, n_h, n_h))
        w1 = np.load(path_name)

        Ws = [w0, w1]

    elif algo == 'random':
        path_name = os.path.join(paths.data_path, 'weight_matrices/', '%s_sm%d.npy'%(dataset, n_hs[0]))
        w0 = np.load(path_name)

        w0 = np.random.randn(*w0.shape)
        w0 /= np.linalg.norm(w0, axis=1)[:,None]

        w1 = np.random.randn(n_hs[1], n_hs[0])
        w1 /= np.linalg.norm(w1, axis=1)[:,None]

        Ws = [w0, w1]

    return Ws[:n_l]


def run_optimisations_pair(dataset='EMNIST', n_hs=25, n_l=None, task_type='1v1', algo='pbmd',
                        n_per_batch=100, n_per_epoch=20, n_epoch=200, n_task=15, w_optim=False):
    if isinstance(n_hs, int):
        n_hs = [n_hs]
    if n_l is None: n_l = len(n_hs)

    so = task_type + '_pair'
    if w_optim: so += '_optW'

    lr = optim.LRLin(200, lr0=0.002, lr1=0.0001)

    Ws = load_weights(n_h, n_l, algo=algo, dataset=dataset)

    # construct the data sampler
    sampler_pair = samplers.SamplerPair(dataset, nb_factor_test=10,
                        n_per_batch=n_per_batch, n_per_epoch=n_per_epoch)
    n_inp = sampler_pair.get_input_dim()

    bias_storage = []
    perfs0 = []
    perfs1 = []
    tasks = []
    for nt in range(n_task):
        task = sample_binary_task(dataset=dataset, task_type=task_type)
        tasks.append(task)
        print('optimizing task ', task)
        sampler_pair.set_tasks(task, target_type='perceptron')

        # construct the initial weights and biases
        ws = Ws[:n_l] + [np.ones((1, n_hs[-1])) / np.sqrt(n_hs[-1])]
        bs = [np.random.randn(n_h, 1) / (10.*n_h) for n_h in range(n_hs)] + [np.random.randn(1, 1) / 10.]

        print([w.shape for w in ws])
        print([b.shape for b in bs])

        w_idx = list(range(n_l+1)) if w_optim else []

        w_final, b_final, perf = optim.run_optim(ws, bs, sampler_pair, n_epoch=n_epoch, w_idx=w_idx, lr=lr)

        bias_storage.append({'w_final': w_final, 'b_final': b_final, 'perf': perf, 'task': task})

        perfs0.append(np.array(perf))

        task = task[::-1]
        print('optimizing task ', task)
        sampler_pair.set_tasks(task, target_type='perceptron')

        # construct the initial weights and biases
        ws = Ws[:n_l] + [np.ones((1, n_hs[n_l-1])) / np.sqrt(n_hs[n_l-1])]
        bs = [np.random.randn(n_hs[ii], 1) / (10.*n_hs[ii]) for ii in range(n_l)] + [np.random.randn(1, 1) / 10.]

        print([w.shape for w in ws])
        print([b.shape for b in bs])

        w_idx = list(range(n_l+1)) if w_optim else []

        w_final, b_final, perf = optim.run_optim(ws, bs, sampler_pair, n_epoch=n_epoch, w_idx=w_idx, lr=lr)

        bias_storage.append({'w_final': w_final, 'b_final': b_final, 'perf': perf, 'task': task})

        perfs1.append(np.array(perf))


    # print('\n>>> average performance per epoch (%s, %s, n_h = %d, n_l = %d):'%(algo, task_type, n_h, n_l))
    # print(np.mean(perfs, 0))
    print('\n>>> average performance per epoch (%s, %s, n_h = %d, n_l = %d):\n'%(algo, task_type, n_h, n_l))
    for tt, p0, p1 in zip(tasks, perfs0, perfs1):
        p0_ = np.mean(p0[-5:])
        p1_ = np.mean(p1[-5:])
        print('task: ' + str(tt) + ' --> perf0 = %.4f, perf1 = %.4f'%(p0_, p1_))


    namestring = '_'.join([str(n_h) for n_h in n_hs])
    with open(paths.data_path + 'bias_storage_%s_nh=%s_%s.p'%(algo, namestring, so), 'wb') as f:
        pickle.dump(bias_storage, f)



def run_optimisations_many(dataset='EMNIST', n_hs=25, n_l=None, task_type='1v1', algo='pbmd',
                        n_per_batch=100, n_per_epoch=20, n_epoch=200, n_task=3, w_optim=False, n_trial=5):
    if isinstance(n_hs, int):
        n_hs = [n_hs]
    if n_l is None: n_l = len(n_hs)

    so = task_type + '_many'
    if w_optim: so += '_optW'

    lr = optim.LRLin(200, lr0=0.002, lr1=0.0001)

    Ws = load_weights(n_h, algo=algo, dataset=dataset)

    # construct the data sampler
    sampler_pair = samplers.SamplerPair(dataset, nb_factor_test=10,
                        n_per_batch=n_per_batch, n_per_epoch=n_per_epoch)
    n_inp = sampler_pair.get_input_dim()

    bias_storage = []
    perfs = [[] for _ in range(n_task)]
    tasks = []
    for nt in range(n_task):
        task = sample_binary_task(dataset=dataset, task_type=task_type)
        tasks.append(task)

        for nn in range(n_trial):

            print('optimizing task ', task)
            sampler_pair.set_tasks(task, target_type='perceptron')

            # construct the initial weights and biases
            ws = Ws[:n_l] + [np.ones((1, n_hs[n_l-1])) / np.sqrt(n_hs[n_l-1])]
            bs = [np.random.randn(n_hs[ii], 1) / (10.*n_hs[ii]) for ii in range(n_l)] + [np.random.randn(1, 1) / 10.]

            print([w.shape for w in ws])
            print([b.shape for b in bs])

            w_idx = list(range(n_l+1)) if w_optim else []

            w_final, b_final, perf = optim.run_optim(ws, bs, sampler_pair, n_epoch=n_epoch, w_idx=w_idx, lr=lr)

            bias_storage.append({'w_final': w_final, 'b_final': b_final, 'perf': perf, 'task': task})

            perfs[nt].append(np.array(perf))


    # print('\n>>> average performance per epoch (%s, %s, n_h = %d, n_l = %d):'%(algo, task_type, n_h, n_l))
    # print(np.mean(perfs, 0))
    print('\n>>> average performance per epoch (%s, %s, n_h = %d, n_l = %d):\n'%(algo, task_type, n_h, n_l))
    for tt, perf in zip(tasks, perfs):
        pstr = 'task: ' + str(tt) + ' --> perf = '
        for pp in perf:
            p_ = np.mean(pp[-5:])
            pstr += '%.4f, '%p_

        print(pstr)


    namestring = '_'.join([str(n_h) for n_h in n_hs])
    with open(paths.data_path + 'bias_storage_%s_nh=%s_%s.p'%(algo, namestring, so), 'wb') as f:
        pickle.dump(bias_storage, f)


def run_optimisations(dataset='EMNIST', n_hs=25, n_l=None, task_type='1v1', algo='pbmd',
                        n_per_batch=100, n_per_epoch=20, n_epoch=200, n_task=15, w_optim=False):
    if isinstance(n_hs, int):
        n_hs = [n_hs]
    if n_l is None: n_l = len(n_hs)
    so = task_type
    if w_optim: so += '_optW'

    lr = optim.LRLin(200, lr0=0.002, lr1=0.0001)

    Ws = load_weights(n_hs, n_l, algo=algo, dataset=dataset)

    # construct the data sampler
    sampler_pair = samplers.SamplerPair(dataset, nb_factor_test=10,
                        n_per_batch=n_per_batch, n_per_epoch=n_per_epoch)
    n_inp = sampler_pair.get_input_dim()

    bias_storage = []
    perfs = []
    for nt in range(n_task):
        task = sample_binary_task(dataset=dataset, task_type=task_type)
        print('optimizing task ', task)
        sampler_pair.set_tasks(task, target_type='perceptron')

        # construct the initial weights and biases
        ws = Ws[:n_l] + [np.ones((1, n_hs[n_l-1])) / np.sqrt(n_hs[n_l-1])]
        bs = [np.random.randn(n_hs[ii], 1) / (10.*n_hs[ii]) for ii in range(n_l)] + [np.random.randn(1, 1) / 10.]

        print([w.shape for w in ws])
        print([b.shape for b in bs])

        w_idx = list(range(0,n_l+1)) if w_optim else []

        w_final, b_final, perf = optim.run_optim(ws, bs, sampler_pair, n_epoch=n_epoch, w_idx=w_idx, lr=lr)

        bias_storage.append({'w_final': w_final, 'b_final': b_final, 'perf': perf, 'task': task})

        perfs.append(np.array(perf))


    namestring = '_'.join([str(n_h) for n_h in n_hs])
    # print('\n>>> average performance per epoch (%s, %s, n_h = %d, n_l = %d):'%(algo, task_type, n_h, n_l))
    # print(np.mean(perfs, 0))
    pstr = '\n>>> average performance per epoch (%s, %s, n_h = %s, n_l = %d):\n'%(algo, task_type, namestring, n_l)
    pstr += str(np.mean(perfs, 0))

    print(pstr)

    with open(paths.data_path + 'bias_storage_%s_nh=%s_%s.p'%(algo, namestring, so), 'wb') as f:
        pickle.dump(bias_storage, f)

    return pstr


def analyze_zerocrossings(dataset='EMNIST', n_h=25, n_l=2, task_type='1v1', algo='pbmd',
                            w_optim=False):
    """
    plot the location and the normal vector where the decision surface crosses zero
    """

    so = task_type
    if w_optim: so += '_optW'
    # so = ''

    sampler = samplers.NTaskSampler(dataset, n_per_batch=400, n_per_epoch=1)

    with open(paths.data_path + 'bias_storage_%s_nh1=%d_nl=%d_%s.p'%(algo, n_h, n_l, so), 'rb') as f:
        bias_storage = pickle.load(f)


    alignments = []

    for res in bias_storage:
        Ws = res['w_final']
        bs = [b[:,0] for b in res['b_final']]
        perf = res['perf']
        task = res['task']

        print('\n' , task)
        print('perf:', perf[-1])

        sampler.set_tasks(task, target_type='perceptron')
        xdata, xtask, xtarget = next(iter(sampler))

        xdat0 = xdata[xtarget < 0].numpy()
        xdat1 = xdata[xtarget > 0].numpy()

        zf = zerofinder.ZeroFinder(Ws, bs)
        rlnet = specificbiasfit.ReLuFit(Ws, res['b_final'])

        o = rlnet.forward(xdata[:20].T, xtask[:20])
        print(1. - losses.binary_loss(o, xtarget[:20]))

        ts = np.linspace(0., 1., 500)
        x0s = []
        xds = []
        wns = []

        f1 = pl.figure(r'perf = %.2f, class = %d'%(np.mean(perf[-10:]), task[0][0]), figsize=(20,5))
        gs0 = GridSpec(5,5)
        gs0.update(top=0.95, bottom=0.05, left=0.02, right=0.24, hspace=0.05, wspace=0.05)
        gs1 = GridSpec(5,5)
        gs1.update(top=0.95, bottom=0.05, left=0.26, right=0.49, hspace=0.05, wspace=0.05)
        gs2 = GridSpec(5,5)
        gs2.update(top=0.95, bottom=0.05, left=0.51, right=0.74, hspace=0.05, wspace=0.05)
        gs3 = GridSpec(5,5)
        gs3.update(top=0.95, bottom=0.05, left=0.76, right=0.98, hspace=0.05, wspace=0.05)

        kk = 0; ll = 0
        while kk < 25 and ll < min(len(xdat0), len(xdat1)):
            try:
                # print(kk, ll)
                # o1 = rlnet.forward(xdat0[ll:ll+1].T, np.zeros(1, dtype=int))
                # o2 = rlnet.forward(xdat1[ll:ll+1].T, np.zeros(1, dtype=int))
                # print(o2, o1)

                x_ = zf.find_zero(xdat0[ll], xdat1[ll])
                wn = zf.find_affine_transform(x_)[0]

                x0s.append(x_)
                xds.append(xdat0[ll] - xdat1[ll])
                wns.append(wn)

                ii = kk//5
                jj = kk%5

                ax0 = pl.subplot(gs0[ii,jj])
                ax0.imshow(utils.to_image_mnist(xdat0[ll]))
                ax0.set_xticks([]); ax0.set_yticks([])

                ax1 = pl.subplot(gs1[ii,jj])
                # ax1.imshow(utils.to_image_mnist(xdat0[ll]))
                ax1.imshow(utils.to_image_mnist(x_))
                ax1.set_xticks([]); ax1.set_yticks([])

                ax2 = pl.subplot(gs2[ii,jj])
                # ax2.imshow(utils.to_image_mnist(xdat1[ll]))
                ax2.imshow(utils.to_image_mnist(wn))
                ax2.set_xticks([]); ax2.set_yticks([])

                ax3 = pl.subplot(gs3[ii,jj])
                fs = np.array([zf.tfunc(t, xdat0[ll], xdat1[ll]) for t in ts])
                ax3.plot(ts, fs)
                ax3.axhline(0., ls='--', c='DarkGrey')
                ax3.set_xticks([]); ax3.set_yticks([])
                # ax3.set_ylim((-.3,.3))

                kk += 1
                ll += 1
            except ValueError:
                ll += 1

        # alignment = np.mean([np.abs(xx @ ww) / (np.linalg.norm(xx)*np.linalg.norm(ww)) for xx, ww in zip(xds, wns)])
        # print('>>> average alignment:', alignment)
        # alignments.append(alignment)

        pl.show()

    # print('\n >>> average alignment all tasks:', np.mean(alignments))




if __name__ == "__main__":
    # run_optimisations(task_type='1v1')

    # ps1 = run_optimisations(task_type='1vall', algo='sm', n_epoch=50)
    # ps2 = run_optimisations(task_type='1vall', algo='sm', n_l=1, n_epoch=50)
    # print(ps1, '\n', ps2)

    # ps1 = run_optimisations(task_type='1vall', algo='random', n_hs=[50,25], n_epoch=50, w_optim=True,)
    # ps2 = run_optimisations(task_type='1vall', algo='pbmd', n_hs=[50,25], n_epoch=50, w_optim=True)
    # ps3 = run_optimisations(task_type='1vall', algo='pbmd', n_hs=[50,25], n_epoch=50)
    # ps4 = run_optimisations(task_type='1vall', algo='pmd', n_hs=[50,25], n_epoch=50)
    # ps5 = run_optimisations(task_type='1vall', algo='scd', n_hs=[50,25], n_epoch=50)
    # print(ps1, '\n', ps2, '\n',ps3, '\n', ps4, '\n', ps5)

    # ps1 = run_optimisations(task_type='1vall', algo='random', n_hs=[50,25], n_l=1, n_epoch=50, w_optim=True,)
    # ps2 = run_optimisations(task_type='1vall', algo='pbmd', n_hs=[50,25], n_l=1, n_epoch=50, w_optim=True)
    # ps3 = run_optimisations(task_type='1vall', algo='pbmd', n_hs=[50,25], n_l=1, n_epoch=50)
    # ps4 = run_optimisations(task_type='1vall', algo='pmd', n_hs=[50,25], n_l=1, n_epoch=50)
    # ps5 = run_optimisations(task_type='1vall', algo='scd', n_hs=[50,25], n_l=1, n_epoch=50)
    # print(ps1, '\n', ps2, '\n',ps3, '\n', ps4, '\n', ps5)

    # ps1 = run_optimisations(task_type='1vall', algo='random', n_hs=[100,25], n_epoch=50, w_optim=True,)
    # ps2 = run_optimisations(task_type='1vall', algo='pbmd', n_hs=[100,25], n_epoch=50, w_optim=True)
    # ps3 = run_optimisations(task_type='1vall', algo='pbmd', n_hs=[100,25], n_epoch=50)
    # ps4 = run_optimisations(task_type='1vall', algo='pmd', n_hs=[100,25], n_epoch=50)
    # ps5 = run_optimisations(task_type='1vall', algo='scd', n_hs=[100,25], n_epoch=50)
    # print(ps1, '\n', ps2, '\n',ps3, '\n', ps4, '\n', ps5)

    # ps1 = run_optimisations(task_type='1vall', algo='random', n_hs=[100,25], n_l=1, n_epoch=50, w_optim=True,)
    # ps2 = run_optimisations(task_type='1vall', algo='pbmd', n_hs=[100,25], n_l=1, n_epoch=50, w_optim=True)
    # ps3 = run_optimisations(task_type='1vall', algo='pbmd', n_hs=[100,25], n_l=1, n_epoch=50)
    # ps4 = run_optimisations(task_type='1vall', algo='pmd', n_hs=[100,25], n_l=1, n_epoch=50)
    # ps5 = run_optimisations(task_type='1vall', algo='scd', n_hs=[100,25], n_l=1, n_epoch=50)
    # print(ps1, '\n', ps2, '\n',ps3, '\n', ps4, '\n', ps5)



    # ps1 = run_optimisations_pair(task_type='1v1', algo='sm', n_l=2, n_epoch=100)
    # ps1 = run_optimisations_pair(task_type='1v1', algo='pbmd', n_l=2, n_epoch=100)

    # ps1 = run_optimisations_many(task_type='1v1', algo='sm', n_l=2, n_epoch=100, n_task=3, n_trial=5)
    # ps1 = run_optimisations_many(task_type='1v1', algo='pbmd', n_l=2, n_epoch=100, n_task=3, n_trial=5)
    # ps1 = run_optimisations_many(task_type='1v1', algo='pbmd', n_l=1, n_epoch=100, n_task=3, n_trial=5)
    # ps1 = run_optimisations_many(task_type='1v1', algo='sm', n_l=1, n_epoch=100, n_task=3, n_trial=5)


    # analyze_zerocrossings(task_type='1vall', algo='pbmd', n_l=1, w_optim=False)
    analyze_zerocrossings(task_type='1vall', algo='pbmd', n_l=2, w_optim=True)
    # analyze_zerocrossings(task_type='1vall', algo='sm', n_l=2, w_optim=True)


