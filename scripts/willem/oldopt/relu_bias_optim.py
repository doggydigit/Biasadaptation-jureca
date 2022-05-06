import numpy as np
import torch
import torch.optim as optim
import scipy.spatial as ssp

import pickle
import copy
import time
import os
import argparse

from biasadaptation.biasfit import biasfit, bgfit
from biasadaptation.utils import samplers, losses

from neat import STree, SNode

from datarep.matplotlibsettings import *
import datarep.paths as paths


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


def get_output_weight(n_h):
    w_vec = np.ones((1,n_h))
    w_vec /= np.linalg.norm(w_vec)

    return w_vec


def get_weights(n_inp, n_h, algo='', dataset='EMNIST'):
    """
    algo: str ('pca', 'ica', 'rp', 'rg', 'sc', 'scd', 'sm', 'random')
    """
    if algo == 'random':
        ws = [np.random.randn(n_h, n_inp),
              np.random.randn(1, n_h)]

        for w in ws:
            w /= np.linalg.norm(w, axis=1)[:,None]

    else:
        # construct the weight vectors
        ws = [load_input_weight(n_h, algo, dataset=dataset),
              get_output_weight(n_h)]

    return ws


def get_biases(n_h, n_task=1):
    bs = [np.random.randn(n_h, n_task) / (10.*n_h)]
    bs.append(np.random.randn(1, n_task) / 10.)

    return bs


def run_binary(ws, bs, sampler_pair, opt_w=False, opt_g=False, n_epoch=100):
    nts_train = sampler_pair.nt_sampler_train
    nts_test = sampler_pair.nt_sampler_test
    # network initialziation
    rlnet = bgfit.ReLuFit(ws, bs, opt_w=opt_w) if opt_g else \
            biasfit.ReLuFit(ws, bs, opt_w=opt_w)
    optimizer = optim.Adam(rlnet.parameters(), lr=0.005, betas=(0.9, 0.999))

    ws_init = copy.deepcopy(ws)

    perf = []
    for nn in range(n_epoch):
        for idx, (xdata, xtask, xtarget) in enumerate(nts_train):
            # compute network output
            optimizer.zero_grad()
            out = rlnet.forward(xdata.T, xtask)
            # compute loss and gradient
            # print(out.shape, xtarget.shape)
            loss = losses.perceptron_loss(out, xtarget)
            loss.backward()
            # perform gradient step
            optimizer.step()

        # performance on train set
        xdata, xtask, xtarget = next(iter(nts_test))
        out = rlnet.forward(xdata.T, xtask)
        lb = losses.binary_loss(out, xtarget)
        print('\n>>> epoch %d --> test accuracy : %.3f'%(nn, 1.-lb))
        perf.append((1.-float(lb))*100)
        # performance on test set
        xdata, xtask, xtarget = next(iter(nts_train))
        out = rlnet.forward(xdata.T, xtask)
        lb = losses.binary_loss(out, xtarget)
        print(out)
        print(xtarget)
        print('             --> train accuracy: %.3f'%(1.-lb))


    print('')
    for ii, (w, w_init) in enumerate(zip(rlnet.ws, ws_init)):
        w = w.detach().numpy()
        print('L%d -> |w_init - w_final| = %.5f'%(ii,np.linalg.norm(w - w_init)))

    w_final = [w.detach().numpy() for w in rlnet.ws]
    b_final = [b.detach().numpy() for b in rlnet.bs]

    return w_final, b_final, perf


def run_single_optim(n_h, task=[(1,), (4,)],
                     opt_w=False, opt_g=False, algo='random',
                     n_per_batch=100, n_per_epoch=20, n_epoch=200):
    # construct the data sampler
    sampler_pair = samplers.SamplerPair('EMNIST', nb_factor_test=10,
                        n_per_batch=n_per_batch, n_per_epoch=n_per_epoch)
    sampler_pair.set_tasks(task, target_type='perceptron')
    n_inp = sampler_pair.get_input_dim()

    # construct the initial weights and biases
    ws = get_weights(n_inp, n_h, dataset='EMNIST', algo=algo)
    bs = get_biases(n_h)

    run_binary(ws, bs, sampler_pair, opt_w=opt_w, n_epoch=n_epoch, opt_g=opt_g)


def optim_tree_random(dataset, n_h, n_epoch=200,
                        opt_g=False,
                        w_timestamp=False, ):
    st = '_%.0f'%time.time() if w_timestamp else ''
    sg = '_g' if opt_g else ''
    f_name = paths.data_path + 'optim_tree_w_random%s_%s_NH=%d%s.p'%(sg, dataset, n_h, st)

    sampler_pair = samplers.SamplerPair(dataset, nb_factor_test=10,
                                          n_per_batch=100, n_per_epoch=20)
    n_inp = sampler_pair.get_input_dim()

    root = SNode(0)
    root.content = {
                    'description': 'Random initial weights and biases',
                    'w_init': get_weights(n_inp, n_h, dataset=dataset, algo='random'),
                    'b_init': get_biases(n_h)
                   }

    # optimization of biases only
    tasks = sample_binary_task()
    print('\n\n>>> Learning task', tasks)
    sampler_pair.set_tasks(tasks, target_type='perceptron')
    w_init, b_opt , perf  = run_binary(copy.deepcopy(root.content['w_init']),
                                       copy.deepcopy(root.content['b_init']),
                                       sampler_pair,
                                       opt_w=False, opt_g=opt_g, n_epoch=n_epoch)

    node1 = SNode(1)
    node1.content = {
                     'description': 'Optimize biases',
                     'w_final': w_init,
                     'b_final': b_opt,
                     'task': tasks,
                     'perf': perf
                    }

    # optimization of weights and biases
    tasks = sample_binary_task()
    sampler_pair.set_tasks(tasks, target_type='perceptron')
    print('\n\n>>> Learning task', tasks)
    w_opt_, b_opt_, perf_ = run_binary(copy.deepcopy(root.content['w_init']),
                                       copy.deepcopy(root.content['b_init']),
                                       sampler_pair,
                                       opt_w=True,  opt_g=opt_g, n_epoch=n_epoch)
    node2 = SNode(2)
    node2.content = {
                     'description': 'Optimize weights and biases',
                     'w_final': w_opt_,
                     'b_final': b_opt_,
                     'w_init': w_opt_,
                     'b_init': get_biases(n_h),
                     'task': tasks,
                     'perf': perf_
                    }

    # new optimization of biases
    tasks = sample_binary_task()
    sampler_pair.set_tasks(tasks, target_type='perceptron')
    print('\n\n>>> Learning task', tasks)
    w_opt_, b_opt_, perf_ = run_binary(copy.deepcopy(node2.content['w_init']),
                                       copy.deepcopy(node2.content['b_init']),
                                       sampler_pair,
                                       opt_w=False, opt_g=opt_g, n_epoch=n_epoch)
    node3 = SNode(3)
    node3.content = {
                     'description': 'Optimize weights and biases',
                     'w_final': w_opt_,
                     'b_final': b_opt_,
                     'task': tasks,
                     'perf': perf_
                    }

    tree = STree(root)
    tree.addNodeWithParent(node1, root)
    tree.addNodeWithParent(node2, root)
    tree.addNodeWithParent(node3, node2)

    print('\n\n--- performances ---')
    for node in tree:
        try:
            print('>', node.index, node.content['perf'])
        except KeyError:
            pass

    with open(f_name, 'wb') as f:
        pickle.dump(tree, f)


def optim_tree_good(dataset, n_h, n_epoch=200,
                    algo='pca', opt_g=False,
                    w_timestamp=False):
    st = '_%.0f'%time.time() if w_timestamp else ''
    sg = '_g' if opt_g else ''
    f_name = paths.data_path + 'optim_tree_w_%s%s_%s_NH=%d%s.p'%(algo, sg, dataset, n_h, st)

    sampler_pair = samplers.SamplerPair(dataset, nb_factor_test=10,
                                        n_per_batch=100, n_per_epoch=20)
    n_inp = sampler_pair.get_input_dim()

    root = SNode(0)
    root.content = {
                    'description': 'Random initial weights and biases',
                    'w_init': get_weights(n_inp, n_h, dataset=dataset, algo=algo),
                    'b_init': get_biases(n_h)
                   }
    tree = STree(root)

    # optimization of biases only
    # good initial condition for weights
    # random initial condition for biases
    tasks = sample_binary_task()
    print('\n\n>>> Learning task', tasks)
    sampler_pair.set_tasks(tasks, target_type='perceptron')
    w_init, b_opt , perf  = run_binary(copy.deepcopy(root.content['w_init']),
                                       copy.deepcopy(root.content['b_init']),
                                       sampler_pair,
                                       opt_w=False,  opt_g=opt_g, n_epoch=n_epoch)

    node1 = SNode(1)
    node1.content = {
                     'description': 'Optimize biases',
                     'w_final': w_init,
                     'b_final': b_opt,
                     'w_init': w_init,
                     'b_init': b_opt,
                     'task': tasks,
                     'perf': perf
                    }
    tree.addNodeWithParent(node1, root)

    # optimization of biases only
    # good initial condition for weights
    # previous biases as initial condition for biases
    tasks = sample_binary_task()
    sampler_pair.set_tasks(tasks, target_type='perceptron')
    print('\n\n>>> Learning task', tasks)
    w_init_, b_opt_, perf_ = run_binary(copy.deepcopy(node1.content['w_init']),
                                       copy.deepcopy(node1.content['b_init']),
                                       sampler_pair,
                                       opt_w=False, opt_g=opt_g, n_epoch=n_epoch)
    node2 = SNode(2)
    node2.content = {
                     'description': 'Optimize weights and biases',
                     'w_final': w_init_,
                     'b_final': b_opt_,
                     'task': tasks,
                     'perf': perf_
                    }
    tree.addNodeWithParent(node2, node1)

    # optimization of weights and biases
    # good initial condition for weights
    # previous biases as initial condition for biases
    tasks = sample_binary_task()
    sampler_pair.set_tasks(tasks, target_type='perceptron')
    print('\n\n>>> Learning task', tasks)
    w_opt_, b_opt_, perf_ = run_binary(copy.deepcopy(node1.content['w_init']),
                                       copy.deepcopy(node1.content['b_init']),
                                       sampler_pair,
                                       opt_w=True, opt_g=opt_g, n_epoch=n_epoch)
    node3 = SNode(3)
    node3.content = {
                     'description': 'Optimize weights and biases',
                     'w_final': w_opt_,
                     'b_final': b_opt_,
                     'task': tasks,
                     'perf': perf_
                    }
    tree.addNodeWithParent(node3, node1)

    # optimization of weights and biases
    # good initial condition for weights
    # random initial condition for biases
    tasks = sample_binary_task()
    print('\n\n>>> Learning task', tasks)
    sampler_pair.set_tasks(tasks, target_type='perceptron')
    w_opt, b_opt , perf  = run_binary(copy.deepcopy(root.content['w_init']),
                                       copy.deepcopy(root.content['b_init']),
                                       sampler_pair,
                                       opt_w=True, opt_g=opt_g, n_epoch=n_epoch)

    node4 = SNode(4)
    node4.content = {
                     'description': 'Optimize biases',
                     'w_final': w_opt,
                     'b_final': b_opt,
                     'w_init': w_opt,
                     'b_init': get_biases(n_h),
                     'task': tasks,
                     'perf': perf
                    }
    tree.addNodeWithParent(node4, root)

    # optimization of biases only
    # previous weights as initial condition for weights
    # previous biasses as initial condition for biases
    tasks = sample_binary_task()
    print('\n\n>>> Learning task', tasks)
    sampler_pair.set_tasks(tasks, target_type='perceptron')
    w_init_, b_opt_, perf_  = run_binary(copy.deepcopy(node4.content['w_init']),
                                       copy.deepcopy(node4.content['b_init']),
                                       sampler_pair,
                                       opt_w=False, opt_g=opt_g, n_epoch=n_epoch)

    node5 = SNode(5)
    node5.content = {
                     'description': 'Optimize biases',
                     'w_final': w_init_,
                     'b_final': b_opt_,
                     'task': tasks,
                     'perf': perf_
                    }
    tree.addNodeWithParent(node5, node4)

    # optimization of weights and biases
    # previous weights as initial condition for weights
    # previous biasses as initial condition for biases
    tasks = sample_binary_task()
    print('\n\n>>> Learning task', tasks)
    sampler_pair.set_tasks(tasks, target_type='perceptron')
    w_opt_, b_opt_, perf_  = run_binary(copy.deepcopy(node4.content['w_init']),
                                       copy.deepcopy(node4.content['b_init']),
                                       sampler_pair,
                                       opt_w=True, opt_g=opt_g, n_epoch=n_epoch)

    node6 = SNode(6)
    node6.content = {
                     'description': 'Optimize biases',
                     'w_final': w_opt_,
                     'b_final': b_opt_,
                     'task': tasks,
                     'perf': perf_
                    }
    tree.addNodeWithParent(node6, node4)


    print('\n\n--- performances ---')
    for node in tree:
        try:
            print('>', node.index, node.content['perf'])
        except KeyError:
            pass

    with open(f_name, 'wb') as f:
        pickle.dump(tree, f)


def run_random_opts(dataset, n_h, opt_g=False, n_repeat=15):
    for ii in range(n_repeat):
        optim_tree_random(dataset, n_h, opt_g=opt_g, w_timestamp=True)


def run_good_opts(dataset, n_h, algo='random', opt_g=False, n_repeat=15):
    for ii in range(n_repeat):
        optim_tree_good(dataset, n_h, opt_g=opt_g, algo=algo, w_timestamp=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action='store_true', help="whether to run single test sim", default=False)
    parser.add_argument("--group", nargs="*", type=int, help="which sim group to run", default=[0])
    parser.add_argument("--nhidden", nargs="*", type=int, help="number of hidden units", default=[10, 25, 50, 100, 250, 500, 1000])
    parser.add_argument("--ntrial", type=int, help="number of trials", default=15)
    args = parser.parse_args()

    # if args.test == True:
    if True:
        run_single_optim(25, [(1,), (4,)], opt_w=False, opt_g=False, algo='scd')

    else:
        if 0 in args.group:

            for n_h in args.nhidden:
                for ii in range(args.ntrial):
                    optim_tree_random('EMNIST', n_h, opt_g=False, w_timestamp=True)
                    optim_tree_random('EMNIST', n_h, opt_g=True, w_timestamp=True)

            # for n_h in args.nhidden:
                if n_h < 784:
                    for ii in range(args.ntrial):
                        optim_tree_good('EMNIST', n_h, opt_g=False, algo='pca', w_timestamp=True)
                        optim_tree_good('EMNIST', n_h, opt_g=True, algo='pca', w_timestamp=True)

            # for n_h in args.nhidden:
                if n_h < 784:
                    for ii in range(args.ntrial):
                        optim_tree_good('EMNIST', n_h, opt_g=False, algo='ica', w_timestamp=True)
                        optim_tree_good('EMNIST', n_h, opt_g=True, algo='ica', w_timestamp=True)

        if 1 in args.group:
            for n_h in args.nhidden:
                for ii in range(args.ntrial):
                    optim_tree_good('EMNIST', n_h, opt_g=False, algo='rg', w_timestamp=True)
                    optim_tree_good('EMNIST', n_h, opt_g=True, algo='rg', w_timestamp=True)

            # for n_h in args.nhidden:
                for ii in range(args.ntrial):
                    optim_tree_good('EMNIST', n_h, opt_g=False, algo='rp', w_timestamp=True)
                    optim_tree_good('EMNIST', n_h, opt_g=True, algo='rp', w_timestamp=True)

        if 2 in args.group:
            for n_h in args.nhidden:
                for ii in range(args.ntrial):
                    optim_tree_good('EMNIST', n_h, opt_g=False, algo='scd', w_timestamp=True)
                    optim_tree_good('EMNIST', n_h, opt_g=True, algo='scd', w_timestamp=True)

            # for n_h in args.nhidden:
                for ii in range(args.ntrial):
                    optim_tree_good('EMNIST', n_h, opt_g=False, algo='sc', w_timestamp=True)
                    optim_tree_good('EMNIST', n_h, opt_g=True, algo='sc', w_timestamp=True)

        if 3 in args.group:
            for n_h in args.nhidden:
                if n_h < 600:
                    for ii in range(args.ntrial):
                        optim_tree_good('EMNIST', n_h, opt_g=False, algo='sm', w_timestamp=True)
                        optim_tree_good('EMNIST', n_h, opt_g=True, algo='sm', w_timestamp=True)


