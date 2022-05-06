"""
optimizations for one hidden layer
"""
import numpy as np
import torch

import argparse
import pickle
import copy

from biasadaptation.utils import preprocessing

import sys
sys.path.append('..')

import optim, helperfuncs

"""
example usage

    python3 biasopt.py --methods scd --ntask 1 --tasktype 1vall --nhidden 50 --nepoch 100 --readout tanh

"""


# original parameters
LRS_0 = [0.005, 0.001, 0.0001]
X_DIV_0 = 'data'
B_DIV_0 = 10.
B_ADD_0 = 0.
# optimized parameters bias adaptation
LRS_B = [0.06, 0.02, 0.008]
# LRS_B = [0.1, 0.02, 0.008]
X_DIV_B = .5 #'data' (CIFAR) #.5 (MNIST)
B_DIV_B = 5.
B_ADD_B = -.5
# optimized parameters bias adaptation from gradient descent matrices
X_DIV_GD = 1.
# optimized parameters weight optimization
LRS_W = [0.002, 0.001, 0.0008]
X_DIV_W = 4.
B_DIV_W = 10.
B_ADD_W = 0.

# READOUT = "sigmoid1"#"linear"#hardtanh"#
torch.set_num_threads(1)

# read command line args and kwargs
parser = argparse.ArgumentParser()
parser.add_argument("--nhidden", nargs="*", type=int, help="number of hidden neurons", default=[25])
parser.add_argument("--methods", nargs="*", help="methods to be applied to create weight matrix", default=['scd'])
parser.add_argument("--wgain", type=int, help="optimize the gains as well", default=0)

parser.add_argument("--ntask", type=int, help="number of different tasks to be sampled", default=15)
parser.add_argument("--tasktype", type=str, help="1v1 or 1vall", default='1vall')
parser.add_argument("--reweighted", type=bool, help="whether to use the reweighted matrices", default=False)
parser.add_argument("--symmetrize", type=int, help="whether to symmetrize the network", default=0)

parser.add_argument("--save", type=bool, help="whether to save results or not", default=True)
parser.add_argument("--path", type=str, help="path to which to save the file", default="")
parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")
parser.add_argument("--datasetpath", type=str, help="path where dataset is downloaded", default='/Users/wybo/Data/')
parser.add_argument("--readout", type=str, help="type of readout", default="linear")

parser.add_argument("--nperbatch", type=int, help="number of datapoints per batch", default=100)
parser.add_argument("--nperepoch", type=int, help="number of batches per epoch", default=20)
parser.add_argument("--nepoch", type=int, help="number of epocsh", default=100)
parser.add_argument("--nbfactor", type=int, help="number of test and validation samples, multiplies batch size", default=10)
parser.add_argument("--suffix", type=str, help="suffix for filename", default="")
args = parser.parse_args()


print(args.methods, args.nhidden, args.ntask, args.tasktype, args.readout)

# # construct the data sampler
# source_train = helperfuncs.get_dataset(args.dataset, train=True, path=args.datasetpath)
# source_test = helperfuncs.get_dataset(args.dataset, train=False, path=args.datasetpath)

for n_h in args.nhidden:
    for algo in args.methods:

        if algo == 'rpw':
            w_idx = [0,1]
            b_idx = [0,1]
        elif algo == 'mr' or algo == 'br':
            w_idx = [1]
            b_idx = [1]
        else:
            w_idx = []
            b_idx = [0,1]

        if args.wgain:
            g_idx = [0,1]
        else:
            g_idx = []

        print(w_idx, b_idx, g_idx)
        suffix = helperfuncs.get_suffix(w_idx, b_idx, g_idx)
        print(suffix)

        # set the parameters
        if algo == 'rpw' or algo == 'mr' or algo == 'br':
            lrs = LRS_W
            b_add = B_ADD_W
            b_div = B_DIV_W
            x_div = X_DIV_W
            rotate = False
            if algo == 'mr' or algo == 'br':
                rotate = True
                x_div = X_DIV_GD
        else:
            lrs = LRS_B
            b_add = B_ADD_B
            b_div = B_DIV_B
            x_div = X_DIV_B
            rotate = False
            if algo == 'mrb' or algo == 'brb' or algo == 'bpo' or algo == 'bp':
                rotate = True
                x_div = X_DIV_GD

        # train and test datasets
        source_train = preprocessing.get_dataset(args.dataset, train=True,  rotate=rotate,
                                    x_div=x_div, path=args.datasetpath)
        source_test  = preprocessing.get_dataset(args.dataset, train=False, rotate=rotate,
                                    x_div=x_div, path=args.datasetpath)

        reslist = []

        seed = np.random.randint(50000)
        tasks = helperfuncs.sample_binary_tasks_(args.ntask,
                        dataset=args.dataset, task_type=args.tasktype, seed=seed
            )

        for task in tasks:
            # initialize weights
            w_in  = helperfuncs.get_weight_matrix_in(n_h, algo, dataset=args.dataset, task=task, reweighted=args.reweighted)
            if args.symmetrize:
                w_in = np.concatenate((w_in, w_in), axis=1)
                w_out = helperfuncs.get_weight_matrix_out(2*n_h, algo=algo, dataset=args.dataset, task=task, bias_opt=len(w_idx)==0)
                w_out[n_h:] *= -1.
                ws = [w_in, w_out]
            else:
                w_out = helperfuncs.get_weight_matrix_out(n_h, algo=algo, dataset=args.dataset, task=task, bias_opt=len(w_idx)==0)
                ws = [w_in, w_out]

            # initilize biasses
            if args.symmetrize:
                bs = [np.random.randn(1, 2*n_h) / (2*n_h), np.random.randn(1, 1)]
            else:
                bs = [np.random.randn(1, n_h) / (n_h), np.random.randn(1, 1)]
            bs = [bs[0] / b_div + b_add,
                  bs[1] / b_div + b_add]

            if algo == 'mr' or algo == 'br':
                bs[0] = helperfuncs.get_bias_1(n_h, algo=algo, dataset=args.dataset, task=task)

            print('\nTask:', task)
            data_loaders = helperfuncs.construct_knm_dataloader_triplet(
                            args.dataset, task, copy.deepcopy(task),
                            source_train, source_test,
                            args.nperbatch, args.nperbatch*args.nbfactor, 10000,
                )

            ws_0, bs_0 = ws, bs
            gs_0 = [np.ones_like(b) for b in bs]
            perf = {'train': [], 'test': 0., 'loss': []}
            for lr in lrs:
                print('--> lr =', lr)

                ws_opt, bs_opt, gs_opt, perf_ = optim.run_optim(
                        ws_0, bs_0, data_loaders, gs=gs_0,
                        lr=lr, readout=args.readout, w_idx=w_idx, b_idx=b_idx, g_idx=g_idx, n_epoch=args.nepoch,
                        verbose=True, return_g=True
                    )

                try:
                    assert not np.max(perf_['train']) > np.max(perf['train'])
                except (ValueError, AssertionError) as e:
                    print('\n---------------')
                    if len(perf['train']) > 0:
                        print('new perf train:', np.max(perf_['train']), '<--> old perf train:', np.max(perf['train']))
                    else:
                        print('new perf train:', np.max(perf_['train']))
                    print('new perf test :', perf_['test'], '<--> old perf test:', perf['test'])

                    ws_out, bs_out, gs_out = ws_opt, bs_opt, gs_opt
                    perf['test'] = perf_['test']

                    print(perf['test'])
                    print('---------------\n')

                perf['train'].extend(perf_['train'])
                perf['loss'].extend(perf_['loss'])
                ws_0, bs_0, gs_0 = ws_opt, bs_opt, gs_opt

            reslist.append({'ws': ws_out,
                            'bs': bs_out,
                            'bs': gs_out,
                            'perf': perf,
                            'task': task,
                            'algo': algo,
                            'n_h': n_h})

        rwstr = '_reweighted' if args.reweighted else ''
        symstr = 'sym' if args.symmetrize else ''

        with open(args.path + "biasopt_1hl_%s_%s%s_%s%s%d_%s_ro=%s%s.p"%(suffix, args.dataset, rwstr, algo, symstr, n_h, args.tasktype, args.readout, args.suffix), 'wb') as file:
            pickle.dump(reslist, file)




