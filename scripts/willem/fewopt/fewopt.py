"""
optimizations for one hidden layer
"""
import numpy as np
import torch

import argparse
import pickle
import sys
sys.path.append('..')

import optim, helperfuncs

"""
example usage

    python3 fewopt.py --methods scd --ntask 1 --tasktype 1vall --nhidden 50 --nsample 10 --readout tanh --datasetpath /Users/wybo/Data/
"""


# hardcoded learning rate and readout
# LRS = [0.001]#[0.005, 0.001, 0.0001]
# READOUT = "sigmoid10"#"linear"#"hardtanh"#

# original parameters
LRS_0 = [0.005, 0.001, 0.0001]
X_DIV_0 = 'data'
B_DIV_0 = 10.
B_ADD_0 = 0.
# optimized parameters bias adaptation
LRS_B = [0.06, 0.02, 0.008]
X_DIV_B = .5
B_DIV_B = 5.
B_ADD_B = -.5
# optimized parameters bias adaptation from gradient descent matrices
X_DIV_GD = 1.
# optimized parameters weight optimization
LRS_W = [0.002, 0.001, 0.0008]
X_DIV_W = 4.
B_DIV_W = 10.
B_ADD_W = 0.

# read command line args and kwargs
parser = argparse.ArgumentParser()
parser.add_argument("--nhidden", nargs="*", type=int, help="number of hidden neurons", default=[10, 25, 50, 100, 250, 500])
parser.add_argument("--methods", nargs="*", help="methods to be applied to create weight matrix", default='all')

parser.add_argument("--ntask", type=int, help="number of different tasks to be sampled", default=15)
parser.add_argument("--nsample", type=int, help="number of samples for each task category", default=1000000)
parser.add_argument("--tasktype", type=str, help="1v1 or 1vall", default='1v1')
parser.add_argument("--readout", type=str, help="readout function", default='linear')

parser.add_argument("--save", type=bool, help="whether to save results or not", default=True)
parser.add_argument("--path", type=str, help="path to which to save the file", default="")
parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")
parser.add_argument("--datasetpath", type=str, help="path where dataset is downloaded", default='/work/users/wybo/Data/')

parser.add_argument("--nperbatch", type=int, help="number of datapoints per batch", default=100)
parser.add_argument("--nperepoch", type=int, help="number of batches per epoch", default=20)
parser.add_argument("--nepoch", type=int, help="number of epocsh", default=100)
parser.add_argument("--nbfactor", type=int, help="number of test and validation samples, multiplies batch size", default=10)
args = parser.parse_args()


# construct the data sampler
source_train = helperfuncs.get_dataset(args.dataset, train=True)
source_test = helperfuncs.get_dataset(args.dataset, train=False)


# suffix 'bias' --> bias adaptation only
# suffix 'full' --> weight and bias optimization
for n_h in args.nhidden:
    for algo in args.methods:
        w_idx = [0,1] if algo == 'rpw' else []

        if algo == 'rpw':
            w_idx = [0,1]
            b_idx = [0,1]
        elif algo == 'mr' or algo == 'br':
            w_idx = [1]
            b_idx = [1]
        else:
            w_idx = []
            b_idx = [0,1]

        print(w_idx, b_idx)

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
        source_train_bias = helperfuncs.get_dataset(args.dataset, train=True,  rotate=rotate,
                                    x_div=X_DIV_B, path=args.datasetpath)
        source_test_bias  = helperfuncs.get_dataset(args.dataset, train=False, rotate=rotate,
                                    x_div=X_DIV_B, path=args.datasetpath)
        source_train_full = helperfuncs.get_dataset(args.dataset, train=True,  rotate=rotate,
                                    x_div=X_DIV_W, path=args.datasetpath)
        source_test_full  = helperfuncs.get_dataset(args.dataset, train=False, rotate=rotate,
                                    x_div=X_DIV_W, path=args.datasetpath)

        reslist_bias = []
        reslist_full = []

        seed = np.random.randint(10000)
        tasks_train = helperfuncs.sample_binary_tasks_(args.ntask, nsample=args.nsample,
                                                                  dataset=args.dataset, task_type=args.tasktype, seed=seed)
        tasks_test = helperfuncs.sample_binary_tasks_(args.ntask, dataset=args.dataset, task_type=args.tasktype, seed=seed)


        for task_train, task_test in zip(tasks_train, tasks_test):

            # initialize weights
            w_in  = helperfuncs.get_weight_matrix_in(n_h, algo, dataset=args.dataset, task=task_train)
            w_out = helperfuncs.get_weight_matrix_out(n_h, algo=algo, dataset=args.dataset, task=task_train, bias_opt=len(w_idx)==0)
            ws_bias = [
                    w_in,
                    w_out
                ]
            ws_full = [
                    helperfuncs.get_weight_matrix_in(n_h, 'rp', dataset=args.dataset),
                    helperfuncs.get_weight_matrix_out(n_h)
                ]

            # initilize biasses
            bs = [np.random.randn(1, n_h) / n_h, np.random.randn(1, 1)]
            bs_bias = [bs[0] / B_DIV_B + B_ADD_B,
                       bs[1] / B_DIV_B + B_ADD_B]

            if algo == 'mr' or algo == 'br':
                bs[0] = helperfuncs.get_bias_1(n_h, algo=algo, dataset=args.dataset, task=task_train)

            bs = [np.random.randn(1, n_h) / n_h, np.random.randn(1, 1)]
            bs_full = [bs[0] / B_DIV_W + B_ADD_W,
                       bs[1] / B_DIV_W + B_ADD_W]

            print('\n', task_train)
            print(task_test)

            data_loaders_bias = helperfuncs.construct_knm_dataloader_triplet(
                                args.dataset, task_train, task_test,
                                source_train_bias, source_test_bias,
                                args.nperbatch, args.nperbatch*args.nbfactor, 10000,
                )

            data_loaders_full = helperfuncs.construct_knm_dataloader_triplet(
                                args.dataset, task_train, task_test,
                                source_train_full, source_test_full,
                                args.nperbatch, args.nperbatch*args.nbfactor, 10000,
                )

            ws_0, bs_0 = ws_bias, bs_bias
            perf = {'train': [], 'test': 0., 'loss': []}
            for lr in LRS_B:
                print('--> lr =', lr)

                ws_opt, bs_opt, gs_opt, perf_ = optim.run_optim(
                        ws_0, bs_0, data_loaders_bias,
                        lr=lr, readout=args.readout, w_idx=[],
                        n_epoch=args.nepoch, n_batch_per_epoch=args.nperepoch,
                        verbose=True
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
                ws_0, bs_0 = ws_opt, bs_opt

            reslist_bias.append({'ws': ws_opt,
                                 'bs': bs_opt,
                                 'gs': gs_opt,
                                 'perf': perf,
                                 'task': task_train,
                                 'algo': algo,
                                 'n_h': n_h})


            ws_0, bs_0 = ws_full, bs_full
            perf = {'train': [], 'test': 0., 'loss': []}
            for lr in LRS_W:
                print('--> lr =', lr)

                ws_opt, bs_opt, gs_opt, perf_ = optim.run_optim(
                        ws_0, bs_0, data_loaders_full,
                        lr=lr, readout=args.readout, w_idx=[0,1],
                        n_epoch=args.nepoch, n_batch_per_epoch=args.nperepoch,
                        verbose=True
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
                ws_0, bs_0 = ws_opt, bs_opt

            reslist_full.append({'ws': ws_opt,
                                 'bs': bs_opt,
                                 'gs': gs_opt,
                                 'perf': perf,
                                 'task': task_train,
                                 'algo': algo,
                                 'n_h': n_h})

        if args.save:
            with open(args.path + "fewopt_1hl_%s_%s%d_%s_nsample=%d_ro=%s.p"%(args.dataset, algo, n_h, args.tasktype, args.nsample, args.readout), 'wb') as file:
                pickle.dump(reslist_bias, file)
                pickle.dump(reslist_full, file)

