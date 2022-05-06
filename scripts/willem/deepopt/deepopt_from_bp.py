"""
optimizations for one hidden layer
"""
import numpy as np
import torch

import argparse
import pickle
import copy

import sys
sys.path.append('..')

import optim, helperfuncs

"""
example usage

    python3 deepopt_from_bp.py --algo1 pmdd --ntask 1 --nhidden1 100 --nhidden2 100 --nepoch 100

"""


# hardcoded learning rate and readout
LRS = [0.005, 0.001, 0.0001]
ALGO_2 = "bp"
torch.set_num_threads(1)

# read command line args and kwargs
parser = argparse.ArgumentParser()
parser.add_argument("--nhidden1", nargs="*", type=int, help="number of hidden neurons in first layer", default=[25])
parser.add_argument("--nhidden2", nargs="*", type=int, help="number of hidden neurons in second layer", default=[25])

parser.add_argument("--algo1", type=str, help="algorithm for input weight matrix", default='scd')
parser.add_argument("--ntask", type=int, help="number of different tasks to be sampled", default=15)
parser.add_argument("--tasktype", type=str, help="1v1 or 1vall", default='1vall')

parser.add_argument("--save", type=bool, help="whether to save results or not", default=True)
parser.add_argument("--path", type=str, help="path to which to save the file", default="")
parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")
parser.add_argument("--readout", type=str, help="type of readout", default="linear")

parser.add_argument("--nperbatch", type=int, help="number of datapoints per batch", default=100)
parser.add_argument("--nperepoch", type=int, help="number of batches per epoch", default=20)
parser.add_argument("--nepoch", type=int, help="number of epocsh", default=100)
parser.add_argument("--nbfactor", type=int, help="number of test and validation samples, multiplies batch size", default=10)
args = parser.parse_args()


# print(args.methods, args.nhidden, args.ntask, args.tasktype)

# construct the data sampler
source_train = helperfuncs.get_dataset(args.dataset, train=True)
source_test = helperfuncs.get_dataset(args.dataset, train=False)

algo_1 = args.algo1
algo_2 = ALGO_2
algo_c = "na"

for n_h1 in args.nhidden1:
    for n_h2 in args.nhidden2:
        w_idx = []

        reslist = []

        seed = np.random.randint(50000)
        tasks = helperfuncs.sample_binary_tasks_(args.ntask,
                        dataset=args.dataset, task_type=args.tasktype, seed=seed
            )

        for task in tasks:

            # initialize weights
            w_in  = helperfuncs.get_weight_matrix_in(n_h1, algo_1, dataset=args.dataset, task=task)
            w_hid = helperfuncs.get_weight_matrix_hidden(n_h1, n_h2, algo_1, algo_2, algo_c, dataset=args.dataset, task=task)
            w_out = helperfuncs.get_weight_matrix_out(n_h2, bias_opt=len(w_idx)==0)
            ws = [w_in, w_hid, w_out]
            for w in ws: print(w.shape)

            # initilize biasses
            bs = [np.random.randn(1, n_h1) / (10.*n_h1),
                  np.random.randn(1, n_h2) / (10.*n_h2),
                  np.random.randn(1, 1) / 10.]

            print('\nTask:', task)
            data_loaders = helperfuncs.construct_knm_dataloader_triplet(
                            args.dataset, task, copy.deepcopy(task),
                            source_train, source_test,
                            args.nperbatch, args.nperbatch*args.nbfactor, 10000,
                )

            ws_0, bs_0 = ws, bs
            perf = {'train': [], 'test': 0., 'loss': []}
            for lr in LRS:
                print('--> lr =', lr)

                ws_opt, bs_opt, gs_opt, perf_ = optim.run_optim(
                        ws_0, bs_0, data_loaders,
                        lr=lr, readout=args.readout, w_idx=w_idx, n_epoch=args.nepoch,
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


            reslist.append({'ws': ws_out,
                            'bs': bs_out,
                            'gs': gs_out,
                            'perf': perf,
                            'task': task,
                            'algo1': algo_1,
                            'algo2': algo_2,
                            'algoc': algo_c,
                            'n_h1': n_h1,
                            'n_h2': n_h2})

        namestring = "deepopt_2hl_%s_algo12c=%s-%s-%s_nh12=%d-%d_%s_ro=%s.p"%(args.dataset, algo_1, algo_2, algo_c, n_h1, n_h2, args.tasktype, args.readout)
        with open(args.path + namestring, 'wb') as file:
            pickle.dump(reslist, file)




