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
import codedataset

"""
example usage

    python3 codeopt.py --algow sc --algoc lstsq --ntask 1 --tasktype 1vall --nhidden 100 --readout tanh

"""


# hardcoded learning rate and readout
# LRS = [0.005, 0.001, 0.0001]
LRS = [0.03, 0.01, 0.005]
B_ADD = 0.
B_DIV = 8.
# READOUT = "sigmoid1"#"linear"#hardtanh"#
torch.set_num_threads(1)

# read command line args and kwargs
parser = argparse.ArgumentParser()
parser.add_argument("--nhidden", type=int, help="number of hidden neurons", default=25)
parser.add_argument("--algow", type=str, help="methods to be applied to create weight matrix", default='scd')
parser.add_argument("--algoc", type=str, help="methods to be applied to create weight matrix", default='sc')

parser.add_argument("--ntask", type=int, help="number of different tasks to be sampled", default=15)
parser.add_argument("--tasktype", type=str, help="1v1 or 1vall", default='1vall')

parser.add_argument("--save", type=bool, help="whether to save results or not", default=True)
parser.add_argument("--path", type=str, help="path to which to save the file", default="")
parser.add_argument("--datapath", type=str, help="path to which to save the file", default="/users/wybo/Data/code_matrices/")
parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")
parser.add_argument("--readout", type=str, help="type of readout", default="linear")

parser.add_argument("--nperbatch", type=int, help="number of datapoints per batch", default=100)
parser.add_argument("--nperepoch", type=int, help="number of batches per epoch", default=20)
parser.add_argument("--nepoch", type=int, help="number of epocsh", default=100)
parser.add_argument("--nbfactor", type=int, help="number of test and validation samples, multiplies batch size", default=10)
args = parser.parse_args()

n_h = args.nhidden
algo_w = args.algow
algo_c = args.algoc

# construct the data sampler
source_train = codedataset.CodeDataset(n_h, algo_w=algo_w, dataset=args.dataset, algo_c=algo_c, path=args.datapath, train=True)
source_test = codedataset.CodeDataset(n_h, algo_w=algo_w, dataset=args.dataset, algo_c=algo_c, path=args.datapath, train=False)

reslist = []

seed = np.random.randint(50000)
tasks = helperfuncs.sample_binary_tasks_(args.ntask,
                dataset=args.dataset, task_type=args.tasktype, seed=seed
    )

for task in tasks:
    # initialize weights
    w_out = helperfuncs.get_weight_matrix_out(n_h, bias_opt=False)
    ws = [w_out]

    # initilize biasses
    bs = [np.random.randn(1, 1) / B_DIV + B_ADD]

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
                lr=lr, readout=args.readout, w_idx=[0], n_epoch=args.nepoch,
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
                    'algo_w': algo_w,
                    'algo_c': algo_c,
                    'n_h': n_h})

with open(args.path + "codeopt_%s_algow-c=%s-%s_nh=%d_%s__ro=%s.p"%(args.dataset, algo_w, algo_c, n_h, args.tasktype, args.readout), 'wb') as file:
    pickle.dump(reslist, file)




