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

    python3 linopt.py --ntask 1 --tasktype 1vall --nhidden 50 --nepoch 100 --readout tanh

"""


# # original parameters
# LRS_0 = [0.005, 0.001, 0.0001]
# X_DIV_0 = 'data'
# B_DIV_0 = 10.
# B_ADD_0 = 0.
# # optimized parameters bias adaptation
# LRS_B = [0.06, 0.02, 0.008]
# # LRS_B = [0.1, 0.02, 0.008]
# X_DIV_B = .5
# B_DIV_B = 5.
# B_ADD_B = -.5
# # optimized parameters bias adaptation from gradient descent matrices
# X_DIV_GD = 1.
# optimized parameters weight optimization
LRS_W = [0.002, 0.001, 0.0008]
X_DIV_W = 4.
B_DIV_W = 10.
B_ADD_W = 0.

# READOUT = "sigmoid1"#"linear"#hardtanh"#
torch.set_num_threads(1)

# read command line args and kwargs
parser = argparse.ArgumentParser()
parser.add_argument("--ntask", type=int, help="number of different tasks to be sampled", default=15)
parser.add_argument("--tasktype", type=str, help="1v1 or 1vall", default='1vall')

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


print(args.ntask, args.tasktype, args.readout)


w_idx = [0]
b_idx = [0]

lrs = LRS_W
b_add = B_ADD_W
b_div = B_DIV_W
x_div = X_DIV_W
rotate = False

# train and test datasets
source_train = helperfuncs.get_dataset(args.dataset, train=True,  rotate=rotate,
                            x_div=x_div, path=args.datasetpath)
source_test  = helperfuncs.get_dataset(args.dataset, train=False, rotate=rotate,
                            x_div=x_div, path=args.datasetpath)

reslist = []

seed = np.random.randint(50000)
tasks = helperfuncs.sample_binary_tasks_(args.ntask,
                dataset=args.dataset, task_type=args.tasktype, seed=seed
    )

for task in tasks:
    # initialize weights
    w_in  = helperfuncs.get_weight_matrix_in(1, 'rpw', dataset=args.dataset, task=task)
    ws = [w_in]

    # initilize biasses
    bs = [np.random.randn(1, 1) / b_div + b_add]

    print('\nTask:', task)
    data_loaders = helperfuncs.construct_knm_dataloader_triplet(
                    args.dataset, task, copy.deepcopy(task),
                    source_train, source_test,
                    args.nperbatch, args.nperbatch*args.nbfactor, 10000,
        )

    ws_0, bs_0 = ws, bs
    perf = {'train': [], 'test': 0., 'loss': []}
    for lr in lrs:
        print('--> lr =', lr)

        ws_opt, bs_opt, gs_opt, perf_ = optim.run_optim(
                ws_0, bs_0, data_loaders,
                lr=lr, readout=args.readout, w_idx=w_idx, b_idx=b_idx, n_epoch=args.nepoch,
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
                    'task': task})

with open(args.path + "linopt_1hl_%s_%s_ro=%s%s.p"%(args.dataset, args.tasktype, args.readout, args.suffix), 'wb') as file:
    pickle.dump(reslist, file)




