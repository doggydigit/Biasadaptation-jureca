"""
optimizations for two hidden layers
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

    python3 deepopt3.py --algo1 scd --algo2 sc --algo3 sc --algoc sc --ntask 1 --tasktype 1vall --nhidden1 100 --nhidden2 100 --nhidden3 100 --readout tanh  --nepoch 5
    python3 deepopt3.py --algo1 br --algo2 br --algo3 br --algoc na --ntask 1  --nhidden1 100 --nhidden2 100 --nhidden3 100 --readout tanh  --nepoch 5

"""


# hardcoded learning rate and readout
# LRS = [0.005, 0.001, 0.0001]
# READOUT = "linear"#"hardtanh"#
torch.set_num_threads(1)

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
parser.add_argument("--nhidden1", nargs="*", type=int, help="number of hidden neurons in first layer", default=[25])
parser.add_argument("--nhidden2", nargs="*", type=int, help="number of hidden neurons in second layer", default=[25])
parser.add_argument("--nhidden3", nargs="*", type=int, help="number of hidden neurons in second layer", default=[25])

parser.add_argument("--algo1", type=str, help="algorithm for input weight matrix", default='scd')
parser.add_argument("--algo2", type=str, help="algorithm for hidden weight matrix", default='sc')
parser.add_argument("--algo3", type=str, help="algorithm for coordinate transform", default='sc')
parser.add_argument("--algoc", type=str, help="algorithm for coordinate transform", default='sc')

parser.add_argument("--ntask", type=int, help="number of different tasks to be sampled", default=15)
parser.add_argument("--tasktype", type=str, help="1v1 or 1vall", default='1vall')
parser.add_argument("--readout", type=str, help="linear, hardtanh, sigmoid1, sigmoid5 or sigmoid10", default='linear')

parser.add_argument("--save", type=bool, help="whether to save results or not", default=True)
parser.add_argument("--path", type=str, help="path to which to save the file", default="")
parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")
parser.add_argument("--datasetpath", type=str, help="path where dataset is downloaded", default='/Users/wybo/Data/')

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
algo_2 = args.algo2
algo_3 = args.algo3
algo_c = args.algoc
if 'rp' in algo_3:
    algo_c_ = 'na'
else:
    algo_c_ = algo_c

algos = [algo_1, algo_2, algo_3]
if 'mrb' in algos or 'brb' in algos or 'bpo' in algos or 'bp' in algos or 'mr' in algos or 'br' in algos:
    assert algo_1 == algo_2
    assert algo_1 == algo_3
    algo_c = "na"
    algo_c_ = "na"

print('!!! ', algo_c, algo_c_)


for n_h1 in args.nhidden1:
    for n_h2 in args.nhidden2:
        for n_h3 in args.nhidden3:
            if 'rpw' in algos:
                w_idx = [0,1,2,3]
                b_idx = [0,1,2,3]
            elif 'mr' in algos or 'br' in algos:
                w_idx = [3]
                b_idx = [3]
            else:
                w_idx = []
                b_idx = [0,1,2,3]

            # set the parameters
            if 'rpw' in algos or 'mr' in algos or 'br' in algos:
                lrs = LRS_W
                b_add = B_ADD_W
                b_div = B_DIV_W
                x_div = X_DIV_W
                rotate = False
                if 'mr' in algos or 'br' in algos:
                    rotate = True
            else:
                lrs = LRS_B
                b_add = B_ADD_B
                b_div = B_DIV_B
                x_div = X_DIV_B
                # rotate = True if algo == 'bpo' or algo == 'bp' else False
                rotate = False
                if 'mrb' in algos or 'brb' in algos or'bpo' in algos or 'bp' in algos:
                    rotate = True
                    x_div = X_DIV_GD


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
                if 'mrb' in algos or 'brb' in algos or 'bpo' in algos or 'bp' in algos or 'mr' in algos or 'br' in algos:
                    ws = helperfuncs.get_multitask_weights(n_h1, n_h2, n_h3, algo=algos[0], dataset=args.dataset, task=task)
                    if 'br' in algos or 'mr' in algos:
                        w_vec = np.random.randn(n_h3,1)
                        w_vec /= np.linalg.norm(w_vec)
                        ws.append(w_vec)
                else:
                    w_1 = helperfuncs.get_weight_matrix_in(n_h1, algo_1, dataset=args.dataset, task=task)
                    w_2 = helperfuncs.get_weight_matrix_hidden(n_h1, n_h2, algo_1, algo_2, algo_c, dataset=args.dataset, task=task)
                    w_3 = helperfuncs.get_weight_matrix_hidden3(n_h1, n_h2, n_h3, algo_1, algo_2, algo_3, algo_c_, dataset=args.dataset)
                    w_o = helperfuncs.get_weight_matrix_out(n_h3, bias_opt=len(w_idx)==0)
                    ws = [w_1, w_2, w_3, w_o]
                for w in ws: print(w.shape)

                if 'br' in algos or 'mr' in algos:
                    bs = helperfuncs.get_multitask_biasses(n_h1, n_h2, n_h3, algo=algos[0], dataset=args.dataset, task=task)
                else:
                    # initilize biasses
                    bs = [np.random.randn(1, n_h1) / n_h1,
                          np.random.randn(1, n_h2) / n_h2,
                          np.random.randn(1, n_h3) / n_h3,
                          np.random.randn(1, 1)]
                    bs = [bs[0] / b_div + b_add,
                          bs[1] / b_div + b_add,
                          bs[2] / b_div + b_add,
                          bs[3] / b_div + b_add]

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
                                'task': task,
                                'algo1': algo_1,
                                'algo2': algo_2,
                                'algo3': algo_3,
                                'algoc': algo_c,
                                'n_h1': n_h1,
                                'n_h2': n_h2,
                                'n_h3': n_h3})

            namestring = "deepopt_3hl_%s_algo123c=%s-%s-%s-%s_nh12=%d-%d-%d_%s_ro=%s.p"%(args.dataset, algo_1, algo_2, algo_3, algo_c_, n_h1, n_h2, n_h3, args.tasktype, args.readout)
            with open(args.path + namestring, 'wb') as file:
                pickle.dump(reslist, file)




