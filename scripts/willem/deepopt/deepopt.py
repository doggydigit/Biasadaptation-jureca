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
from biasadaptation.utils import preprocessing


"""
example usage

    python3 deepopt.py --algo1 scd --algo2 sc --algoc sc --ntask 1 --tasktype 1vall --nhidden1 100 --readout tanh --nhidden2 100 --nepoch 100

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
# optimized parameters weight optimization
LRS_W = [0.002, 0.001, 0.0008]
X_DIV_W = 4.
B_DIV_W = 10.
B_ADD_W = 0.


# read command line args and kwargs
parser = argparse.ArgumentParser()
parser.add_argument("--nhidden1", nargs="*", type=int, help="number of hidden neurons in first layer", default=[25])
parser.add_argument("--nhidden2", nargs="*", type=int, help="number of hidden neurons in second layer", default=[25])

parser.add_argument("--algo1", type=str, help="algorithm for input weight matrix", default='scd')
parser.add_argument("--algo2", type=str, help="algorithm for hidden weight matrix", default='sc')
parser.add_argument("--algoc", type=str, help="algorithm for coordinate transform", default='sc')
parser.add_argument("--reweighted", type=bool, help="whether to use the reweighted matrices", default=False)
parser.add_argument("--enriched", type=bool, help="whether to use the enriched matrices", default=False)
parser.add_argument("--symmetrize", type=int, help="whether to symmetrize the network", default=0)

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


symstr = 'sym' if args.symmetrize else ''

algo_1 = args.algo1
algo_2 = args.algo2
algo_c = args.algoc
if algo_2 == 'bp' or algo_2 == 'bpo' or 'rp' in algo_2:
    algo_c = 'na'

for n_h1 in args.nhidden1:
    for n_h2 in args.nhidden2:
        w_idx = [0,1,2] if (algo_1 == 'rpw' or algo_2 == 'rpw') else []

        # set the parameters
        if algo_1 == 'rpw' or algo_2 == 'rpw':
            lrs = LRS_W
            b_add = B_ADD_W
            b_div = B_DIV_W
            x_div = X_DIV_W
            rotate = False
        else:
            lrs = LRS_B
            b_add = B_ADD_B
            b_div = B_DIV_B
            x_div = X_DIV_B
            # rotate = True if algo == 'bpo' or algo == 'bp' else False
            rotate = False


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
            w_in  = helperfuncs.get_weight_matrix_in(n_h1, algo_1, dataset=args.dataset, task=task)
            w_hid = helperfuncs.get_weight_matrix_hidden(n_h1, n_h2, algo_1+symstr, algo_2, algo_c, dataset=args.dataset, task=task,
                                                         reweighted=args.reweighted, enriched=args.enriched)
            # w_out = helperfuncs.get_weight_matrix_out(n_h2, bias_opt=len(w_idx)==0)
            # ws = [w_in, w_hid, w_out]

            if args.symmetrize:
                w_in = np.concatenate((w_in, w_in), axis=1)
                if 'rp' in algo_2:
                    w_hid = helperfuncs.get_weight_matrix_hidden(2*n_h1, n_h2, algo_1+symstr, algo_2, algo_c, dataset=args.dataset, task=task,
                                                                 reweighted=args.reweighted, enriched=args.enriched)
                w_hid = np.concatenate((w_hid, w_hid), axis=1)
                w_out = helperfuncs.get_weight_matrix_out(2*n_h2, algo=algo_2, dataset=args.dataset, task=task, bias_opt=len(w_idx)==0)
                w_out[n_h2:] *= -1.
                ws = [w_in, w_hid, w_out]
            else:
                w_out = helperfuncs.get_weight_matrix_out(n_h2, algo=algo_2, dataset=args.dataset, task=task, bias_opt=len(w_idx)==0)
                ws = [w_in, w_hid, w_out]
            for w in ws: print(w.shape)

            # initilize biasses

            if args.symmetrize:

                bs = [np.random.randn(1, 2*n_h1) / (2*n_h1),
                      np.random.randn(1, 2*n_h2) / (2*n_h2),
                      np.random.randn(1, 1)]
            else:
                bs = [np.random.randn(1, n_h1) / n_h1,
                      np.random.randn(1, n_h2) / n_h2,
                      np.random.randn(1, 1)]
            bs = [bs[0] / b_div + b_add,
                  bs[1] / b_div + b_add,
                  bs[2] / b_div + b_add]
            for b in bs: print(b.shape)

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

                ws_opt, bs_opt, perf_ = optim.run_optim(
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

                    ws_out, bs_out = ws_opt, bs_opt
                    perf['test'] = perf_['test']

                    print(perf['test'])
                    print('---------------\n')

                perf['train'].extend(perf_['train'])
                perf['loss'].extend(perf_['loss'])
                ws_0, bs_0 = ws_opt, bs_opt


            reslist.append({
                            'ws': ws_out,
                            'bs': bs_out,
                            'perf': perf,
                            'task': task,
                            'algo1': algo_1,
                            'algo2': algo_2,
                            'algoc': algo_c,
                            'n_h1': n_h1,
                            'n_h2': n_h2})

        if args.reweighted:
            rwstr = '_reweighted'
        elif args.enriched:
            rwstr = '_enriched'
        else:
            rwstr = ''
        namestring = "deepopt_2hl_%s%s_algo12c=%s%s-%s%s-%s_nh12=%d-%d_%s_ro=%s.p"%(args.dataset, rwstr, algo_1, symstr, algo_2, symstr, algo_c, n_h1, n_h2, args.tasktype, args.readout)
        with open(args.path + namestring, 'wb') as file:
            pickle.dump(reslist, file)




