"""
optimizations for two hidden layers
"""
import numpy as np
import torch

import argparse
import pickle
import copy
import os
import sys
sys.path.append('..')

import optim, helperfuncs


from datarep import paths

"""
example usage
  - take the first layer biases as fixed
    python3 fixedbiasopt.py --algo1 pmdd --algo2 scdfs --algoc ha --ntask 1 --tasktype 1vall --nhidden1 25 --readout tanh --nhidden2 25 --prefix 1hl --biases 1 2 --order 1 --nepoch 100
  - take the second and output layer biases as fixed
    python3 fixedbiasopt.py --algo1 pmdd --algo2 scdfs --algoc ha --ntask 1 --tasktype 1vall --nhidden1 25 --readout tanh --nhidden2 25 --prefix 2hl-w-b12 --biases 0 --order 2 --nepoch 100
  - take nothing as fixed
    python3 fixedbiasopt.py --algo1 pmdd --algo2 scdfs --algoc ha --ntask 1 --tasktype 1vall --nhidden1 25 --readout tanh --nhidden2 25 --biases 0 1 2 --order 1 --nepoch 100

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

parser.add_argument("--biases", nargs="*", type=int, help="bias layers to optimize", default=[0,1,2])
parser.add_argument("--prefix", type=str, help="file from which to take fixed biases", default="1hl")
parser.add_argument("--order", type=int, help="order number for initial conditions", default=0)
parser.add_argument("--weights", nargs="*", type=int, help="weight layers to optimize", default=[])

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
algo_c = args.algoc
if algo_2 == 'bp' or algo_2 == 'bpo' or 'rp' in algo_2:
    algo_c = 'na'

if args.order > 1:
    args.prefix += "-o%d"%(args.order-1)

def get_file_name(prefix, n_h1, n_h2):
    if "1hl" in prefix:
        fname = "biasopt_%s_%s_%s%d_%s_ro=%s.p"%(prefix, args.dataset, algo_1, n_h1, args.tasktype, args.readout)

    elif "2hl" in prefix:
        fname = "deepopt_%s_%s_algo12c=%s-%s-%s_nh12=%d-%d_%s_ro=%s.p"%(prefix, args.dataset, algo_1, algo_2, algo_c, n_h1, n_h2, args.tasktype, args.readout)

    return fname


def get_weight_and_bias(task, file_name, layer_idx):
    class_idx = list(task[-1][args.dataset].keys())[0]

    # load the 1hl data file
    with open(os.path.join(paths.result_path, "biasopt/", file_name) , 'rb') as file:
        reslist = pickle.load(file)

    for res in reslist:
        cres_idx = list(res['task'][-1][args.dataset].keys())[0]

        if cres_idx == class_idx:
            w = res['ws'][layer_idx]
            b = res['bs'][layer_idx]

            break

    return w, b


for n_h1 in args.nhidden1:
    for n_h2 in args.nhidden2:
        # w_idx = [0,1,2] if (algo_1 == 'rpw' or algo_2 == 'rpw') else []
        w_idx = args.weights

        b_idx = args.biases
        bfix_idx = [l_idx for l_idx in [0,1,2] if l_idx not in b_idx]

        prefix = args.prefix + \
                 "w" + "".join([str(idx) for idx in w_idx]) + \
                "-b" + "".join([str(idx) for idx in b_idx])

        parent_file_name = get_file_name(args.prefix, n_h1, n_h2)

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
            # load or instantiate weights or biases
            if 0 in bfix_idx:
                w_in, b_in = get_weight_and_bias(task, parent_file_name, 0)
            else:
                w_in = helperfuncs.get_weight_matrix_in(n_h1, algo_1, dataset=args.dataset, task=task)
                b_in = np.random.randn(1, n_h1) / n_h1
                b_in = b_in / b_div + b_add

            if 1 in bfix_idx:
                w_hid, b_hid = get_weight_and_bias(task, parent_file_name, 1)
            else:
                w_hid = helperfuncs.get_weight_matrix_hidden(n_h1, n_h2, algo_1, algo_2, algo_c, dataset=args.dataset, task=task, reweighted=args.reweighted, enriched=args.enriched)
                b_hid = np.random.randn(1, n_h2) / n_h2
                b_hid = b_hid / b_div + b_add

            if 2 in bfix_idx:
                w_out, b_out = get_weight_and_bias(task, parent_file_name, 2)
            else:
                w_out = helperfuncs.get_weight_matrix_out(n_h2, bias_opt=len(w_idx)==0)
                b_out = np.random.randn(1, 1) / b_div + b_add

            # initialize weights
            ws = [w_in, w_hid, w_out]
            for w in ws: print(w.shape)

            # initilize biasses
            bs = [b_in, b_hid, b_out]

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
                            'n_h2': n_h2,
                            'parent': parent_file_name})

        if args.reweighted:
            rwstr = '_reweighted'
        elif args.enriched:
            rwstr = '_enriched'
        else:
            rwstr = ''
        optstring = "w" + "".join([str(idx) for idx in w_idx]) + \
                   "-b" + "".join([str(idx) for idx in b_idx])
        if args.order > 0:
            optstring += "-o%d"%args.order
        namestring = "deepopt_2hl-%s_%s%s_algo12c=%s-%s-%s_nh12=%d-%d_%s_ro=%s.p"%(optstring, args.dataset, rwstr, algo_1, algo_2, algo_c, n_h1, n_h2, args.tasktype, args.readout)
        with open(args.path + namestring, 'wb') as file:
            pickle.dump(reslist, file)

