"""
optimizations for one hidden layer
"""
import numpy as np
import torch

import argparse
import pickle
import copy

from datarep import paths

import optim_new, helperfuncs_new


"""
example usage

    python3 biasopt.py --algo pmdd --ntask 1 --tasktype 1vall --nhidden 50 --nepoch 100 --readout tanh

"""


# hardcoded learning rate and readout
LRS = [0.03, 0.01, 0.001]
torch.set_num_threads(1)

# read command line args and kwargs
parser = argparse.ArgumentParser()
parser.add_argument("--nhidden", nargs="*", type=int, help="number of hidden neurons", default=[25])


parser.add_argument("--ntask", type=int, help="number of different tasks to be sampled", default=15)
parser.add_argument("--tasktype", type=str, help="1v1 or 1vall", default='1vall')
parser.add_argument("--algo", type=str, help="methods to be applied to create weight matrix", default='scd')

parser.add_argument("--save", type=bool, help="whether to save results or not", default=True)
parser.add_argument("--path", type=str, help="path to which to save the file", default="")
parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")
parser.add_argument("--readout", type=str, help="type of readout: \'linear\', \'hardtanh\', \'tanh\', \'sigmoidX\'", default="linear")

parser.add_argument("--nperbatch", type=int, help="number of datapoints per batch", default=100)
parser.add_argument("--nperepoch", type=int, help="number of batches per epoch", default=20)
parser.add_argument("--nepoch", type=int, help="number of epocsh", default=100)
parser.add_argument("--nbfactor", type=int, help="number of test and validation samples, multiplies batch size", default=10)
args = parser.parse_args()


print(args.algo, args.nhidden, args.ntask, args.tasktype, args.readout)

# construct the data sampler
source_train = helperfuncs_new.get_dataset(args.dataset, train=True, rotate=False)
source_test = helperfuncs_new.get_dataset(args.dataset, train=False, rotate=False)

for n_h in args.nhidden:
    w_idx = []

    reslist = []

    seed = np.random.randint(50000)
    tasks = helperfuncs_new.sample_binary_tasks_(args.ntask,
                    dataset=args.dataset, task_type=args.tasktype, seed=seed
        )

    for task in tasks:
        # initialize weights
        w_in  = helperfuncs_new.get_weight_matrix_in(n_h, args.algo, dataset=args.dataset, task=task)
        w_out = helperfuncs_new.get_weight_matrix_out(n_h, algo=args.algo, dataset=args.dataset, task=task, bias_opt=len(w_idx)==0)
        ws = [w_in, w_out]

        # initilize biasses
        bs = [np.random.randn(1, n_h) / (n_h), np.random.randn(1, 1)]

        print('\nTask:', task)
        data_loaders = helperfuncs_new.construct_knm_dataloader_triplet(
                        args.dataset, task, copy.deepcopy(task),
                        source_train, source_test,
                        args.nperbatch, args.nperbatch*args.nbfactor, 10000,
            )

        ws_0, bs_0 = ws, bs
        perf = {'train': [], 'test': 0., 'loss': []}
        for lr in LRS:
            print('--> lr =', lr)

            ws_opt, bs_opt, gs_opt, perf_ = optim_new.run_optim(
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
                        'algo': args.algo,
                        'n_h': n_h})

    with open(args.path + "biasopt_newparams_1hl_%s_%s%d_%s_ro=%s.p"%(args.dataset, args.algo, n_h, args.tasktype, args.readout), 'wb') as file:
        pickle.dump(reslist, file)




