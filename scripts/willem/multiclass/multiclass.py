import numpy as np
import torch

import os
import pickle

import sys
sys.path.append('..')

import datarep.paths as paths
import helperfuncs

from biasadaptation.biasfit import multiclassfit

NHS = [500]
ALGO = "pmdd"
DATAPATH = "/Users/wybo/Data/results/biasopt/"
DATASET = "EMNIST"
TASKTYPE = "1vall"
READOUT = "sigmoid10"

dataset_test = helperfuncs.get_dataset(DATASET, train=False, rotate=False)

perfs = []
for n_h in NHS:
    # load binary optimization result file
    print(os.path.join(DATAPATH, "biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(DATASET, ALGO, n_h, TASKTYPE, READOUT)))
    with open(os.path.join(DATAPATH, "biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(DATASET, ALGO, n_h, TASKTYPE, READOUT)), 'rb') as file:
        reslist = pickle.load(file)

    print("avg binary perf:", np.mean([res['perf']['test']['all'] for res in reslist]))

    # construct the task list, corresponding to the class indices
    task = [list(res['task'][-1]['EMNIST'].keys())[0] for res in reslist]

    # construct bias matrices of shape (n_class, n_hidden)
    bs = [np.concatenate([res['bs'][0] for res in reslist], axis=0),
          np.concatenate([res['bs'][1] for res in reslist], axis=0)]

    # initialize weights
    w_in  = helperfuncs.get_weight_matrix_in(n_h, ALGO, dataset=DATASET)
    w_out = helperfuncs.get_weight_matrix_out(n_h, bias_opt=True)
    ws = [w_in, w_out]

    # multiclass relu network
    mc = multiclassfit.MultiClass(ws, bs)
    # data loader
    dl = torch.utils.data.DataLoader(dataset_test, batch_size=10000, shuffle=True)

    xdata, tclass = next(iter(dl))

    # compute class prediction
    out = mc.forward(xdata)
    xclass = torch.FloatTensor([task[idx] for idx in torch.argmin(out, dim=1)])

    print("frac correct:", float(torch.sum(tclass == xclass) / len(tclass)))
