
import numpy as np

import os
import pickle

DATAPATH = "/Users/wybo/Data/results/biasopt/"
DATASET = "EMNIST"
TASKTYPE = "1vall"
READOUT = "tanh"
SUFFIX=''


fname = os.path.join(DATAPATH,
                     "linopt_1hl_%s_%s_ro=%s%s.p"%(DATASET, TASKTYPE, READOUT, SUFFIX))
try:
    with open(fname, 'rb') as file:
        reslist = pickle.load(file)
    # print([res['perf']['test'] for res in reslist])
    perf = np.mean([res['perf']['test']['all'] for res in reslist])
except FileNotFoundError:
    perf = np.nan

print('perf =', perf)