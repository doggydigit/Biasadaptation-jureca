import numpy as np

import os
import pickle
import glob


import sys
sys.path.append('..')
sys.path.insert(0, '../biasopt/metaopt')
# sys.path.insert(0,'/home/anand/')

import optim_new, helperfuncs
import codedataset
import metaopt, deapopt, tools, algorithms

from datarep.matplotlibsettings import *
import datarep.paths as paths


NHS = [250]
METHODS = ["code"]
DATAPATH = "/Users/wybo/Data/results/biasopt/"
DATASET = "EMNIST"
TASKTYPE = "1vall"
READOUT = "tanh"

ll = 0
algo = METHODS[0]
nh_str = "-".join([str(n_h) for n_h in NHS])


glob_name = os.path.join(DATAPATH, "metaopt_interm_v2_%s_%s_%s_nh=%s_ro=%s/*"%(DATASET, algo, TASKTYPE, nh_str, READOUT))


pl.figure('Perf %s ro=%s'%(TASKTYPE, READOUT), figsize=(14,6))
gs = GridSpec(1, 4)
gs.update(top=0.90, bottom=0.15, left=0.15, right=0.95, hspace=1.1, wspace=0.4)


lr, b_add, b_div, x_div = [], [], [], []
frac_pos0, frac_pos1 = [], []

xdict = {'lr': [], 'b_add': [], 'b_div': [], 'x_div': []}

perf = []
print(glob_name)
for file_name in glob.glob(glob_name):
    print(file_name)
    with open(file_name, 'rb') as file:
        res = pickle.load(file)

        xdict['lr'].append(res['params']['lr'].value)
        xdict['b_add'].append(res['params']['b_add'].value)
        xdict['b_div'].append(res['params']['b_div'].value)
        xdict['x_div'].append(res['params']['x_div'].value)

        perf.append(np.mean(res['perf']))


for ii, (key, xdata) in enumerate(xdict.items()):
    jj = ii // 4
    kk = ii % 4

    ax = myAx(pl.subplot(gs[jj,kk]))
    ax.set_title(key)

    xmin, xmax = np.min(xdata), np.max(xdata)
    print(xmin, xmax)

    rect = Rectangle((xmin, 95.), xmax, 5., color='DarkGrey', alpha=.3)
    ax.add_patch(rect)
    ax.axhline(90, ls='--', lw=lwidth/2., c='k')
    ax.axhline(80, ls='--', lw=lwidth/2., c='k')

    ax.scatter(xdata, perf, s=3., alpha=.6)

pl.show()
