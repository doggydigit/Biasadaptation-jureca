import torch
import torchvision.transforms as transforms

import argparse
import sys
sys.path.append('..')

from datarep.matplotlibsettings import *
from biasadaptation.utils import utils

import helperfuncs


GRID_DICT={10: [2,5],
           25: [5,5],
           50: [5,10],
           100: [10,10],
           250: [10,25],
           500: [20,25],
           1000: [33, 30],
           2500: [50, 50],
           }

"""
example usage

python plot_weightmat.py --dataset K49 --method pmdd --nhidden 100
"""

# read command line args and kwargs
parser = argparse.ArgumentParser()
parser.add_argument("--nhidden", type=int, help="number of hidden neurons", default=25)
parser.add_argument("--method", type=str, help="methods to be applied to create weight matrix", default='scd')
parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")
parser.add_argument("--reweighted", type=bool, help="reweighted matrix", default=False)
args = parser.parse_args()

# load the weight matrix
w_mat = helperfuncs.get_weight_matrix_in(args.nhidden, args.method, dataset=args.dataset, reweighted=args.reweighted).T

pl.figure('W_mat', figsize=(18,11))
nrow, ncol = GRID_DICT[args.nhidden]
nplot = nrow * ncol
gs = GridSpec(nrow, ncol)
gs.update(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.05, wspace=0.05)


for ii, w_vec in enumerate(w_mat[:nplot]):
    jj = ii%ncol
    kk = ii//ncol

    ax = noFrameAx(pl.subplot(gs[kk,jj]))
    if "MNIST" in args.dataset or args.dataset == "K49":
        ax.imshow(utils.to_image_mnist(w_vec))
    elif "CIFAR" in args.dataset:
        img = utils.to_image_cifar(torch.FloatTensor(w_vec), remove_normalization=False)
        img = transforms.ToPILImage()(img)
        ax.imshow(img)

pl.show()