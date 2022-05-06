import torch
import numpy as np

import random
import argparse
import sys
sys.path.append('..')

import optim, helperfuncs

from biasadaptation.utils import utils


"""
example usage

    python3 allopt.py --methods pmdd --nhidden 10 --ndata 500000 --datasets K49 --path ~/Data/weight_matrices2 --datapath ~/Data
    python3 allopt.py --methods pmdd --nhidden 10 --ndata 60000 --datasets CIFAR10 --datapath "/work/users/wybo/Data/"

"""

# read command line args and kwargs
parser = argparse.ArgumentParser()
parser.add_argument("--nhidden", nargs="*", type=int, help="number of hidden neurons", default=[10, 25, 50, 100, 250, 500, 1000])
parser.add_argument("--methods", nargs="*", help="methods to be applied to create weight matrix", default='all')
parser.add_argument("--ndata", type=int, help="number of data points", default=1000000000)
parser.add_argument("--save", type=bool, help="whether to save results or not", default=True)
parser.add_argument("--path", type=str, help="path to which to save the file", default="")
parser.add_argument("--datapath", type=str, help="path to which the data sets are saved", default="./datasets")
parser.add_argument("--datasets", nargs="*", type=str, help="datasets to load", default="EMNIST")
args = parser.parse_args()

# # data import
# print("loading data")
# data_loader = utils.load_data(dataset=args.dataset, data_path=args.datapath)
# if sum([m in args.methods for m in ['pca', 'ica', 'pca', 'sc', 'scd', 'rg', 'rp', 'pmd', 'pmdd', 'bmd', 'bmdd', 'all']]):
#     # only load data  matrix when necessary for algorithm
#     data_matrix = utils.get_big_data_matrix(data_loader)

#     if args.ndata < data_matrix.shape[0]:
#         inds = np.random.choice(np.arange(data_matrix.shape[0]), args.ndata, replace=False)
#         data_matrix = data_matrix[inds]

#     print(data_matrix.shape)

#     n_in_features = data_matrix.shape[1]

dname = "-".join(args.datasets)

data = []
for dataset in args.datasets:
    # get source dataset, EMNIST is rotated
    rotate = True if dataset == "EMNIST" else False
    ds = helperfuncs.get_dataset(dataset, rotate=rotate, path=args.datapath)

    # load the data matrix
    dl = torch.utils.data.DataLoader(ds, batch_size=args.ndata, shuffle=True)
    X_, _ = next(iter(dl))
    data.append(X_)

data_matrix = torch.vstack(data).numpy()

print(np.linalg.norm(data_matrix, axis=1))
print(data_matrix.shape)
print(dataset, ' norm:', np.mean(np.linalg.norm(data_matrix, axis=1)))


# save the weights
def save_weight_matrix(path_name, file_name, W):
    import os
    np.save(os.path.join(path_name, file_name), W)


# weight matrix creation with different methods
if 'pca' in args.methods or 'all' in args.methods:
    from biasadaptation.weightmatrices import pca
    print("creating weight matrix using PCA")
    for n_h in args.nhidden:
        if n_h <= n_in_features: # Number of requested components <= input dimensionality
            W_pca = pca.get_weightmatrix_pca(data_loader, n_h)
            if args.save:
                save_weight_matrix(args.path, dname+'_pca'+str(n_h), W_pca)

if 'ica' in args.methods or 'all' in args.methods:
    from  biasadaptation.weightmatrices import ica
    print("creating weight matrix using ICA")
    for n_h in args.nhidden:
        if n_h <= n_in_features: # Number of requested components <= input dimensionality
            W_ica = ica.get_weightmatrix_ica(data_matrix, n_h)
            if args.save:
                save_weight_matrix(args.path, dname+'_ica'+str(n_h), W_ica)

if 'sc' in args.methods or 'all' in args.methods:
    from biasadaptation.weightmatrices import sc
    print("creating weight matrix using SC")
    for n_h in args.nhidden:
        W_sc = sc.get_weightmatrix_sc(data_matrix, n_h, getsparsity=False)
        if args.save:
            save_weight_matrix(args.path, dname+'_sc'+str(n_h), W_sc)

if 'scd' in args.methods or 'all' in args.methods:
    from biasadaptation.weightmatrices import sc
    print("creating weight matrix using SC of the input differences")
    for n_h in args.nhidden:
        W_scd = sc.get_weightmatrix_scd(data_matrix, n_h, getsparsity=False)
        if args.save:
            save_weight_matrix(args.path, dname+'_scd'+str(n_h), W_scd)

if 'rg' in args.methods or 'all' in args.methods:
    from biasadaptation.weightmatrices import rg
    print("creating weight matrix using RG")
    for n_h in args.nhidden:
        W_rg = rg.get_weightmatrix_rg(data_matrix, n_h)
        if args.save:
            save_weight_matrix(args.path, dname+'_rg'+str(n_h), W_rg)

if 'rp' in args.methods or 'all' in args.methods:
    from biasadaptation.weightmatrices import rp
    print("creating weight matrix using RP")
    for n_h in args.nhidden:
        W_rp = rp.get_weightmatrix_rp(data_matrix, n_h)
        if args.save:
            save_weight_matrix(args.path, dname+'_rp'+str(n_h), W_rp)

# deprecated, should take data matrix
if 'sm' in args.methods or 'all' in args.methods:
    from biasadaptation.weightmatrices import sm
    print("creating weight matrix using SM")
    data_loader_sm = utils.load_data_sm(dataset=dname, data_path=args.datapath, batch_size=1000)
    for n_h in args.nhidden:
        W_sm = sm.get_weightmatrix_sm(data_loader_sm, n_h)
        if args.save:
            save_weight_matrix(args.path, dname+'_sm'+str(n_h), W_sm)

if 'pmd' in args.methods or 'all' in args.methods:
    from biasadaptation.weightmatrices import pmd
    print("creating weight matrix using PMD")
    for n_h in args.nhidden:
        W_pmd = pmd.get_weightmatrix_pmd(data_matrix, n_h)
        if args.save:
            save_weight_matrix(args.path, dname+'_pmd'+str(n_h), W_pmd)

if 'pmdd' in args.methods or 'all' in args.methods:
    from biasadaptation.weightmatrices import pmd
    print("creating weight matrix using PMD of input differences")
    for n_h in args.nhidden:
        W_pmdd = pmd.get_weightmatrix_pmdd(data_matrix, n_h)
        if args.save:
            save_weight_matrix(args.path, dname+'_pmdd'+str(n_h), W_pmdd)

if 'bmd' in args.methods or 'all' in args.methods:
    from biasadaptation.weightmatrices import bmd
    print("creating weight matrix using BMD")
    for n_h in args.nhidden:
        W_bmd = bmd.get_weightmatrix_bmd(data_matrix, n_h)
        if args.save:
            save_weight_matrix(args.path, dname+'_bmd'+str(n_h), W_bmd)

if 'bmdd' in args.methods or 'all' in args.methods:
    from biasadaptation.weightmatrices import bmd
    print("creating weight matrix using BMD of input differences")
    for n_h in args.nhidden:
        W_bmdd = bmd.get_weightmatrix_bmdd(data_matrix, n_h)
        if args.save:
            save_weight_matrix(args.path, dname+'_bmdd'+str(n_h), W_bmdd)


# # jump to interactive mode
# import code
# code.interact(local=locals())


# sample plotting
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt

#W = W_pca
#plt.ion()
#plt.imshow(W[random.sample(range(0, args.nhidden[-1]), 1)[0], :].reshape(28, 28), cmap = 'gray')
#plt.show()
