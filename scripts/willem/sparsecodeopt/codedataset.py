import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import pickle
import argparse
import os
import sys
sys.path.append('..')

import helperfuncs


'''
example usage

python3 codedataset.py --nhidden 25  --algow sc --algoc sc
'''

class CodeDataset(Dataset):
    """
    A dataset containing codes of elements from source in the provided
    dictionary
    """
    def __init__(self, n_h, algo_w='sc', dataset='EMNIST', algo_c='sc', path='', train=True, recompute=False):
        suffix = 'train' if train else 'test'
        # file name of code file
        f_name = os.path.join(path, '%s_codedata_%s_%s_%d_%s.npy'%(dataset, algo_w, algo_c, n_h, suffix))

        # source dataset
        source = helperfuncs.get_dataset(dataset, train=train)
        self.targets = source.targets

        # input weights
        W_in = helperfuncs.get_weight_matrix_in(n_h, algo=algo_w).T

        try:
            if recompute:
                raise FileNotFoundError
            self.data = np.load(f_name, mmap_mode='r+')
        except FileNotFoundError as e:
            print('Computing code matrix %d %s %s'%(n_h, algo_w, algo_c))
            # get the code matrix
            X_data = helperfuncs.get_data_matrix(dataset, return_diff=False, train=train)
            C_data = helperfuncs.get_coordinates(algo_c, X_data, W_in)

            print('Done, code matrix shape:', C_data.shape)

            # save the code matrix to disk and delete
            np.save(f_name, C_data)
            del C_data

            # load the code matrix in memmap-mode
            self.data = np.load(f_name, mmap_mode='r+')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), int(self.targets[idx])


def create_code_matrices():
    # read command line args and kwargs
    parser = argparse.ArgumentParser()
    parser.add_argument("--nhidden", type=int, help="number of hidden neurons", default=25)
    parser.add_argument("--algow", type=str, help="methods to be applied to create weight matrix", default='sc')
    parser.add_argument("--algoc", type=str, help="methods to be applied to create weight matrix", default='sc')
    # parser.add_argument("--ndata", type=int, help="number of data points", default=100000)
    parser.add_argument("--path", type=str, help="path to which to save the files", default="")
    parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")
    args = parser.parse_args()

    cd_train = CodeDataset(args.nhidden, dataset=args.dataset, algo_w=args.algow, algo_c=args.algoc, path=args.path, train=True, recompute=True)
    cd_test = CodeDataset(args.nhidden, dataset=args.dataset, algo_w=args.algow, algo_c=args.algoc, path=args.path, train=False, recompute=True)


if __name__ == "__main__":
    create_code_matrices()







