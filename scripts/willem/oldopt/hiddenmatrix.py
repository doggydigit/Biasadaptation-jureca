import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as tfunc

import pickle
import os

from datarep import paths
from datarep.matplotlibsettings import *

from biasadaptation.utils import samplers, utils
from biasadaptation.weightmatrices import sm


def load_data(n_h, algo):
    f_name = paths.data_path + 'bias_storage_%s_nh=%d.p'%(algo, n_h)

    with open(f_name, 'rb') as file:
        bias_storage = pickle.load(file)

    # print(bias_storage)
    tasks = [resdict['task'] for resdict in bias_storage]
    perfs = [resdict['perf'][-1] for resdict in bias_storage]

    biasses = [resdict['b_final'][0] for resdict in bias_storage]

    # print(tasks)
    # print(biasses)
    # print(perfs)
    print(np.mean(perfs))
    print(np.median(perfs))



    return tasks, biasses


class HiddenDataLoader:
    def __init__(self, n_h, tasks, biasses_h, biasses_o, perfs, algo, dataset='EMNIST', batch_size=100, n_per_epoch=10000):
        """
        Parameters
        ----------
        n_h: int
            number of hidden units
        tasks: list of list of tuples
            the different task on which the biasses where trained
        biasses_h: list of np.ndarray
            the list of obtained biasses (same length as outer list of ``tasks``)
            of the hidden layer
        biasses_o: list of np.ndarray
            the list of obtained biasses (same length as outer list of ``tasks``)
            of the output unit
        algo: str
            The algorithm to load
        dataset: str
            the dataset
        batch_size: int
            the batch size
        """
        # load the input weights
        path_name = os.path.join(paths.data_path, 'weight_matrices/', '%s_%s%d.npy'%(dataset, algo, n_h))
        self.w_0 = torch.FloatTensor(np.load(path_name))

        self.biasses_h = [torch.FloatTensor(bias) for bias in biasses_h]
        self.biasses_o = [torch.FloatTensor(bias) for bias in biasses_o]
        self.samplers = [samplers.NTaskSampler(dataset, [task],
                                               n_per_batch=batch_size, n_per_epoch=n_per_epoch, target_type='index') \
                        for task in tasks]

        self.tasks = tasks
        self.perfs = perfs

        self.batch_size = batch_size


    def __iter__(self):

        isamplers = [iter(sampler) for sampler in self.samplers]

        print('!!!')
        print(len(self.biasses_h))
        print(len(self.tasks))

        w_1 = torch.ones((1, self.w_0.shape[0]))
        w_1 /= torch.norm(w_1, dim=1)

        # sample batches of datapoints for each tasks
        for x_tuples in zip(*isamplers):
            print(len(x_tuples))
            hidden_list = []
            output_list = []
            target_list = []

            task_list = []
            bias_h_list = []
            bias_o_list = []
            perf_list = []

            for bias_h, bias_o, task, perf, (x_data, _, x_targets) in \
                    zip(self.biasses_h, self.biasses_o, self.tasks, self.perfs, x_tuples):
                # compute the hidden unit differences
                x_data_t = torch.transpose(x_data, 0, 1)
                h_data_t = tfunc.relu(torch.mm(self.w_0, x_data_t) + bias_h)
                o_data_t = torch.mm(w_1, h_data_t) + bias_o

                h_data = torch.transpose(h_data_t, 0, 1)
                o_data = o_data_t[0]

                hidden_list.append(h_data)
                output_list.append(o_data)
                target_list.append(x_targets)

                bias_h_list.append(bias_h)
                bias_o_list.append(bias_o)
                task_list.append(task)
                perf_list.append(perf)

            yield hidden_list, output_list, target_list, task_list, bias_h_list, bias_o_list, perf_list


class HiddenDataLoaderDiff(HiddenDataLoader):
    def __iter__(self):

        isamplers = [iter(sampler) for sampler in self.samplers]

        # sample batches of datapoints for each tasks
        for x_tuples in zip(*isamplers):
            h_diff_list = []

            for bias, (x_data, x_tasks, x_targets) in zip(self.biasses, x_tuples):
                # compute the hidden unit differences
                x_data_t = torch.transpose(x_data, 0, 1)
                h_data_t = tfunc.relu(torch.mm(self.w_0, x_data_t) + bias)

                h_data = torch.transpose(h_data_t, 0, 1)
                h_diff = utils.differences_torch(h_data, self.batch_size)

                h_diff_list.append(h_diff)

            yield torch.cat(h_diff_list, dim=0)

    def get_data_matrix(self):
        return torch.cat([h_diff for h_diff in self]).numpy()


def test_hidden_data_loader(n_h, algo):
    tasks, biasses = load_data(n_h, algo)

    hdl = HiddenDataLoaderDiff(n_h, tasks, biasses, algo, batch_size=1000)

    for ii, h_diff in enumerate(hdl):
        print('\n--> ', ii)
        print(h_diff.shape)

    h_mat = hdl.get_data_matrix()

    print(h_mat.shape)


def find_hidden_subspace(data_loader, n_h,
                  n_max_iter=2000, reg=None):
    """
    Match the ``n_h``-dimensional subspace to the distribution of difference
    vectors of the EMNIST dataset

    Parameters
    ----------
    data_loader: torch.dataloader
        the data loader object
    n_h: int
        the number of basis vectors that span the subspace
    n_max_iter: int
        maximum number of iterations
    reg: None or dict {str: float}
        parameters for regularization term. 'alpha' is the regularization strength
        and 'p' order of the matrix norm. If ``None``, no regularization is applied.

    Returns
    -------
    np.ndarray (n_h, input_dimension)
        The basis vectors that minimaze orthogonal distance between differences
        and subspace
    """
    # get input dimension
    batch = next(iter(data_loader))
    batch_size = batch.shape[0]
    input_dim = batch.view(batch_size, -1).shape[1]

    # subspace optimizer
    w_init = np.random.randn(input_dim, n_h)
    sso = sm.SubSpaceOpt(w_init)

    # stochastic gradien descent with MSEloss
    optimizer = optim.Adam(sso.parameters(), lr=0.005, betas=(0.9, 0.999))
    mse = nn.MSELoss()

    kk = 0
    while kk < n_max_iter:
        for ii, h_diff in enumerate(data_loader):
            optimizer.zero_grad()

            # compute projection, loss, gradient
            h_ = sso.project(h_diff)
            loss = mse(h_, h_diff)
            if reg is not None:
                loss += reg['alpha'] * torch.norm(sso.w, p=reg['p'])
            loss.backward()
            # perform gradient step
            optimizer.step()

            print('\n>>> iter %d --> loss = %.8f <<<'%(kk+ii, loss))

            if kk+ii > n_max_iter:
                break
        kk += ii

    return sso.w.detach().numpy().T



def create_hidden_matrix(n_h1, n_h2, algo, dataset='EMNIST'):
    tasks, biasses = load_data(n_h1, algo)
    hdl = HiddenDataLoaderDiff(n_h1, tasks, biasses, algo, batch_size=100)

    if algo == 'sm':
        w_hid = find_hidden_subspace(hdl, n_h2, reg={'p': 1, 'alpha':.1})
    if algo == 'scd':
        from biasadaptation.weightmatrices import sc
        w_hid = sc.get_weightmatrix_sc(hdl.get_data_matrix(), n_h2)

    # print(w_hid)

    f_name = paths.data_path + 'weight_matrices/' + 'hidden_%s_%s_nh1=%d_nh2=%d.npy'%(dataset, algo, n_h1, n_h2)
    np.save(f_name, w_hid)

    pl.figure('w_hid', figsize=(6,6))
    gs = GridSpec(5,5)

    kk = 0
    for ii in range(3):
        for jj in range(3):
            ax = pl.subplot(gs[ii,jj])

            w_plot = np.reshape(w_hid[kk,:], (5,5))
            ax.imshow(w_plot)

            kk += 1

            ax.set_xticks([])
            ax.set_yticks([])

    pl.show()



if __name__ == "__main__":
    load_data(25, 'sc')

    # test_hidden_data_loader(25, 'sm')

    # create_hidden_matrix(25, 25, 'scd')


