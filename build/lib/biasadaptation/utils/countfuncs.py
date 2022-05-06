import numpy as np
import sklearn.decomposition as skdec
import torch
import torch.optim as optim
import torch.nn.functional as tfunc

import pickle

from biasadaptation.utils import utils
import helperfuncs


def rearrange_to_task_ind(tasks, *args):
    all_dicts = [{} for _ in args]
    for tup in zip(tasks, *args):
        for kk in range(len(args)):
            all_dicts[kk][tup[0]] = tup[kk+1]

    return all_dicts


def count_usage_single_task(ws, bs, data_loader):
    """
    Count the number of times the neurons are active

    Parameters
    ----------
    ws: list of torch.FloatTensor or np.ndarray
        The weight matrices of the network. Each matrix has shape (Nhi-1, Nhi).
        The last layer has 1 hidden unit, thus last weight matrix has shape
        (Nhk-1, 1)
    bs: list of torch.FloatTensor or np.ndarray
        The list of biasses for each layer. For each layer, the biasses have
        shape (Nt, Nhi)
    data_loader: torch.DataLoader
        loader for the input data of the network

    Returns
    -------
    counts: list of torch.LongTensor
        The number of times a units in the network are \'used\' (i.e. non-zero)
    """
    ws = [torch.FloatTensor(w) for w in ws]
    bs = [torch.FloatTensor(b) for b in bs]

    # to store the usage count, same structure as `bs`
    counts = [torch.zeros(b.shape[1], dtype=int) for b in bs]

    xdata, _ = next(iter(data_loader))

    o = xdata
    # compute forward pass
    for w, b, count in zip(ws, bs, counts):
        o = tfunc.relu(torch.mm(o, w) + b[0, :])

        count += torch.sum((o > 1e-10).int(), 0)

    return counts


def count_usage(fname, data_loader, dataset, verbose=False):
    """
    Count the amount of times units are used within the network

    Parameters
    ----------
    fname: str
        the complete path to the optimization results file
    data_loader: torch.DataLoader
        dataloader for the dataset that was used for the optimization
    dataset: str
        the name of the dataset

    Returns
    -------
    ws_dict: dict
        dictionary with task indices as keys and network weights as values
    bs_dict: dict
        dictionary with task indices as keys and network biasses as values
    counts_dict: dict
        dictionary with task indices as keys and unit usage counts as values
    verbose: bool
        wither or not to print stuff
    """

    # load the relevant data from the results file
    with open(fname, 'rb') as file:
        reslist = pickle.load(file)
    ws_list = [res['ws'] for res in reslist]
    bs_list = [res['bs'] for res in reslist]
    task_list = [list(res['task'][-1][dataset].keys())[0] for res in reslist]

    # do the actual counting
    counts_list = []
    for ws, bs in zip(ws_list, bs_list):
        counts_list.append(count_usage_single_task(ws, bs, data_loader))

    # rearrange to task indices
    ws_dict, bs_dict, counts_dict = rearrange_to_task_ind(task_list, ws_list, bs_list, counts_list)

    return ws_dict, bs_dict, counts_dict


def maybe_count(fname_count, fname_res, data_loader, dataset="EMNIST", recompute=False, hl=1):
    try:
        assert not recompute
        with open(fname_count, 'rb') as file:
            counts_dict = pickle.load(file)
    except (FileNotFoundError, AssertionError) as e:
        _, _, counts_dict = count_usage(fname_res, data_loader, dataset, verbose=True)

        with open(fname_count, 'wb') as file:
            pickle.dump(counts_dict, file)

    tot_count = torch.zeros(counts_dict[0][hl-1].shape[0])
    for kk in range(helperfuncs.N_CLASSES[dataset]):
        count = counts_dict[kk][hl-1]
        tot_count += count

    return tot_count


def differences_weighted(data_loader, w_mat, count, n_diff, n_batch=1000, verbose=True):
    """
    Sample differences between datapoints (assumed K-dimensional), weighted by
    the maximal alignment to count

    Parameters
    ----------
    x: torch.FloatTensor (shape=(n_batch, K))
        the input data points
    w_mat: torch.FloatTensor (shape=(K, N_unit))
        the receptive fields of the ``N_unit`` units
    count: torch.FloatTensor or torch.LongTensor (shape=(N_unit,))
        the amount of times a unit is used
    n_diff: int
        The minal number of differences that are to be returned
    n_batch: int
        batch size for the computation

    Returns
    -------
    torch.FloatTensor (shape=(nx, K))
        difference vectors between datapoints (``nx`` >= ``n_diff``)
    """
    assert w_mat.shape[1] == count.shape[0]
    prob = count / torch.sum(count)

    n_tot = 0
    x_list = []

    data_iter = iter(data_loader)

    while n_tot < n_diff:
        try:
            x_data, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            x_data, _ = next(data_iter)

        x_diff = utils.differences_torch(x_data, n_batch)
        n_sample = x_diff.shape[0]

        # find maximal alignment
        ip = torch.abs(torch.matmul(x_diff, w_mat))
        idp = torch.argmax(ip, 1)

        # store directions proportional to maximal alignment
        idx = torch.where(torch.rand(n_sample) < prob[idp])
        x_list.append(x_diff[idx])
        n_tot += len(idx[0])

        if verbose: print(n_tot)

    return torch.cat(x_list, 0)


def get_coordinates(algo, X_data, W_in):
    """
    Get the coordinates of the datapoints in X_data in the basis spanned by the
    rows of W_in

    Parameters
    ----------
    algo: 'lstsq' or 'sc'
        The algorithm used to estimate the coordinates
    X_data: np.ndarray (shape=(k, n))
        k datapoints with n features
    W_in: np.ndarray(nh, n)
        the nh vectors that consitute the basis of the coordinates

    Returns
    -------
    np.ndarray (shape=(k,nh))
        The coordinates
    """
    if algo == 'lstsq':
        C_oo = np.linalg.lstsq(W_in.T, X_data.T, rcond=None)[0].T
    elif algo == 'sc':
        C_oo = skdec.sparse_encode(X_data, W_in, algorithm='omp', alpha=0.1)
    else:
        raise ValueError('[algo] should be \'lstsq\' or \'sc\'')
    return C_oo


def get_coordinates_torch(X_data, W_in):
    """
    Get the coordinates of the datapoints in X_data in the basis spanned by the
    rows of W_in

    Parameters
    ----------
    X_data: torch.FloatTensor (shape=(k, n))
        k datapoints with n features
    W_in: torch.FloatTensor (shape=(n, nh))
        the nh vectors that consitute the basis of the coordinates

    Returns
    -------
    np.ndarray (shape=(k,nh))
        The coordinates
    """
    C_oo = torch.linalg.lstsq(W_in, X_data.T, rcond=None)[0].T
    return C_oo


def coordinates_weighted(data_loader, w_mat1, w_mat2, count, algoc,
                         n_diff=1000, n_batch=1000, verbose=True):


    assert w_mat2.shape[1] == count.shape[0]
    prob = count / torch.sum(count)

    n_tot = 0
    c_list = []

    data_iter = iter(data_loader)

    while n_tot < n_diff:
        try:
            x_data, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            x_data, _ = next(data_iter)

        c_oo = get_coordinates(algoc, x_data.numpy(), w_mat1.numpy().T)
        c_oo = torch.FloatTensor(c_oo)
        n_sample = c_oo.shape[0]

        # find maximal alignment
        ip = torch.abs(torch.matmul(c_oo, w_mat2))
        idp = torch.argmax(ip, 1)

        # store directions proportional to maximal alignment
        idx = torch.where(torch.rand(n_sample) < prob[idp])
        c_list.append(c_oo[idx])
        n_tot += len(idx[0])

        if verbose: print(n_tot)

    return torch.cat(c_list, 0)


def get_activation_profile(X_data, W_in, b_in):
    """
    Parameters
    ----------
    X_data: torch.FloatTensor (k x n)
        k datapoints with n features
    W_in: torch.FloatTensor (n x nh)
        input weights with n features to nh hidden neurons
    b_in: torch.FloatTensor (nt x nh)
        nh biases for nt tasks

    Returns
    -------
    torch.BoolTensor (k x nh x nt)
        binary tensor with for each datapoint a 2d tensor indicating the which of
        the nh hidden neurons were active for each of the nt tasks
    """
    return ((X_data @ W_in)[:,:,None] + b_in.T[None,:,:]) > 0.



