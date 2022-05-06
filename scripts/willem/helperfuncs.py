import numpy as np
import torch
import torchvision.datasets as tdatasets
import torchvision.transforms as ttransforms
import torchvision.transforms.functional as tfunctional
from sklearn.preprocessing import normalize
import sklearn.decomposition as skdec
from tqdm import tqdm

import os
import pickle

from datarep import paths

from biasadaptation.utils import utils, preprocessing
from biasadaptation.utils import k_task_n_class_m_dataset_data as knm
import biasadaptation.datasets as bdatasets

# N_CLASSES = {'MNIST': 10,
#              'EMNIST': 47}
N_CLASSES = {'MNIST': 10,
             'EMNIST': 47,
             'K49': 49,
             'CIFAR10': 10,
             'CIFAR100': 100}

# original parameters
LRS_0 = [0.005, 0.001, 0.0001]
X_DIV_0 = 'data'
B_DIV_0 = 10.
B_ADD_0 = 0.
# optimized parameters bias adaptation
LRS_B = [0.06, 0.02, 0.008]
# LRS_B = [0.1, 0.02, 0.008]
X_DIV_B = .5
B_DIV_B = 5.
B_ADD_B = -.5
# optimized parameters bias adaptation from gradient descent matrices
X_DIV_GD = 1.
# optimized parameters weight optimization
LRS_W = [0.002, 0.001, 0.0008]
X_DIV_W = 4.
B_DIV_W = 10.
B_ADD_W = 0.


def get_dataset_with_options(algo, dataset, path):
    # set options based on algorithm
    if algo == 'rpw' or algo == 'mr' or algo == 'br':
        x_div = X_DIV_W
        rotate = False
        if algo == 'mr' or algo == 'br':
            rotate = True
            x_div = X_DIV_GD
    else:
        x_div = X_DIV_B
        rotate = False
        if algo == 'mrb' or algo == 'brb' or algo == 'bpo' or algo == 'bp':
            rotate = True
            x_div = X_DIV_GD

    # source dataset
    tdataset = preprocessing.get_dataset(dataset,
                           train=True,  rotate=rotate,
                           x_div=x_div, path=path)

    return tdataset


def construct_knm_dataloader_triplet(dataset, task_train, task_test,
                                     source_train, source_test,
                                     batch_size_train, batch_size_validate, batch_size_test):
    """
    Assumes `task_train` and `task_test` represent the same task, only that the
    number of samples per task are differnt
    """
    size_factor = 2*len(task_train[-1][dataset]) * len(task_train[1][dataset])

    size_train = size_factor
    while size_train < len(source_train):
        size_train += size_factor

    size_test = size_factor
    while size_test < len(source_test):
        size_test += size_factor

    ds_train = knm.KTaskNClassMDatasetData(size=size_train, tasks=[task_train], datasets={dataset: source_train}, cache_suffix="train")
    ds_validate = knm.KTaskNClassMDatasetData(size=size_train, tasks=[task_test], datasets={dataset: source_train}, cache_suffix="train")
    ds_test = knm.KTaskNClassMDatasetData(size=size_test, tasks=[task_test], datasets={dataset: source_test}, cache_suffix="test")

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size_train, shuffle=True)
    dl_validate = torch.utils.data.DataLoader(ds_validate, batch_size=batch_size_validate, shuffle=True)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size_test, shuffle=True)

    return dl_train, dl_test, dl_validate


def construct_knm_dataloader_triplet_transfer(task_train, task_test,
                                              sources_train, sources_test,
                                              batch_size_train, batch_size_validate, batch_size_test):
    """
    Assumes `task_train` and `task_test` represent the same task, only that the
    number of samples per task are differnt
    """

    size_factor = 2 * sum([len(task_classes) for task_classes in task_train[-1].values()]) * \
                      sum([len(task_classes) for task_classes  in task_train[1].values()])

    len_train = np.sum([len(source) for source in sources_train.values()])
    len_test = np.sum([len(source) for source in sources_test.values()])

    size_train = size_factor
    while size_train < len_train:
        size_train += size_factor

    size_test = size_factor
    while size_test < len_test:
        size_test += size_factor

    ds_train = knm.KTaskNClassMDatasetData(size=size_train, tasks=[task_train], datasets=sources_train, cache_suffix="train")
    ds_validate = knm.KTaskNClassMDatasetData(size=size_train, tasks=[task_test], datasets=sources_train, cache_suffix="train")
    ds_test = knm.KTaskNClassMDatasetData(size=size_test, tasks=[task_test], datasets=sources_test, cache_suffix="test")

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size_train, shuffle=True)
    dl_validate = torch.utils.data.DataLoader(ds_validate, batch_size=batch_size_validate, shuffle=True)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size_test, shuffle=True)

    return dl_train, dl_test, dl_validate


def get_weight_matrix_in(n_h, algo, dataset='EMNIST', task=None, reweighted=False, wm_path="weight_matrices/"):
    """
    Parameters
    ----------
    n_h: int
        number of hidden units
    algo: str ('pca', 'ica', 'rp', 'rg', 'sc', 'scd', 'sm', 'pmd', 'pmdd', 'bmd', 'bmdd', 'bp', 'bpo')
        name of the algorithm
    dataset: str ('EMNIST')
        the name of the dataset
    task: dict
        the task, should be provided if the algorithm is 'bp'
    reweighted: bool
        whether to use the reweighted matrix (only works for 'pmdd' and 'scd')
    """
    if 'rp' in algo:
        try:
            path_name = os.path.join(paths.data_path, '%s'%wm_path, '%s_pmdd%d.npy'%(dataset, n_h))
            w_mat = np.load(path_name).T

            w_mat = np.random.randn(*w_mat.shape)
        except FileNotFoundError:
            w_mat = np.random.randn(784, n_h)

        w_mat = normalize(w_mat)

    elif algo == 'bp' or algo == 'bpo':
        class_idx = list(task[-1][dataset].keys())[0]

        # path_name = os.path.join(paths.data_path, 'weight_matrices/%s_bp/l1o_weights/'%(dataset),
        #                          'biaslearner_[%d]_testclass_%d_seed_0.pickle'%(n_h, class_idx))
        path_name = os.path.join(paths.data_path, '%s'%wm_path, '%s_tsai/'%(dataset),
                                 'biaslearner_[%d]_testclass_%d_seed_0.pickle'%(n_h, class_idx))

        print('input weights:', path_name)
        with open(path_name, 'rb') as f:
            ws, bs = pickle.load(f)
        w_mat = ws[0]

    elif algo == 'mr' or algo == 'mrb':
        class_idx = list(task[-1][dataset].keys())[0]

        # path_name = os.path.join(paths.data_path, 'weight_matrices/%s_mr/'%(dataset),
        #                          'multireadout_[%d]_testclass_%d_seed_0.pickle'%(n_h, class_idx))

        path_name = os.path.join(paths.data_path, '%s'%wm_path, '%s_tsai/'%(dataset),
                                 'multireadout_[%d]_testclass_%d_seed_0.pickle'%(n_h, class_idx))

        print('input weights:', path_name)
        with open(path_name, 'rb') as f:
            data = pickle.load(f)

        w_mat = data[0][0]

    elif algo == 'br' or algo == 'brb':
        class_idx = list(task[-1][dataset].keys())[0]

        # path_name = os.path.join(paths.data_path, 'weight_matrices/%s_mr/'%(dataset),
        #                          'multireadout_[%d]_testclass_%d_seed_0.pickle'%(n_h, class_idx))

        path_name = os.path.join(paths.data_path, '%s'%wm_path, '/%s_tsai/'%(dataset),
                                 'binarymr_[%d]_testclass_%d_seed_0.pickle'%(n_h, class_idx))

        print('input weights:', path_name)
        with open(path_name, 'rb') as f:
            data = pickle.load(f)
            print(data[0][0].shape)

        w_mat = data[0][0]

    else:
        if reweighted:
            dataset = dataset + '_reweighted'
        path_name = os.path.join(paths.data_path, '%s'%wm_path, '%s_%s%d.npy'%(dataset, algo, n_h))
        w_mat = np.load(path_name).T

    return w_mat


def _to_str(idxlist):
    return [str(val) for val in idxlist]


def get_suffix(w_idx, b_idx, g_idx=[]):
    return "w%s-b%s-g%s"%(';'.join(_to_str(w_idx)),
                          ';'.join(_to_str(b_idx)),
                          ';'.join(_to_str(g_idx)))


def _get_indices_subset(char, suffix):
    idx0 = suffix.find(char)
    if idx0 == -1:
        idxs = None
    else:
        idx1 = suffix[idx0+1:].find('-')
        suffix_ = suffix[idx0+1:][:idx1] if idx1 != -1 else suffix[idx0+1:]
        idxs = [int(val) for val in suffix_.split(';') if len(val) > 0 ]
    return idxs


def get_indices_from_suffix(suffix):
    w_idx = _get_indices_subset('w', suffix)
    b_idx = _get_indices_subset('b', suffix)
    g_idx = _get_indices_subset('g', suffix)

    return w_idx, b_idx, g_idx


def get_weight_matrix_hidden(n_h1, n_h2, algo_1, algo_2, algo_c, dataset='EMNIST', task=None, reweighted=False, enriched=False, wm_path="weight_matrices/"):
    """
    Parameters
    ----------
    n_h1: int
        number of hidden units in first layer
    n_h2: int
        number of hidden units in second layer
    algo_1: str ('rpw', 'scd', 'pmdd')
        name of the with which the input weight matrix is derived
    algo_2: str ('rpw', 'sc', 'pmd')
        name of the with which the hidden weight matrix is derived
    algo_c: str ('sc', 'lstsq')
        name of the algorithm with which the hidden coordinates are derived
    dataset: str ('EMNIST')
        the name of the dataset
    task: dict
        the task, should be provided if the algorithm is 'bp'
    reweighted: bool
        whether to use the reweighted matrix (only works for 'pmdd' and 'scd')
    """
    if reweighted:
        namestring = '%s_reweighted_%s-%s-%s%d_%d.npy'%(dataset, algo_1, algo_2, algo_c, n_h1, n_h2)
        w_mat = np.load(os.path.join(paths.data_path, '%s'%wm_path, namestring)).T
        print('!!!', w_mat.shape)

    elif algo_c == 'ha':
        if 'rp' in algo_2:
            w_mat = np.random.randn(n_h1, n_h2)
            w_mat = normalize(w_mat)

        else:
            namestring = '%s_%s_algos12=%s-%s_nh12=%d-%d.npy'%(dataset, algo_c, algo_1, algo_2, n_h1, n_h2)
            w_mat = np.load(os.path.join(paths.data_path, '%s'%wm_path, namestring)).T

    else:
        enstr = '_enriched' if enriched else ''
        namestring = '%s%s_algos12C=%s-%s-%s_nh12=%d-%d.p'%(dataset, enstr, algo_1, algo_2, algo_c, n_h1, n_h2)

        if 'rp' in algo_1 or 'rp' in algo_2:
            w_mat = np.random.randn(n_h1, n_h2)
            w_mat = normalize(w_mat)

        elif algo_2 == 'bp' or algo_2 == 'bpo':
            class_idx = list(task[-1][dataset].keys())[0]

            path_name = os.path.join(paths.data_path, '%s'%wm_path, '%s_bp/deepened_weights/'%dataset,
                                     'deepened_%s_[%d, %d]_%s_willem_testclass_%d_seed_0.pickle'%(algo_1, n_h1, n_h2, dataset, class_idx))

            with open(path_name, 'rb') as f:
                ws, bs = pickle.load(f)
            w_mat = ws[1]

        else:
            path_name = os.path.join(paths.data_path, '%s'%wm_path, namestring)
            with open(path_name, 'rb') as f:
                w_mat = pickle.load(f).T

    return w_mat


def get_weight_matrix_hidden3(n_h1, n_h2, n_h3, algo_1, algo_2, algo_3, algo_c, dataset='EMNIST', wm_path="weight_matrices/"):
    """
    Parameters
    ----------
    n_h1: int
        number of hidden units in first layer
    n_h2: int
        number of hidden units in second layer
    n_h3: int
        number of hidden units in third layer
    algo_1: str ('rpw', 'scd', 'pmdd')
        name of the with which the input weight matrix is derived
    algo_2: str ('rpw', 'sc', 'pmd')
        name of the with which the hidden weight matrix is derived
    algo_3: str ('rpw', 'sc', 'pmd')
        name of the with which the hidden weight matrix is derived
    algo_c: str ('sc', 'lstsq')
        name of the algorithm with which the hidden coordinates are derived
    dataset: str ('EMNIST')
        the name of the dataset
    """
    namestring = '%s_algos123C=%s-%s-%s-%s_nh123=%d-%d-%d.p'%(dataset, algo_1, algo_2, algo_3, algo_c, n_h1, n_h2, n_h3)
    if 'rp' in algo_3:
        w_mat = np.random.randn(n_h2, n_h3)
        w_mat = normalize(w_mat)
    else:
        path_name = os.path.join(paths.data_path, '%s'%wm_path, namestring)
        with open(path_name, 'rb') as f:
            w_mat = pickle.load(f).T

    return w_mat


def get_weight_matrix_out(n_h, algo='', dataset='EMNIST', task=None, bias_opt=True, wm_path="weight_matrices/"):
    if bias_opt:
        if algo == "bpo":
            class_idx = list(task[-1][dataset].keys())[0]

            # path_name = os.path.join(paths.data_path, 'weight_matrices/%s_bp/l1o_weights/'%(dataset),
            #                          'biaslearner_[%d]_testclass_%d_seed_0.pickle'%(n_h, class_idx))
            path_name = os.path.join(paths.data_path, '%s'%wm_path, '%s_tsai/'%(dataset),
                                     'biaslearner_[%d]_testclass_%d_seed_0.pickle'%(n_h, class_idx))

            print('output weights:', path_name)
            with open(path_name, 'rb') as f:
                ws, bs = pickle.load(f)
            w_vec = ws[-1]

        else:
            w_vec = np.ones((n_h,1))
    else:
        w_vec = np.random.randn(n_h,1)
    w_vec /= np.linalg.norm(w_vec)

    return w_vec


def get_multitask_weights(*args, algo='', dataset='EMNIST', task=None, seed=0, wm_path="weight_matrices/"):
    nh_str = ", ".join([str(a) for a in args])

    if algo == 'br' or algo == 'brb':
        specifier = 'binarymr'
    elif algo == 'mr' or algo == 'mrb':
        specifier = 'multireadout'
    elif algo == 'bp' or algo == 'bpo':
        specifier = 'biaslearner'
    else:
        raise Exception('algorithm not implemented')

    class_idx = list(task[-1][dataset].keys())[0]

    fname = os.path.join(paths.data_path, '%s'%wm_path, '%s_tsai/'%(dataset),
                                          '%s_[%s]_testclass_%d_seed_0.pickle'%(specifier, nh_str, class_idx))

    with open(fname, 'rb') as f:
        data = pickle.load(f)

    return data[0]


def get_multitask_biasses(*args, algo='', dataset='EMNIST', task=None, seed=0, wm_path="weight_matrices/"):
    nh_str = ", ".join([str(a) for a in args])

    if algo == 'br' or algo == 'brb':
        specifier = 'binarymr'
    elif algo == 'mr' or algo == 'mrb':
        specifier = 'multireadout'
    else:
        raise Exception('algorithm not implemented')

    class_idx = list(task[-1][dataset].keys())[0]

    fname = os.path.join(paths.data_path, '%s'%wm_path, '%s_tsai/'%(dataset),
                                          '%s_[%s]_testclass_%d_seed_0.pickle'%(specifier, nh_str, class_idx))

    with open(fname, 'rb') as f:
        data = pickle.load(f)

    return [d[None,:] for d in data[1][:-1]] + [data[1][-1][class_idx:class_idx+1][None,:]]


def get_bias_1(n_h, algo='', dataset='EMNIST', task=None, wm_path="weight_matrices/"):
    class_idx = list(task[-1][dataset].keys())[0]

    if algo == 'mr' or algo == 'mrb':
        # path_name = os.path.join(paths.data_path, 'weight_matrices/%s_mr/'%(dataset),
        #                          'multireadout_[%d]_testclass_%d_seed_0.pickle'%(n_h, class_idx))
        path_name = os.path.join(paths.data_path, '%s'%wm_path, '%s_tsai/'%(dataset),
                                 'multireadout_[%d]_testclass_%d_seed_0.pickle'%(n_h, class_idx))

        with open(path_name, 'rb') as f:
            data = pickle.load(f)
        b_vec = data[1][0][None,:]

    elif algo == 'br' or algo == 'brb':
        # path_name = os.path.join(paths.data_path, 'weight_matrices/%s_mr/'%(dataset),
        #                          'multireadout_[%d]_testclass_%d_seed_0.pickle'%(n_h, class_idx))
        path_name = os.path.join(paths.data_path, '%s'%wm_path, '%s_tsai/'%(dataset),
                                 'binarymr_[%d]_testclass_%d_seed_0.pickle'%(n_h, class_idx))

        with open(path_name, 'rb') as f:
            data = pickle.load(f)
        b_vec = data[1][0][None,:]

    else:
        fname = "biasopt_1hl_%s_%s%d_%s_ro=%s.p"%(args.dataset, algo, n_h, "1vall", "tanh")
        path_name = os.path.join(paths.data_path, wm_path, fname)
        with open(path_name, 'rb') as file:
            reslist = pickle.load(file)

        for res in reslist:
            cres_idx = list(reslist['task'][-1][dataset].keys())[0]
            if cres_idx == class_idx:
                b_vec = res['bs']
                break

    return b_vec


def sample_1vall_tasks(ntask, nsample=1000000, t0_datasets=['EMNIST'], t1_datasets=['EMNIST'], seed=42):
    """
    Get random set of binary '1 vs all' tasks

    Parameters
    ----------
    ntask: int
        the number of task. If `C` is the number of classes in the dataset,
        should be smaller than `C`*(`C`-1) for '1v1' tasks and smaller than `C`
        for '1vall' tasks
    nsample: int
        the number of samples per task category
    t0_datasets: str or list of string
        the datasets used for the one class
    t1_datasets: str or list of string
        the datasets used for the all class
    seed: int
        seed for the random generator used for task sampling

    Returns
    -------
    tasks: list of dicts
        the tasks, containing entries for all classes
    """
    np.random.seed(seed)

    # distribute tasks of t0_datasets
    tot_classes = sum([N_CLASSES[dataset] for dataset in t0_datasets])


    tasks = []
    for dataset0 in t0_datasets:
        nclass = N_CLASSES[dataset0]

        ntask_ = int(np.round(ntask * nclass / tot_classes))

        print('--> ', dataset0, ', ntask =', ntask_)

        class_idx = np.arange(nclass)
        c0 = np.random.choice(class_idx, ntask_, replace=False)

        for c_ in c0:
            task_m1 = {dataset0: {c_: nsample}}

            task_p1 = {}
            for dataset1 in t1_datasets:

                nclass_ = N_CLASSES[dataset1]
                class_idx_ = np.arange(nclass_)

                if dataset0 == dataset1:
                    task_p1[dataset1] = [cc for cc in class_idx_ if cc != c_]
                else:
                    task_p1[dataset1] = [cc for cc in class_idx_]

            tasks.append({-1: task_m1, 1: task_p1})

    return tasks


def sample_binary_tasks_(ntask, nsample=1000000, dataset='EMNIST', task_type='1v1', seed=42):
    """
    Get random set of binary tasks

    Parameters
    ----------
    ntask: int
        the number of task. If `C` is the number of classes in the dataset,
        should be smaller than `C`*(`C`-1) for '1v1' tasks and smaller than `C`
        for '1vall' tasks
    nsample: int
        the number of samples per task category
    dataset: str or list if string
        the dataset
    task_type: str ('1v1' or '1vall')
        the task type
    seed: int
        seed for the random generator used for task sampling
    """
    nclass = N_CLASSES[dataset]

    np.random.seed(seed)

    if task_type == '1v1':
        class_pairs = [(idx0, idx1) for idx0 in range(nclass) for idx1 in range(nclass) if idx1 != idx0]
        pair_idxs = np.arange(len(class_pairs))

        task_pairs = np.random.choice(pair_idxs, ntask, replace=False)

        return [
            {-1: {dataset: {class_pairs[idx][0]: nsample}}, 1: {dataset: {class_pairs[idx][1]: nsample}}} \
                for idx in task_pairs
                ]

    elif task_type == '1vall':
        class_idx = np.arange(nclass)
        c1 = np.random.choice(class_idx, ntask, replace=False)

        return [{-1: {dataset: {c_: nsample}}, 1: {dataset: [cc for cc in class_idx if cc != c_]}} for c_ in c1]

    else:
        raise Exception('Invalid `task_type`, choose \'1v1\' or \'1vall\'')


def get_data_matrix(dataset, return_diff=True, train=True):
    """
    """
    data = preprocessing.get_dataset(dataset, x_div='data', train=train)

    data_loader = torch.utils.data.DataLoader(
                         dataset=data,
                         batch_size=100000,
                         shuffle=False,
                         drop_last=True)

    # data_matrix = utils.get_big_data_matrix(data_loader)

    data_list = []
    for (d, t) in tqdm(data_loader):
        data_list.append(d.numpy())

    data_matrix = np.vstack(data_list)
    del data_list

    if return_diff:
        diff_matrix = utils.differences_numpy(data_matrix, data_matrix.shape[0])
        return diff_matrix
    else:
        return data_matrix


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



