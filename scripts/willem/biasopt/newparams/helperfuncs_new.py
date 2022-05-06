import numpy as np
import  torch
import torchvision.datasets as tdatasets
import torchvision.transforms as ttransforms
import torchvision.transforms.functional as tfunctional
from sklearn.preprocessing import normalize


from torchvision.transforms.functional import hflip, rotate
from torchvision.transforms import Compose, ToTensor
from torchvision import datasets

import os
import pickle

from datarep import paths

from biasadaptation.utils import samplers
from biasadaptation.utils import k_task_n_class_m_dataset_data as knm



DATA_NORMS = {'MNIST': 9.2147,
              'EMNIST': 10.3349}

# N_CLASSES = {'MNIST': 10,
#              'EMNIST': 47}
N_CLASSES = {'MNIST': 10,
             'EMNIST': 47}



class ReshapeTransform:
    """
    Transform class to reshape tensors
    """
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


def get_dataset(dataset, rotate=True, train=True):
    """
    Return the torch dataset
    """
    # transform_list = [ttransforms.ToTensor()]
    # if normed:
    #     data_norm = DATA_NORMS[dataset]
    #     transform_list.append(lambda x: x/data_norm)

    # data_transform = ttransforms.Compose(transform_list)
    if rotate:
        data_transform = ttransforms.Compose([lambda img: tfunctional.rotate(img, -90),
                                              lambda img: tfunctional.hflip(img),
                                              ttransforms.ToTensor(),
                                              ReshapeTransform((-1,)),
                                              ])
    else:
        data_transform = ttransforms.Compose([ttransforms.ToTensor(),
                                              ReshapeTransform((-1,)),
                                              ])

    kwargs = dict(download=True, transform=data_transform)
    if dataset == 'EMNIST':
        kwargs['split'] = 'bymerge'
        # kwargs['split'] = 'byclass'
    kwargs['train'] = True if train else False
    tdataset = eval('tdatasets.%s(paths.data_path, **kwargs)'%dataset)

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


def get_weight_matrix_in(n_h, algo, dataset='EMNIST', task=None):
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
    """
    if 'rp' in algo:
        path_name = os.path.join(paths.data_path, 'weight_matrices/', '%s_rp%d.npy'%(dataset, n_h))
        w_mat = np.load(path_name).T

        w_mat = np.random.randn(*w_mat.shape)
        w_mat = normalize(w_mat)

    elif algo == 'bp' or algo == 'bpo':
        class_idx = list(task[-1][dataset].keys())[0]

        path_name = os.path.join(paths.data_path, 'weight_matrices/%s_bp/l1o_weights/'%(dataset),
                                 'biaslearner_[%d]_testclass_%d_seed_0.pickle'%(n_h, class_idx))

        with open(path_name, 'rb') as f:
            ws, bs = pickle.load(f)
        w_mat = ws[0]

    else:
        path_name = os.path.join(paths.data_path, 'weight_matrices/', '%s_%s%d.npy'%(dataset, algo, n_h))
        w_mat = np.load(path_name).T

    print('input weights:', path_name)

    return w_mat


def get_weight_matrix_out(n_h, algo='', dataset='EMNIST', task=None, bias_opt=True):
    if bias_opt:
        if algo == "bpo":
            class_idx = list(task[-1][dataset].keys())[0]

            path_name = os.path.join(paths.data_path, 'weight_matrices/%s_bp/l1o_weights/'%(dataset),
                                     'biaslearner_[%d]_testclass_%d_seed_0.pickle'%(n_h, class_idx))

            print('output weights:', path_name)
            with open(path_name, 'rb') as f:
                ws, bs = pickle.load(f)
            w_vec = ws[1]

        else:
            w_vec = np.ones((n_h,1))
    else:
        w_vec = np.random.randn(n_h,1)
    w_vec /= np.linalg.norm(w_vec)

    return w_vec


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
    dataset: str
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
            {-1: {'EMNIST': {class_pairs[idx][0]: nsample}}, 1: {'EMNIST': {class_pairs[idx][1]: nsample}}} \
                for idx in task_pairs
                ]

    elif task_type == '1vall':
        class_idx = np.arange(nclass)
        c1 = np.random.choice(class_idx, ntask, replace=False)

        return [{-1: {'EMNIST': {c_: nsample}}, 1: {'EMNIST': [cc for cc in class_idx if cc != c_]}} for c_ in c1]

    else:
        raise Exception('Invalid `task_type`, choose \'1v1\' or \'1vall\'')


