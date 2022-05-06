import numpy as np
import torch
import torchvision.datasets as tdatasets
import torchvision.transforms as ttransforms
import torch.utils.data as tdata

import random
import copy
import os


if os.uname()[0] == 'Darwin':
    DATASET_PATH =  '/Users/wybo/Data/'
elif os.uname()[0] == 'Linux' and (os.uname()[1] == 'pc59' or os.uname()[1] == 'pc58'):
    DATASET_PATH =  '/home/wybo/Data/'
else:
    # assumes we are running on hambach
    DATASET_PATH =  '/users/wybo/Data/'


DATA_NORMS = {'MNIST': 9.2147,
              'EMNIST': 10.3349}

N_CLASSES = {'MNIST': 10,
             'EMNIST': 47}

class NBinarySampler:
    """
    Give data batches suitable for multiple binary seperation meta-learning

    Attributes
    ----------
    n_per_batch: int
        the number of images per batch
    n_per_epoch: int
        the number of batches per epoch
    data_set: `torch.Dataset`
        the data set
    data_norm: float
        Global avarage norm of the datapoints
    ds: `torch.Dataset`
        The active dataset. Either `self.data_set_train` or `self.data_set_test`
    n_task: int
        The number of binary classification tasks
    ids: list of int
        The indices of the datapoints that figure in at least one of the binary
        seperation tasks
    targets: list of int
        Contains at its i'th position the corresponding target (-1 or +1) for the
        task the i'th datapoint in `self.ids` is involved in
    tasks: list of int
        Contains at its i'th position the index of the tasks
    """
    def __init__(self, d_name, n_per_batch=100, n_per_epoch=20, tasks=None,
                       normed=True, train=True):
        """
        Parameters
        ----------
        d_name: str ('MNIST' or 'EMNIST')
        n_per_batch: int
            number of samples per batch
        n_per_epoch: int
            number of batches per epoch
        tasks: None or list of tuples
            Pairs of class indices represent binary seperation tasks. First
            element of pair is the class index for which the task target is -1,
            second element is the class index for which the task target is +1.
            If None, lists `self.ids`, `self.targets` and `self.tasks` are not
            initialized and calling `self.__iter__` will result in an error.
            Call `self.set_nbinary_tasks(tasks)` to (re)initialize those lists.
        normed: bool
            Divide datapoints by the average global norm
        train: bool
            Whether to use train set (``True``) or test set (``False``)
        """
        self.n_per_batch = n_per_batch
        self.n_per_epoch = n_per_epoch
        self.d_name = d_name

        transform_list = [ttransforms.ToTensor()]
        if normed:
            data_norm = self.get_norm()
            transform_list.append(lambda x: x/data_norm)

        train_transform = ttransforms.Compose(transform_list)

        kwargs = dict(download=True, transform=train_transform)
        if d_name == 'EMNIST':
            kwargs['split'] = 'bymerge'
        kwargs['train'] = True if train else False
        self.data_set = eval('tdatasets.%s(DATASET_PATH, **kwargs)'%d_name)

        if tasks is not None:
            self.set_tasks(tasks)

    def get_input_dim(self):
        """
        Get the number of input dimensions
        """
        ds = self.data_set

        return np.prod(list(ds.data.shape)[1:])

    def get_norm(self):
        """
        Return and, if necessary, compute and set the average norm of the
        datapoints
        """
        if self.d_name in DATA_NORMS:
            return DATA_NORMS[self.d_name]

        elif hasattr(self, "data_norm"):
            return self.data_norm

        else:
            ds = self.data_set
            n_b = ds.data.shape[0]

            x_data = torch.squeeze(next(iter(tdata.DataLoader(ds, batch_size=n_b)))[0])
            self.data_norm = torch.mean(torch.norm(x_data, dim=(1, 2)))

            return self.data_norm

    def set_tasks(self, tasks):
        """
        Parameters
        ----------
        tasks: list of tuples
            Pairs of class indices. First element of pair is the class index
            for which the class label is -1, second element is the class index
            for which the class label is +1
        """
        ds = self.data_set
        ts = ds.targets.numpy()
        self.n_task = len(tasks)

        targets = [] # the target for the binary classification task
        task_ids = [] # the task index
        ids = [] # the index of the data point

        for jj, task in enumerate(tasks):
            for ii, class_id in enumerate(task):
                idx = np.where(np.equal(ts, class_id))[0]
                task = jj * np.ones_like(idx)
                target = 2*ii * np.ones_like(idx) - 1

                ids.extend(idx.tolist())
                task_ids.extend(task.tolist())
                targets.extend(target.tolist())

        self.ids = np.array(ids)
        self.tasks = np.array(task_ids)
        self.targets = np.array(targets)

    def __iter__(self):
        ds = self.data_set
        n_b = self.n_per_batch
        n_e = self.n_per_epoch

        idx = np.arange(len(self.ids))
        np.random.shuffle(idx)
        n_iter = max(min(len(idx) // n_b, n_e),1)

        for ii in range(n_iter):
            i0 = ii * n_b
            i1 = (ii+1) * n_b
            inds = idx[i0:i1]

            # load data corresponding to the indices
            data_loader = tdata.DataLoader(tdata.Subset(ds, self.ids[inds]),
                                           batch_size=n_b)
            x_data = next(iter(data_loader))[0]
            x_data = x_data.reshape(x_data.shape[0], np.prod(list(x_data.shape[1:])))

            x_tasks = torch.LongTensor(self.tasks[inds])
            x_targets = torch.LongTensor(self.targets[inds])

            yield x_data, x_tasks, x_targets



class SamplerPair_:
    """
    Pair of `NBinarySampler` for training and test set

    Attributes
    ----------
    nb_sampler_train: `NBinarySampler`
        The sampler for the training set
    nb_sampler_test: `NBinarySampler`
        The sampler for the testing set
    """
    def __init__(self, d_name, nb_factor_test=10, **kwargs):
        """
        Parameters
        ----------
        d_name: str ('MNIST' or 'EMNIST')
            the data set
        nb_factor_test: int
            batch size for testing is ``nb_factor_test`` times batch size in
            `kwargs`
        **kwargs:
            keyword arguments for the `NBinarySampler` initialization, with the
            exception of 'train'
        """
        if 'train' in kwargs:
            del kwargs['train']
        kwargs_train = copy.deepcopy(kwargs)
        kwargs_test = copy.deepcopy(kwargs)
        if 'n_per_batch' in kwargs_train:
            kwargs_test['n_per_batch'] = nb_factor_test * kwargs_train['n_per_batch']
        else:
            kwargs_test['n_per_batch'] = 1000

        self.nb_sampler_train = NBinarySampler(d_name, train=True, **kwargs_train)
        self.nb_sampler_test = NBinarySampler(d_name, train=False, **kwargs_test)

    def set_tasks(self, tasks):
        """
        Parameters
        ----------
        tasks: list of tuples
            Pairs of class indices. First element of pair is the class index
            for which the class label is -1, second element is the class index
            for which the class label is +1
        """
        self.nb_sampler_train.set_tasks(tasks)
        self.nb_sampler_test.set_tasks(tasks)

    def get_input_dim(self):
        """
        Get the number of input dimensions
        """
        dim_train = self.nb_sampler_train.get_input_dim()
        dim_test = self.nb_sampler_test.get_input_dim()
        assert dim_train == dim_test

        return dim_train


class NTaskSampler:
    """
    Give data batches suitable for multiple binary seperation meta-learning

    Attributes
    ----------
    n_per_batch: int
        the number of images per batch
    n_per_epoch: int
        the number of batches per epoch
    data_set: `torch.Dataset`
        the data set
    data_norm: float
        Global avarage norm of the datapoints
    ds: `torch.Dataset`
        The active dataset. Either `self.data_set_train` or `self.data_set_test`
    n_task: int
        The number of binary classification tasks
    ids: list of int
        The indices of the datapoints that figure in at least one of the binary
        seperation tasks
    targets: list of int
        Contains at its i'th position the corresponding target (-1 or +1) for the
        task the i'th datapoint in `self.ids` is involved in
    tasks: list of int
        Contains at its i'th position the index of the tasks
    target_type: str ('index', 'one hot' or 'perceptron')
        The format in which the targets are returned. Perceptron can only
        be chosen for binary classifications
    """
    def __init__(self, d_name,
                       tasks=None,
                       n_per_batch=100, n_per_epoch=20,
                       normed=True, target_type='index',
                       train=True):
        """
        Parameters
        ----------
        d_name: str ('MNIST' or 'EMNIST')
        n_per_batch: int
            number of samples per batch
        n_per_epoch: int
            number of batches per epoch
        tasks: None or list of tuples
            Pairs of class indices represent binary seperation tasks. First
            element of pair is the class index for which the task target is -1,
            second element is the class index for which the task target is +1.
            If None, lists `self.ids`, `self.targets` and `self.tasks` are not
            initialized and calling `self.__iter__` will result in an error.
            Call `self.set_tasks(tasks)` to (re)initialize those lists.
        normed: bool
            Divide datapoints by the average global norm
        train: bool
            Whether to use train set (``True``) or test set (``False``)
        target_type: str ('index', 'one hot' or 'perceptron')
            The format in which the targets are returned. Perceptron can only
            be chosen for binary classifications. Is only be set if `tasks`
            is not ``None``
        """
        self.n_per_batch = n_per_batch
        self.n_per_epoch = n_per_epoch
        self.d_name = d_name

        transform_list = [ttransforms.ToTensor()]
        if normed:
            data_norm = self.get_norm()
            transform_list.append(lambda x: x/data_norm)

        train_transform = ttransforms.Compose(transform_list)

        kwargs = dict(download=True, transform=train_transform)
        if d_name == 'EMNIST':
            kwargs['split'] = 'bymerge'
        kwargs['train'] = True if train else False
        self.data_set = eval('tdatasets.%s(DATASET_PATH, **kwargs)'%d_name)

        if tasks is not None:
            self.set_tasks(*tasks)
            self.set_target_type(target_type)

    def get_input_dim(self):
        """
        Get the number of input dimensions
        """
        ds = self.data_set

        return np.prod(list(ds.data.shape)[1:])

    def get_norm(self):
        """
        Return and, if necessary, compute and set the average norm of the
        datapoints
        """
        if self.d_name in DATA_NORMS:
            return DATA_NORMS[self.d_name]

        elif hasattr(self, "data_norm"):
            return self.data_norm

        else:
            ds = self.data_set
            n_b = ds.data.shape[0]

            x_data = torch.squeeze(next(iter(tdata.DataLoader(ds, batch_size=n_b)))[0])
            self.data_norm = torch.mean(torch.norm(x_data, dim=(1,2)))

            return self.data_norm

    def set_tasks(self, *tasks, target_type='index'):
        """
        Parameters
        ----------
        tasks: list of tuples
            Each argument is a list of tuples. The list represents the task,
            and each tuple is a class for that task. Each tuple consists of the
            class indices in the dataset that constitute a task class
        target_type: str ('index', 'one hot' or 'perceptron')
            The format in which the targets are returned. Perceptron can only
            be chosen for binary classifications
        """
        ds = self.data_set
        ts = ds.targets.numpy()
        self.n_task = len(tasks)
        self.n_class = len(tasks[0])

        self.set_target_type(target_type)

        n_max = 0
        idx_list_list = []

        # we first find the maximum number of samples in any given class, to
        # allow for proper balancing of the input datapoints, so that every
        # class in every tasks receive approximately the same number of samples
        for jj, task in enumerate(tasks):
            # check that all tasks have the same number of classes
            assert len(task) == self.n_class
            idx_list = []

            for ii, class_ids in enumerate(task):
                idx_ = [np.where(np.equal(ts, class_id))[0] for class_id in class_ids]
                idx = np.concatenate(idx_)

                n_max = len(idx) if len(idx) > n_max else n_max
                idx_list.append(idx)

            idx_list_list.append(idx_list)

        targets = [] # the target for the classification task
        task_ids = [] # the task index
        ids = [] # the index of the data point

        for jj, idx_list in enumerate(idx_list_list):
            for ii, idx in enumerate(idx_list):

                idx = np.concatenate((idx, np.random.choice(idx, n_max-len(idx))))

                task = jj * np.ones_like(idx)
                target = ii * np.ones_like(idx)

                ids.extend(idx.tolist())
                task_ids.extend(task.tolist())
                targets.extend(target.tolist())

        self.ids = np.array(ids)
        self.tasks = np.array(task_ids)
        self.targets = np.array(targets)

    def set_target_type(self, target_type='index'):
        """
        Parameters
        ----------
        target_type: str ('index', 'one hot' or 'perceptron')
            The format in which the targets are returned. Perceptron can only
            be chosen for binary classifications
        """
        if target_type == 'perceptron':
            assert self.n_class == 2

        self.target_type = target_type

    def __iter__(self):
        ds = self.data_set
        n_b = self.n_per_batch
        n_e = self.n_per_epoch

        idx = np.arange(len(self.ids))
        np.random.shuffle(idx)
        n_iter = max(min(len(idx) // n_b, n_e),1)

        for ii in range(n_iter):
            i0 = ii * n_b
            i1 = (ii+1) * n_b
            inds = idx[i0:i1]

            # load data corresponding to the indices
            data_loader = tdata.DataLoader(tdata.Subset(ds, self.ids[inds]),
                                           batch_size=n_b)
            x_data = next(iter(data_loader))[0]
            x_data = x_data.reshape(x_data.shape[0], np.prod(list(x_data.shape[1:])))

            x_tasks = torch.LongTensor(self.tasks[inds])

            if self.target_type == 'index':
                x_targets = torch.LongTensor(self.targets[inds])
            elif self.target_type == 'one hot':
                x_targets = torch.zeros((self.n_per_batch, self.n_class))
                x_targets[np.arange(self.n_per_batch), self.targets[inds]] += 1.
            elif self.target_type == 'perceptron':
                x_targets = torch.LongTensor(2.*self.targets[inds] - 1.)

            yield x_data, x_tasks, x_targets


class SamplerTriplet:
    """
    Pair of `NTaskSampler` for training and test set

    Attributes
    ----------
    nt_sampler_train: `NTaskSampler`
        The sampler for the training set
    nt_sampler_test: `NTaskSampler`
        The sampler for the testing set
    """
    def __init__(self, d_name, nb_factor_test=10, nb_factor_validate=10, **kwargs):
        """
        Parameters
        ----------
        d_name: str ('MNIST' or 'EMNIST')
            the data set
        nt_factor_test: int
            batch size for testing is ``nb_factor_test`` times batch size in
            `kwargs`
        **kwargs:
            keyword arguments for the `NBinarySampler` initialization, with the
            exception of 'train'
        """
        if 'train' in kwargs:
            del kwargs['train']
        kwargs_train = copy.deepcopy(kwargs)
        kwargs_validate = copy.deepcopy(kwargs)
        kwargs_test = copy.deepcopy(kwargs)
        if 'n_per_batch' in kwargs_train:
            kwargs_validate['n_per_batch'] = nb_factor_validate * kwargs_train['n_per_batch']
            kwargs_test['n_per_batch'] = nb_factor_test * kwargs_train['n_per_batch']
        else:
            kwargs_validate['n_per_batch'] = 1000
            kwargs_test['n_per_batch'] = 1000

        self.nt_sampler_train = NTaskSampler(d_name, train=True, **kwargs_train)
        self.nt_sampler_validate = NTaskSampler(d_name, train=True, **kwargs_validate)
        self.nt_sampler_test = NTaskSampler(d_name, train=False, **kwargs_test)

    def set_tasks(self, tasks, target_type='index'):
        """
        Parameters
        ----------
        tasks: list of tuples
            Each argument is a list of tuples. The list represents the task,
            and each tuple is a class for that task. Each tuple consists of the
            class indices in the dataset that constitute a task class
        target_type: str ('index', 'one hot' or 'perceptron')
            The format in which the targets are returned. Perceptron can only
            be chosen for binary classifications
        """
        self.nt_sampler_train.set_tasks(tasks, target_type=target_type)
        self.nt_sampler_validate.set_tasks(tasks, target_type=target_type)
        self.nt_sampler_test.set_tasks(tasks, target_type=target_type)

    def set_target_type(self, target_type='index'):
        """
        Parameters
        ----------
        target_type: str ('index', 'one hot' or 'perceptron')
            The format in which the targets are returned. Perceptron can only
            be chosen for binary classifications
        """
        self.nt_sampler_train.set_target_type(target_type=target_type)
        self.nt_sampler_validate.set_target_type(target_type=target_type)
        self.nt_sampler_test.set_target_type(target_type=target_type)

    def get_input_dim(self):
        """
        Get the number of input dimensions
        """
        dim_train = self.nt_sampler_train.get_input_dim()
        dim_validate = self.nt_sampler_validate.get_input_dim()
        dim_test = self.nt_sampler_test.get_input_dim()
        assert dim_train == dim_test and dim_train == dim_validate

        return dim_train





