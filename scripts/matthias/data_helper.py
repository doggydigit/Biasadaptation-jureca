from torch import Generator, as_tensor, reshape
from torch.nn.functional import one_hot
from torch.utils.data import random_split
from torchvision.datasets import MNIST, QMNIST, EMNIST, KMNIST, CIFAR100
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.transforms.functional import hflip, rotate
from biasadaptation.utils.k_task_n_class_m_dataset_data import KTaskNClassMDatasetData
from tasks_2d_helper import get_all_data_2dtasks
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import requests
from tqdm import tqdm

import os


K49_URLS = ['http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
            'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz']


def download_k49(root, url_list):
    """
    Adapted from

    https://github.com/rois-codh/kmnist/blob/master/download_data.py
    """
    for url in url_list:
        path = url.split('/')[-1]
        f_name = os.path.join(root, path)

        r = requests.get(url, stream=True)
        with open(f_name, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))

            for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
                if chunk:
                    f.write(chunk)
    print('All dataset files downloaded!')


class K49(Dataset):
    """
    MNIST-like implementation of Kuzushiji-49 dataset
    """
    def __init__(self, root, train=True, download=True, transform=None, target_transform=None):
        label = 'train' if train else 'test'

        root = os.path.join(root, self.__class__.__name__)
        if not os.path.isdir(root):
            os.makedirs(root)

        self.f_name_data = os.path.join(root, 'k49-%s-imgs.npz' % label)
        self.f_name_targets = os.path.join(root, 'k49-%s-labels.npz' % label)

        if not self._check_exists() and download:
            download_k49(root, K49_URLS)

        # load the data arrays in memmap mode
        self.data = np.load(self.f_name_data, mmap_mode='r+')['arr_0']
        self.targets = np.load(self.f_name_targets, mmap_mode='r+')['arr_0']

        self.transform = transform
        self.target_transform = target_transform

    def _check_exists(self):
        return (os.path.exists(self.f_name_data) and
                os.path.exists(self.f_name_targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Adapted from torchvision.MNIST

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class ReshapeTransform:
    """
    Transform class to reshape tensors
    """
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return reshape(img, self.new_size)


def get_number_classes(dataset):
    """
    Get the original number of classes of a dataset
    Parameters
    ----------
    dataset: Name of the dataset

    Returns the number of classes of the dataset
    -------

    """
    nr_classes_per_dataset = {"MNIST": 10, "QMNIST": 10, "EMNIST": 47, "EMNIST_letters": 26, "EMNIST_bymerge": 47,
                              "EMNIST_bymerge_bw": 47, "EMNIST_willem": 47, "KMNIST": 10, "TASKS2D": 48, "K49": 49,
                              "CIFAR100": 100}
    return nr_classes_per_dataset[dataset]


def get_labels(dataset):
    """
    Get labels of a dataset
    Parameters
    ----------
    dataset: Name of the dataset

    Returns a list of all the labels of the dataset
    -------

    """
    nr_labels = get_number_classes(dataset)
    if dataset == "EMNIST_letters":
        label_list = list(range(1, nr_labels + 1))
    else:
        label_list = list(range(nr_labels))
    return label_list


def get_dataset(dataset, root_dir, train):
    """
    Getter for dataset objects with standardized preprocessing,
    Parameters
    ----------
    dataset:    Name of the dataset
    root_dir:   Path to the root directory of BiasAdaptation project
    train:      Whether to load the training set or (if False) the testing set

    Returns dataset object
    -------

    """
    root = root_dir + "biasadaptation/utils/data/"
    if dataset == "MNIST":
        mnist_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,)), ReshapeTransform((-1,))])
        data = MNIST(root, train=train, transform=mnist_transform, download=True)

    elif dataset == "QMNIST":
        mnist_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,)), ReshapeTransform((-1,))])
        data = QMNIST(root, train=train, transform=mnist_transform, download=True)

    elif dataset == "EMNIST":
        emnist_transf = Compose([lambda img: rotate(img, -90), lambda img: hflip(img), ToTensor(),
                                 ReshapeTransform((-1,))])
        data = EMNIST(root, "balanced", train=train, transform=emnist_transf, download=True)

    elif dataset == "EMNIST_bymerge":
        emnist_transf = Compose([lambda img: rotate(img, -90), lambda img: hflip(img), ToTensor(),
                                 ReshapeTransform((-1,))])
        data = EMNIST(root, "bymerge", train=train, transform=emnist_transf, download=True)

    elif dataset == "EMNIST_willem":
        emnist_transf = Compose([ToTensor(), ReshapeTransform((-1,))])
        data = EMNIST(root, "bymerge", train=train, transform=emnist_transf, download=True)

    elif dataset == "EMNIST_letters":
        emnist_transf = Compose([lambda img: rotate(img, -90), lambda img: hflip(img), ToTensor(),
                                 ReshapeTransform((-1,))])
        data = EMNIST(root, "letters", train=train, transform=emnist_transf, download=True)

    elif dataset == "KMNIST":
        data = KMNIST(root, train=train, transform=Compose([ToTensor(), ReshapeTransform((-1,))]), download=True)

    elif dataset == "K49":
        data = K49(root, train=train, transform=Compose([ToTensor(), ReshapeTransform((-1,))]), download=True)

    elif dataset == "CIFAR100":
        data = CIFAR100(
            root, train=train, transform=Compose([ToTensor(),
                                                  Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                  ReshapeTransform((-1,))]), download=True
        )

    else:
        raise NotImplementedError("Wrong dataset '{}'. Available are MNIST, QMNIST, EMNIST and EMNIST_letters."
                                  "".format(dataset))

    return data


def get_dataset_nrclasses_labels(dataset, root_dir, train):
    """
    Gets a tuple of the dataset, the number of classes and a list of labels of the dataset
    Parameters
    ----------
    dataset:    Name of the dataset
    root_dir:   Path to the root directory of BiasAdaptation project
    train:      Whether to load the training set or (if False) the testing set

    Returns dataset object, the number of classes and a list of labels of the dataset
    -------

    """
    return get_dataset(dataset, root_dir, train), get_number_classes(dataset), get_labels(dataset)


def get_dataset_size(dataset, train):
    """

    Parameters
    ----------
    dataset: Name of the dataset
    train:   Whether get size of train data or (if False) test data


    Returns the original train set size and the maximal (final) train set size if we set apart a validation set.
    -------

    """
    if train:
        traindataset_sizes = {"MNIST": 60000, "QMNIST": 60000, "EMNIST": 112800, "EMNIST_letters": 124800,
                              "EMNIST_bymerge": 697932, "EMNIST_willem": 697932, "KMNIST": 60000, "K49": 232365,
                              "CIFAR100": 50000}

        max_data_train_sizes = {"MNIST": 50000, "QMNIST": 50000, "EMNIST": 100000, "EMNIST_letters": 100000,
                                "EMNIST_bymerge": 650000, "EMNIST_willem": 650000, "KMNIST": 50000, "K49": 200000,
                                "CIFAR100": 41000}
        return traindataset_sizes[dataset], max_data_train_sizes[dataset]
    else:
        testdataset_sizes = {"MNIST": 10000, "QMNIST": 60000, "EMNIST": 18800, "EMNIST_letters": 20800,
                             "EMNIST_bymerge": 116323, "EMNIST_willem": 116323, "KMNIST": 10000, "K49": 38547,
                             "CIFAR100": 10000}
        return testdataset_sizes[dataset], None


def data_splitter(min_nr, splitting, data, seed, debug, tasks, dataset, train):
    """
    Produce KTaskNClassMDatasetData object for the desired data. Will produce two in case split is True that can be used
    as training and validation sets.
    Parameters
    ----------
    min_nr:     Minimal size of the training and validation sets; used as train size in case of debugging
    splitting:  Bool whether to produce a training and validation set or just a single KTaskNClassMDatasetData object
    data:       Dataset object to process
    seed:       Random seed for split to ensure it can be reproduced throughout later simulations
    debug:      Bool defining whether you are in debug mode
    tasks:      List of task dictionaries for the KTaskNClassMDatasetData object (see k_task_n_class_m_dataset_data.py)
    dataset:    String with the name of the new dataset
    train:      Whether to load train data or (if False) test data

    Returns Either one or two KTaskNClassMDatasetData objects with the new dataset
    -------

    """

    assert train or not splitting  # There should be no reason to ever split the test set

    total_size, max_train_size = get_dataset_size(dataset, train)
    if splitting:
        train_nr = max_train_size - (max_train_size % min_nr)
        max_valsize = total_size - train_nr
        valid_nr = max_valsize - (max_valsize % min_nr)
        split = [train_nr, total_size - train_nr]
        training_set, validation_set = random_split(
            data, split, generator=Generator().manual_seed(seed)
        )
        if debug:
            train_nr = min_nr
            valid_nr = min_nr
        train_data = KTaskNClassMDatasetData(size=train_nr * 2, tasks=tasks, datasets={dataset: training_set},
                                             reinitialize_cache=True, cache_suffix="train")
        validation_data = KTaskNClassMDatasetData(size=valid_nr * 2, tasks=tasks, datasets={dataset: validation_set},
                                                  reinitialize_cache=True, cache_suffix="valid")
        return train_data, validation_data

    else:
        if debug:
            data_nr = min_nr
        else:
            data_nr = total_size - (total_size % min_nr)
        return KTaskNClassMDatasetData(size=data_nr * 2, tasks=tasks, datasets={dataset: data},
                                       reinitialize_cache=True)


def get_all_data(dataset, model_type="biaslearner", splitting=True, train=True, seed=0, debug=False, root_dir="../../"):
    """
    Creates new dataset of all possible 1 vs. all tasks for the desired dataset.
    Parameters
    ----------
    dataset:    Name of dataset
    model_type: Name of the type of model that will use the data
    splitting:  Bool whether to split dataset in training and validation set
    train:      Whether to load train data or (if False) test data
    seed:       Random seed for reproducibility
    debug:      Bool defining whether you want to run in debug mode
    root_dir:   Path to the root directory of BiasAdaptation project

    Returns either one or two KTaskNClassMDatasetData objects with all 1 vs. all tasks for the desired dataset
    -------
    """

    if dataset == "TASKS2D":
        return get_all_data_2dtasks(splitting=splitting, train=train, seed=seed)

    data, totalnrclasses, traindigits = get_dataset_nrclasses_labels(dataset, root_dir, train)

    if model_type in ["biaslearner", "binarymr", "gainlearner", "bglearner", "xshiftlearner"]:
        min_datasize = totalnrclasses * (totalnrclasses - 1)
        tasks = [{-1: {dataset: traindigits[:i] + traindigits[i + 1:totalnrclasses]},
                  1: {dataset: [traindigits[i]]}} for i in range(totalnrclasses)]

    elif model_type == "multireadout":
        min_datasize = totalnrclasses
        tasks = [{}]
        for t in range(totalnrclasses):
            label = one_hot(as_tensor(t), num_classes=totalnrclasses)
            tasks[0][label] = {dataset: [traindigits[t]]}
    else:
        raise ValueError(model_type)

    return data_splitter(min_datasize, splitting, data, seed, debug, tasks, dataset, train)


def get_leave1out_traindata(testclass, model_type, dataset, splitting=True, train=True, seed=0, debug=False,
                            root_dir="../../"):
    """
    Creates new dataset of all possible 1 vs. all tasks for the desired dataset but leaving out one class.
    Parameters
    ----------
    testclass:  The id of the class to leave out
    dataset:    Name of dataset
    model_type: Name of the type of model that will use the data
    splitting:  Bool whether to split dataset in training and validation set
    train:      Whether to load train data or (if False) test data
    seed:       Random seed for reproducibility
    debug:      Bool defining whether you want to run in debug mode
    root_dir:   Path to the root directory of BiasAdaptation project

    Returns either one or two KTaskNClassMDatasetData objects with all 1 vs. all tasks for the desired dataset
    -------
    """
    data, totalnrclasses, label_list = get_dataset_nrclasses_labels(dataset, root_dir, train)
    if model_type in ["biaslearner", "binarymr"]:
        min_datasize = (totalnrclasses - 1) * (totalnrclasses - 2)
        traindigits = label_list[:testclass] + label_list[testclass+1:]
        tasks = [{-1: {dataset: traindigits[:i] + traindigits[i+1:totalnrclasses-1]},
                  1: {dataset: [traindigits[i]]}} for i in range(totalnrclasses-1)]
    elif model_type == "multireadout":
        min_datasize = totalnrclasses - 1
        tasks = [{}]
        traindigits = label_list[:testclass] + label_list[testclass+1:]
        for t in range(totalnrclasses - 1):
            label = one_hot(as_tensor(t), num_classes=totalnrclasses-1)
            tasks[0][label] = {dataset: [traindigits[t]]}
    else:
        raise ValueError(model_type)

    return data_splitter(min_datasize, splitting, data, seed, debug, tasks, dataset, train)


def get_leave1out_transferdata(testclass, model_type, dataset, splitting=True, train=True, seed=0, debug=False,
                               root_dir="../../"):
    """
    Creates new dataset with one class vs. all the other tasks from the desired dataset.
    Parameters
    ----------
    testclass:  The id of the class to stage against the remaining ones
    dataset:    Name of dataset
    model_type: Name of the type of model that will use the data
    splitting:  Bool whether to split dataset in training and validation set
    train:      Whether to load train data or (if False) test data
    seed:       Random seed for reproducibility
    debug:      Bool defining whether you want to run in debug mode
    root_dir:    Path to the root directory of BiasAdaptation project

    Returns either one or two KTaskNClassMDatasetData objects with all 1 vs. all tasks for the desired dataset
    -------

    """
    if model_type in ["biaslearner", "binarymr"]:
        data, totalnrclasses, digits = get_dataset_nrclasses_labels(dataset, root_dir, train)
        min_datasize = 2 * (totalnrclasses - 1)
        tasks = [{-1: {dataset: digits[:testclass] + digits[testclass + 1:]}, 1: {dataset: [digits[testclass]]}}]

        return data_splitter(min_datasize, splitting, data, seed, debug, tasks, dataset, train)
    elif model_type == "multireadout":
        return get_all_data(dataset=dataset, model_type=model_type, splitting=splitting, train=train,
                            seed=seed, debug=debug, root_dir=root_dir)
    else:
        raise ValueError(model_type)


def get_singular_data(class_id, dataset, splitting=True, train=True, seed=0, debug=False, root_dir="../../"):
    """
    Creates new dataset of all possible 1 vs. all tasks for the desired dataset.
    Parameters
    ----------
    class_id:   ID of the class of interest
    dataset:    Name of dataset
    splitting:  Bool whether to split dataset in training and validation set
    train:      Whether to load train data or (if False) test data
    seed:       Random seed for reproducibility
    debug:      Bool defining whether you want to run in debug mode
    root_dir:   Path to the root directory of BiasAdaptation project

    Returns either one or two KTaskNClassMDatasetData objects with the 1 vs. all tasks for the desired class the dataset
    -------

    """

    data, totalnrclasses, traindigits = get_dataset_nrclasses_labels(dataset, root_dir, train)
    min_datasize = 2 * (totalnrclasses - 1)
    tasks = [{-1: {dataset: traindigits[:class_id] + traindigits[class_id + 1:totalnrclasses]},
              1: {dataset: [traindigits[class_id]]}}]

    return data_splitter(min_datasize, splitting, data, seed, debug, tasks, dataset, train)


def get_dataloader(dataset, max_batch_size=10000):
    nr_samples = len(dataset)
    batch_size = min(max_batch_size, nr_samples)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader, nr_samples
    

if __name__ == '__main__':
    for ds in ["K49", "CIFAR100", "EMNIST_bymerge"]:
        get_all_data(dataset=ds)
        
