
import torch
from tqdm import tqdm
import numpy as np
from torchvision import datasets, transforms


CIFAR_CHANNEL_AVGS = (0.4914, 0.4822, 0.4465)
CIFAR_CHANNEL_STDS = (0.2023, 0.1994, 0.2010)


def load_data(dataset='EMNIST', batch_size=100000, data_path='./datasets'):
    if dataset=='EMNIST':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
        data = datasets.EMNIST(data_path, 'bymerge', transform=trans, download=True)
        data_loader = torch.utils.data.DataLoader(
                         dataset=data,
                         batch_size=batch_size,
                         shuffle=False,
                         drop_last=True)

    return data_loader


def load_data_sm(dataset='EMNIST', batch_size=100000, data_path='./datasets'):
    if dataset=='EMNIST':
        data_set = datasets.EMNIST(data_path, train=True, download=True, split="bymerge", transform=np.array)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size)

    return data_loader


def get_big_data_matrix(data_loader):
    print("create big data matrix out of data loader (this takes a minute or so)...")
    X = []
    for (d, t) in tqdm(data_loader):
        s = d.shape
        X.append(d.numpy().reshape(data_loader.batch_size, s[-1]*s[-2]))

    return np.vstack(X)


def to_image_mnist(x):
    """
    Convert input vector to image

    Parameters
    ----------
    x: np.array of shape=(784,) or torch.Tensor of shape=(784,)
        input vector

    Returns
    -------
    np.array of shape=(28,28)
    """
    # x = 0.5 * (x + 1.)
    # x = np.clip(x, 0., 1.)
    if isinstance(x, np.ndarray):
        return np.reshape(x, (28, 28))
    elif isinstance(x, torch.Tensor):
        return torch.reshape(x, (28, 28))
    else:
        raise IOError("unknown data type, can not convert to MNIST image")


def to_image_cifar(x, remove_normalization=True):
    """
    Convert input vector to image

    Parameters
    ----------
    x: torch.tensor of shape=(3072,)
        input vector

    Returns
    -------
    np.array of shape=(28,28)
    """
    if isinstance(x, np.ndarray):
        img = np.reshape(x, (3, 32, 32))
        if remove_normalization:
            img *= np.array(CIFAR_CHANNEL_STDS)[:,None,None]
            img += np.array(CIFAR_CHANNEL_AVGS)[:,None,None]
    elif isinstance(x, torch.Tensor):
        img = torch.reshape(x, (3, 32, 32))
        if remove_normalization:
            img *= torch.FloatTensor(CIFAR_CHANNEL_STDS)[:,None,None]
            img += torch.FloatTensor(CIFAR_CHANNEL_AVGS)[:,None,None]
    else:
        raise IOError("unknown data type, can not convert to CIFAR image")

    return img


def differences_numpy(x, n_diff):
    """
    Sample differences between datapoints

    Parameters
    ----------
    x: numpy array (batch_size, input_dim)
        the data points
    n_diff: int
        the number of differences

    Returns
    -------
    numpy array (n_diff, input_dim)
        Random sample of the difference vectors
    """

    id1 = np.random.randint(x.shape[0], size=n_diff)
    id2 = np.random.randint(x.shape[0], size=n_diff)

    idx = np.where(np.not_equal(id1, id2))

    id1 = id1[idx]
    id2 = id2[idx]

    return x[id1,:] - x[id2,:]


def differences_torch(x, n_diff):
    """
    Sample differences between datapoints

    Parameters
    ----------
    x: torch.tensor (batch_size, input_dim)
        the data points
    n_diff: int
        the number of differences

    Returns
    -------
    torch.tensor (n_diff, input_dim)
        Random sample of the difference vectors
    """

    id1 = np.random.randint(x.shape[0], size=n_diff)
    id2 = np.random.randint(x.shape[0], size=n_diff)

    idx = np.where(np.not_equal(id1, id2))

    id1 = torch.LongTensor(id1[idx])
    id2 = torch.LongTensor(id2[idx])

    return x[id1,:] - x[id2,:]

def differences_and_midpoints_torch(x, n_diff):
    """
    Sample differences between datapoints

    Parameters
    ----------
    x: torch.tensor (batch_size, input_dim)
        the data points
    n_diff: int
        the number of differences

    Returns
    -------
    xdiff: torch.tensor (n_diff, input_dim)
        Random sample of the difference vectors
    xmidp: torch.tensor (n_diff, input_dim)
        Corresponding midpoints
    """

    id1 = np.random.randint(x.shape[0], size=n_diff)
    id2 = np.random.randint(x.shape[0], size=n_diff)

    idx = np.where(np.not_equal(id1, id2))

    id1 = torch.LongTensor(id1[idx])
    id2 = torch.LongTensor(id2[idx])

    return x[id1,:] - x[id2,:], (x[id1,:] + x[id2,:]) / 2.

