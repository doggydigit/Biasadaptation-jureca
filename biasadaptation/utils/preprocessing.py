import torch
import torchvision.datasets as tdatasets
import torchvision.transforms as ttransforms
import torchvision.transforms.functional as tfunctional

import biasadaptation.datasets as bdatasets

# average norm of the datapoints
DATA_NORMS = {'MNIST': 9.2147,
              'EMNIST': 10.3349,
              'K49': 10.546548,
              'CIFAR10': 66.87808,
              'CIFAR100': 70.78967}


class ReshapeTransform:
    """
    Transform class to reshape tensors
    """
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


def get_dataset(dataset, rotate=False, train=True, x_add=0., x_div=1., path=''):
    """
    Return the torch dataset

    Parameters
    ----------
    dataset: str
        The dataset name
    rotate: bool
        whether or not to rotate the data
    train: bool
        whether to load the train (``True``) or test (``False``) data
    x_add: float
        constant added to the datapoints
    x_div: float
        constant that divides the datapoints (x -> x / x_div + x_add)
    path: string
        path where the dataset is or will be stored
    """
    transform_list = []
    if rotate:
        transform_list.extend([ lambda img: tfunctional.rotate(img, -90),
                                lambda img: tfunctional.hflip(img),
                              ])


    if x_div == 'data':
        x_div = DATA_NORMS[dataset]

    if "MNIST" in dataset or "K49" in dataset:
        # dataset is an MNIST variation

        transform_list.extend([ ttransforms.ToTensor(),
                                lambda x: x / x_div + x_add,
                                ReshapeTransform((-1,)),
                              ])

        kwargs = dict(download=True, transform=ttransforms.Compose(transform_list))
        if dataset == 'EMNIST':
            kwargs['split'] = 'bymerge'

    elif "CIFAR" in dataset:

        transform_list.extend([
                ttransforms.ToTensor(),
                ttransforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                lambda x: x / x_div + x_add,
                ReshapeTransform((-1,))])

        kwargs = dict(download=True, transform=ttransforms.Compose(transform_list))

    kwargs['train'] = True if train else False
    try:
        tdataset = eval('tdatasets.%s(path, **kwargs)'%dataset)
    except AttributeError:
        # torchvision did not implement this dataset,
        # look for it in biasadaptation.datasets
        tdataset = eval('bdatasets.%s(path, **kwargs)'%dataset)

    return tdataset