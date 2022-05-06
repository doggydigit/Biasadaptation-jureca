import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import requests
from tqdm import tqdm

import os


K49_URLS = [ \
        'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz'
]


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

        self.f_name_data    = os.path.join(root, 'k49-%s-imgs.npz'%label)
        self.f_name_targets = os.path.join(root, 'k49-%s-labels.npz'%label)

        if not self._check_exists() and download:
            download_k49(root, K49_URLS)

        # load the data arrays in memmap mode
        self.data    = np.load(self.f_name_data, mmap_mode='r+')['arr_0']
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


