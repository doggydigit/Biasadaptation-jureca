from math import pi
from os.path import isfile as path_isfile
from pickle import dump as pickle_dump
from pickle import load as pickle_load
from pickle import HIGHEST_PROTOCOL
from random import seed as rd_seed
from numpy.random import seed as np_seed
from torch import manual_seed as torch_seed
from torch import Generator
from torch import sum as tsum
from torch import sin as tsin
from torch import empty as t_empty
from torch.utils.data import random_split
from torch.utils.data import Dataset
from biasadaptation.utils.k_task_n_class_m_dataset_data import KTaskNClassMDatasetData


def task_2d_label(task, input_data):
    """
    Get labels for one of the 2d tasks for the desired inputs
    Parameters
    ----------
    task: task id
    input_data: tensor of inputs with shape nx2 for which to compute n labels

    Returns labels for the given input for the desired task
    -------

    """

    if task == 0:
        # Horizontal Line
        output = input_data[:, 1] > 0.5

    elif task == 1:
        # Vertical Line
        output = input_data[:, 0] < 0.5

    elif task == 2:
        # Y is smaller X
        output = input_data[:, 1] < input_data[:, 0]

    elif task == 3:
        # Y is bigger X
        output = input_data[:, 1] > input_data[:, 0]

    elif task == 4:
        # Diagonal from upper left to bottom right
        output = input_data[:, 1] > (-input_data[:, 0] + 1.)

    elif task == 5:
        # Below diagonal from upper left to bottom right
        output = input_data[:, 1] < (-input_data[:, 0] + 1.)

    elif task == 6:
        # Within horizontal band
        output = (0.25 < input_data[:, 1]) & (input_data[:, 1] < 0.75)

    elif task == 7:
        # Outside horizontal band
        output = (0.25 > input_data[:, 1]) | (input_data[:, 1] > 0.75)

    elif task == 8:
        # Within vertical band
        output = (0.25 < input_data[:, 0]) & (input_data[:, 0] < 0.75)

    elif task == 9:
        # Outside vertical band
        output = (0.25 > input_data[:, 0]) | (input_data[:, 0] > 0.75)

    elif task == 10:
        # Within diagonal band from lower left to upper right
        x = input_data[:, 1] + input_data[:, 0]
        output = (0.70710678118 < x) & (x < 1.29289321881)

    elif task == 11:
        # Outside diagonal band from lower left to upper right
        x = input_data[:, 1] + input_data[:, 0]
        output = (0.70710678118 > x) | (x > 1.29289321881)

    elif task == 12:
        # Within diagonal band from upper left to bottom right
        x = input_data[:, 1] - input_data[:, 0] + 1.
        output = (0.70710678118 < x) & (x < 1.29289321881)

    elif task == 13:
        # Outside diagonal band from upper left to bottom right
        x = input_data[:, 1] - input_data[:, 0] + 1.
        output = (0.70710678118 > x) | (x > 1.29289321881)

    elif task == 14:
        # Inside bottom left diamond
        output = (2 * input_data[:, 1] < input_data[:, 0] + 1) & (input_data[:, 1] + 1 > 2 * input_data[:, 0])

    elif task == 15:
        # Inside top right diamond
        output = (2 * input_data[:, 1] > input_data[:, 0]) & (input_data[:, 1] < 2 * input_data[:, 0])

    elif task == 16:
        # Inside bottom right diamond
        output = (input_data[:, 1] > - 2 * input_data[:, 0] + 1) & (2 * input_data[:, 1] - 2 < - input_data[:, 0])

    elif task == 17:
        # Inside top left diamond
        output = (2 * input_data[:, 1] - 1 > - input_data[:, 0]) & (input_data[:, 1] - 2 < -2 * input_data[:, 0])

    elif task == 18:
        # Outside bottom left diamond
        output = (2 * input_data[:, 1] > input_data[:, 0] + 1) ^ (input_data[:, 1] + 1 < 2 * input_data[:, 0])

    elif task == 19:
        # Outside top right diamond
        output = (2 * input_data[:, 1] < input_data[:, 0]) ^ (input_data[:, 1] > 2 * input_data[:, 0])

    elif task == 20:
        # Outside bottom right diamond
        output = (input_data[:, 1] < - 2 * input_data[:, 0] + 1) ^ (2 * input_data[:, 1] - 2 > - input_data[:, 0])

    elif task == 21:
        # Outside top left diamond
        output = (2 * input_data[:, 1] - 1 < - input_data[:, 0]) ^ (input_data[:, 1] - 2 > -2 * input_data[:, 0])

    elif task == 22:
        # Outside rightarrow
        output = (-2 * input_data[:, 1] + 2 > input_data[:, 0]) ^ (2 * input_data[:, 1] > input_data[:, 0])

    elif task == 23:
        # Inside rightarrow
        output = (-2 * input_data[:, 1] + 2 > input_data[:, 0]) ^ (2 * input_data[:, 1] < input_data[:, 0])

    elif task == 24:
        # Outside leftarrow
        output = (input_data[:, 0] + 1 < 2 * input_data[:, 1]) ^ (-2 * input_data[:, 1] + 1 > input_data[:, 0])

    elif task == 25:
        # Inside leftarrow
        output = (input_data[:, 0] + 1 < 2 * input_data[:, 1]) ^ (-2 * input_data[:, 1] + 1 < input_data[:, 0])

    elif task == 26:
        # XOR Operation
        output = (input_data[:, 0] > 0.5) ^ (input_data[:, 1] > 0.5)

    elif task == 27:
        # NXOR Operation
        output = (input_data[:, 0] < 0.5) ^ (input_data[:, 1] > 0.5)

    elif task == 28:
        # Diagonal XOR 1
        output = (input_data[:, 1] > (-input_data[:, 0] + 1.)) ^ (input_data[:, 1] < input_data[:, 0])

    elif task == 29:
        # Diagonal XOR 2
        output = (input_data[:, 1] < (-input_data[:, 0] + 1.)) ^ (input_data[:, 1] < input_data[:, 0])

    elif task == 30:
        # Inside circle of radius 0.7978845608 with center (0, 0)
        output = tsum(input_data * input_data, dim=1) < 0.7978845608

    elif task == 31:
        # Outside circle of radius 0.7978845608 with center (0, 0)
        output = tsum(input_data * input_data, dim=1) > 0.7978845608

    elif task == 32:
        # Inside circle of radius 0.7978845608 with center (1, 1)
        output = tsum((input_data - 1.) * (input_data - 1.), dim=1) < 0.7978845608

    elif task == 33:
        # Outside circle with radius 0.39894228 and center (1, 1)
        output = tsum((input_data - 1.) * (input_data - 1.), dim=1) > 0.7978845608

    elif task == 34:
        # Y is above quadratic function f(X)
        a, c = 64. / 9., 16. / 9.
        x = input_data[:, 0]
        output = input_data[:, 1] > a * x * x - a * x + c

    elif task == 35:
        # Y is below quadratic function f(X)
        a, c = 64. / 9., 16. / 9.
        x = input_data[:, 0]
        output = input_data[:, 1] < a * x * x - a * x + c

    elif task == 36:
        # Y is above negative quadratic function f(X)
        a, c = 64. / 9., 7. / 9.
        x = input_data[:, 0]
        output = input_data[:, 1] > -a * x * x + a * x - c

    elif task == 37:
        # Y is below negative quadratic function f(X)
        a, c = 64. / 9., 7. / 9.
        x = input_data[:, 0]
        output = input_data[:, 1] < -a * x * x + a * x - c

    elif task == 38:
        # Inside circle with radius 0.39894228 and center (0.5, 0.5)
        output = (input_data[:, 1] - 0.5) ** 2 + (input_data[:, 0] - 0.5) ** 2 < 0.15915494309

    elif task == 39:
        # Outside circle with radius 0.39894228 and center (0.5, 0.5)
        output = (input_data[:, 1] - 0.5) ** 2 + (input_data[:, 0] - 0.5) ** 2 > 0.15915494309

    elif task == 40:
        # Within ring
        x = (input_data[:, 1] - 0.5) ** 2 + (input_data[:, 0] - 0.5) ** 2
        output = (0.0908450569 < x) & (x < 0.25)

    elif task == 41:
        # Outside ring
        x = (input_data[:, 1] - 0.5) ** 2 + (input_data[:, 0] - 0.5) ** 2
        output = (0.0908450569 > x) | (x > 0.25)

    elif task == 42:
        # Above Sinus
        output = tsin(2 * pi * input_data[:, 0]) < 2 * input_data[:, 1] - 1.

    elif task == 43:
        # Below Sinus
        output = tsin(2 * pi * input_data[:, 0]) > 2 * input_data[:, 1] - 1.

    elif task == 44:
        # Right of Sinus
        output = tsin(2 * pi * input_data[:, 1]) < 2 * input_data[:, 0] - 1.

    elif task == 45:
        # Left of Sinus
        output = tsin(2 * pi * input_data[:, 1]) > 2 * input_data[:, 0] - 1.

    elif task == 46:
        # Outside corner circles
        output = (tsum(input_data * input_data, dim=1) > 0.15915494309) ^ \
                 ((input_data[:, 0] - 1) ** 2 + input_data[:, 1] ** 2 > 0.15915494309) ^ \
                 ((input_data[:, 1] - 1) ** 2 + input_data[:, 0] ** 2 > 0.15915494309) ^ \
                 (tsum((input_data - 1) * (input_data - 1.), dim=1) > 0.15915494309)

    elif task == 47:
        # Inside corner circles
        output = (tsum(input_data * input_data, dim=1) > 0.15915494309) & \
                 ((input_data[:, 0] - 1) ** 2 + input_data[:, 1] ** 2 > 0.15915494309) & \
                 ((input_data[:, 1] - 1) ** 2 + input_data[:, 0] ** 2 > 0.15915494309) & \
                 (tsum((input_data - 1) * (input_data - 1.), dim=1) > 0.15915494309)

    else:
        raise NotImplementedError

    return output.int()


class TASKS2D(Dataset):
    """Dataset for one of the 48 binary classification tasks with 2D inputs."""

    def __init__(self, task, root_dir="../../biasadaptation/utils/data/TASKS2D/", train=True):
        """
        Args:
            task (int): ID of the task
            root_dir (string): Directory where dataset is stored.
            train: whether to load train or test set.
        """
        self.root_dir = root_dir

        if train:
            train_string = "training"
        else:
            train_string = "test"
        file_name = root_dir + "task_2d_{}_{}.pickle".format(task, train_string)

        if not path_isfile(file_name):
            if train:
                seed = 0
            else:
                seed = 1
            rd_seed(seed)
            np_seed(seed)
            torch_seed(seed)
            d_size = 24576
            x = t_empty(d_size, 2).uniform_(0., 1.)
            while True:
                labels = task_2d_label(task, x).tolist()
                if sum(labels) == d_size / 2:
                    break
                else:
                    if sum(labels) > d_size / 2:
                        new_target = 0
                        i = labels.index(1)
                    else:
                        new_target = 1
                        i = labels.index(0)
                    while True:
                        new_x = t_empty(1, 2).uniform_(0., 1.)
                        new_label = task_2d_label(task, new_x).tolist()[0]
                        if new_label == new_target:
                            x[i, :] = new_x
                            labels[i] = new_label
                            break

            with open(file_name, 'wb') as handle:
                pickle_dump((x, task_2d_label(task, x).tolist()), handle, protocol=HIGHEST_PROTOCOL)

        with open(file_name, 'rb') as handle:
            self.samples, self.labels = pickle_load(handle)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx], self.labels[idx]


def get_all_data_2dtasks(splitting=True, train=True, seed=0):
    """
    Creates the KNM multitask dataset containing all 48 tasks of binary classification with 2D inputs.
    Parameters
    ----------
    splitting: Whether to split the train set into train and validation set
    train: Whether to get train or test set
    seed: random seed to use for the splitting into train and validation set

    Returns KNM dataset of all 2D tasks
    -------

    """
    nrtasks = 48
    tasks = [{-1: {"task2d_{}".format(t): [0]}, 1: {"task2d_{}".format(t): [1]}} for t in range(nrtasks)]
    if train:
        if splitting:
            trainsets = {}
            validsets = {}
            train_nr = 18048
            valid_nr = 4512
            split = [train_nr, 24576 - train_nr]
            for t in range(nrtasks):
                data = TASKS2D(t)
                trainset, validset = random_split(
                    data, split, generator=Generator().manual_seed(seed)
                )
                trainsets["task2d_{}".format(t)] = trainset
                validsets["task2d_{}".format(t)] = validset

            train_data = KTaskNClassMDatasetData(size=train_nr, tasks=tasks, reinitialize_cache=True,
                                                 datasets=trainsets, cache_suffix='train')
            validation_data = KTaskNClassMDatasetData(size=valid_nr, tasks=tasks, reinitialize_cache=True,
                                                      datasets=validsets, cache_suffix='valid')
            return train_data, validation_data
        else:
            raise NotImplementedError
    else:
        datasets = {}
        for t in range(nrtasks):
            datasets["task2d_{}".format(t)] = TASKS2D(t, train=False)
        return KTaskNClassMDatasetData(size=24576, tasks=tasks, datasets=datasets)
