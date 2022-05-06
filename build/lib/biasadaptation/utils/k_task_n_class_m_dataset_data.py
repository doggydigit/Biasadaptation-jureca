import pickle
import warnings
from itertools import islice, cycle

import torch
import numpy as np


class CustomClass:
    def __init__(self, *, n_samples, sample_indices_by_dataset_by_class, **class_definition):
        """
        A custom class constructed from several datasets.
        """

        self._n_samples = n_samples
        self._sample_indices_by_dataset_by_class = sample_indices_by_dataset_by_class
        self._n_subclasses = self._compute_n_subclasses(class_definition)
        self._samples_map = []

        self._construct_class(class_definition)

    @property
    def _n_samples_per_subclass(self):
        n_samples_per_subclass = self._n_samples // self._n_subclasses

        if n_samples_per_subclass * self._n_subclasses != self._n_samples:
            raise ValueError("could not split {} samples evenly across {} subclasses".format(self._n_samples,
                                                                                             self._n_subclasses))

        return n_samples_per_subclass

    def _compute_n_subclasses(self, class_definition):
        return sum(len(subclasses) for subclasses in class_definition.values())

    def _construct_class(self, class_definition):
        # we store a list of tuples, where each tuple contains a
        # maximal index, a dataset name and a subclass label; the
        # maximal index indicates that all samples below this value
        # are part of the same subclass of the same dataset
        maximal_idx = 0
        for dataset_name in class_definition:

            for subclass_label in class_definition[dataset_name]:

                n_samples_in_total = len(
                        self._sample_indices_by_dataset_by_class[dataset_name][subclass_label]
                    )

                if isinstance(class_definition[dataset_name], dict):
                    # if the number of samples to be used for sampling is
                    # provided, use it to constrain the number of samples,
                    # but only if this number is less than the samples per subclass
                    n_samples_for_sampling = min(
                            class_definition[dataset_name][subclass_label], self._n_samples_per_subclass
                        )
                else:
                    # if the number of samples is not provided, the number of
                    # samples will either be the total number of samples in
                    # the subclass, or the number of samples per subclass,
                    # whichever is smaller
                    n_samples_for_sampling =  min(
                            n_samples_in_total, self._n_samples_per_subclass
                        )

                if n_samples_for_sampling <= n_samples_in_total:
                    # we subsample the total set of samples for the subclass,
                    # and then repeat that list until we reach the desired number
                    # of samples
                    sample_idxs = np.random.choice(
                            self._sample_indices_by_dataset_by_class[dataset_name][subclass_label],
                            n_samples_for_sampling,
                            replace=False
                        ).tolist()

                    sample_idxs = list(islice(
                            cycle(sample_idxs), self._n_samples_per_subclass
                        ))
                else:
                    # we repeat the samples of the subclass until we reach the
                    # desired number of samples
                    sample_idxs = list(
                            self._sample_indices_by_dataset_by_class[dataset_name][subclass_label]
                        )

                    sample_idxs = list(islice(
                            cycle(sample_idxs), self._n_samples_per_subclass
                        ))

                maximal_idx += self._n_samples_per_subclass
                self._samples_map.append((maximal_idx, dataset_name, subclass_label, sample_idxs))

    def _find_dataset_and_subclass(self, idx):
        for maximal_idx, dataset_name, subclass_label, sample_idxs in self._samples_map:
            if idx < maximal_idx:
                return dataset_name, subclass_label, sample_idxs
        assert False  # should never be reached

    def _determine_subclass_idx(self, idx):
        return idx % self._n_samples_per_subclass

    def _determine_sample_idx(self, subclass_idx):
        return

    def __getitem__(self, idx):
        dataset_name, subclass_label, sample_idxs = self._find_dataset_and_subclass(idx)
        subclass_idx = self._determine_subclass_idx(idx)
        sample_idx = sample_idxs[subclass_idx]
        return dataset_name, subclass_label, sample_idx


class KTaskNClassMDatasetData(torch.utils.data.Dataset):
    def __init__(self, *, size, tasks, datasets,
                     reinitialize_cache=False, cache_suffix=''):
        """A custom data set consisting of K tasks, each containing N classes
        from M datasets.

        Parameters
        ----------
        size : int
            The total number of samples in the dataset.

        tasks : list
            A list of dictionaries mapping class labels to
            dictionaries. These dictionaries define the source of
            samples for this particular class. Each source is defined
            by a string, denoting the dataset, and a list or dict defining
            the class labels from that data set.

            If it is a list, each entry specifies a class from the source dataset.
            The total number of samples per class from the source dataset is
            identical to `size / (number of tasks * number of classes *
            number of source classes`) or by the total number of
            samples present in the source
            data set of the corresponding class, whichever is smaller.

            If it is a dict, keys then correspond to class labels of the source
            dataset and values to the total number of samples from the
            source dataset. Is only used if this number is smaller than the
            two values referred to above.

            The strings need to match the keys of the "datasets" parameter.

        datasets : dict
            A dictionary mapping strings to torch Datasets.

        reinitialize_cache : bool, optional
            Sorted original datasets are stored on disk to avoid
            reordering them for each new instance of this class
            created. If this flag is True the old cache file is
            deleted and a new one created. Defaults to False.

        Examples
        --------
        >>> KTaskNClassMDatasetData(
        >>> size=900,
        >>> tasks=[
        >>>     {-1: {"MNIST": {0:1}}, 0: {"QMNIST": {2:2}}, 1: {"EMNIST": {27:10000000}}},
        >>>     {-1: {"MNIST": {0:1}, "EMNIST": {25: 100000000, 45: 1}}, 0: {"QMNIST": {2:1}}, 1: {"EMNIST": [14,23]}},
        >>>     ],
        >>> datasets={
        >>>    "MNIST": mnist_dataset, "QMNIST": qmnist_dataset, "EMNIST": emnist_dataset}
        >>>     }
        >>> )

        """
        self._size = size
        self._datasets = datasets
        self._reinitialize_cache = reinitialize_cache
        self._cache_suffix = cache_suffix

        self._assert_valid_task_definitions(tasks, datasets)

        self._n_tasks = len(tasks)
        self._n_classes = len(tasks[0])
        self._sample_indices_by_dataset_by_class = {}
        self._tasks = {}
        self._class_idx_to_class_label_by_task = {}

        self._determine_sample_indices_by_dataset_by_class()
        self._assert_valid_subclass_labels(tasks)

        self._construct_tasks(tasks)

    @staticmethod
    def _assert_valid_task_definitions(tasks, datasets):

        for task_idx in range(len(tasks)):

            if len(tasks[task_idx]) != len(tasks[0]):
                raise ValueError(
                    f"all tasks must define the same number of classes. task {task_idx} defines "
                    f"{len(tasks[task_idx])} while task 0 defines "
                    f"{len(tasks[0])} classes"
                )

            for class_label in tasks[task_idx]:
                for dataset_name in tasks[task_idx][class_label]:
                    if dataset_name not in datasets:
                        raise ValueError(
                            f'undefined dataset "{dataset_name}" in class {class_label} '
                            f"of task {task_idx}"
                        )

                if class_label not in tasks[0]:
                    raise ValueError(
                        f"all tasks must define the same class labels. task {task_idx} defines "
                        f"class {class_label} which is not defined in task 0"
                    )

    def _assert_valid_subclass_labels(self, tasks):
        for task_idx in range(len(tasks)):
            for class_label in tasks[task_idx]:
                for dataset_name in tasks[task_idx][class_label]:
                    for subclass_label in tasks[task_idx][class_label][dataset_name]:
                        if (
                            subclass_label
                            not in self._sample_indices_by_dataset_by_class[
                                dataset_name
                            ]
                        ):
                            raise ValueError(
                                f'unknown class label "{subclass_label}" for dataset {dataset_name}; possible class labels are: {sorted(self._sample_indices_by_dataset_by_class[dataset_name].keys())}'
                            )

    @property
    def _n_samples_per_task(self):
        n_samples_per_task = self._size // self._n_tasks

        return n_samples_per_task

    @property
    def _n_samples_per_class(self):
        n_samples_per_class = self._size // (self._n_classes * self._n_tasks)

        if n_samples_per_class * self._n_classes * self._n_tasks != self._size:
            raise ValueError("could not split {} samples evenly across {} classes and {} tasks".format(
                self._size, self._n_classes, self._n_tasks))

        return n_samples_per_class

    def _determine_sample_indices_by_dataset_by_class(self):
        for dataset_name in self._datasets:
            self._sample_indices_by_dataset_by_class[
                dataset_name
            ] = self._determine_sample_indices_by_class(dataset_name)

    def _determine_sample_indices_by_class(self, dataset_name):
        """
        Creates dictionary with as keys original dataset class labels and as
        values list of the original dataset sample indices that belong to the
        class corresponding to the key
        """

        # try to load sorted dataset from disk if desired
        if not self._reinitialize_cache:
            try:
                with open(f".cache-knm-dataset-{dataset_name}_{self._cache_suffix}.pkl", "rb") as f:
                    return pickle.load(f)
            except FileNotFoundError:
                pass

        # otherwise sort dataset
        sample_indices_by_class = {}
        for sample_idx, (_, class_label) in enumerate(self._datasets[dataset_name]):
            if class_label not in sample_indices_by_class:
                sample_indices_by_class[class_label] = []
            sample_indices_by_class[class_label].append(sample_idx)

        with open(f".cache-knm-dataset-{dataset_name}_{self._cache_suffix}.pkl", "wb") as f:
            pickle.dump(sample_indices_by_class, f)

        return sample_indices_by_class

    def _construct_tasks(self, tasks):
        for task_idx in range(self._n_tasks):
            self._tasks[task_idx] = {}
            self._class_idx_to_class_label_by_task[task_idx] = {}
            for class_idx, class_label in enumerate(tasks[task_idx]):
                self._tasks[task_idx][class_label] = CustomClass(
                    n_samples=self._n_samples_per_class,
                    sample_indices_by_dataset_by_class=self._sample_indices_by_dataset_by_class,
                    **(tasks[task_idx][class_label]),
                )
                self._class_idx_to_class_label_by_task[task_idx][
                    class_idx
                ] = class_label

    def __len__(self):
        return self._size

    def _idx_to_task_idx(self, idx):
        return idx // self._n_samples_per_task

    def _idx_to_class_idx(self, idx):
        return (idx % self._n_samples_per_task) // self._n_samples_per_class

    def _idx_to_sample_idx(self, idx):
        return idx % self._n_samples_per_class

    def __getitem__(self, idx):
        # first, we translate the linear index of this dataset into a
        # task index, class index and a sample index
        task_idx = self._idx_to_task_idx(idx)
        class_idx = self._idx_to_class_idx(idx)
        class_label = self._class_idx_to_class_label_by_task[task_idx][class_idx]
        sample_idx = self._idx_to_sample_idx(idx)

        # second, using the task and class index we determine the
        # identity of the sample by querying the custom class
        dataset_name, subclass_label, sample_idx = self._tasks[task_idx][class_label][
            sample_idx
        ]

        # finally we return the specific sample from the specific
        # dataset
        sample, original_class_label = self._get_sample_from_dataset(
            dataset_name, subclass_label, sample_idx
        )

        return (sample, original_class_label), (task_idx, class_label)

    def _get_sample_from_dataset(self, dataset_name, subclass_label, sample_idx):
        sample = self._datasets[dataset_name][sample_idx]

        assert subclass_label == sample[1]  # consistency check

        return sample

