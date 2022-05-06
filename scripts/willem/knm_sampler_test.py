import torch
import torchvision

from datarep import paths

from biasadaptation.utils.k_task_n_class_m_dataset_data import KTaskNClassMDatasetData


def test_binary_minst_qmnist_two_tasks():

    torch.manual_seed(123)

    path = paths.data_path

    mnist_dataset = torchvision.datasets.MNIST(
        path, train=True, transform=torchvision.transforms.ToTensor(), download=True
    )
    qmnist_dataset = torchvision.datasets.QMNIST(
        path, train=True, transform=torchvision.transforms.ToTensor(), download=True
    )
    emnist_dataset = torchvision.datasets.EMNIST(
        path, "balanced", train=True, transform=torchvision.transforms.ToTensor(), download=True
    )

    metalearn_data = KTaskNClassMDatasetData(
        size=900,
        tasks=[
            {-1: {"MNIST": [0]}, 0: {"QMNIST": [2]}, 1: {"EMNIST": [27]}},
        ],
        datasets={"MNIST": mnist_dataset, "QMNIST": qmnist_dataset, "EMNIST": emnist_dataset},
    )

    data_loader = torch.utils.data.DataLoader(
        metalearn_data, batch_size=100, shuffle=True
    )

    for batch_idx, (data, (target_task, target_class)) in enumerate(data_loader):
        print("batch: ", batch_idx, ", batch shape:", data[0], ", class:", target_class)
        # plt.clf()
        # plt.imshow(data[0].numpy().reshape(28, 28))
        # plt.show()


test_binary_minst_qmnist_two_tasks()