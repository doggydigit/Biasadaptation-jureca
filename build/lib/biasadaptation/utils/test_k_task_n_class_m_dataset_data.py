import matplotlib.pyplot as plt
import torch
import torchvision


from k_task_n_class_m_dataset_data import KTaskNClassMDatasetData


def test_binary_minst_qmnist_two_tasks():

    torch.manual_seed(123)

    root = "./data/"

    mnist_dataset = torchvision.datasets.MNIST(
        root, train=True, transform=torchvision.transforms.ToTensor(), download=True
    )
    qmnist_dataset = torchvision.datasets.QMNIST(
        root, train=True, transform=torchvision.transforms.ToTensor(), download=True
    )
    emnist_dataset = torchvision.datasets.EMNIST(
        root, "balanced", train=True, transform=torchvision.transforms.ToTensor(), download=True
    )

    metalearn_data = KTaskNClassMDatasetData(
        size=3000,
        tasks=[
            {-1: {"MNIST": [0, 4, 6, 8], "QMNIST": [0]}, 1: {"MNIST": [1, 3, 5, 7, 9]}},
            {-1: {"MNIST": [0, 2, 4, 6, 8]}, 1: {"MNIST": [1, 3, 5], "QMNIST": [7, 9]}},
            {-1: {"EMNIST": [0, 1, 2, 3, 4]}, 1: {"EMNIST": [5, 6, 7, 8, 9]}},
        ],
        datasets={"MNIST": mnist_dataset, "QMNIST": qmnist_dataset, "EMNIST": emnist_dataset},
    )

    data_loader = torch.utils.data.DataLoader(
        metalearn_data, batch_size=100, shuffle=True
    )

    for batch_idx, (data, (target_task, target_class)) in enumerate(data_loader):
        print("batch", batch_idx, "task", target_task, "class", target_class)
        # plt.clf()
        # plt.imshow(data[0].numpy().reshape(28, 28))
        # plt.show()


test_binary_minst_qmnist_two_tasks()
