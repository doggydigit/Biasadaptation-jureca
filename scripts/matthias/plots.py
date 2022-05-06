import torch
import pickle
import numpy as np
from numpy import arange as np_arange
from torch import mean as torch_mean
from torch import std as torch_std
import matplotlib.pyplot as plt
from pathlib import Path
from biaslearning_helper import get_biaslearner_training_params
from multireadout_helper import get_multireadout_training_params, get_binarymr_training_params
from plot_helper import get_dirs, plot_scan_lr, plot_train_test_performances, plot_best_2d_lr, multitask_all_plot
from plot_helper import multitask_emnist_1hl_plot, multitask_plot, fig_2, k49_plot, get_performances


def plot_batch_scan_trainbw(dataset="MNIST", saving=False, root_dir="../../", subdir="scan_trainbw_batch"):
    """

    Parameters
    ----------
    dataset: Dataset used for the training. One of "MNIST", "QMNIST", "EMNIST"
    nr_hidden: list of the number of hidden neurons in each hidden layer. One of [10], [100], [10, 10], [100, 100]
    saving: Whether to save the plot to file
    root_dir:
    subdir:

    Returns
    -------

    """
    assert dataset in ["MNIST", "QMNIST", "EMNIST", "EMNIST_letters"]

    loss_function = "mse"
    readout_function = "hardtanh"
    nr_hiddens = [[10], [100], [10, 10], [100, 100]]
    load_dir, save_dir = get_dirs(root_dir, subdir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2)
    plot_types = ["train_performance", "train_loss", "validation_performance", "validation_loss"]
    legends = []
    axid = [[0, 0], [0, 1], [1, 0], [1, 1]]

    for nr_hidden in nr_hiddens:
        save_name = "network_{}_{}_{}_{}.pickle".format(nr_hidden, dataset, loss_function, readout_function)
        with open(load_dir + save_name, "rb") as f:
            results = pickle.load(f)
        x = [float(lr) for lr in results.keys()]
        for i in range(len(plot_types)):
            y = [torch.mean(torch.stack(results[lr][plot_types[i]])) for lr in results.keys()]
            err = [torch.std(torch.stack(results[lr][plot_types[i]])) for lr in results.keys()]
            axs[axid[i][0], axid[i][1]].errorbar(x, y, err)
            # axs[axid[i][0], axid[i][1]].plot(x, y)
        legends += [save_name]
    fig.canvas.manager.full_screen_toggle()
    for i in range(4):
        axs[axid[i][0], axid[i][1]].legend(legends)
        axs[axid[i][0], axid[i][1]].set_title(plot_types[i])
        axs[axid[i][0], axid[i][1]].set_xscale('log')
        axs[axid[i][0], axid[i][1]].set_xlabel('batch size')
    for i in range(2):
        axs[i, 0].set_ylim([0., 100.])
    plt.suptitle("bw training outcome on {}".format(dataset))
    if saving:
        fig.set_size_inches(16, 9)
        plt.savefig(save_dir + "scan_trainbw_lr_{}.svg".format(dataset))
        plt.savefig(save_dir + "scan_trainbw_lr_{}.png".format(dataset))
        plt.close()
    else:
        plt.show()


def plot_scan_trainbw_lr(datasets=None, nr_hiddens=None, saving=False, root_dir="../../", subdir="scan_trainbw_lr",
                         mse_only=True, individual=True, lrs=None, early_stopping=None):
    """

    Parameters
    ----------
    datasets: Dataset used for the training. One of "MNIST", "QMNIST", "EMNIST"
    nr_hiddens: list of the number of hidden neurons in each hidden layer. One of [10], [100], [10, 10], [100, 100]
    saving: Whether to save the plot to file
    root_dir:
    subdir:

    Returns
    -------

    """
    if nr_hiddens is None:
        nr_hiddens = [[10], [100], [500], [10, 10], [100, 100], [500, 500]]
    if lrs is None:
        lrs = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
    if early_stopping is None:
        ess = ["", "_es"]
    else:
        if early_stopping:
            ess = ["_es"]
        else:
            ess = [""]
    if datasets is None:
        datasets = ["MNIST", "QMNIST", "EMNIST", "EMNIST_letters", "EMNIST_bymerge"]

    if mse_only:
        loss_functions = ["mse"]
        readout_functions = ["tanh"]
    else:
        loss_functions = ["l1", "mse", "squared_hinge"]
        readout_functions = ["linear", "tanh", "hardtanh"]
    load_dir, save_dir = get_dirs(root_dir, subdir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2)
    plot_types = ["train_performance", "train_loss", "validation_performance", "validation_loss"]
    legends = []
    axid = [[0, 0], [0, 1], [1, 0], [1, 1]]
    maxes = []

    for dataset in datasets:
        for nr_hidden in nr_hiddens:
            for es in ess:
                for loss_function in loss_functions:
                    for readout_function in readout_functions:
                        if individual:
                            ys = {}
                            errs = {}
                            for i in plot_types:
                                ys[i] = []
                                errs[i] = []
                            for lr in lrs:
                                path = load_dir + "individual/network_{}_{}_{}_{}{}_lr_{}.pickle" \
                                                  "".format(nr_hidden, dataset, loss_function, readout_function, es, lr)
                                with open(path, "rb") as f:
                                    result = pickle.load(f)
                                for i in plot_types:
                                    ys[i] += [np.mean(result[i])]
                                    errs[i] += [np.std(result[i])]
                            for i in range(len(plot_types)):
                                axs[axid[i][0], axid[i][1]].errorbar(lrs, ys[plot_types[i]], errs[plot_types[i]])

                            m = max(zip(ys["validation_performance"], range(len(ys["validation_performance"]))))[1]
                            maxes += [[float(lrs[m]), ys["validation_performance"][m]]]

                        else:
                            path = load_dir + "network_{}_{}_{}_{}.pickle".format(nr_hidden, dataset, loss_function, readout_function)
                            with open(path, "rb") as f:
                                results = pickle.load(f)
                            x = [float(lr) for lr in results.keys()]
                            for i in range(len(plot_types)):
                                y = [torch.mean(torch.stack(results[lr][plot_types[i]])) for lr in results.keys()]
                                err = [torch.std(torch.stack(results[lr][plot_types[i]])) for lr in results.keys()]
                                axs[axid[i][0], axid[i][1]].errorbar(x, y, err)
                                # axs[axid[i][0], axid[i][1]].plot(x, y)
                        legends += ["{}{}_{}".format(nr_hidden, es, dataset)]

    # Plotting Specifics
    fig.canvas.manager.full_screen_toggle()
    for i in range(4):
        axs[axid[i][0], axid[i][1]].legend(legends)
        axs[axid[i][0], axid[i][1]].set_title(plot_types[i])
        axs[axid[i][0], axid[i][1]].set_xscale('log')
        axs[axid[i][0], axid[i][1]].set_xlabel('learning rate')
    for i in range(2):
        axs[i, 0].set_ylim([0., 100.])
    plt.suptitle("bw training outcome for network {} on {}".format(nr_hidden, dataset))
    for m in maxes:
        axs[axid[2][0], axid[2][1]].plot(m[0], m[1], "xk")
    if saving:
        fig.set_size_inches(16, 9)
        plt.savefig(save_dir + "scan_trainbw_lr_{}_{}.svg".format(nr_hidden, dataset))
        plt.savefig(save_dir + "scan_trainbw_lr_{}_{}.png".format(nr_hidden, dataset))
        plt.close()
    else:
        plt.show()


def plot_multireadout_scan(dataset="MNIST", nr_hidden=None, saving=False, root_dir="../../"):
    """

    Parameters
    ----------
    dataset: Dataset used for the training. One of "MNIST", "QMNIST", "EMNIST"
    nr_hidden: list of the number of hidden neurons in each hidden layer. One of [10], [100], [10, 10], [100, 100]
    saving: Whether to save the plot to file
    root_dir

    Returns
    -------

    """
    if nr_hidden is None:
        nr_hidden = [10]
    assert dataset in ["MNIST", "QMNIST", "EMNIST", "EMNIST_letters"]
    assert nr_hidden in [[], [10], [100], [10, 10], [100, 100]]

    loss_functions = ["l1", "mse", "squared_hinge"]
    readout_functions = ["softmax"]
    load_dir, save_dir = get_dirs(root_dir, "scan_multi_readout_lr")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2)
    plot_types = ["train_performance", "train_loss", "validation_performance", "validation_loss"]
    legends = []
    axid = [[0, 0], [0, 1], [1, 0], [1, 1]]

    for loss_function in loss_functions:
        for readout_function in readout_functions:
            save_name = "network_{}_{}_{}_{}.pickle".format(nr_hidden, dataset, loss_function, readout_function)
            with open(load_dir + save_name, "rb") as f:
                results = pickle.load(f)
            x = [float(lr) for lr in results.keys()]
            for i in range(len(plot_types)):
                y = [torch.mean(torch.stack(results[lr][plot_types[i]])) for lr in results.keys()]
                axs[axid[i][0], axid[i][1]].plot(x, y)
            legends += [save_name]

    fig.canvas.manager.full_screen_toggle()
    for i in range(4):
        axs[axid[i][0], axid[i][1]].legend(legends)
        axs[axid[i][0], axid[i][1]].set_title(plot_types[i])
        axs[axid[i][0], axid[i][1]].set_xscale('log')
        axs[axid[i][0], axid[i][1]].set_xlabel('learning rate')
    for i in range(2):
        axs[i, 0].set_ylim([0., 100.])
    plt.suptitle("bw training outcome for network {} on {}".format(nr_hidden, dataset))
    if saving:
        fig.set_size_inches(16, 9)
        plt.savefig(save_dir + "scan_lr_{}_{}.svg".format(nr_hidden, dataset))
        plt.savefig(save_dir + "scan_lr_{}_{}.png".format(nr_hidden, dataset))
        plt.close()
    else:
        plt.show()


def plot_trainbw(loss_function="mse", readout_function="hardtanh", bwlr=0.001, saving=False, root_dir="../../"):
    """

    Parameters
    ----------
    loss_function: One of "l1", "mse", "squared_hinge", "perceptron"
    readout_function: One of "linear", "tanh", "hardtanh"
    saving: Whether to save the plot to file
    root_dir

    Returns
    -------

    """
    datasets = ["MNIST", "QMNIST", "EMNIST", "EMNIST_letters"]
    nr_hiddens = [[10], [25], [50], [100], [250], [500], [10, 10], [25, 25], [50, 50], [100, 100], [250, 250], [500, 500]]
    load_dir, save_dir = get_dirs(root_dir, "trainbw")

    fig, axs = plt.subplots(2, 2)
    plot_types = ["train_performance", "train_loss", "validation_performance", "validation_loss"]
    axid = [[0, 0], [0, 1], [1, 0], [1, 1]]
    width = 1. / (len(nr_hiddens) + 1)
    x_offset = [i * width - 0.5 for i in range(1, len(nr_hiddens) + 1)]
    x = np.arange(len(datasets))

    for h in range(len(nr_hiddens)):
        for i in range(len(plot_types)):
            y = []
            for dataset in datasets:
                save_name = "{}/network_{}_{}_{}_bwlr_{}.pickle" \
                            "".format(dataset, nr_hiddens[h], loss_function, readout_function, bwlr)
                with open(load_dir + save_name, "rb") as f:
                    results = pickle.load(f)
                y += [torch.mean(torch.stack(results[plot_types[i]]))]
            axs[axid[i][0], axid[i][1]].bar(x + x_offset[h], y, width, label="number hidden " + str(nr_hiddens[h]))

    for i in range(4):
        axs[axid[i][0], axid[i][1]].set_title(plot_types[i])
        axs[axid[i][0], axid[i][1]].set_xticks(x)
        axs[axid[i][0], axid[i][1]].set_xticklabels(datasets)
        axs[axid[i][0], axid[i][1]].legend()

    for i in range(2):
        axs[i, 0].set_ylim([60., 100.])
    plt.suptitle("bw training")
    fig.canvas.manager.full_screen_toggle()
    if saving:
        fig.set_size_inches(16, 9)
        plt.savefig(save_dir + "trainbw.svg")
        plt.savefig(save_dir + "trainbw.png")
        plt.close()
    else:
        plt.show()


def plot_willeminit_trainbw(saving=False, root_dir="../../"):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file
    root_dir

    Returns
    -------

    """
    datasets = ["MNIST", "QMNIST", "EMNIST", "EMNIST_letters"]
    nr_hiddens = [50, 100]
    load_dir, save_dir = get_dirs(root_dir, "willem_weights")

    fig, axs = plt.subplots(2, 2)
    plot_types = ["train_performance", "train_loss", "validation_performance", "validation_loss"]
    axid = [[0, 0], [0, 1], [1, 0], [1, 1]]
    width = 1. / (len(nr_hiddens) + 1)
    x_offset = [i * width - 0.5 for i in range(1, len(nr_hiddens) + 1)]
    x = np.arange(len(datasets))

    for h in range(len(nr_hiddens)):
        for i in range(len(plot_types)):
            y = []
            for dataset in datasets:
                save_name = "{}/network_{}.pickle".format(dataset, nr_hiddens[h])
                with open(load_dir + save_name, "rb") as f:
                    results = pickle.load(f)
                y += [torch.mean(torch.stack(results[plot_types[i]]))]
            axs[axid[i][0], axid[i][1]].bar(x + x_offset[h], y, width, label="number hidden " + str(nr_hiddens[h]))

    for i in range(4):
        axs[axid[i][0], axid[i][1]].set_title(plot_types[i])
        axs[axid[i][0], axid[i][1]].set_xticks(x)
        axs[axid[i][0], axid[i][1]].set_xticklabels(datasets)
        axs[axid[i][0], axid[i][1]].legend()

    for i in range(2):
        axs[i, 0].set_ylim([0., 100.])
    plt.suptitle("bw training")
    fig.canvas.manager.full_screen_toggle()
    if saving:
        fig.set_size_inches(16, 9)
        plt.savefig(save_dir + "willeminit_trainbw.svg")
        plt.savefig(save_dir + "willeminit_trainbw.png")
        plt.close()
    else:
        plt.show()


def plot_leave1out_trainbw(loss_function="mse", readout_function="hardtanh", saving=False, root_dir="../../"):
    """

    Parameters
    ----------
    loss_function: One of "l1", "mse", "squared_hinge", "perceptron"
    readout_function: One of "linear", "tanh", "hardtanh"
    saving: Whether to save the plot to file
    root_dir

    Returns
    -------

    """
    datasets = ["MNIST", "QMNIST", "EMNIST", "EMNIST_letters"]
    nr_hiddens = [[10], [100], [10, 10], [100, 100]]
    load_dir, save_dir = get_dirs(root_dir, "leave1out_trainbw")

    fig, axs = plt.subplots(2, 2)
    plot_types = ["train_performance", "train_loss", "validation_performance", "validation_loss"]
    nr_classes_per_dataset = {"MNIST": 10, "QMNIST": 10, "EMNIST": 47, "EMNIST_letters": 26}
    axid = [[0, 0], [0, 1], [1, 0], [1, 1]]
    width = 1. / (len(nr_hiddens) + 1)
    x_offset = [i * width - 0.5 for i in range(1, len(nr_hiddens) + 1)]
    x = np.arange(len(datasets))

    for h in range(len(nr_hiddens)):
        for i in range(len(plot_types)):
            y = []
            for dataset in datasets:
                save_name = "network_{}_{}_{}_{}.pickle".format(nr_hiddens[h], dataset, loss_function, readout_function)
                with open(load_dir + save_name, "rb") as f:
                    results = pickle.load(f)
                r = [torch.mean(torch.stack(results[j][plot_types[i]])) for j in range(nr_classes_per_dataset[dataset])]
                y += [sum(r)/len(r)]
            axs[axid[i][0], axid[i][1]].bar(x + x_offset[h], y, width, label="number hidden " + str(nr_hiddens[h]))

    for i in range(4):
        axs[axid[i][0], axid[i][1]].set_title(plot_types[i])
        axs[axid[i][0], axid[i][1]].set_xticks(x)
        axs[axid[i][0], axid[i][1]].set_xticklabels(datasets)
        axs[axid[i][0], axid[i][1]].legend()

    for i in range(2):
        axs[i, 0].set_ylim([0., 100.])
    plt.suptitle("bw training leaving one class out")
    fig.canvas.manager.full_screen_toggle()
    if saving:
        fig.set_size_inches(16, 9)
        plt.savefig(save_dir + "leave1out_trainbw.svg")
        plt.savefig(save_dir + "leave1out_trainbw.png")
        plt.close()
    else:
        plt.show()


def plot_leave1out_trainb(loss_function="mse", readout_function="hardtanh", saving=False, root_dir="../../"):
    """

    Parameters
    ----------
    loss_function: One of "l1", "mse", "squared_hinge", "perceptron"
    readout_function: One of "linear", "tanh", "hardtanh"
    saving: Whether to save the plot to file
    root_dir

    Returns
    -------

    """
    datasets = ["MNIST", "QMNIST", "EMNIST", "EMNIST_letters"]
    nr_hiddens = [[10], [100], [10, 10], [100, 100]]
    load_dir, save_dir = get_dirs(root_dir, "leave1out_trainb")

    fig, axs = plt.subplots(2, 2)
    plot_types = ["train_performance", "train_loss", "validation_performance", "validation_loss"]
    nr_classes_per_dataset = {"MNIST": 10, "QMNIST": 10, "EMNIST": 47, "EMNIST_letters": 26}
    axid = [[0, 0], [0, 1], [1, 0], [1, 1]]
    width = 1. / (len(nr_hiddens) + 1)
    x_offset = [i * width - 0.5 for i in range(1, len(nr_hiddens) + 1)]
    x = np.arange(len(datasets))

    for h in range(len(nr_hiddens)):
        for i in range(len(plot_types)):
            y = []
            for dataset in datasets:
                save_name = "network_{}_{}_{}_{}.pickle".format(nr_hiddens[h], dataset, loss_function, readout_function)
                with open(load_dir + save_name, "rb") as f:
                    results = pickle.load(f)
                r = [torch.mean(torch.stack(results["class_results"][j][plot_types[i]])) for j in range(nr_classes_per_dataset[dataset])]
                y += [sum(r)/len(r)]
            axs[axid[i][0], axid[i][1]].bar(x + x_offset[h], y, width, label="number hidden " + str(nr_hiddens[h]))

    for i in range(4):
        axs[axid[i][0], axid[i][1]].set_title(plot_types[i])
        axs[axid[i][0], axid[i][1]].set_xticks(x)
        axs[axid[i][0], axid[i][1]].set_xticklabels(datasets)
        axs[axid[i][0], axid[i][1]].legend()

    for i in range(2):
        axs[i, 0].set_ylim([0., 100.])
    plt.suptitle("bw training leaving one class out")
    fig.canvas.manager.full_screen_toggle()
    if saving:
        fig.set_size_inches(16, 9)
        plt.savefig(save_dir + "leave1out_trainbw.svg")
        plt.savefig(save_dir + "leave1out_trainbw.png")
        plt.close()
    else:
        plt.show()


def plot_scan_leave1out_trainb_lr(dataset="MNIST", saving=False, root_dir="../../"):
    """

    Parameters
    ----------
    dataset: Dataset used for the training. One of "MNIST", "QMNIST", "EMNIST"
    saving: Whether to save the plot to file
    root_dir

    Returns
    -------

    """

    # Basic Parameters
    assert dataset in ["MNIST", "QMNIST", "EMNIST", "EMNIST_letters"]
    nr_hiddens = [[10], [100], [10, 10], [100, 100]]
    loss_function = "mse"
    readout_function = "hardtanh"
    load_dir, save_dir = get_dirs(root_dir, "scan_trainb_l1o_lr")
    nr_classes = {"MNIST": 10, "QMNIST": 10, "EMNIST": 47, "EMNIST_letters": 26}

    # Plot initializations
    fig, axs = plt.subplots(2, 2)
    plot_types = ["train_performance", "train_loss", "validation_performance", "validation_loss"]
    axid = [[0, 0], [0, 1], [1, 0], [1, 1]]
    maxes = []

    # Loading results and plotting them
    for nr_hidden in nr_hiddens:
        save_name = "network_{}_{}_{}_{}.pickle".format(nr_hidden, dataset, loss_function, readout_function)
        with open(load_dir + save_name, "rb") as f:
            results = pickle.load(f)
        lrs = [lr for lr in results.keys() if lr[0] == '0']
        x = [float(lr) for lr in lrs]
        for i in range(len(plot_types)):
            y = []
            for lr in lrs:
                y += [torch.mean(torch.stack([torch.stack(results[lr][t][plot_types[i]])
                                              for t in range(nr_classes[dataset])]))]
            axs[axid[i][0], axid[i][1]].plot(x, y)
            if i == 2:
                m = max(zip(y, range(len(y))))[1]
                maxes += [[float(lrs[m]), y[m]]]
                print("For {} network {} found its best validation performance of {} for learning rate {}"
                      "".format(dataset, nr_hidden, y[m], lrs[m]))

    # Plotting Specifics
    for m in maxes:
        axs[axid[2][0], axid[2][1]].plot(m[0], m[1], "xk")
    fig.canvas.manager.full_screen_toggle()
    for i in range(4):
        axs[axid[i][0], axid[i][1]].legend(nr_hiddens + ["best validation performances"])
        axs[axid[i][0], axid[i][1]].set_title(plot_types[i])
        axs[axid[i][0], axid[i][1]].set_xscale('log')
        axs[axid[i][0], axid[i][1]].set_xlabel('learning rate')
    for i in range(2):
        axs[i, 0].set_ylim([0., 100.])
    plt.suptitle("b training outcome on {}".format(dataset))

    # Saving / showing plot
    if saving:
        fig.set_size_inches(16, 9)
        plt.savefig(save_dir + "scan_trainb_l1o_lr_{}.svg".format(dataset))
        plt.savefig(save_dir + "scan_trainb_l1o_lr_{}.png".format(dataset))
        plt.close()
    else:
        plt.show()


def plot_train_transfer(maindir, x_labels, load_name_base_1, load_name_base_2, suptitle, save_name, loss_function="mse",
                        readout_function="hardtanh", saving=False, root_dir="../../"):
    """

    Parameters
    ----------
    maindir
    x_labels
    load_name_base_1
    load_name_base_2
    suptitle
    save_name
    loss_function: One of "l1", "mse", "squared_hinge", "perceptron"
    readout_function: One of "linear", "tanh", "hardtanh"
    saving: Whether to save the plot to file
    root_dir

    Returns
    -------

    """

    b_lr = 0.01
    bw_lr = 0.001
    nr_hiddens = [[10], [100], [10, 10], [100, 100]]
    load_dir, save_dir = get_dirs(root_dir, maindir)

    fig, axs = plt.subplots(2, 2)
    plot_types = ["train_performance", "train_loss", "validation_performance", "validation_loss"]
    axid = [[0, 0], [0, 1], [1, 0], [1, 1]]
    width = 1. / (len(nr_hiddens) + 1)
    x_offset = [i * width - 0.5 for i in range(1, len(nr_hiddens) + 1)]
    x = np.arange(len(x_labels))

    for h in range(len(nr_hiddens)):
        load_name1 = root_dir + load_name_base_1.format(nr_hiddens[h], loss_function, readout_function, bw_lr)
        load_name2 = load_dir + load_name_base_2.format(nr_hiddens[h], loss_function, readout_function, b_lr)
        load_names = [load_name1, load_name2]

        for i in range(len(plot_types)):
            y = []

            for j in range(2):
                with open(load_names[j], "rb") as f:
                    results = pickle.load(f)
                r = results[plot_types[i]]
                y += [sum(r)/len(r)]
            axs[axid[i][0], axid[i][1]].bar(x + x_offset[h], y, width, label="number hidden " + str(nr_hiddens[h]))

    for i in range(4):
        axs[axid[i][0], axid[i][1]].set_title(plot_types[i])
        axs[axid[i][0], axid[i][1]].set_xticks(x)
        axs[axid[i][0], axid[i][1]].set_xticklabels(x_labels)
        axs[axid[i][0], axid[i][1]].legend()

    for i in range(2):
        axs[i, 0].set_ylim([0., 100.])
        axs[i, 0].set_ylabel("[%]")
    plt.suptitle(suptitle)
    fig.canvas.manager.full_screen_toggle()
    if saving:
        fig.set_size_inches(16, 9)
        plt.savefig(save_dir + save_name + ".svg")
        plt.savefig(save_dir + save_name + ".png")
        plt.close()
    else:
        plt.show()


def plot_train_letters_transfer_mnist(loss_function="mse", readout_function="hardtanh", saving=False, root_dir="../../"):
    maindir = "train_letters_transfer_mnist"
    x_labels = ["Weight Training on EMNIST_letters", "Bias Transfer on MNIST"]
    load_name_base_1 = "results/trainbw/EMNIST_letters/network_{}_{}_{}_bwlr_{}.pickle"
    load_name_base_2 = "network_{}_{}_{}_blr_{}.pickle"
    suptitle = "Transfer Learning from EMNIST letters to MNIST digits"
    save_name = "train_letters_transfer_mnist"

    plot_train_transfer(maindir, x_labels, load_name_base_1, load_name_base_2, suptitle, save_name,  loss_function,
                        readout_function, saving, root_dir)


def plot_train_mnist_transfer_letters(loss_function="mse", readout_function="hardtanh", saving=False, root_dir="../../"):
    maindir = "train_mnist_transfer_letters"
    x_labels = ["Weight Training on MNIST", "Bias Transfer on EMNIST letters"]
    load_name_base_1 = "results/trainbw/MNIST/network_{}_{}_{}_bwlr_{}.pickle"
    load_name_base_2 = "network_{}_{}_{}_blr_{}.pickle"
    suptitle = "Transfer Learning from MNIST digits to EMNIST letters"
    save_name = "train_mnist_transfer_letters"

    plot_train_transfer(maindir, x_labels, load_name_base_1, load_name_base_2, suptitle, save_name,  loss_function,
                        readout_function, saving, root_dir)


def replot_all(saving=True):
    for dataset in ["MNIST", "QMNIST", "EMNIST", "EMNIST_letters"]:
        plot_scan_leave1out_trainb_lr(dataset, saving=saving)
        plot_batch_scan_trainbw(dataset, saving=saving, subdir="scan_trainbw_batch_1000epochs")
        for nr_hidden in [[10], [100], [10, 10], [100, 100]]:
            # noinspection PyTypeChecker
            plot_scan_trainbw_lr(dataset, nr_hidden, saving)
            # noinspection PyTypeChecker
            plot_scan_trainbw_lr(dataset, nr_hidden, saving, subdir="scan_trainbw_lr_10_epochs", mse_only=True)
            # noinspection PyTypeChecker
            plot_scan_trainbw_lr(dataset, nr_hidden, saving, subdir="scan_trainbw_lr_100_epochs", mse_only=True)
            # noinspection PyTypeChecker
            plot_multireadout_scan(dataset, nr_hidden, saving=saving)
    plot_trainbw(saving=saving)
    plot_willeminit_trainbw(saving=saving)
    plot_leave1out_trainbw(saving=saving)
    plot_leave1out_trainb(saving=saving)
    plot_train_letters_transfer_mnist(saving=saving)

    for dataset in ["MNIST", "EMNIST", "EMNIST_letters"]:
        plot_batch_scan_trainbw(dataset, True)

    print("All plots were successfully reproduced.")


def bias_vs_gain_plot(saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """

    # datasets = ["QMNIST", "KMNIST", "EMNIST_bymerge", "EMNIST_letters"]
    datasets = ["CIFAR100", "EMNIST_bymerge"]
    dataset_name = ["CIFAR-100", "EMNIST"]
    nr_hiddens = [[100], [100, 100], [100, 100, 100]]
    traintypes = ["train_b_w", "train_g_bw", "train_bg_w"]
    lcolors = ["royalblue", "forestgreen", "lightseagreen"]
    legends = ["Contextual Biases", "Contextual Gains", "Contextual Biases & Gains"]

    load_dir = "{}results/train_full_dataset/".format(root_dir)
    title = "Multitask learning with gains vs. biases"

    fig, axs = plt.subplots(len(datasets), 1, sharex='all')
    nr_bars = len(traintypes)
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(nr_hiddens))
    perflog = {0: [], 1: [], 2: []}
    for row in range(len(datasets)):
        prog_params = {"dataset": datasets[row]}
        for tt in range(len(traintypes)):
            prog_params["training_type"] = traintypes[tt]
            hs = 25
            y = []
            y_err = []
            for nh in range(len(nr_hiddens)):
                prog_params["nr_hidden"] = nr_hiddens[nh]
                train_params = get_biaslearner_training_params(prog_params=prog_params, highseed=hs)
                performances = get_performances(prog_name=traintypes[tt] + "_full", load_info=load_dir,
                                                nr_hidden=nr_hiddens[nh], dataset=datasets[row],
                                                train_params=train_params, performance_type="test_performance")
                perflog[tt] += [performances]
                y += [torch_mean(performances)]
                y_err += [torch_std(performances)]
            axs[row].bar(x + x_offset[tt], y, width, yerr=y_err, label=legends[tt], zorder=3, color=lcolors[tt])
        axs[row].set_title(dataset_name[row])
        axs[row].set_xticks(x)
        axs[row].grid(axis='y', alpha=0.4, zorder=0)
        axs[row].set_ylabel("Test Performance [%]")
    axs[0].set_ylim([60, 90])
    axs[1].set_ylim([90, 100])
    horiz_center = 0.44
    fig.text(0.4, 0.015, 'Network Architecture', ha='center')
    if titling:
        fig.text(horiz_center, 0.96, title, fontsize=14, ha='center')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
    axs[-1].set_xticklabels(nr_hiddens)
    if titling:
        top = 0.88
    else:
        top = 0.93
    fig.subplots_adjust(left=0.09, right=0.72, top=top, bottom=0.08, hspace=0.26)
    fig.set_size_inches(9, 7)
    from scipy.stats import ttest_ind
    for i in range(6):
        if i > 2.5:
            d = "EMNIST"
        else:
            d = "CIFAR100"
        print(d, "b vs g", ttest_ind(perflog[0][i], perflog[1][i]))
        print(d, "b vs bg", ttest_ind(perflog[0][i], perflog[2][i]))
        print(d, "g vs bg", ttest_ind(perflog[1][i], perflog[2][i]))
    if saving:
        plt.savefig("{}plots/train_full_dataset/svg/bias_vs_gain.svg".format(root_dir))
        plt.savefig("{}plots/train_full_dataset/bias_vs_gain.png".format(root_dir))
        plt.close()
    else:
        plt.show()


def bias_vs_mr_plot(saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """

    # datasets = ["QMNIST", "KMNIST", "EMNIST_bymerge", "EMNIST_letters"]
    datasets = ["EMNIST_bymerge"]
    dataset_name = ["EMNIST"]
    nr_hiddens = [[100], [100, 100], [100, 100, 100]]
    traintypes = ["train_b_w", "train_binarymr", "train_bw"]
    lcolors = ["#0072B2", "#009E73", "#D55E00"]
    perf_traintypes = ["polish_b_full", "test_train_bmr_full", "train_sg_full"]
    legends = ["Task-dependent biases", "Task-dependent outputs", "Task-dependent networks"]

    load_dir = "{}results/train_full_dataset/".format(root_dir)
    title = "Multitask learning with biases vs. readouts"

    fig, axs = plt.subplots(len(datasets), 1, sharex='all')
    nr_bars = len(traintypes)
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(nr_hiddens))
    perflog = {0: [], 1: [], 2: []}
    for row in range(len(datasets)):
        prog_params = {"dataset": datasets[row]}
        for tt in range(len(traintypes)):
            prog_params["training_type"] = traintypes[tt]
            y = []
            y_err = []
            for nh in range(len(nr_hiddens)):
                prog_params["nr_hidden"] = nr_hiddens[nh]
                if tt == 0:
                    train_params = get_biaslearner_training_params(prog_params=prog_params, highseed=25)
                elif tt == 1:
                    train_params = get_binarymr_training_params(prog_params=prog_params, highseed=25)
                else:
                    train_params = get_biaslearner_training_params(prog_params=prog_params, highseed=3)
                performances = get_performances(prog_name=perf_traintypes[tt], load_info=load_dir,
                                                nr_hidden=nr_hiddens[nh], dataset=datasets[row],
                                                train_params=train_params, performance_type="test_performance")
                perflog[tt] += [performances]
                y += [torch_mean(performances)]
                y_err += [torch_std(performances)]
            axs.bar(x + x_offset[tt], y, width, yerr=y_err, label=legends[tt], zorder=3, color=lcolors[tt])
        # axs.set_title(dataset_name[row])
        axs.set_xticks(x)
        # axs.grid(axis='y', alpha=0.4, zorder=0)
        axs.set_ylabel("Test Performance [%]")
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.set_ylim([90, 100])
    horiz_center = 0.44
    fig.text(0.53, 0.015, 'Hidden Layers', ha='center')
    if titling:
        fig.text(horiz_center, 0.96, title, fontsize=14, ha='center')
    plt.legend(loc="lower right")
    axs.set_xticklabels([len(nh) for nh in nr_hiddens])
    if titling:
        top = 0.88
    else:
        top = 0.94
    fig.subplots_adjust(left=0.21, right=0.95, top=top, bottom=0.18, hspace=0.26)
    fig.set_size_inches(3.5, 3)
    from scipy.stats import ttest_ind
    for i in range(1):
        if i > 2.5:
            d = "EMNIST"
        else:
            d = "CIFAR100"
        print(d, "b vs mr", ttest_ind(perflog[0][i], perflog[1][i]))
    if saving:
        plt.savefig("{}plots/train_full_dataset/svg/bias_vs_mr.svg".format(root_dir))
        plt.savefig("{}plots/train_full_dataset/bias_vs_mr.png".format(root_dir))
        plt.close()
    else:
        plt.show()


def transfer_plot(saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """

    # datasets = ["QMNIST", "KMNIST", "EMNIST_bymerge", "EMNIST_letters"]
    datasets = ["EMNIST_bymerge"]
    dataset_name = ["EMNIST"]
    nr_hiddens = [[250], [250, 250], [250, 250, 250]]
    traintypes = ["train_b_w", "train_b_w"]
    lcolors = ["royalblue", "blueviolet"]
    perf_traintypes = ["polish_b_full", "transfer_b_l1o_b_w"]
    legends = ["Mulit-task", "Transfer learning"]

    load_dirs = ["{}results/train_full_dataset/".format(root_dir), "{}results/leave_1_out/".format(root_dir)]
    title = "Multitask learning with biases vs. readouts"

    fig, axs = plt.subplots(len(datasets), 1, sharex='all')
    nr_bars = len(traintypes)
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(nr_hiddens))
    perflog = {0: [], 1: []}
    for row in range(len(datasets)):
        prog_params = {"dataset": datasets[row]}
        for tt in range(len(traintypes)):
            prog_params["training_type"] = traintypes[tt]
            y = []
            y_err = []
            for nh in range(len(nr_hiddens)):
                prog_params["nr_hidden"] = nr_hiddens[nh]
                if tt == 0:
                    train_params = get_biaslearner_training_params(prog_params=prog_params, highseed=25)
                else:
                    train_params = get_biaslearner_training_params(prog_params=prog_params, highseed=3)
                performances = get_performances(prog_name=perf_traintypes[tt], load_info=load_dirs[tt],
                                                nr_hidden=nr_hiddens[nh], dataset=datasets[row],
                                                train_params=train_params, performance_type="test_performance")
                perflog[tt] += [performances]
                y += [torch_mean(performances)]
                y_err += [torch_std(performances)]
            axs.bar(x + x_offset[tt], y, width, yerr=y_err, label=legends[tt], zorder=3, color=lcolors[tt])
        axs.set_title(dataset_name[row])
        axs.set_xticks(x)
        axs.grid(axis='y', alpha=0.4, zorder=0)
        axs.set_ylabel("Test Performance [%]")
    axs.set_ylim([90, 101])
    horiz_center = 0.44
    fig.text(0.4, 0.015, 'Network Architecture', ha='center')
    if titling:
        fig.text(horiz_center, 0.96, title, fontsize=14, ha='center')
    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
    axs.set_xticklabels(nr_hiddens)
    if titling:
        top = 0.88
    else:
        top = 0.93
    fig.subplots_adjust(left=0.09, right=0.8, top=top, bottom=0.08, hspace=0.26)
    fig.set_size_inches(9, 7)
    from scipy.stats import ttest_ind
    for i in range(1):
        if i > 2.5:
            d = "EMNIST"
        else:
            d = "CIFAR100"
        print(d, "b vs mr", ttest_ind(perflog[0][i], perflog[1][i]))
    if saving:
        plt.savefig("{}plots/leave_1_out/svg/transfer.svg".format(root_dir))
        plt.savefig("{}plots/leave_1_out/transfer.png".format(root_dir))
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # plot_leave1out_trainb(saving=False)

    # replot_all()
    # for dataset in ["MNIST", "QMNIST", "EMNIST", "EMNIST_letters"]:

    #
    # for dataset in ["MNIST", "QMNIST", "EMNIST", "EMNIST_letters"]:
    #     plot_scan_leave1out_trainb_lr_variance(dataset, saving=False)
    #     for nr_hidden in [[10], [100], [10, 10], [100, 100]]:
    #         plot_scan_trainbw_lr(dataset, nr_hidden, False, subdir="scan_trainbw_lr_100_epochs", mse_only=True)
    #         plot_scan_trainbw_lr(dataset, nr_hidden, False, subdir="scan_trainbw_lr_10_epochs", mse_only=True)
    #         plot_scan_trainbw_lr(dataset, nr_hidden, False, mse_only=True)
    #         plot_multireadout_scan(dataset, nr_hidden, saving=False)
    # for dataset in ["MNIST", "EMNIST", "EMNIST_letters"]:
    #     plot_batch_scan_trainbw(dataset, saving=False, subdir="scan_trainbw_batch_1000epochs")
    #     plot_batch_scan_trainbw(dataset, True)
    # plot_multireadout_scan("EMNIST",  [], saving=False)

    # plot_batch_scan_trainbw("MNIST", saving=False, subdir="scan_trainbw_batch_1000epochs")
    # plot_scan_trainbw_lr("EMNIST", [100, 100], False, mse_only=True)
    # plot_scan_trainbw_lr()
    # plot_willeminit_trainbw(True)
    # plot_train_mnist_transfer_letters(saving=False)

    # replot_all()
    # plot_trainbw(saving=False)
    # nh = [[500, 500]]
    # nh = [[10], [100], [500], [10, 10], [100, 100], [500, 500]]
    # ds = ["MNIST", "QMNIST", "EMNIST_bymerge"]
    # ds = ["MNIST", "QMNIST"]
    # ds = ["MNIST", "QMNIST", "EMNIST", "EMNIST_letters", "EMNIST_bymerge"]
    # plot_scan_trainbw_lr(ds, nr_hiddens=nh, early_stopping=None)
    # plot_leave1out_trainb()
    # mnist_l1o_toy_plot()
    # plot_leave1out_trainbw()

    """ Final plots. """
    saving = False
    tt = True
    plt.style.use("seaborn-colorblind")

    # plot_best_2d_lr(prog_name="scan_train_blr_wlr", saving=saving, titling=tt)
    # plot_best_2d_lr(prog_name="scan_train_bmr_lr", saving=saving, titling=tt)
    # plot_scan_lr("train_b_w", early_stopping=True, titling=tt, saving=saving)
    # plot_scan_lr("train_bw", early_stopping=True, titling=tt, saving=saving)
    # plot_scan_lr("train_bw_hardtanh", early_stopping=True, titling=tt, saving=saving)
    # plot_scan_lr("train_mr", early_stopping=True, titling=tt, saving=saving)
    # plot_scan_lr("train_bmr", early_stopping=True, titling=tt, saving=saving)
    # plot_scan_lr("train_sg", early_stopping=True, titling=tt, saving=saving)
    # plot_scan_lr("train_sg_bigbatch", early_stopping=True, titling=tt, saving=saving)
    # plot_scan_lr("transfer_b_l1o", early_stopping=True, titling=tt, saving=saving)
    # plot_scan_lr("transfer_mr_l1o", early_stopping=True, titling=tt, saving=saving)
    # plot_scan_lr("transfer_bmr_l1o", early_stopping=True, titling=tt, saving=saving)
    # plot_train_test_performances("train_b_w_full", titling=tt, saving=saving)
    # plot_train_test_performances("train_sg_full", titling=tt, saving=saving)
    # plot_train_test_performances("train_mr_full", titling=tt, saving=saving)
    # plot_train_test_performances("train_bmr_full", titling=tt, saving=saving)
    # plot_train_test_performances("train_b_w_l1o", titling=tt, saving=saving)
    # plot_train_test_performances("train_mr_l1o", titling=tt, saving=saving)
    # plot_train_test_performances("train_bmr_l1o", titling=tt, saving=saving)
    # plot_train_test_performances("transfer_b_l1o", titling=tt, saving=saving)
    # plot_train_test_performances("transfer_mr_l1o", titling=tt, saving=saving)
    # multitask_all_plot(saving=saving, titling=tt)
    # multitask_emnist_1hl_plot(saving=saving, titling=tt)
    # multitask_plot(saving=saving, titling=tt)
    # fig_2(saving=saving, titling=tt)
    # k49_plot(saving=saving, titling=tt)
    # bias_vs_gain_plot(saving=False, titling=False)
    # bias_vs_mr_plot(saving=True, titling=False)
    # transfer_plot(saving=saving, titling=False)
