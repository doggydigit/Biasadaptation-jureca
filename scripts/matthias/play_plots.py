import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy import arange as np_arange
from torch import mean as torch_mean
from torch import std as torch_std
from tasks_2d_helper import task_2d_label
from plot_helper import get_dirs, get_performances, plot_comparison, get_hidden_layer_info
from biaslearning_helper import get_biaslearner_training_params
from multireadout_helper import get_multireadout_training_params, get_binarymr_training_params


def plot_trainbw_readout_comparison(saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """
    train_params = {"tanh": get_biaslearner_training_params(readout_function="tanh", highseed=20),
                    "hardtanh": get_biaslearner_training_params(readout_function="hardtanh", highseed=20)}
    plot_params = {"subdir": "scan_trainbw_lr",
                   "prog_name": "scan_train_bw_lr",
                   "perf_type": "validation_performance",
                   "y_label": "Validation Performance [%]",
                   "labels": ["tanh", "hardtanh"],
                   "ymin": 50,
                   "right_adjust": 0.915,
                   "title": "Comparing readout functions for train bw",
                   "save_name": "train_bw_readout_comparison"}
    datasets = ["QMNIST", "EMNIST_bymerge", "EMNIST_letters", "KMNIST"]
    nr_hiddens = [[10], [100], [500], [10, 10], [100, 100], [500, 500]]
    plot_comparison(train_params=train_params, plot_params=plot_params, datasets=datasets, nr_hiddens=nr_hiddens,
                    saving=saving, root_dir=root_dir, titling=titling)


def plot_train_bw_vs_b_w_comparison(saving=False, root_dir="../../", titling=False):
    """
    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """
    nr_hiddens = [[10], [100], [500], [10, 10], [100, 100], [500, 500]]
    train_params = get_biaslearner_training_params(highseed=20)
    load_dirs = ["scan_trainbw_lr", "scan_train_blr_wlr"]
    prog_names = ["scan_train_bw_lr", "scan_train_blr_wlr"]
    labels = ["same lr for w and b", "different lr for w and b"]
    hidden_layer_nrs, hidden_neuron_per_layer_nrs = get_hidden_layer_info(nr_hiddens=nr_hiddens)
    nr_rows = len(hidden_layer_nrs)
    nr_cols = len(hidden_neuron_per_layer_nrs)
    fig, axs = plt.subplots(nr_rows, nr_cols, sharex='all', sharey='all')
    nr_bars = len(load_dirs)
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    datasets = ["QMNIST", "KMNIST", "EMNIST_bymerge", "EMNIST_letters"]
    x = np_arange(len(datasets))

    for r in range(nr_rows):
        for c in range(nr_cols):
            for i in range(nr_bars):
                nr_hidden = hidden_layer_nrs[r] * [hidden_neuron_per_layer_nrs[c]]
                y = []
                y_err = []
                for dataset in datasets:
                    load_dir = "{}results/{}/".format(root_dir, load_dirs[i])
                    performances = get_performances(
                        prog_name=prog_names[i], load_info=load_dir, nr_hidden=nr_hidden, dataset=dataset,
                        train_params=train_params, performance_type="validation_performance"
                    )
                    y += [torch_mean(performances)]
                    y_err += [torch_std(performances)]
                axs[r, c].bar(x + x_offset[i], y, width, yerr=y_err, label=labels[i], zorder=3)
            axs[r, c].set_title(
                "{} x {} hidden neurons".format(hidden_layer_nrs[r], hidden_neuron_per_layer_nrs[c]))
            axs[r, c].set_xticks(x)
            axs[r, c].grid(axis='y', alpha=0.4, zorder=0)
            axs[r, c].set_ylim(50, 105)

        axs[r, 0].set_ylabel("Validation Performance [%]")
    for c in range(len(hidden_layer_nrs)):
        axs[-1, c].set_xticklabels([d.replace("_", "\n") for d in datasets])

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    fig.subplots_adjust(left=0.05, right=0.85, top=0.91, wspace=0.05, hspace=0.16)
    fig.set_size_inches(16, 9)
    if titling:
        plt.suptitle("Comparing weight training with equal or unequal learning rate for biases", fontsize=14)
    if saving:
        plt.savefig("{}plots/scan_params/svg/{}.svg".format(root_dir, "train_bw_vs_b_w"))
        plt.savefig("{}plots/scan_params/{}.png".format(root_dir, "train_bw_vs_b_w"))
        plt.close()
    else:
        plt.show()


def plot_trainbw_early_stopping_comparison(saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """
    train_params = {"early": get_biaslearner_training_params(early_stopping=True, highseed=20),
                    "late": get_biaslearner_training_params(early_stopping=False, highseed=20)}
    plot_params = {"subdir": "scan_trainbw_lr",
                   "prog_name": "scan_train_bw_lr",
                   "perf_type": "validation_performance",
                   "y_label": "Validation Performance [%]",
                   "labels": ["+ early stopping", "- early stopping"],
                   "ymin": 50,
                   "right_adjust": 0.885,
                   "title": "Comparing train bw with and without early stopping",
                   "save_name": "train_bw_early_stopping_comparison"}
    datasets = ["QMNIST", "EMNIST_bymerge", "EMNIST_letters", "KMNIST"]
    nr_hiddens = [[10], [100], [500], [10, 10], [100, 100], [500, 500]]
    plot_comparison(train_params=train_params, plot_params=plot_params, datasets=datasets, nr_hiddens=nr_hiddens,
                    saving=saving, root_dir=root_dir, titling=titling)


def plot_layer_comparison(saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """

    plot_params = {"prog_names": ["train_bw_l1o", "transfer_b_l1o"],
                   "labels": ["1 hidden layer", "2 hidden layers"],
                   "ymin": 50,
                   "right_adjust": 0.893,
                   "title": "Comparing leave 1 out performances depending on number of hidden layers",
                   "save_name": "l1o_nr_layer_comparison"}
    y_labels = ["Train Test Performance [%]", "Transfer Test Performance [%]"]
    train_params = [get_biaslearner_training_params(), get_biaslearner_training_params(transfering=True)]
    # datasets = ["MNIST", "QMNIST", "KMNIST", "EMNIST_bymerge", "EMNIST_letters", "EMNIST"]
    datasets = ["QMNIST", "KMNIST", "EMNIST_bymerge", "EMNIST_letters"]
    load_dir = "{}results/leave_1_out/".format(root_dir)
    hidden_layer_nrs = [1, 2]
    hidden_neuron_per_layer_nrs = [10, 25, 50, 100, 250, 500]
    nr_rows = len(hidden_layer_nrs)
    nr_cols = len(datasets)
    fig, axs = plt.subplots(nr_rows, nr_cols, sharex='all', sharey='all')
    x = np_arange(len(hidden_neuron_per_layer_nrs))

    for r in range(nr_rows):
        for c in range(nr_cols):
            for hln in range(len(hidden_layer_nrs)):
                y = []
                y_err = []
                for hnpln in range(len(hidden_neuron_per_layer_nrs)):
                    nr_hidden = hidden_layer_nrs[hln] * [hidden_neuron_per_layer_nrs[hnpln]]
                    performances = get_performances(
                        prog_name=plot_params["prog_names"][r], load_info=load_dir, nr_hidden=nr_hidden,
                        dataset=datasets[c], train_params=train_params[r], performance_type="test_performance"
                    )
                    y += [torch_mean(performances)]
                    y_err += [torch_std(performances)]
                axs[r, c].errorbar(x, y, y_err, label=plot_params["labels"][hln], zorder=3)

            axs[r, c].set_xticks(x)
            axs[r, c].grid(axis='y', alpha=0.4, zorder=0)
            axs[r, c].set_ylim(plot_params["ymin"], 100)

        axs[r, 0].set_ylabel(y_labels[r])
    fig.text(0.47, 0.055, 'Number of neurons in each hidden layer', ha='center')
    for c in range(nr_cols):
        axs[0, c].set_title(datasets[c])
        axs[-1, c].set_xticklabels([hnpln for hnpln in hidden_neuron_per_layer_nrs])

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    fig.subplots_adjust(left=0.05, right=plot_params["right_adjust"], top=0.91, wspace=0.05, hspace=0.05)
    fig.set_size_inches(16, 9)
    if titling:
        plt.suptitle(plot_params["title"], fontsize=14)
    if saving:
        plt.savefig("{}plots/leave_1_out/svg/{}.svg".format(root_dir, plot_params["save_name"]))
        plt.savefig("{}plots/leave_1_out/{}.png".format(root_dir, plot_params["save_name"]))
        plt.close()
    else:
        plt.show()


def mnist_l1o_toy_plot(saving=False):
    loss_function = "mse"
    readout_function = "hardtanh"
    root_dir = "../../"
    dataset = "MNIST"
    nr_hiddens = [[10], [100], [10, 10], [100, 100]]
    xticklabels = ["1x10", "1x100", "2x10", "2x100"]

    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=200)
    train_types = ["leave1out_trainbw", "leave1out_trainb"]
    width = 1. / (len(train_types) + 1)
    x_offset = [i * width - 0.5 for i in range(1, len(train_types) + 1)]
    x = np.arange(len(nr_hiddens))
    bar_colors = ["r", "g"]
    legends = ["digits seen during weight training", "digits left out during weight training"]
    for tt in range(len(train_types)):
        load_dir, _ = get_dirs(root_dir, "old/" + train_types[tt])
        y = []
        std = []
        for nr_hidden in nr_hiddens:
            save_name = "network_{}_{}_{}_{}.pickle".format(nr_hidden, dataset, loss_function, readout_function)

            with open(load_dir + save_name, "rb") as f:
                results = pickle.load(f)
            if train_types[tt] == "leave1out_trainbw":
                result = results
            else:
                result = results["class_results"]
            r = torch.cat([torch.stack(result[j]["validation_performance"]) for j in range(10)])
            y += [torch.mean(r)]
            std += [torch.std(r)]
        plt.bar(x + x_offset[tt], y, width, yerr=std, color=bar_colors[tt], label=legends[tt])

    # plt.set_title(plot_type)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    plt.ylabel("Test Performance [%]")
    plt.xlabel("Network Architecture")
    plt.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()

    plt.ylim([0., 100.])
    save_dir = "../../plots/toyplots/"
    if saving:
        fig.set_size_inches(16, 9)
        plt.savefig(save_dir + "gcb_transfer_plot.png", bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def willem_init_toyplot(saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """

    dataset = "EMNIST_bymerge"
    nr_hiddens = [[10], [25], [50], [100], [250], [500]]
    wtypes = ["biaslearner", "pmdd_winit_biaslearner", "scd_winit_biaslearner"]
    legends = ["Gaussian", "PMDD", "SCD"]

    train_params = get_biaslearner_training_params()
    subdir = "train_full_dataset"
    title = "Training with different weight initializations"
    load_dir, save_dir = get_dirs(root_dir, subdir)

    fig = plt.figure(figsize=(16, 4))
    nr_bars = len(wtypes)
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(nr_hiddens))

    for wt in range(len(wtypes)):
        wtype = wtypes[wt]
        y = []
        y_err = []
        for nh in range(len(nr_hiddens)):
            performances = get_performances(prog_name="train_bw_full", load_info=load_dir, nr_hidden=nr_hiddens[nh],
                                            dataset=dataset, train_params=train_params,
                                            performance_type="test_performance", model_type=wtype)
            y += [torch_mean(performances)]
            y_err += [torch_std(performances)]
        plt.bar(x + x_offset[wt], y, width, yerr=y_err, label=legends[wt], zorder=3)
    plt.xticks(x, labels=nr_hiddens)
    plt.grid(axis='y', alpha=0.4, zorder=0)
    plt.ylim([75, 100])
    plt.legend(framealpha=1, loc="upper left")
    plt.xlabel("Network Architecture")
    plt.ylabel("Test Performance [%]")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, wspace=0.05, hspace=0.06)
    if titling:
        plt.suptitle(title, fontsize=14, y=0.96)
    if saving:
        plt.savefig(save_dir + "svg/willem_init_train_test_performance_plot.svg")
        plt.savefig(save_dir + "willem_init_train_test_performance_plot.png")
        plt.close()
    else:
        plt.show()


def willem_deepened_toyplot(saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """

    # dataset = "EMNIST_bymerge"
    dataset = "EMNIST_willem"
    final = True
    if final:
        nr_hiddenss = [[[10, 100], [25, 100], [50, 100], [100, 100], [250, 100], [500, 100]],
                       [[100, 10], [100, 25], [100, 50], [100, 100], [100, 250], [100, 500]],
                       [[10, 10], [25, 25], [50, 50], [100, 100], [250, 250], [500, 500]]]
        subtitles = ["Architectures [x, 100]",
                     "Architectures [100, x]",
                     "Architectures [x, x]"]
    else:
        nr_hiddenss = [[[10, 100], [25, 100], [50, 100], [100, 100], [250, 100], [500, 100]],
                       [[10, 10], [25, 10], [50, 10], [100, 10], [250, 10], [500, 10]],
                       [[10, 10], [25, 25], [50, 50], [100, 100], [250, 250], [500, 500]]]
        subtitles = ["Architectures [x, 100]",
                     "Architectures [x, 10]",
                     "Architectures [x, x]"]
    wtypes = ["pmdd", "scd"]

    train_params = get_biaslearner_training_params(highseed=1)
    subdir = "willem_weights"
    title = "Finding deep weights with backprop"
    load_dir, save_dir = get_dirs(root_dir, subdir)

    nr_cols = len(nr_hiddenss)
    fig, axs = plt.subplots(1, nr_cols, sharex='all', sharey='all')
    nr_bars = len(nr_hiddenss[0])
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(wtypes))

    for c in range(nr_cols):
        for nh in range(len(nr_hiddenss[c])):
            nr_hidden = nr_hiddenss[c][nh]
            y = []
            y_err = []
            for wt in range(len(wtypes)):
                load_info = "{}deepen_networks/deepened_{}".format(load_dir, wtypes[wt])
                performances = get_performances(prog_name="willem_deepened", load_info=load_info, nr_hidden=nr_hidden,
                                                dataset=dataset, train_params=train_params,
                                                performance_type="validation_performance")
                y += [torch_mean(performances)]
                y_err += [torch_std(performances)]
            axs[c].bar(x + x_offset[nh], y, width, yerr=y_err, label=str(nr_hidden))
        axs[c].set_xticks(x)
        axs[c].grid(axis='y', alpha=0.4)
        axs[c].set_xticklabels(wtypes)
        axs[c].set_title(subtitles[c])
        axs[c].set_ylim([50, 100])
        axs[c].legend(framealpha=1, loc="lower right")

    axs[0].set_ylabel("Validation Performance [%]")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.8, wspace=0.05, hspace=0.06)
    fig.set_size_inches(16, 4)
    if titling:
        plt.suptitle(title, fontsize=16, y=0.96)
    if saving:
        plt.savefig(save_dir + "svg/willem_deepened_train_test_performance_plot.svg")
        plt.savefig(save_dir + "willem_deepened_train_test_performance_plot.png")
        plt.close()
    else:
        plt.show()


def l1o_all_plot(saving=False, root_dir="../../", titling=False):
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
    nr_hiddens = [[10], [25], [50], [100], [250], [500],
                  [10, 10], [25, 25], [50, 50], [100, 100], [250, 250], [500, 500]]
    traintypes = ["train_sg_full", "train_b_w_l1o", "transfer_b_l1o_b_w", "train_bmr_l1o", "transfer_bmr_l1o"]
    tts = ["train_bw", "train_b_w", "train_b_w", "train_binarymr", "train_binarymr"]
    legends = ["Singular approach", "Train weights + biases", "Transfer learn biases", "Train multireadout",
               "Transfer learn readout"]

    load_dirs = ["{}results/train_full_dataset/".format(root_dir)] + 4 * ["{}results/leave_1_out/".format(root_dir)]
    title = "Leave one out transfer learning performances"

    fig, axs = plt.subplots(len(datasets), 1, sharex='all', sharey='all')
    nr_bars = len(traintypes)
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(nr_hiddens))

    for row in range(len(datasets)):
        prog_params = {"dataset": datasets[row]}
        for tt in range(len(traintypes)):
            prog_params["training_type"] = tts[tt]
            if tt == 0:
                hs = 3
            else:
                hs = 25
            transfering = tt in [2, 4]
            y = []
            y_err = []
            for nh in range(len(nr_hiddens)):
                prog_params["nr_hidden"] = nr_hiddens[nh]
                if tt > 2.5:
                    train_params = get_binarymr_training_params(prog_params=prog_params, highseed=hs)
                else:
                    train_params = get_biaslearner_training_params(prog_params=prog_params, transfering=transfering,
                                                                   highseed=hs)
                performances = get_performances(prog_name=traintypes[tt], load_info=load_dirs[tt],
                                                nr_hidden=nr_hiddens[nh], dataset=datasets[row],
                                                train_params=train_params, performance_type="test_performance")
                y += [torch_mean(performances)]
                y_err += [torch_std(performances)]
            axs[row].bar(x + x_offset[tt], y, width, yerr=y_err, label=legends[tt], zorder=3)
        axs[row].set_title(datasets[row])
        axs[row].set_xticks(x)
        axs[row].grid(axis='y', alpha=0.4, zorder=0)
        axs[row].set_ylim([50, 105])
        axs[row].set_ylabel("Test Performance [%]")
    horiz_center = 0.44
    fig.text(horiz_center, 0.015, 'Network Architecture', ha='center')
    if titling:
        fig.text(horiz_center, 0.96, title, fontsize=14, ha='center')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    axs[-1].set_xticklabels(nr_hiddens)
    fig.subplots_adjust(left=0.06, right=0.83, top=0.91, bottom=0.06, hspace=0.26)
    fig.set_size_inches(13, 10)
    if saving:
        plt.savefig("{}plots/leave_1_out/svg/l1o_comparisons.svg".format(root_dir))
        plt.savefig("{}plots/leave_1_out/l1o_comparisons.png".format(root_dir))
        plt.close()
    else:
        plt.show()


def l1o_asym_toyplot(saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """

    datasets = ["QMNIST", "KMNIST", "EMNIST_bymerge", "EMNIST_letters"]
    nr_hiddens = [[10, 100], [25, 100], [50, 100], [100, 100], [250, 100], [500, 100],
                  [100, 10], [100, 25], [100, 50], [100, 100], [100, 250], [100, 500]]
    traintypes = ["train_b_w_l1o", "train_mr_l1o"]
    legends = ["Train weights + biases", "Train multireadout"]

    load_dirs = 2 * ["{}results/leave_1_out/".format(root_dir)]
    train_params = [get_biaslearner_training_params(highseed=3),
                    get_biaslearner_training_params(),
                    get_biaslearner_training_params(transfering=True),
                    get_multireadout_training_params(),
                    get_multireadout_training_params()]
    title = "Leave one out transfer learning performances"

    fig, axs = plt.subplots(len(datasets), 1, sharex='all', sharey='all')
    nr_bars = len(traintypes)
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(nr_hiddens))

    for row in range(len(datasets)):
        for tt in range(len(traintypes)):
            y = []
            y_err = []
            for nh in range(len(nr_hiddens)):
                performances = get_performances(prog_name=traintypes[tt], load_info=load_dirs[tt],
                                                nr_hidden=nr_hiddens[nh], dataset=datasets[row],
                                                train_params=train_params[tt], performance_type="test_performance")
                y += [torch_mean(performances)]
                y_err += [torch_std(performances)]
            axs[row].bar(x + x_offset[tt], y, width, yerr=y_err, label=legends[tt], zorder=3)
        axs[row].set_title(datasets[row])
        axs[row].set_xticks(x)
        axs[row].grid(axis='y', alpha=0.4, zorder=0)
        axs[row].set_ylim([50, 105])
        axs[row].set_ylabel("Test Performance [%]")
    horiz_center = 0.44
    fig.text(horiz_center, 0.015, 'Network Architecture', ha='center')
    if titling:
        fig.text(horiz_center, 0.96, title, fontsize=14, ha='center')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    axs[-1].set_xticklabels(nr_hiddens)
    fig.subplots_adjust(left=0.06, right=0.83, top=0.91, bottom=0.06, hspace=0.26)
    fig.set_size_inches(13, 10)
    if saving:
        plt.savefig("{}plots/leave_1_out/svg/l1o_comparisons.svg".format(root_dir))
        plt.savefig("{}plots/leave_1_out/l1o_comparisons.png".format(root_dir))
        plt.close()
    else:
        plt.show()


def bw_vs_b_w_l1o(saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """

    datasets = ["QMNIST", "KMNIST", "EMNIST_bymerge", "EMNIST_letters"]
    nr_hiddens = [[10], [25], [50], [100], [250], [500],
                  [10, 10], [25, 25], [50, 50], [100, 100], [250, 250], [500, 500]]
    traintypes = ["train_sg_full", "train_b_w_l1o", "transfer_b_l1o_b_w", "train_bw_l1o", "transfer_b_l1o_bw"]
    legends = ["Singular approach", "Train b w", "Transfer b w", "Train bw", "Transfer bw"]

    load_dirs = ["{}results/train_full_dataset/".format(root_dir)] + 2 * ["{}results/leave_1_out/".format(root_dir)]
    load_dirs += 2 * ["{}results/old/train_bw_stash/leave_1_out/".format(root_dir)]
    train_params = [get_biaslearner_training_params(highseed=3),
                    get_biaslearner_training_params(highseed=3),
                    get_biaslearner_training_params(highseed=3, transfering=True),
                    get_biaslearner_training_params(highseed=3),
                    get_biaslearner_training_params(highseed=3, transfering=True)]
    title = "Leave one out transfer learning performances"

    fig, axs = plt.subplots(len(datasets), 1, sharex='all', sharey='all')
    nr_bars = len(traintypes)
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(nr_hiddens))

    for row in range(len(datasets)):
        for tt in range(len(traintypes)):
            y = []
            y_err = []
            for nh in range(len(nr_hiddens)):
                performances = get_performances(prog_name=traintypes[tt], load_info=load_dirs[tt],
                                                nr_hidden=nr_hiddens[nh], dataset=datasets[row],
                                                train_params=train_params[tt], performance_type="test_performance")
                y += [torch_mean(performances)]
                y_err += [torch_std(performances)]
            axs[row].bar(x + x_offset[tt], y, width, yerr=y_err, label=legends[tt], zorder=3)
        axs[row].set_title(datasets[row])
        axs[row].set_xticks(x)
        axs[row].grid(axis='y', alpha=0.4, zorder=0)
        axs[row].set_ylim([60, 105])
        axs[row].set_ylabel("Test Performance [%]")
    horiz_center = 0.44
    fig.text(horiz_center, 0.015, 'Network Architecture', ha='center')
    if titling:
        fig.text(horiz_center, 0.96, title, fontsize=14, ha='center')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    axs[-1].set_xticklabels(nr_hiddens)
    fig.subplots_adjust(left=0.06, right=0.83, top=0.91, bottom=0.06, hspace=0.26)
    fig.set_size_inches(13, 10)
    if saving:
        plt.savefig("{}plots/leave_1_out/svg/train_bw_vs_b_w.svg".format(root_dir))
        plt.savefig("{}plots/leave_1_out/train_bw_vs_b_w.png".format(root_dir))
        plt.close()
    else:
        plt.show()


def l1o_bymerge_plot(saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """

    datasets = ["EMNIST_bymerge"]
    nr_hiddens = [[10], [25], [50], [100], [250], [500]]
    traintypes = ["train_sg_full", "train_b_w_l1o", "transfer_b_l1o_b_w", "train_bmr_l1o", "transfer_bmr_l1o"]
    legends = ["Singular approach", "Train weights + biases", "Transfer learn biases", "Train multireadout",
               "Transfer learn readout"]

    load_dirs = ["{}results/train_full_dataset/".format(root_dir)] + 4 * ["{}results/leave_1_out/".format(root_dir)]
    train_params = [get_biaslearner_training_params(highseed=3),
                    get_biaslearner_training_params(highseed=1),
                    get_biaslearner_training_params(highseed=1, transfering=True),
                    get_binarymr_training_params(highseed=1),
                    get_binarymr_training_params(highseed=1)]
    title = "Leave one out transfer learning performances"

    fig, axs = plt.subplots(len(datasets), 1, sharex='all', sharey='all')
    nr_bars = len(traintypes)
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(nr_hiddens))

    for row in range(len(datasets)):
        for tt in range(len(traintypes)):
            y = []
            y_err = []
            for nh in range(len(nr_hiddens)):
                performances = get_performances(prog_name=traintypes[tt], load_info=load_dirs[tt],
                                                nr_hidden=nr_hiddens[nh], dataset=datasets[row],
                                                train_params=train_params[tt], performance_type="test_performance")
                y += [torch_mean(performances)]
                y_err += [torch_std(performances)]
            axs.bar(x + x_offset[tt], y, width, yerr=y_err, label=legends[tt], zorder=3)
        axs.set_title(datasets[row])
        axs.set_xticks(x)
        axs.grid(axis='y', alpha=0.4, zorder=0)
        axs.set_ylim([70, 105])
        axs.set_ylabel("Test Performance [%]")
    horiz_center = 0.44
    fig.text(horiz_center, 0.015, 'Network Architecture', ha='center')
    if titling:
        fig.text(horiz_center, 0.96, title, fontsize=14, ha='center')
    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    axs.set_xticklabels(nr_hiddens)
    fig.subplots_adjust(left=0.06, right=0.83, top=0.91, bottom=0.06, hspace=0.26)
    fig.set_size_inches(13, 10)
    if saving:
        plt.savefig("{}plots/leave_1_out/svg/l1o_comparisons.svg".format(root_dir))
        plt.savefig("{}plots/leave_1_out/l1o_comparisons.png".format(root_dir))
        plt.close()
    else:
        plt.show()


def plot_2d_tasks(saving=True, model=None, root_dir="../../", savename="Tasks48", debug=False):
    nr_tasks = 48
    tasks = list(range(nr_tasks))
    datamin, datamax, inputsize, nrpixperdim = 0., 1., 2, 300
    if debug:
        nrpixperdim = 30
    aa = torch.linspace(datamin, datamax, nrpixperdim)
    input_x, input_y = torch.meshgrid(aa, aa)
    input_x = torch.reshape(input_x, (nrpixperdim * nrpixperdim,))
    input_y = torch.reshape(input_y, (nrpixperdim * nrpixperdim,))
    input_data = torch.stack((input_x, input_y), 1)
    nrrows = min(4, nr_tasks)
    nrcols = 1 + int((nr_tasks - 1) / nrrows)
    single_col = nrcols < 1.1
    figwidth = 3 * nrcols
    fig, axes = plt.subplots(nrows=min(nrrows, nr_tasks), ncols=nrcols, figsize=(nrcols, nrrows))
    plt.subplots_adjust(bottom=0.05, top=0.95, wspace=0.1)
    for task in tasks:
        if model:
            data_labels = model.forward(input_data, task)
            g = data_labels > 0.
            g = g.squeeze()
            data_labels = task_2d_label(task, input_data)
            truth = data_labels > 0.5
            errors = torch.logical_xor(truth, g)
        else:
            data_labels = task_2d_label(task, input_data)
            g = data_labels > 0.5
        r = [not i for i in g]
        if single_col:
            a = axes[task]
        else:
            a = axes[task % nrrows][int(task / nrrows)]
        if debug:
            marksize = 2.
            fmarksize = 1.
            c1 = 'lightsalmon'
            c2 = "lightseagreen"
        else:
            marksize = 0.15
            fmarksize = 0.1
            c1 = 'lightsalmon'
            c2 = "lightseagreen"
        a.plot(input_data[g, 0], input_data[g, 1], 's', color=c1, markersize=marksize)
        a.plot(input_data[r, 0], input_data[r, 1], 's', color=c2,  markersize=marksize)
        if model:
            a.plot(input_data[errors, 0], input_data[errors, 1], 's', color="k", markersize=fmarksize)
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.xaxis.set_ticks_position('none')
        a.yaxis.set_ticks_position('none')
    for ax in range(nrrows * nrcols):
        if single_col:
            a = axes[ax]
        else:
            a = axes[ax % nrrows][int(ax / nrrows)]
        a.axis('off')
    # fig.set_size_inches(18, 6)
    fig.subplots_adjust(left=0., right=1., wspace=0., hspace=0., top=1., bottom=0.)
    if saving:
        plt.savefig("{}plots/toyplots/{}.png".format(root_dir, savename), dpi=800)
        plt.close()
    else:
        plt.show()


def plot_best_2d_lr_tasks2d(prog_name="scan_train_blr_wlr", root_dir="../../", titling=True):
    """
    Plot best weight and bias learning rates for training biaslearner.
    Parameters
    ----------
    prog_name:
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    """

    # Process the types of network architectures to plot, to know the number of subplots to draw
    if prog_name == "scan_train_blr_wlr":
        multicol = True
        if multicol:
            if True:
                nr_hiddens = [[50], [50, 50], [50, 50, 50], [500], [500, 500], [500, 500, 500]]
                lrs1 = [0.3, 0.2, 0.1, 0.06, 0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0001, 0.00003, 0.00001]
                lrs2 = [0.5, 0.3, 0.2, 0.1, 0.06, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
            else:
                nr_hiddens = [[100, 100], [100, 100, 100], [1000, 1000], [1000, 1000, 1000]]
                lrs1 = [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
                lrs2 = [0.5, 0.3, 0.2, 0.1, 0.06, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
        else:
            nr_hiddens = [[50], [50, 50], [50, 50, 50], [50, 50, 50, 50]]
            lrs1 = [0.3, 0.2, 0.1, 0.06, 0.03, 0.01, 0.006, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0001, 0.00003, 0.00001]
            lrs2 = [0.5, 0.3, 0.2, 0.1, 0.06, 0.03, 0.01, 0.006, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
        train_params = get_biaslearner_training_params(highseed=20)
        subdir = "scan_train_blr_wlr"
        min_lr = 0.00001
        lr1 = "lr"
        lr2 = "b_lr"
        xlab = "weight learning rate"
        ylab = "bias learning rate"
    elif prog_name == "scan_train_bmr_lr":
        multicol = True
        if multicol:
            nr_hiddens = [[50], [50, 50], [50, 50, 50], [500], [500, 500], [500, 500, 500]]
            lrs1 = [0.3, 0.1, 0.03, 0.01, 0.006, 0.003, 0.001, 0.0006, 0.0003, 0.0001, 0.00001]
            lrs2 = [0.3, 0.1, 0.06, 0.03, 0.01, 0.001, 0.0001, 0.00001]
        else:
            nr_hiddens = [[50], [50, 50], [50, 50, 50], [50, 50, 50, 50]]
            lrs1 = [0.3, 0.1, 0.03, 0.01, 0.006, 0.003, 0.001, 0.0006, 0.0003, 0.0001, 0.00001]
            lrs2 = [0.3, 0.1, 0.06, 0.03, 0.01, 0.001, 0.0001, 0.00001]
        train_params = get_biaslearner_training_params(highseed=20)
        subdir = "scan_train_bmr_lr"
        min_lr = 0.00001
        lr1 = "lr"
        lr2 = "r_lr"
        xlab = "weight learning rate"
        ylab = "readout learning rate"

    hidden_layer_nrs, hidden_neuron_per_layer_nrs = get_hidden_layer_info(nr_hiddens=nr_hiddens)
    row_ids = []
    col_ids = []
    if multicol:
        fig, axs = plt.subplots(len(hidden_layer_nrs), len(hidden_neuron_per_layer_nrs))
        for nr_hidden in nr_hiddens:
            row_ids += [hidden_layer_nrs.index(len(nr_hidden))]
            col_ids += [hidden_neuron_per_layer_nrs.index(nr_hidden[0])]
    else:
        fig, axs = plt.subplots(len(hidden_neuron_per_layer_nrs), len(hidden_layer_nrs))
        for nr_hidden in nr_hiddens:
            row_ids += [hidden_neuron_per_layer_nrs.index(nr_hidden[0])]
            col_ids += [hidden_layer_nrs.index(len(nr_hidden))]
    datasets = ["TASKS2D"]
    markers = {"TASKS2D": "1"}
    load_dir, save_dir = get_dirs(root_dir, subdir)

    for nh in range(len(nr_hiddens)):
        row = row_ids[nh]
        col = col_ids[nh]
        for dataset in datasets:
            ys = []
            lrs = []
            for wlr in lrs1:
                train_params[lr1] = wlr
                for blr in lrs2:
                    lrs += [[wlr, blr]]
                    train_params[lr2] = blr
                    performances = get_performances(prog_name=prog_name, load_info=load_dir,
                                                    nr_hidden=nr_hiddens[nh], dataset=dataset,
                                                    train_params=train_params,
                                                    performance_type="validation_performance")

                    ys += [torch_mean(performances)]
            m = max(zip(ys, range(len(ys))))[1]
            print(nr_hiddens[nh], ys[m], float(lrs[m][0]), float(lrs[m][1]))
            if multicol:
                axs[row, col].plot(float(lrs[m][0]), float(lrs[m][1]), markers[dataset], markersize=8, label=dataset)
                axs[row, col].plot([min_lr, 1.], [min_lr, 1.], "k:", linewidth=0.7)
                axs[row, col].set_title("Network {}".format(nr_hiddens[nh]))
                axs[row, col].set_xscale('log')
                axs[row, col].set_yscale('log')
                axs[row, col].set_xlim([min_lr, 1.])
                axs[row, col].set_ylim([min_lr, 1.])
                axs[row, col].grid(which="both")
            else:
                axs[col].plot(float(lrs[m][0]), float(lrs[m][1]), markers[dataset], markersize=8, label=dataset)
                axs[col].plot([min_lr, 1.], [min_lr, 1.], "k:", linewidth=0.7)
                axs[col].set_title("Network {}".format(nr_hiddens[nh]))
                axs[col].set_xscale('log')
                axs[col].set_yscale('log')
                axs[col].set_xlim([min_lr, 1.])
                axs[col].set_ylim([min_lr, 1.])
                axs[col].grid(which="both")

    # Plotting Specifics
    if multicol:
        for col in range(len(hidden_neuron_per_layer_nrs)):
            axs[len(hidden_layer_nrs) - 1, col].set_xlabel(xlab)
        for row in range(len(hidden_layer_nrs)):
            axs[row, 0].set_ylabel(ylab)
        handles, labels = axs[0, 0].get_legend_handles_labels()
    else:
        for col in range(len(hidden_neuron_per_layer_nrs)):
            axs[len(hidden_layer_nrs) - 1].set_xlabel(xlab)
        for row in range(len(hidden_layer_nrs)):
            axs[row].set_ylabel(ylab)
        handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    if multicol:
        fig.set_size_inches(16, 9)
        fig.subplots_adjust(left=0.05, right=0.875, wspace=0.15, hspace=0.26)
    else:
        fig.set_size_inches(18, 5)
        fig.subplots_adjust(left=0.06, right=0.92, wspace=0.15, hspace=0.47)

    if titling:
        plt.suptitle("Scanning separate bias and weight learning rates")
    plt.show()


def plot_scan_lr_2d(train_type, train_params=None, saving=False, datasets=None, nr_hiddens=None, lrs=None,
                    early_stopping=None, root_dir="../../", titling=False):
    """
    Plot validation performances for different learning rates, different datasets, different network architectures and
    possibly different training stopping mechanisms.
    Parameters
    ----------
    train_type: type of training to plot
    train_params: specific training parameters used during training
    saving: Whether to save the plot to file or just show it
    datasets: List of datasets used for training.
    nr_hiddens: list of the number of hidden neurons in the hidden layers.
    lrs: list of learning rates to plot
    early_stopping: whether early stopping was used. None will lead to both possibilities being plotted
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    """

    if datasets is None:
        # datasets = ["MNIST", "QMNIST", "KMNIST", "EMNIST_bymerge", "EMNIST_letters", "EMNIST"]
        datasets = ["TASKS2D"]
    if train_type == "train_b_w":
        # Some argument dependent default initializations
        multicol = False

        if multicol:
            if True:
                nr_hiddens = [[50], [50, 50], [50, 50, 50], [500], [500, 500], [500, 500, 500]]
                lrs = [0.3, 0.2, 0.1, 0.06, 0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0001, 0.00003,
                       0.00001]  # w_lr
                # lrs = [0.5, 0.3, 0.2, 0.1, 0.06, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]  # blr
                train_params = get_biaslearner_training_params(new_train_params=train_params, blr=0.01, highseed=20)
            else:
                nr_hiddens = [[100, 100], [1000, 1000], [100, 100, 100], [1000, 1000, 1000]]
                lrs = [0.3, 0.1, 0.06, 0.03, 0.01, 0.006, 0.003, 0.001]  # w_lr
                train_params = get_biaslearner_training_params(new_train_params=train_params, blr=0.06, highseed=20)
        else:
            nr_hiddens = [[50], [50, 50], [50, 50, 50], [50, 50, 50, 50]]
            nr_hiddens = [[500], [500, 500], [500, 500, 500], [500, 500, 500, 500]]
            lrs = [0.3, 0.2, 0.1, 0.06, 0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0001, 0.00003, 0.00001]  # w_lr
            train_params = get_biaslearner_training_params(new_train_params=train_params, blr=0.01, highseed=20)
        subdir = "scan_train_blr_wlr"
        prog_name = "scan_train_blr_wlr"
    elif train_type == "train_bmr":
        multicol = True
        if multicol:
            nr_hiddens = [[50], [50, 50], [50, 50, 50], [500], [500, 500], [500, 500, 500]]
            lrs = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00001]  # w_lr
            train_params = get_binarymr_training_params(new_train_params=train_params, r_lr=0.01, highseed=20)

            # lrs = [0.3, 0.1, 0.06, 0.03, 0.01, 0.001, 0.0001, 0.00001]  # r_lr
            # train_params = get_binarymr_training_params(new_train_params=train_params, lr=0.001, highseed=20)
        else:
            nr_hiddens = [[50], [50, 50], [50, 50, 50], [50, 50, 50, 50]]
            train_params = get_binarymr_training_params(new_train_params=train_params, r_lr=0.1, highseed=20)
            lrs = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00001]  # w_lr
        subdir = "scan_train_bmr_lr"
        prog_name = "scan_train_bmr_lr"
    else:
        raise ValueError(train_type)
    load_dir, save_dir = get_dirs(root_dir, subdir)

    # Process the types of network architectures to plot, to know the number of subplots to draw
    hidden_layer_nrs, hidden_neuron_per_layer_nrs = get_hidden_layer_info(nr_hiddens=nr_hiddens)
    fig, axs = plt.subplots(len(hidden_layer_nrs), len(hidden_neuron_per_layer_nrs))
    row_ids = []
    col_ids = []
    for nr_hidden in nr_hiddens:
        row_ids += [hidden_layer_nrs.index(len(nr_hidden))]
        col_ids += [hidden_neuron_per_layer_nrs.index(nr_hidden[0])]

    if early_stopping is None:
        early_stopping = [True, False]
        label_es = [" with early stopping", " without early stopping"]
    else:
        early_stopping = [early_stopping]
        label_es = [""]

    for nh in range(len(nr_hiddens)):
        row = row_ids[nh]
        col = col_ids[nh]
        if multicol:
            ax = axs[row, col]
        else:
            ax = axs[row]
        for dataset in datasets:
            for es in range(len(early_stopping)):
                train_params["early_stopping"] = early_stopping[es]
                ys = []
                errs = []
                for lr in lrs:
                    train_params["lr"] = lr
                    performances = get_performances(prog_name=prog_name, load_info=load_dir, nr_hidden=nr_hiddens[nh],
                                                    dataset=dataset, train_params=train_params,
                                                    performance_type="validation_performance")

                    ys += [torch_mean(performances)]
                    errs += [torch_std(performances)]
                ax.errorbar(lrs, ys, errs, label=dataset + label_es[es])
                m = max(zip(ys, range(len(ys))))[1]
                ax.plot(float(lrs[m]), ys[m], "xk")

            ax.set_title("Network {}".format(nr_hiddens[nh]))
            ax.set_xscale('log')
            ax.set_ylim([80., 100.])

            # Plotting Specifics
        if train_type == "train_b_w":
            x_label = "bias learning rate"
        else:
            x_label = "learning rate"
        if multicol:
            for col in range(len(hidden_neuron_per_layer_nrs)):
                axs[len(hidden_layer_nrs) - 1, col].set_xlabel(x_label)
            for row in range(len(hidden_layer_nrs)):
                axs[row, 0].set_ylabel('Validation Performance [%]')
            handles, labels = axs[0, 0].get_legend_handles_labels()
        else:
            for col in range(len(hidden_neuron_per_layer_nrs)):
                axs[len(hidden_layer_nrs) - 1].set_xlabel(x_label)
            for row in range(len(hidden_layer_nrs)):
                axs[row].set_ylabel('Validation Performance [%]')
            handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')
        fig.set_size_inches(16, 9)
        fig.subplots_adjust(left=0.05, right=0.875, wspace=0.15, hspace=0.26)

    if titling:
        plt.suptitle("Scanning learning rates for {}".format(train_type))
    if saving:
        plt.savefig(root_dir + "plots/scan_params/svg/{}_performance_plot.svg".format(prog_name))
        plt.savefig(root_dir + "plots/scan_params/{}_performance_plot.png".format(prog_name))
        plt.close()
    else:
        plt.show()


def plot_2d_model_output(saving=True, debug=False):
    from biasadaptation.biasfit.biasfit import ReLuFit
    net, seed = [50], 4
    # net, seed = [50, 50, 50, 50], 2
    # net, seed = [25, 25, 25, 25], 2

    with open("../../results/train_full_dataset/TASKS2D/final_weights/biaslearner_{}_seed_{}.pickle".format(net, seed),
              'rb') as file:
        ws, bs = pickle.load(file)
    plot_2d_tasks(saving=saving, model=ReLuFit(ws, bs, True, readout="tanh"), savename="bl_{}_2d".format(net), debug=debug)


def plot_asym(saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """

    datasets = ["EMNIST_bymerge"]
    nr_hiddens = [[25, 25], [25, 50], [25, 100], [25, 250], [25, 500]]
    # traintypes = ["train_sg_full", "polish_b_full", "train_bmr_full", "transfer_b_l1o_b_w", "transfer_bmr_l1o"]
    # legends = ["Singular approach", "Train weights + biases", "Train multireadout", "Transfer learn biases",
    #            "Transfer learn readout"]
    # colors = ["mediumpurple", "royalblue", "olivedrab", "cornflowerblue", "yellowgreen"]
    # load_dirs = ["{}results/train_full_dataset/".format(root_dir)] + 2 * ["{}results/train_full_dataset/".format(root_dir)] + 2 * ["{}results/leave_1_out/".format(root_dir)]
    # train_params = [get_biaslearner_training_params(highseed=3),
    #                 get_biaslearner_training_params(),
    #                 get_binarymr_training_params(),
    #                 get_biaslearner_training_params(highseed=3, transfering=True),
    #                 get_binarymr_training_params(highseed=3)]

    traintypes = ["train_b_w_full", "train_bmr_full"]
    legends = ["Task spec biases", "Task spec readouts"]
    colors = ["mediumpurple", "royalblue"]

    load_dirs = 2 * ["{}results/train_full_dataset/".format(root_dir)]
    train_params = [get_biaslearner_training_params(), get_binarymr_training_params()]

    title = "Expanding network performances"

    fig, axs = plt.subplots(len(datasets), 1, sharex='all', sharey='all')
    nr_bars = len(traintypes)
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(nr_hiddens))

    for row in range(len(datasets)):
        for tt in range(len(traintypes)):
            y = []
            y_err = []
            for nh in range(len(nr_hiddens)):
                # if tt in [2]:
                #     dataset = "EMNIST_bymerge_bw"
                # else:
                #     dataset = datasets[row]
                # performances = get_performances(prog_name=traintypes[tt], load_info=load_dirs[tt],
                #                                 nr_hidden=nr_hiddens[nh], dataset=dataset,
                #                                 train_params=train_params[tt], performance_type="test_performance")
                performances = get_performances(prog_name=traintypes[tt], load_info=load_dirs[tt],
                                                nr_hidden=nr_hiddens[nh], dataset=datasets[row],
                                                train_params=train_params[tt], performance_type="test_performance")
                y += [torch_mean(performances)]
                y_err += [torch_std(performances)]
                # if traintypes[tt] == "transfer_b_l1o_b_w":
                #     print(traintypes, nr_hiddens[nh])
                #     print(performances)
                #     print(y_err)
            axs.bar(x + x_offset[tt], y, width, yerr=y_err, label=legends[tt], zorder=3, color=colors[tt])
        axs.set_xticks(x)
        axs.grid(axis='y', alpha=0.4, zorder=0)
        axs.set_ylim([75, 100])
        axs.set_ylabel("Test Performance [%]")
    horiz_center = 0.44
    fig.text(horiz_center, 0.015, '# Hidden Neurons', ha='center')
    if titling:
        fig.text(horiz_center, 0.94, title, fontsize=14, ha='center')
    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    axs.set_xticklabels(nr_hiddens)
    fig.subplots_adjust(left=0.06, right=0.83, top=0.91, bottom=0.06, hspace=0.26)
    fig.set_size_inches(13, 10)
    if saving:
        plt.savefig("{}plots/train_full_dataset/svg/multitask_all.svg".format(root_dir))
        plt.savefig("{}plots/train_full_dataset/multitask_all.png".format(root_dir))
        plt.close()
    else:
        plt.show()


def plot_dataset_examples(dataset="CIFAR100", nrrows=9, nrcols=4, seed=0):
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor, Compose
    from train_helper import all_seeds

    all_seeds(seed)
    if dataset == "CIFAR100":
        from torchvision.datasets import CIFAR100
        data = CIFAR100("../../biasadaptation/utils/data/", train=True, transform=ToTensor(), download=True)
    elif dataset == "EMNIST_bymerge":
        from torchvision.datasets import EMNIST
        from torchvision.transforms.functional import hflip, rotate
        data = EMNIST("../../biasadaptation/utils/data/", "bymerge", train=True, transform=Compose([
            lambda img: rotate(img, -90), lambda img: hflip(img), ToTensor()]), download=True)
    else:
        raise NotImplementedError(dataset)
    dl = DataLoader(data, nrrows * nrcols, shuffle=True)
    images, _ = next(iter(dl))
    fig, axs = plt.subplots(nrrows, nrcols, figsize=(nrcols * 2, nrrows * 2))
    for i in range(nrrows):
        for j in range(nrcols):
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
            if dataset == "CIFAR100":
                axs[i, j].imshow(images[i * nrcols + j].permute(1, 2, 0))
            else:
                from torch import squeeze
                axs[i, j].imshow(squeeze(images[i * nrcols + j]), cmap="magma")
    plt.tight_layout()
    fig.subplots_adjust(left=0., right=1., wspace=0.06, hspace=0.06, top=1., bottom=0.)
    # plt.show()
    plt.savefig("../../plots/presentation/{}_examples.svg".format(dataset))
    plt.close()


def plot_dataset_examples_spec(dataset="CIFAR100", nrrows=4, nrcols=4, task=0):
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor, Compose
    from train_helper import all_seeds

    all_seeds(task)
    if dataset == "CIFAR100":
        from torchvision.datasets import CIFAR100
        data = CIFAR100("../../biasadaptation/utils/data/", train=True, transform=ToTensor(), download=True)
    elif dataset == "EMNIST_bymerge":
        from torchvision.datasets import EMNIST
        from torchvision.transforms.functional import hflip, rotate
        data = EMNIST("../../biasadaptation/utils/data/", "bymerge", train=True, transform=Compose([
            lambda img: rotate(img, -90), lambda img: hflip(img), ToTensor()]), download=True)
    else:
        raise NotImplementedError(dataset)
    nrimgs = nrrows * nrcols
    dl = DataLoader(data, nrimgs * 200, shuffle=True)
    images, labels = next(iter(dl))
    ids1 = []
    ids2 = []
    for l in range(len(labels)):
        if labels[l] == task:
            ids1 += [l]
            if len(ids1) == nrimgs:
                break
        elif len(ids2) != nrimgs:
            ids2 += [l]

    fig1, axs1 = plt.subplots(nrrows, nrcols, figsize=(nrcols, nrrows))
    fig2, axs2 = plt.subplots(nrrows, nrcols, figsize=(nrcols, nrrows))
    for i in range(nrrows):
        for j in range(nrcols):
            axs1[i, j].set_yticks([])
            axs1[i, j].set_xticks([])
            axs2[i, j].set_yticks([])
            axs2[i, j].set_xticks([])
            if dataset == "CIFAR100":
                axs1[i, j].imshow(images[ids1[i * nrcols + j]].permute(1, 2, 0))
                axs2[i, j].imshow(images[ids2[i * nrcols + j]].permute(1, 2, 0))
            else:
                from torch import squeeze
                axs1[i, j].imshow(squeeze(images[ids1[i * nrcols + j]]), cmap="magma")
                axs2[i, j].imshow(squeeze(images[ids2[i * nrcols + j]]), cmap="magma")
    plt.tight_layout()
    fig1.subplots_adjust(left=0., right=1., wspace=0.06, hspace=0.06, top=1., bottom=0.)
    fig2.subplots_adjust(left=0., right=1., wspace=0.06, hspace=0.06, top=1., bottom=0.)
    fig1.savefig("../../plots/presentation/{}_examples_task_{}_1.svg".format(dataset, task))
    fig2.savefig("../../plots/presentation/{}_examples_task_{}_2.svg".format(dataset, task))
    plt.close()


def plot_dataset_examples_multiclass(dataset="CIFAR100", nrrows=9, seed=0):
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor, Compose
    from train_helper import all_seeds

    all_seeds(seed)
    if dataset == "CIFAR100":
        from torchvision.datasets import CIFAR100
        data = CIFAR100("../../biasadaptation/utils/data/", train=True, transform=ToTensor(), download=True)
    elif dataset == "EMNIST_bymerge":
        from torchvision.datasets import EMNIST
        from torchvision.transforms.functional import hflip, rotate
        data = EMNIST("../../biasadaptation/utils/data/", "bymerge", train=True, transform=Compose([
            lambda img: rotate(img, -90), lambda img: hflip(img), ToTensor()]), download=True)
    else:
        raise NotImplementedError(dataset)
    dl = DataLoader(data, nrrows * 200, shuffle=True)
    images, labels = next(iter(dl))
    ids = []
    targets = [10, 11, 8, 9]
    plotrows = [0, 2, 6, 8]
    targetids = [0, -66, 1, -66, -66, -66, 2, -66, 3]
    for t in targets:
        for l in range(len(labels)):
            if labels[l] == t:
                ids += [l]
                break

    fig1, axs1 = plt.subplots(nrrows, 1, figsize=(1, 9))
    for i in range(nrrows):
        axs1[i].set_yticks([])
        axs1[i].set_xticks([])
        if i in plotrows:
            if dataset == "CIFAR100":
                axs1[i].imshow(images[ids[i]].permute(1, 2, 0))
            else:
                from torch import squeeze
                axs1[i].imshow(squeeze(images[ids[targetids[i]]]), cmap="magma")
        else:
            axs1[i].spines['top'].set_visible(False)
            axs1[i].spines['right'].set_visible(False)
            axs1[i].spines['bottom'].set_visible(False)
            axs1[i].spines['left'].set_visible(False)
    plt.tight_layout()
    fig1.subplots_adjust(left=0., right=1., wspace=0.06, hspace=0.06, top=1., bottom=0.)
    # plt.show()
    fig1.savefig("../../plots/presentation/{}_examples_multiclass.svg".format(dataset))
    plt.close()


if __name__ == '__main__':
    # mnist_l1o_toy_plot(saving=False)
    # willem_deepened_toyplot(titling=True, saving=False)
    # willem_init_toyplot(titling=True, saving=True)
    # plot_trainbw_readout_comparison(titling=True, saving=False)
    # plot_trainbw_early_stopping_comparison(titling=True, saving=False)
    # plot_layer_comparison(titling=True, saving=False)
    # l1o_all_plot(titling=True, saving=False)
    # bw_vs_b_w_l1o(titling=True, saving=False)
    # l1o_asym_toyplot(titling=True, saving=False)
    # plot_train_bw_vs_b_w_comparison(titling=True, saving=False)
    # l1o_bymerge_plot(titling=True, saving=False)
    # plot_2d_tasks(saving=False)
    # plot_best_2d_lr_tasks2d("scan_train_blr_wlr")
    # plot_best_2d_lr_tasks2d("scan_train_bmr_lr")
    # plot_scan_lr_2d("train_b_w", early_stopping=True)
    # plot_scan_lr_2d("train_bmr", early_stopping=True)
    # plot_2d_model_output(saving=True, debug=False)
    # plot_asym()

    # for d in ["CIFAR100", "EMNIST_bymerge"]:
    #     plot_dataset_examples(d)
    # plot_dataset_examples("EMNIST_bymerge")
    # plot_dataset_examples_spec("EMNIST_bymerge", task=9)
    # plot_dataset_examples_multiclass("EMNIST_bymerge")

    plt.style.use("seaborn-colorblind")
    for i in range(10):
        p = plt.plot([0, 1, 2, 3], 4 * [i], linewidth=9)
        print(p[0].get_color())

    plt.show()
