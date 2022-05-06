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
    nr_hiddens = [[100], [100, 100], [100, 100, 100]]
    traintypes = ["train_b_w", "train_g_bw", "train_bg_w"]
    lcolors = ["#0072B2", "#009E73", "yellowgreen"]
    legends = ["Contextual Biases", "Contextual Gains", "Contextual Biases & Gains"]

    load_dir = "{}results/train_full_dataset/".format(root_dir)
    title = "Multitask learning with gains vs. biases"

    fig, axs = plt.subplots(1, 1, sharex='all')
    nr_bars = len(traintypes)
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(nr_hiddens))
    perflog = {0: [], 1: [], 2: []}
    prog_params = {"dataset": "EMNIST_bymerge"}
    for tt in range(len(traintypes)):
        prog_params["training_type"] = traintypes[tt]
        hs = 25
        y = []
        y_err = []
        for nh in range(len(nr_hiddens)):
            prog_params["nr_hidden"] = nr_hiddens[nh]
            train_params = get_biaslearner_training_params(prog_params=prog_params, highseed=hs)
            performances = get_performances(prog_name=traintypes[tt] + "_full", load_info=load_dir,
                                            nr_hidden=nr_hiddens[nh], dataset="EMNIST_bymerge",
                                            train_params=train_params, performance_type="test_performance")
            perflog[tt] += [performances]
            y += [torch_mean(performances)]
            y_err += [torch_std(performances)]
        axs.bar(x + x_offset[tt], y, width, yerr=y_err, label=legends[tt], zorder=3, color=lcolors[tt])
    axs.set_xticks(x)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.set_ylabel("Test Performance [%]")
    axs.set_ylim([94, 100])
    horiz_center = 0.44
    fig.text(0.57, 0.03, 'Hidden Layers', ha='center')
    # handles, labels = axs.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower left')
    plt.legend(loc="upper right")
    axs.set_xticklabels(["1", "2", "3"])
    fig.subplots_adjust(left=0.20, right=0.98, top=0.84, bottom=0.25, hspace=0.26)
    # fig.subplots_adjust(left=0.52, right=0.98, top=0.84, bottom=0.24, hspace=0.26)
    fig.set_size_inches(3, 2)
    if saving:
        plt.savefig("{}plots/train_full_dataset/svg/bias_vs_gain.svg".format(root_dir))
        plt.savefig("{}plots/train_full_dataset/bias_vs_gain.png".format(root_dir))
        plt.close()
    else:
        plt.show()


def transfer_plot(saving=False, root_dir="../../"):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    -------
    """

    datasets = ["EMNIST_bymerge"]
    nr_hiddens = [[250], [250, 250], [250, 250, 250]]
    traintypes = ["train_b_w", "train_b_w"]
    lcolors = ["#0072B2", "mediumorchid"]
    perf_traintypes = ["polish_b_full", "transfer_b_l1o_b_w"]
    legends = ["Mulit-task", "Transfer learning"]
    load_dirs = ["{}results/train_full_dataset/".format(root_dir), "{}results/leave_1_out/".format(root_dir)]

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
        axs.set_xticks(x)
        axs.set_yticks([90, 95, 100])
        axs.set_ylabel("test perf (%)")
    axs.set_ylim([90, 100])
    axs.tick_params(axis=u'both', which=u'both', length=0)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    fig.text(0.4, 0.015, 'no. hidden layers', ha='center')
    # handles, labels = axs.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower right')
    plt.legend(loc="lower center")
    axs.set_xticklabels([1, 2, 3])
    top = 0.93
    fig.subplots_adjust(left=0.3, right=0.81, top=top, bottom=0.16, hspace=0.26)
    fig.set_size_inches(2, 2)
    if saving:
        plt.savefig("{}plots/leave_1_out/svg/transfer.svg".format(root_dir))
        plt.savefig("{}plots/leave_1_out/transfer.png".format(root_dir))
        plt.close()
    else:
        plt.show()


def track_plot(saving=False, debug=False):
    dataset = "EMNIST_bymerge"
    plot_types = ["train_loss", "datadif_weight_span"]
    fig, axs = plt.subplots(len(plot_types), 1, sharex="all")
    batchresx = list(range(10)) + list(range(10, 100, 10)) + list(range(100, 1000, 100)) + [1000, 1500, 2000, 2500,
                                                                                            3000]
    nrbatches_per_epoch = 6486.
    batch_in_epochs = [float(b) / nrbatches_per_epoch for b in batchresx]
    nrspochsmax = 50

    colors = ["#0072B2", "#D55E00"]
    meancolors = ["#0072B2", "#D55E00"]
    if debug:
        nrseeds = [2, 2]
        dsnrc = {"EMNIST_bymerge": 2, "CIFAR100": 2, "K49": 2}
    else:
        nrseeds = [3, 25]
        dsnrc = {"EMNIST_bymerge": 47, "CIFAR100": 100, "K49": 49}
    nrclasses = [dsnrc[dataset], 1]
    for k in range(2):
        all_loss = np.zeros((nrclasses[k] * nrseeds[k], len(batchresx) - 1 + nrspochsmax))
        all_l2 = np.zeros((nrclasses[k] * nrseeds[k], len(batchresx) + nrspochsmax))
        all_loss[:, :] = np.nan
        all_l2[:, :] = np.nan
        for seed in range(nrseeds[k]):
            for t in range(nrclasses[k]):
                if k == 1:
                    batch_result_name = "../../results/trackepoch/biaslearner_[100]_{}_seed_{}.pickle" \
                                        "".format(dataset, seed)
                    epoch_result_name = "../../results/tracktraining/biaslearner_[100]_{}_seed_{}.pickle" \
                                        "".format(dataset, seed)
                else:
                    batch_result_name = "../../results/trackepoch/singular_{}_[100]_{}_seed_{}.pickle" \
                                        "".format(t, dataset, seed)
                    epoch_result_name = "../../results/tracktraining/singular_{}_[100]_{}_seed_{}.pickle" \
                                        "".format(t, dataset, seed)
                with open(batch_result_name, "rb") as f:
                    batch_results = pickle.load(f)
                with open(epoch_result_name, "rb") as f:
                    epoch_results = pickle.load(f)
                for i in range(len(plot_types)):
                    plot_type = plot_types[i]
                    batch_result = batch_results[plot_type]
                    epoch_result = epoch_results[plot_type]
                    if plot_type == "train_loss":
                        x = batch_in_epochs[1:] + list(range(1, len(epoch_result) + 1))
                        y = [nrbatches_per_epoch * b * 0.00001 for b in batch_result]
                        y += [sum(loss) * 0.00001 for loss in epoch_result]
                        all_loss[t * nrseeds[k] + seed, :len(y)] = y
                    else:
                        x = batch_in_epochs + list(range(1, len(epoch_result)))
                        y = batch_result + epoch_result[1:]
                        y = [yi * 0.01 for yi in y]
                        all_l2[t * nrseeds[k] + seed, :len(y)] = y
                    axs[i].plot(x, y, colors[k], alpha=0.1)
        for i in range(2):
            if i == 0:
                x = batch_in_epochs[1:]
                y = all_loss[:, ~np.all(np.isnan(all_loss), axis=0)]
            else:
                x = batch_in_epochs.copy()
                y = all_l2[:, ~np.all(np.isnan(all_l2), axis=0)]
            x += list(range(1, y.shape[1] - len(x) + 1))
            axs[i].plot(x, np.nanmean(y, 0), meancolors[k], linewidth=1.5)

    ylabels = ["train loss", "L2 ΔX-span"]
    for i in range(len(plot_types)):
        axs[i].set_ylabel(ylabels[i])
        axs[i].set_xscale('symlog')
        # axs[i].set_ylim([0., 3.])
    axs[len(axs) - 1].plot([0., 50.], [14.55, 14.55], "k--")
    axs[len(axs) - 1].set_xlabel("Epoch")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.24, hspace=0.26)
    fig.set_size_inches(5.5, 2.5)

    if saving:
        plt.savefig("../../plots/track_weight_diff/track_weight_diff_{}.svg".format(dataset))
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # bias_vs_mr_plot(saving=True, titling=False)
    # bias_vs_gain_plot(saving=True, titling=False)
    # transfer_plot(saving=False)
    track_plot(saving=True, debug=False)
