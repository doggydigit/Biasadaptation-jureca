import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from re import search
from os import listdir
from pathlib import Path
from matplotlib import rc as pltrc
from numpy import arange as np_arange
from torch import mean as torch_mean
from torch import std as torch_std
from matplotlib.ticker import LogLocator, NullFormatter
from tasks_2d_helper import task_2d_label
from plot_helper import get_dirs, get_performances, get_hidden_layer_info, plot_scan_lr
from biaslearning_helper import get_biaslearner_training_params, get_best_params
from multireadout_helper import get_binarymr_training_params


def plot_2d_tasks(saving=True, model=None, root_dir="../../", savename="Tasks48", nrow=4, nrxdim=100):
    nr_tasks = 48
    tasks = list(range(nr_tasks))
    datamin, datamax, inputsize = 0., 1., 2
    aa = torch.linspace(datamin, datamax, nrxdim)
    input_x, input_y = torch.meshgrid(aa, aa)
    input_x = torch.reshape(input_x, (nrxdim * nrxdim,))
    input_y = torch.reshape(input_y, (nrxdim * nrxdim,))
    input_data = torch.stack((input_x, input_y), 1)
    nrrows = min(nrow, nr_tasks)
    nrcols = 1 + int((nr_tasks - 1) / nrrows)
    single_col = nrcols < 1.1
    figwidth = 3 * nrcols
    fig, axes = plt.subplots(nrows=min(nrrows, nr_tasks), ncols=nrcols, figsize=(figwidth, 10))
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
            errors = None
        r = [not i for i in g]
        if single_col:
            a = axes[task]
        else:
            a = axes[task % nrrows][int(task / nrrows)]
        a.plot(input_data[g, 0], input_data[g, 1], 's', color='orange', markersize=0.9)
        a.plot(input_data[r, 0], input_data[r, 1], 's', color="deepskyblue",  markersize=0.9)
        if model:
            a.plot(input_data[errors, 0], input_data[errors, 1], 's', color="k", markersize=0.1)
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.xaxis.set_ticks_position('none')
        a.yaxis.set_ticks_position('none')
        a.axis('off')
    for ax in range(nr_tasks, nrrows * nrcols):
        if single_col:
            a = axes[ax]
        else:
            a = axes[ax % nrrows][int(ax / nrrows)]
        a.axis('off')
    if nrow == 4:
        fig.set_size_inches(18, 6)
    elif nrow == 3:
        fig.set_size_inches(16, 3)
    elif nrow == 6:
        fig.set_size_inches(12, 9)
    else:
        raise ValueError(nrow)
    fig.subplots_adjust(left=0., right=1., wspace=0., hspace=0., top=1., bottom=0.)
    if saving:
        plt.savefig("{}plots/toyplots/{}.svg".format(root_dir, savename))
        plt.savefig("{}plots/toyplots/{}.png".format(root_dir, savename))
        plt.close()
    else:
        plt.show()


def plot_2d_scan(prog_name="scan_train_blr_wlr", dataset="EMNIST_bymerge", saving=False, debug=False):
    """
    Plot best weight and bias learning rates for training biaslearner.
    Parameters
    ----------
    debug:
    prog_name:
    dataset:
    saving: Whether to save the plot to file or just show it
    """

    # Process the types of network architectures to plot, to know the number of subplots to draw
    if dataset == "TASKS2D":
        nr_hiddens = [[25], [50], [100], [25, 25], [50, 50], [100, 100], [25, 25, 25], [50, 50, 50], [100, 100, 100],
                      [25, 25, 25, 25], [50, 50, 50, 50], [100, 100, 100, 100]]
    else:
        nr_hiddens = [[25], [100], [500], [25, 25], [100, 100], [500, 500], [25, 25, 25], [100, 100, 100],
                      [500, 500, 500]]
    # nr_hiddens = [[25], [100], [500], [25, 25], [100, 100], [500, 500], [25, 25, 25], [500, 500, 500]]
    param_names = {"scan_train_blr_wlr": "Bias", "scan_train_glr_bwlr": "Gain",
                   "scan_train_glr_xwlr": "Gain (with xshift)", "scan_train_bglr_wlr": "Bias+Gain",
                   "scan_train_bmr_lr": "Readout"}
    train_names = {"scan_train_blr_wlr": "train_b_w", "scan_train_glr_bwlr": "train_g_bw",
                   "scan_train_glr_xwlr": "train_g_xw",
                   "scan_train_bglr_wlr": "train_bg_w", "scan_train_bmr_lr": "train_binarymr"}
    l2, l1 = get_best_params(train_names[prog_name], dataset=dataset)
    allchoice = [l1, l2]
    title = "Task-specific {} Modulation for {}".format(param_names[prog_name], dataset)
    if prog_name in ["scan_train_blr_wlr", "scan_train_glr_bwlr", "scan_train_glr_xwlr", "scan_train_bglr_wlr"]:
        subdir = prog_name
        # lrs1 = [0.1, 0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0001, 0.00003]
        # lrs2 = [0.5, 0.3, 0.2, 0.1, 0.06, 0.03, 0.01, 0.003, 0.001, 0.0003]
        # lrs2 = [0.5, 0.3, 0.2, 0.1, 0.06, 0.03, 0.02, 0.01, 0.006, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0001,
        # 0.00003]
        # lrs1 = [0.1, 0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0002, 0.0001, 0.00006, 0.00003, 0.00002,
        # 0.00001]
        if (dataset in ["TASKS2D", "K49"]) or prog_name in ["scan_train_glr_xwlr"]:
            if dataset in ["TASKS2D"]:
                lrs1 = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
                lrs2 = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
            else:
                lrs1 = None
                lrs2 = None
                netstring = str([25, 25])
                files = listdir("../../results/" + prog_name + "/individual/")
                files = [f for f in files if netstring in f and dataset in f]
                if prog_name == "scan_train_blr_wlr":
                    l1, l2 = "blr", "wlr"
                elif prog_name in ["scan_train_glr_bwlr", "scan_train_glr_xwlr"]:
                    l1, l2 = "glr", "wlr"
                elif prog_name == "scan_train_bglr_wlr":
                    l1, l2 = "bglr", "wlr"
                elif prog_name == "bmr":
                    l1, l2 = "rlr", "lr"
                else:
                    raise ValueError(prog_name)
                lrparams = [[float(search("{}_(.*?).pickle".format(l1), f).group(1)),
                             float(search("{}_(.*?)_{}".format(l2, l1), f).group(1))] for f in files]
        else:
            lrs1 = [0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0002, 0.0001, 0.00006, 0.00003, 0.00002, 0.00001,
                    0.000006, 0.000003]
            lrs2 = [0.3, 0.2, 0.1, 0.06, 0.03, 0.02, 0.01, 0.006, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0001, 0.00003]
            lrparams = None
        print(lrparams)
        train_params = get_biaslearner_training_params(highseed=20)
        lr1_name = "lr"
        lr2_name = "b_lr"
        xlab = "Weight learning rate"
        ylab = "Context learning rate"
        # choice = [0.0006, 0.06]
        majoryticks = (0.0001, 0.001, 0.01, 0.1)
        majoryticklabels = ("$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$")
        majorxticks = (0.00001, 0.0001, 0.001, 0.01)
        majorxticklabels = ("$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$")
        # if prog_name == "scan_train_blr_wlr":
        #     if dataset == "EMNIST_bymerge":
        #         allchoice = [0.0006, 0.01]
        #     elif dataset == "CIFAR100":
        #         allchoice = [0.0002, 0.03]
        #     else:
        #         raise ValueError(dataset)
        # elif prog_name == "scan_train_glr_bwlr":
        #     if dataset == "EMNIST_bymerge":
        #         allchoice = [0.0001, 0.01]
        #     elif dataset == "CIFAR100":
        #         allchoice = [0.0003, 0.03]
        #     else:
        #         raise ValueError(dataset)
        # elif prog_name == "scan_train_bglr_wlr":
        #     if dataset == "EMNIST_bymerge":
        #         allchoice = [0.0003, 0.003]
        #     elif dataset == "CIFAR100":
        #         allchoice = [0.0001, 0.01]
        #     else:
        #         raise ValueError(dataset)
        # else:
        #     raise ValueError(prog_name)
    elif prog_name == "scan_train_bmr_lr":
        subdir = "scan_train_bmr_lr"
        l1, l2 = "rlr", "lr"
        if dataset in ["TASKS2D", "K49"] or prog_name in ["scan_train_glr_xwlr"]:
            lrs1 = None
            lrs2 = None
            netstring = str([25, 25])
            files = listdir("../../results/" + prog_name + "/individual/")
            files = [f for f in files if netstring in f and dataset in f]
            lrparams = [[float(search("{}_(.*?).pickle".format(l1), f).group(1)),
                         float(search("{}_(.*?)_{}".format(l2, l1), f).group(1))] for f in files]
        else:
            # lrs1 = [0.1, 0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0001, 0.00003]
            # lrs2 = [0.1, 0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0001, 0.00003]
            lrs1 = [0.1, 0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0002, 0.0001, 0.00006, 0.00003]
            lrs2 = [0.1, 0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0002, 0.0001, 0.00003]
        train_params = get_binarymr_training_params(highseed=20)
        lr1_name = "lr"
        lr2_name = "r_lr"
        xlab = "Hidden neuron learning rate"
        ylab = "Readout neuron learning rate"
        # if dataset == "EMNIST_bymerge":
        #     allchoice = [0.0002, 0.001]
        # elif dataset == "CIFAR100":
        #     allchoice = [0.0002, 0.0006]
        # else:
        #     raise ValueError(dataset)
        majoryticks = (0.0001, 0.001, 0.01, 0.1)
        majoryticklabels = ("$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "")
        majorxticks = (0.0001, 0.001, 0.01, 0.1)
        majorxticklabels = ("$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "")

    else:
        raise ValueError(prog_name)
    with open("../../results/{}/bestparams.pickle".format(prog_name), 'rb') as file:
        bestparams = pickle.load(file)[dataset]
    hidden_layer_nrs, hidden_neuron_per_layer_nrs = get_hidden_layer_info(nr_hiddens=nr_hiddens)
    fig, axs = plt.subplots(len(hidden_layer_nrs), len(hidden_neuron_per_layer_nrs))
    row_ids = []
    col_ids = []
    for nr_hidden in nr_hiddens:
        row_ids += [hidden_layer_nrs.index(len(nr_hidden))]
        col_ids += [hidden_neuron_per_layer_nrs.index(nr_hidden[0])]
    if debug:
        v = np.linspace(50., 100., 51, endpoint=True)
    else:
        v = np.linspace(50., 100., 201, endpoint=True)
    load_dir, save_dir = get_dirs("../../", subdir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    locmin = LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=5)

    cntr = None
    for nh in range(len(nr_hiddens)):
        row = row_ids[nh]
        col = col_ids[nh]
        ys = []
        lrs1s = []
        lrs2s = []
        valperfs = []
        if dataset in ["K49"] or prog_name in ["scan_train_glr_xwlr"]:
            for lrs in lrparams:
                lrs1s += [lrs[1]]
                lrs2s += [lrs[0]]
                train_params[lr1_name] = lrs[1]
                train_params[lr2_name] = lrs[0]
                performances = get_performances(prog_name=prog_name, load_info=load_dir, nr_hidden=nr_hiddens[nh],
                                                dataset=dataset, train_params=train_params,
                                                performance_type="validation_performance")
                valperfs += [max(torch_mean(performances).item(), 50.)]
                ys += [torch_mean(performances)]
        else:
            for lr1 in lrs1:
                train_params[lr1_name] = lr1
                for lr2 in lrs2:
                    lrs1s += [lr1]
                    lrs2s += [lr2]
                    train_params[lr2_name] = lr2
                    performances = get_performances(prog_name=prog_name, load_info=load_dir, nr_hidden=nr_hiddens[nh],
                                                    dataset=dataset, train_params=train_params,
                                                    performance_type="validation_performance")
                    # The max 50. is just for clipping purposes in the tricontourf plot (some mean perfs are 49%)
                    valperfs += [max(torch_mean(performances).item(), 50.)]
                    ys += [torch_mean(performances)]
        z = np.array(valperfs)
        cntr = axs[row, col].tricontourf(lrs1s, lrs2s, z, v, cmap="RdBu_r", extend="both")
        axs[row, col].plot(lrs1s, lrs2s, 'ko', ms=1)
        choice = bestparams[str(nr_hiddens[nh])]
        if debug:
            axs[row, col].plot(choice[1], choice[0], 'kx', markersize=8, label=dataset)
        axs[row, col].plot(allchoice[0], allchoice[1], 'k+', markersize=8, label=dataset)
        axs[row, col].set_xscale('log')
        axs[row, col].set_yscale('log')
        axs[row, col].set_title("{}".format(nr_hiddens[nh]))
        axs[row, col].set_xticks(majorxticks)
        axs[row, col].xaxis.set_minor_locator(locmin)
        axs[row, col].xaxis.set_minor_formatter(NullFormatter())
        axs[row, col].set_yticks(majoryticks)
        axs[row, col].yaxis.set_minor_locator(locmin)
        axs[row, col].yaxis.set_minor_formatter(NullFormatter())

        # axs[row, col].tick_params(axis='both', which='minor')
        # axs[row, col].set_xticks([0.00001, 0.0001, 0.001, 0.01, 0.1])
    cbar_ax = fig.add_axes([0.91, 0.1, 0.01, 0.227])
    cbar = fig.colorbar(cntr, cax=cbar_ax)
    cbar.set_ticks([50, 60, 70, 80, 90, 100])
    cbar.set_ticklabels(["50%", "60%", "70%", "80%", "90%", "100%"])

    # Plotting Specifics
    axs[len(hidden_layer_nrs) - 1, 1].set_xlabel(xlab)
    axs[1, 0].set_ylabel(ylab)
    for col in range(len(hidden_neuron_per_layer_nrs)):
        axs[-1, col].xaxis.set_ticklabels(majorxticklabels)
        for row in range(len(hidden_layer_nrs) - 1):
            axs[row, col].set_xticks([])
            axs[row, col].set_xticks([], minor=True)
    for row in range(len(hidden_layer_nrs)):
        axs[row, 0].yaxis.set_ticklabels(majoryticklabels)
        for col in range(1, len(hidden_neuron_per_layer_nrs)):
            axs[row, col].set_yticks([])
            axs[row, col].set_yticks([], minor=True)
    # fig.set_size_inches(9, 9)
    # fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, wspace=0.15, hspace=0.26)
    fig.set_size_inches(6, 6)
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.13, right=0.90, wspace=0.15, hspace=0.26)
    plt.suptitle(title)

    if saving:
        plt.savefig("../../plots/scan_params/svg/{}_{}_performance_plot.svg".format(prog_name, dataset))
        plt.savefig("../../plots/scan_params/{}_{}_performance_plot.png".format(prog_name, dataset))
        plt.close()
    else:
        plt.show()


def panel_s1a(saving=True, nrow=4, nrxdim=100):
    plot_2d_tasks(saving=saving, model=None, root_dir="../../", savename="Tasks48", nrow=nrow, nrxdim=nrxdim)


def panel_s1b(saving=True):
    from biasadaptation.biasfit.biasfit import ReLuFit
    net = [50]
    seed = 4
    with open("../../results/train_full_dataset/TASKS2D/final_weights/biaslearner_{}_seed_{}.pickle".format(net, seed),
              'rb') as file:
        ws, bs = pickle.load(file)
    plot_2d_tasks(saving=saving, model=ReLuFit(ws, bs, True, readout="tanh"), savename="bl_{}_2d".format(net))


def panel_s1c(saving=True):
    from biasadaptation.biasfit.biasfit import ReLuFit
    net = [50, 50, 50, 50]
    seed = 2
    with open("../../results/train_full_dataset/TASKS2D/final_weights/biaslearner_{}_seed_{}.pickle".format(net, seed),
              'rb') as file:
        ws, bs = pickle.load(file)
    plot_2d_tasks(saving=saving, model=ReLuFit(ws, bs, True, readout="tanh"), savename="bl_{}_2d".format(net))


def panel_s1d(saving=True):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    -------
    """
    root_dir = "../../"
    dataset = "TASKS2D"
    nr_hiddens = [[25], [50], [100]]
    nr_layers = [1, 2, 3, 4]
    traintype = "train_b_w_full"
    legends = ["{} hidden layers".format(n) for n in nr_layers]
    colors = ["mediumpurple", "royalblue", "olivedrab", "goldenrod"]

    load_dir = "{}results/train_full_dataset/".format(root_dir)
    train_params = get_biaslearner_training_params()

    fig, axs = plt.subplots(1, 1, sharex='all', sharey='all')
    pltrc({'size': 44})
    nr_bars = len(nr_layers)
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(nr_hiddens))

    for row in range(1):
        for tt in range(len(nr_layers)):
            y = []
            y_err = []
            for nh in range(len(nr_hiddens)):
                performances = get_performances(prog_name=traintype, load_info=load_dir,
                                                nr_hidden=nr_layers[tt] * nr_hiddens[nh], dataset=dataset,
                                                train_params=train_params, performance_type="test_performance")
                y += [torch_mean(performances)]
                y_err += [torch_std(performances)]
            axs.bar(x + x_offset[tt], y, width, yerr=y_err, label=legends[tt], zorder=3, color=colors[tt])
        axs.set_xticks(x)
        axs.grid(axis='y', alpha=0.4, zorder=0)
        axs.set_ylim([75, 100])
        axs.set_ylabel("Test Performance [%]")
    horiz_center = 0.43
    fig.text(horiz_center, 0.02, 'Neurons in hidden layers', ha='center')
    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    axs.set_xticklabels([nh[0] for nh in nr_hiddens])
    axs.set_yticks([75, 80, 85, 90, 95, 100])
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.78, top=0.94, bottom=0.21, hspace=0.26)
    fig.set_size_inches(8, 2.3)
    if saving:
        plt.savefig("{}plots/train_full_dataset/svg/multitask_all.svg".format(root_dir))
        plt.savefig("{}plots/train_full_dataset/multitask_all.png".format(root_dir))
        plt.close()
    else:
        plt.show()


def panel_s2a(saving=True):
    plot_2d_scan("scan_train_blr_wlr", saving=saving)


def panel_s2b(saving=True):
    plot_2d_scan("scan_train_bmr_lr", saving=saving)


def panel_s2c(saving=False):
    """
    Plot validation performances for different learning rates, different datasets, different network architectures and
    possibly different training stopping mechanisms.
    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    """
    nr_hiddens = [[25], [100], [500],
                  [25, 25], [100, 100], [500, 500],
                  [25, 25, 25], [100, 100, 100], [500, 500, 500]]
    lrs = [0.1, 0.03, 0.02, 0.01, 0.006, 0.003, 0.002, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
    hidden_layer_nrs, hidden_neuron_per_layer_nrs = get_hidden_layer_info(nr_hiddens=nr_hiddens)
    fig, axs = plt.subplots(len(hidden_layer_nrs), len(hidden_neuron_per_layer_nrs))
    row_ids = []
    col_ids = []
    for nr_hidden in nr_hiddens:
        row_ids += [hidden_layer_nrs.index(len(nr_hidden))]
        col_ids += [hidden_neuron_per_layer_nrs.index(nr_hidden[0])]
    train_params = get_biaslearner_training_params(highseed=20)
    load_dir, _ = get_dirs("../../", "scan_train_sg_lr")
    for nh in range(len(nr_hiddens)):
        row = row_ids[nh]
        col = col_ids[nh]
        ys = []
        errs = []
        for lr in lrs:
            train_params["lr"] = lr
            performances = get_performances(prog_name="scan_train_sg_lr", load_info=load_dir,
                                            nr_hidden=nr_hiddens[nh],
                                            dataset="EMNIST_bymerge", train_params=train_params,
                                            performance_type="validation_performance")
            ys += [torch_mean(performances)]
            errs += [torch_std(performances)]
        axs[row, col].errorbar(lrs, ys, errs)
        axs[row, col].plot(float(lrs[5]), ys[5], "xk")
        axs[row, col].set_title("{}".format(nr_hiddens[nh]))
        axs[row, col].set_xscale('log')
        axs[row, col].set_ylim([45., 105.])
        axs[row, col].grid(which="both")
        axs[row, col].set_xticks([0.00001, 0.0001, 0.001, 0.01, 0.1])
        axs[row, col].set_yticks([50, 60, 70, 80, 90, 100])

    # Plotting Specifics
    for col in range(len(hidden_neuron_per_layer_nrs)):
        axs[-1, col].xaxis.set_ticklabels(["$10^{-5}$", "", "$10^{-3}$", "", "$10^{-1}$"])
        for row in range(len(hidden_layer_nrs) - 1):
            axs[row, col].xaxis.set_ticklabels([])
    for row in range(len(hidden_layer_nrs)):
        axs[row, 0].yaxis.set_ticklabels(["", 60, "", 80, "", 100])
        for col in range(1, len(hidden_neuron_per_layer_nrs)):
            axs[row, col].yaxis.set_ticklabels([])
    axs[len(hidden_layer_nrs) - 1, 1].set_xlabel("Learning rate")
    axs[1, 0].set_ylabel('Validation Performance [%]')
    fig.set_size_inches(6, 4.8)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.88, wspace=0.20, hspace=0.37)
    plt.suptitle("Task-specific networks")

    if saving:
        plt.savefig("../../plots/scan_params/svg/scan_sg_lr.svg")
        plt.savefig("../../plots/scan_params/scan_sg_lr.png")
        plt.close()
    else:
        plt.show()


# def track_plot(dataset="EMNIST_bymerge", batchres=False, saving=False):
#     # fig, axs = plt.subplots(3, 1, sharex=True)
#     # plot_types = ["train_loss", "valid_perf", "datadif_weight_span"]
#     if batchres:
#         loaddir = "../../results/trackepoch/"
#         plot_types = ["train_loss", "datadif_weight_span"]
#         if dataset == "CIFAR100":  # 396 batches per epoch
#             batchresx = list(range(10)) + list(range(10, 100, 10)) + [150, 200]
#             batch_in_epochs = [float(b) / 396. for b in batchresx]
#         elif dataset == "K49":  # 2000
#             batchresx = list(range(10)) + list(range(10, 100, 10)) + list(range(100, 1000, 100))
#             batch_in_epochs = [float(b) / 2000. for b in batchresx]
#         elif dataset == "EMNIST_bymerge":  # 6486
#             batchresx = list(range(10)) + list(range(10, 100, 10)) + list(range(100, 1000, 100)) \
#                         + [1000, 1500, 2000, 2500, 3000]
#             batch_in_epochs = [float(b) / 6486. for b in batchresx]
#         else:
#             raise ValueError(dataset)
#     else:
#         loaddir = "../../results/tracktraining/"
#         plot_types = ["train_loss", "valid_perf", "datadif_weight_span"]
#         batchresx = None
#     fig, axs = plt.subplots(len(plot_types), 1, sharex="all")
#
#     colors = ["r", "b"]
#     meancolors = ["darkred", "darkblue"]
#     # forl = [1, 1]
#     nrseeds = [3, 25]
#     dsnrc = {"EMNIST_bymerge": 47, "CIFAR100": 100, "K49": 49}
#     nrclasses = [dsnrc[dataset], 1]
#     for k in range(2):
#         if batchres:
#             all_loss = np.zeros((nrclasses[k]*nrseeds[k], len(batchresx)-1))
#             all_l2 = np.zeros((nrclasses[k] * nrseeds[k], len(batchresx)))
#         else:
#             all_l2 = None
#             all_loss = None
#         for seed in range(nrseeds[k]):
#             for t in range(nrclasses[k]):
#                 if k == 1:
#                     result_name = "{}biaslearner_[100]_{}_seed_{}.pickle".format(loaddir, dataset, seed)
#                 else:
#                     result_name = "{}singular_{}_[100]_{}_seed_{}.pickle".format(loaddir, t, dataset, seed)
#                 with open(result_name, "rb") as f:
#                     results = pickle.load(f)
#                 for i in range(len(plot_types)):
#                     plot_type = plot_types[i]
#                     y = results[plot_type]
#                     if plot_type == "train_loss":
#                         if batchres:
#                             x = batchresx[1:]
#                             all_loss[t*nrseeds[k]+seed, :] = y
#                         else:
#                             x = list(range(1, len(y) + 1))
#                             y = [sum(loss) for loss in y]
#                     elif plot_type == "valid_perf":
#                         if batchres:
#                             x = batchresx[1:]
#                         else:
#                             x = list(range(1, len(y) + 1))
#                     else:
#                         if batchres:
#                             x = batchresx
#                             all_l2[t * nrseeds[k] + seed, :] = y
#                         else:
#                             x = list(range(len(y)))
#                     axs[i].plot(x, y, colors[k], alpha=0.1)
#         if batchres:
#             axs[0].plot(batchresx[1:], np.mean(all_loss, 0), meancolors[k], linewidth=1.5)
#             axs[1].plot(batchresx, np.mean(all_l2, 0), meancolors[k], linewidth=1.5)
#
#     for i in range(len(plot_types)):
#         axs[i].set_ylabel(plot_types[i])
#         axs[i].set_xscale('symlog')
#
#     if batchres:
#         axs[len(axs)-1].plot([0., 3000.], [1455, 1455], "k--")
#         # axs[len(axs) - 1].set_ylim([1000, 3000])
#     else:
#         axs[len(axs)-1].plot([0., 50.], [1455, 1455], "k--")
#
#     if batchres:
#         axs[len(axs) - 1].set_xlabel("Batch")
#     else:
#         axs[len(axs) - 1].set_xlabel("Epoch")
#     if saving:
#         if batchres:
#             st = "batch"
#         else:
#             st = "epoch"
#         plt.savefig("../../plots/track_weight_diff/track_weight_diff_{}_{}.svg".format(dataset, st))
#         plt.close()
#     else:
#         plt.show()


def track_plot(dataset="EMNIST_bymerge", saving=False, debug=False):
    plot_types = ["train_loss", "datadif_weight_span"]
    fig, axs = plt.subplots(len(plot_types), 1, sharex="all")
    if dataset == "CIFAR100":  # 396 batches per epoch
        batchresx = list(range(10)) + list(range(10, 100, 10)) + [150, 200]
        nrbatches_per_epoch = 396.
    elif dataset == "K49":  # 2000
        batchresx = list(range(10)) + list(range(10, 100, 10)) + list(range(100, 1000, 100))
        nrbatches_per_epoch = 2000.
    elif dataset == "EMNIST_bymerge":  # 6486
        batchresx = list(range(10)) + list(range(10, 100, 10)) + list(range(100, 1000, 100)) \
                    + [1000, 1500, 2000, 2500, 3000]
        nrbatches_per_epoch = 6486.
    else:
        raise ValueError(dataset)
    batch_in_epochs = [float(b) / nrbatches_per_epoch for b in batchresx]
    nrspochsmax = 50

    # colors = ["r", "b"]
    # meancolors = ["darkred", "darkblue"]
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
                    batch_result_name = "../../results/trackepoch/biaslearner_[100]_{}_seed_{}.pickle".format(dataset, seed)
                    epoch_result_name = "../../results/tracktraining/biaslearner_[100]_{}_seed_{}.pickle".format(dataset, seed)
                else:
                    batch_result_name = "../../results/trackepoch/singular_{}_[100]_{}_seed_{}.pickle".format(t, dataset, seed)
                    epoch_result_name = "../../results/tracktraining/singular_{}_[100]_{}_seed_{}.pickle".format(t, dataset, seed)
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
                        y = [nrbatches_per_epoch * b for b in batch_result] + [sum(loss) for loss in epoch_result]
                        all_loss[t*nrseeds[k]+seed, :len(y)] = y
                    else:
                        x = batch_in_epochs + list(range(1, len(epoch_result)))
                        y = batch_result + epoch_result[1:]
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

    for i in range(len(plot_types)):
        axs[i].set_ylabel(plot_types[i])
        axs[i].set_xscale('symlog')
    axs[len(axs)-1].plot([0., 50.], [1455, 1455], "k--")
    axs[len(axs) - 1].set_xlabel("Epoch")
    if saving:
        plt.savefig("../../plots/track_weight_diff/track_weight_diff_{}.svg".format(dataset))
        plt.close()
    else:
        plt.show()


def panel_cosyne_1c(saving=True):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    -------
    """
    root_dir = "../../"
    dataset = "TASKS2D"
    nr_hiddens = 50
    nr_layers = [1, 2, 3, 4]
    traintype = "train_b_w_full"
    # legends = ["{} hidden layers".format(n) for n in nr_layers]

    load_dir = "{}results/train_full_dataset/".format(root_dir)
    train_params = get_biaslearner_training_params()

    fig, axs = plt.subplots(1, 1, sharex='all', sharey='all')
    pltrc({'size': 44})
    # nr_bars = len(nr_layers)
    # width = 1. / (nr_bars + 1)
    # x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = list(range(len(nr_layers)))
    for tt in x:
        y = []
        y_err = []
        performances = get_performances(prog_name=traintype, load_info=load_dir,
                                        nr_hidden=nr_layers[tt] * [nr_hiddens], dataset=dataset,
                                        train_params=train_params, performance_type="test_performance")
        y += [torch_mean(performances)]
        y_err += [torch_std(performances)]
        axs.bar(nr_layers[tt], y[0]-80, 0.9, yerr=y_err, bottom=80, zorder=3, color="grey")
        # print(y[0])
    axs.set_xticks(nr_layers)
    axs.grid(axis='y', alpha=0.4, zorder=0)
    axs.set_ylim([80, 100])
    axs.set_ylabel("Test Performance [%]")
    horiz_center = 0.62
    fig.text(horiz_center, 0.02, 'Number of hidden layers', ha='center')
    # handles, labels = axs.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center right')
    # axs.set_xticklabels([nh[0] for nh in nr_hiddens])
    axs.set_yticks([80, 85, 90, 95, 100])
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    fig.subplots_adjust(left=0.26, right=0.98, top=0.94, bottom=0.24, hspace=0.26)
    fig.set_size_inches(2.5, 2.3)
    if saving:
        plt.savefig("{}plots/train_full_dataset/svg/multitask_all.svg".format(root_dir))
        plt.savefig("{}plots/train_full_dataset/multitask_all.png".format(root_dir))
        plt.close()
    else:
        plt.show()


def panel_cosyne_1d(saving=True):
    from biasadaptation.biasfit.biasfit import ReLuFit
    net = [50, 50, 50, 50]
    seed = 2
    with open("../../results/train_full_dataset/TASKS2D/final_weights/biaslearner_{}_seed_{}.pickle"
              "".format(net, seed), 'rb') as file:
        ws, bs = pickle.load(file)
    plot_2d_tasks(saving=saving, model=ReLuFit(ws, bs, True, readout="tanh"), savename="bl_{}_2d".format(net), nrow=3)


if __name__ == '__main__':
    save = True
    plt.style.use("seaborn-colorblind")
    # panel_s1a(True, nrxdim=100)

    # panel_s1a(save)
    # panel_s1b(save)
    # panel_s1c(save)
    # panel_s1d(save)
    # panel_s2a(save)
    # panel_s2b(save)
    # panel_s2c(save)
    # track_plot(dataset="EMNIST_bymerge", saving=False, debug=True)
    # for ds in ["EMNIST_bymerge", "CIFAR100", "K49"]:
    #     track_plot(dataset=ds, saving=save)
    # panel_cosyne_1c(save)
    # panel_cosyne_1d(save)

    # 2D Scans
    debug = False
    for ds in ["EMNIST_bymerge", "CIFAR100"]:
    # for ds in ["K49"]:
        plot_2d_scan("scan_train_blr_wlr", dataset=ds, saving=save, debug=debug)
        plot_2d_scan("scan_train_glr_bwlr", dataset=ds, saving=save, debug=debug)
        plot_2d_scan("scan_train_bglr_wlr", dataset=ds, saving=save, debug=debug)
        plot_2d_scan("scan_train_bmr_lr", dataset=ds, saving=save, debug=debug)
    # plot_scan_lr("train_sg", early_stopping=True, titling=False, saving=save)

    # plot_2d_scan("scan_train_glr_bwlr", dataset="EMNIST_bymerge", saving=save, debug=debug)
    # plot_2d_scan("scan_train_bglr_wlr", dataset="CIFAR100", saving=save, debug=debug)
    # plot_2d_scan("scan_train_blr_wlr", dataset="TASKS2D", saving=save, debug=debug)
    # for ds in ["EMNIST_bymerge", "CIFAR100", "K49"]:
    #     plot_2d_scan("scan_train_glr_xwlr", dataset=ds, saving=save, debug=debug)
