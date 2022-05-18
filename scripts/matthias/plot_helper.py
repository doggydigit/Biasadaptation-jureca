import matplotlib.pyplot as plt
from pathlib import Path
from pickle import load as pickle_load
from pickle import dump as pickle_dump
from pickle import HIGHEST_PROTOCOL
from numpy import zeros as npzeros
from numpy import mean as npmean
from numpy import reshape as npreshape
from numpy import argmax as npargmax
from numpy import arange as np_arange
from torch import tensor
from torch import mean as torch_mean
from torch import std as torch_std
from torch import stack as torch_stack
from torch import from_numpy as torch_from_numpy
from data_helper import get_number_classes
from biaslearning_helper import get_biaslearner_training_params
from multireadout_helper import get_multireadout_training_params, get_binarymr_training_params


def get_dirs(root_dir, spec_dir):
    load_dir = root_dir + "results/" + spec_dir + "/"
    save_dir = root_dir + "plots/" + spec_dir + "/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return load_dir, save_dir


def get_hidden_layer_info(nr_hiddens):
    hidden_layer_nrs = []
    hidden_neuron_per_layer_nrs = []
    for nr_hidden in nr_hiddens:
        if len(nr_hidden) not in hidden_layer_nrs:
            hidden_layer_nrs += [len(nr_hidden)]
        if nr_hidden[0] not in hidden_neuron_per_layer_nrs:
            hidden_neuron_per_layer_nrs += [nr_hidden[0]]
    hidden_layer_nrs.sort()
    hidden_neuron_per_layer_nrs.sort()
    return hidden_layer_nrs, hidden_neuron_per_layer_nrs


def get_performances(prog_name, load_info, nr_hidden, dataset, train_params, performance_type, model_type=None,
                     bestseed=False, task_average=True):
    if model_type is None:
        if prog_name in ["train_b_w_full", "train_b_w_l1o", "train_bw_full", "train_bw_l1o", "transfer_b_l1o_b_w",
                         "scan_transfer_b_lr", "transfer_b_l1o_bw"]:
            model_type = "biaslearner"
        elif prog_name in ["train_bg_w_full"]:
            model_type = "bglearner"
        elif prog_name in ["train_g_bw_full"]:
            model_type = "gainlearner"
        elif prog_name in ["train_mr_full", "train_mr_l1o", "transfer_mr_l1o", "scan_transfer_mr_lr"]:
            model_type = "multireadout"
        elif prog_name in ["train_bmr_full", "train_bmr_l1o", "scan_transfer_bmr_lr", "transfer_bmr_l1o",
                           "test_train_bmr_full"]:
            model_type = "binarymr"
        elif prog_name == "train_sg_full":
            model_type = "singular"
        elif prog_name == "transfer_b_scd_l1o":
            model_type = "scd"
        elif prog_name == "transfer_b_pmdd_l1o":
            model_type = "pmdd"
        elif prog_name == "transfer_b_scd_hardtanh_l1o":
            model_type = "scd_readout"
        elif prog_name == "transfer_b_pmdd_hardtanh_l1o":
            model_type = "pmdd_readout"
        elif prog_name == "polish_b_full":
            model_type = "biaslearner_polished"

    if prog_name in ["scan_train_bw_lr", "scan_train_mr_lr"]:
        if train_params["early_stopping"]:
            es = "_es"
        else:
            es = ""
        path = load_info + "individual/network_{}_{}_{}_{}{}_lr_{}.pickle".format(
            nr_hidden, dataset, train_params["loss_function"], train_params["readout_function"],
            es, train_params["lr"]
        )
        with open(path, "rb") as f:
            result = pickle_load(f)
            performances = result[performance_type]

    elif prog_name in ["scan_train_blr_wlr", "scan_train_glr_bwlr", "scan_train_bglr_wlr", "scan_train_glr_xwlr"]:
        if train_params["early_stopping"]:
            es = "_es"
        else:
            es = ""
        if prog_name == "scan_train_blr_wlr":
            pathbase = load_info + "individual/network_{}_{}_{}_{}{}_wlr_{}_blr_{}.pickle"
        elif prog_name in ["scan_train_glr_bwlr", "scan_train_glr_xwlr"]:
            pathbase = load_info + "individual/network_{}_{}_{}_{}{}_wlr_{}_glr_{}.pickle"
        elif prog_name == "scan_train_bglr_wlr":
            pathbase = load_info + "individual/network_{}_{}_{}_{}{}_wlr_{}_bglr_{}.pickle"
        else:
            raise None
        path = pathbase.format(nr_hidden, dataset, train_params["loss_function"], train_params["readout_function"], es,
                               train_params["lr"], train_params["b_lr"])
        with open(path, "rb") as f:
            result = pickle_load(f)
            performances = result[performance_type]

    elif prog_name == "scan_train_bmr_lr":
        if train_params["early_stopping"]:
            es = "_es"
        else:
            es = ""
        path = load_info + "individual/network_{}_{}_{}_{}{}_lr_{}_rlr_{}.pickle".format(
            nr_hidden, dataset, train_params["loss_function"], train_params["readout_function"],
            es, train_params["lr"], train_params["r_lr"]
        )
        with open(path, "rb") as f:
            result = pickle_load(f)
            performances = result[performance_type]

    elif prog_name in ["scan_transfer_b_lr", "scan_transfer_bmr_lr", "scan_transfer_mr_lr"]:
        performances = []
        for tc in range(get_number_classes(dataset)):
            path = load_info + "individual/{}_{}_lr_{}_{}_testclass_{}.pickle".format(
                model_type, dataset, train_params["lr"], nr_hidden, tc
            )
            try:
                with open(path, "rb") as f:
                    result = pickle_load(f)
                    performances += result[performance_type][:train_params["highseed"]]
            except FileNotFoundError:
                print(path)

    elif prog_name in ["scan_train_sg_lr"]:
        if train_params["early_stopping"]:
            es = "_es"
        else:
            es = ""
        performances = []
        for tc in range(get_number_classes(dataset)):
            path = load_info + "individual/network_{}_{}_{}_{}{}_lr_{}_testclass_{}.pickle".format(
                nr_hidden, dataset, train_params["loss_function"], train_params["readout_function"], es,
                train_params["lr"], tc
            )
            with open(path, "rb") as f:
                result = pickle_load(f)
                performances += result[performance_type][:train_params["highseed"]]

    elif prog_name in ["train_b_w_full", "train_g_bw_full", "train_bg_w_full", "train_bw_full", "train_mr_full", "train_bmr_full"]:
        load_path = "{}{}/{}_{}.pickle".format(load_info, dataset, model_type, nr_hidden)
        with open(load_path, "rb") as f:
            results = pickle_load(f)
            if bestseed:
                performances = [max(results[performance_type])]
            else:
                performances = results[performance_type]

    elif prog_name in ["transfer_bmr_full"]:
        load_path = "{}_{}.pickle".format(load_info, nr_hidden)
        with open(load_path, "rb") as f:
            results = pickle_load(f)
            if bestseed:
                performances = [max(results[performance_type])]
            else:
                performances = results[performance_type]

    elif prog_name == "train_sg_full":
        performances = []
        for tc in range(get_number_classes(dataset)):
            path = "{}{}/{}_{}_testclass_{}.pickle".format(load_info, dataset, model_type, nr_hidden, tc)
            with open(path, "rb") as f:
                result = pickle_load(f)
                if bestseed:
                    performances += [max(result[performance_type][:train_params["highseed"]])]
                else:
                    performances += result[performance_type][:train_params["highseed"]]

    elif prog_name in ["train_b_w_l1o", "train_bw_l1o", "train_mr_l1o", "train_bmr_l1o", "transfer_b_l1o_b_w",
                       "transfer_b_l1o_bw", "transfer_mr_l1o", "transfer_bmr_l1o", "transfer_b_scd_l1o",
                       "transfer_b_pmdd_l1o", "transfer_b_scd_hardtanh_l1o", "transfer_b_pmdd_hardtanh_l1o"]:
        if prog_name in ["train_bw_l1o", "train_b_w_l1o", "train_mr_l1o", "train_bmr_l1o"]:
            t_dir = "train"
        else:
            t_dir = "transfer"
        performances = []
        for tc in range(get_number_classes(dataset)):
            load_path = "{}{}/{}/{}_{}_testclass_{}.pickle".format(load_info, dataset, t_dir, model_type, nr_hidden, tc)
            with open(load_path, "rb") as f:
                results = pickle_load(f)
                if bestseed:
                    task_perfs = results[performance_type][:train_params["highseed"]]
                    performances += [max(task_perfs)]
                else:
                    performances += results[performance_type][:train_params["highseed"]]

    elif prog_name == "willem_deepened":
        performances = []
        for tc in range(get_number_classes(dataset)):
            load_path = "{}_{}_{}_testclass_{}.pickle".format(load_info, nr_hidden, dataset, tc)
            with open(load_path, "rb") as f:
                results = pickle_load(f)
                performances += results[performance_type]

    elif prog_name == "polish_b_full":
        performances = npzeros((25, get_number_classes(dataset)))
        for tc in range(get_number_classes(dataset)):
            load_path = "{}{}/{}_{}_task_{}.pickle".format(load_info, dataset, model_type, nr_hidden, tc)
            with open(load_path, "rb") as f:
                results = pickle_load(f)
                performances[:, tc] = results[performance_type]
        if task_average:
            performances = npmean(performances, axis=1)
        if bestseed:
            return torch_from_numpy(performances[npargmax(npmean(performances, axis=1)), :])
        else:
            return torch_from_numpy(npreshape(performances, -1))

    elif prog_name == "test_train_bmr_full":
        performances = npzeros((25, get_number_classes(dataset)))
        for s in range(25):
            load_path = "{}{}/{}_{}_seed_{}.pickle".format(load_info, dataset, model_type, nr_hidden, s)
            with open(load_path, "rb") as f:
                if task_average:
                    performances[s, :] = npmean(pickle_load(f))
                else:
                    performances[s, :] = pickle_load(f)

        if bestseed:
            return torch_from_numpy(performances[npargmax(npmean(performances, axis=1)), :])
        else:
            return torch_from_numpy(npreshape(performances, -1))

    else:
        raise ValueError(prog_name)

    return torch_stack(performances)


def plot_best_2d_lr(prog_name, saving=False, nr_hiddens=None, root_dir="../../", titling=True):
    """
    Plot best weight and bias learning rates for training biaslearner.
    Parameters
    ----------
    prog_name:
    saving: Whether to save the plot to file or just show it
    nr_hiddens: list of the number of hidden neurons in the hidden layers.
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    """

    # Process the types of network architectures to plot, to know the number of subplots to draw
    if nr_hiddens is None:
        nr_hiddens = [[10], [100], [500], [10, 10], [100, 100], [500, 500]]

    if prog_name == "scan_train_blr_wlr":
        subdir = "scan_train_blr_wlr"
        if True:
            nr_hiddens = [[10], [100], [500], [10, 10], [100, 100], [500, 500], [10, 10, 10], [100, 100, 100], [500, 500, 500]]
            lrs1 = [0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0001]
            lrs2 = [0.5, 0.3, 0.2, 0.1, 0.06, 0.03, 0.01, 0.003, 0.001]
        else:
            lrs1 = [0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0001, 0.00003]
            lrs2 = [0.5, 0.3, 0.2, 0.1, 0.06, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]
        train_params = get_biaslearner_training_params(highseed=20)
        min_lr = 0.00001
        lr1 = "lr"
        lr2 = "b_lr"
        xlab = "weight learning rate"
        ylab = "bias learning rate"
    elif prog_name == "scan_train_bmr_lr":
        subdir = "scan_train_bmr_lr"
        lrs1 = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
        lrs2 = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]
        train_params = get_binarymr_training_params(highseed=20)
        min_lr = 0.000001
        lr1 = "lr"
        lr2 = "r_lr"
        xlab = "deep learning rate"
        ylab = "readout learning rate"
    else:
        raise ValueError(prog_name)

    hidden_layer_nrs, hidden_neuron_per_layer_nrs = get_hidden_layer_info(nr_hiddens=nr_hiddens)
    fig, axs = plt.subplots(len(hidden_layer_nrs), len(hidden_neuron_per_layer_nrs))
    row_ids = []
    col_ids = []
    for nr_hidden in nr_hiddens:
        row_ids += [hidden_layer_nrs.index(len(nr_hidden))]
        col_ids += [hidden_neuron_per_layer_nrs.index(nr_hidden[0])]

    datasets = ["QMNIST", "KMNIST", "EMNIST_bymerge", "EMNIST_letters"]
    markers = {"QMNIST": "1", "KMNIST": "2", "EMNIST_bymerge": "3", "EMNIST_letters": "4"}
    load_dir, save_dir = get_dirs(root_dir, subdir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

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
            axs[row, col].plot(float(lrs[m][0]), float(lrs[m][1]), markers[dataset], markersize=8, label=dataset)
            axs[row, col].plot([min_lr, 1.], [min_lr, 1.], "k:", linewidth=0.7)

            axs[row, col].set_title("Network {}".format(nr_hiddens[nh]))
            axs[row, col].set_xscale('log')
            axs[row, col].set_yscale('log')
            axs[row, col].set_xlim([min_lr, 1.])
            axs[row, col].set_ylim([min_lr, 1.])
            axs[row, col].grid(True, which="both")

    # Plotting Specifics
    for col in range(len(hidden_neuron_per_layer_nrs)):
        axs[len(hidden_layer_nrs)-1, col].set_xlabel(xlab)
    for row in range(len(hidden_layer_nrs)):
        axs[row, 0].set_ylabel(ylab)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    fig.set_size_inches(16, 9)
    fig.subplots_adjust(left=0.05, right=0.875, wspace=0.15, hspace=0.26)

    if titling:
        plt.suptitle("Scanning separate bias and weight learning rates")
    if saving:
        plt.savefig(root_dir + "plots/scan_params/svg/{}_performance_plot.svg".format(prog_name))
        plt.savefig(root_dir + "plots/scan_params/{}_performance_plot.png".format(prog_name))
        plt.close()
    else:
        plt.show()


def plot_scan_lr(train_type, train_params=None, saving=False, datasets=None, nr_hiddens=None, lrs=None,
                 early_stopping=None, root_dir="../../", titling=False, nrhd100=True):
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

    # Process the types of network architectures to plot, to know the number of subplots to draw
    if nr_hiddens is None:
        if nrhd100:
            nr_hiddens = nr_hiddens = [[100], [100, 100], [100, 100, 100]]
        else:
            nr_hiddens = [[25], [100], [500], [25, 25], [100, 100], [500, 500], [25, 25, 25], [100, 100, 100],
                          [500, 500, 500]]
    # Some argument dependent default initializations
    if lrs is None:
        if train_type in ["train_bw", "train_bw_hardtanh", "train_mr", "train_sg_bigbatch"]:
            lrs = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
        # elif train_type in ["transfer_mr_l1o"]:
        #     lrs = [0.9, 0.6, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
        elif train_type in ["transfer_b_l1o", "transfer_bmr_l1o", "transfer_mr_l1o"]:
            nr_hiddens = [[25], [100], [500], [25, 25], [100, 100], [500, 500], [25, 25, 25], [100, 100, 100], [500, 500, 500]]
            lrs = [0.5, 0.4, 0.3, 0.2, 0.1, 0.06, 0.03, 0.02, 0.01, 0.006, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0001, 0.00003]
        elif train_type == "train_b_w":
            if True:
                nr_hiddens = [[10], [100], [500], [10, 10], [100, 100], [500, 500], [10, 10, 10], [100, 100, 100],
                              [500, 500, 500]]
                # lrs = [0.5, 0.3, 0.2, 0.1, 0.06, 0.03, 0.01, 0.003, 0.001]
                lrs = [0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0001]
            else:
                lrs = [0.5, 0.3, 0.2, 0.1, 0.06, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]
        elif train_type == "train_bmr":
            # lrs = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]
            lrs = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]
        elif train_type == "train_sg":
            # nr_hiddens = [[25], [100], [500],
            #               [25, 25], [100, 100], [500, 500],
            #               [25, 25, 25], [100, 100, 100], [500, 500, 500]]
            if nrhd100:
                lrs = [0.1, 0.03, 0.02, 0.01, 0.006, 0.003, 0.002,  0.001, 0.0003, 0.0001, 0.00003, 0.00002, 0.00001,
                       0.000006, 0.000003, 0.000001]
            else:
                lrs = [0.1, 0.03, 0.02, 0.01, 0.006, 0.003, 0.002, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
        else:
            raise ValueError(train_type)
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

    if datasets is None:
        # datasets = ["MNIST", "QMNIST", "KMNIST", "EMNIST_bymerge", "EMNIST_letters", "EMNIST"]
        # datasets = ["QMNIST", "KMNIST", "EMNIST_bymerge", "EMNIST_letters"]
        datasets = ["EMNIST_bymerge", "K49", "CIFAR100"]
    if train_type == "train_b_w":
        train_params = get_biaslearner_training_params(new_train_params=train_params, highseed=20)
        subdir = "scan_train_blr_wlr"
        prog_name = "scan_train_blr_wlr"
    elif train_type == "train_bw":
        train_params = get_biaslearner_training_params(new_train_params=train_params, highseed=20)
        subdir = "scan_trainbw_lr"
        prog_name = "scan_train_bw_lr"
    elif train_type == "train_bw_hardtanh":
        train_params = get_biaslearner_training_params(new_train_params=train_params, readout_function="hardtanh",
                                                       highseed=20)
        subdir = "scan_trainbw_lr"
        prog_name = "scan_train_bw_lr"
    elif train_type == "train_mr":
        train_params = get_multireadout_training_params(new_train_params=train_params, highseed=20)
        subdir = "scan_train_mr_lr"
        prog_name = "scan_train_mr_lr"
    elif train_type == "train_bmr":
        train_params = get_binarymr_training_params(new_train_params=train_params, highseed=20)
        subdir = "scan_train_bmr_lr"
        prog_name = "scan_train_bmr_lr"
    elif train_type == "train_sg":
        train_params = get_biaslearner_training_params(new_train_params=train_params, highseed=20)
        subdir = "scan_train_sg_lr"
        prog_name = "scan_train_sg_lr"
    elif train_type == "train_sg_bigbatch":
        train_params = get_biaslearner_training_params(new_train_params=train_params, highseed=20)
        subdir = "scan_train_sg_bigbatch_lr"
        prog_name = "scan_train_sg_lr"
    elif train_type == "transfer_b_l1o":
        train_params = get_biaslearner_training_params(new_train_params=train_params, highseed=3)
        subdir = "scan_transfer_b_lr"
        prog_name = "scan_transfer_b_lr"
    elif train_type == "transfer_mr_l1o":
        train_params = get_multireadout_training_params(new_train_params=train_params, highseed=3)
        subdir = "scan_transfer_mr_lr"
        prog_name = "scan_transfer_mr_lr"
    elif train_type == "transfer_bmr_l1o":
        train_params = get_binarymr_training_params(new_train_params=train_params, highseed=3)
        subdir = "scan_transfer_bmr_lr"
        prog_name = "scan_transfer_bmr_lr"
    else:
        raise ValueError(train_type)
    load_dir, save_dir = get_dirs(root_dir, subdir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for nh in range(len(nr_hiddens)):
        row = row_ids[nh]
        col = col_ids[nh]
        for dataset in datasets:
            for es in range(len(early_stopping)):
                train_params["early_stopping"] = early_stopping[es]
                ys = []
                errs = []
                for lr in lrs:
                    if train_type == "train_b_w":
                        train_params["lr"] = lr
                    elif train_type == "train_bmr":
                        train_params["r_lr"] = lr
                        train_params["lr"] = lr
                    else:
                        train_params["lr"] = lr
                    performances = get_performances(prog_name=prog_name, load_info=load_dir, nr_hidden=nr_hiddens[nh],
                                                    dataset=dataset, train_params=train_params,
                                                    performance_type="validation_performance")

                    ys += [torch_mean(performances)]
                    errs += [torch_std(performances)]
                m = max(zip(ys, range(len(ys))))[1]

                if nrhd100:
                    axs[row].errorbar(lrs, ys, errs, label=dataset + label_es[es])
                    axs[row].plot(float(lrs[m]), ys[m], "xk")
                else:
                    axs[row, col].errorbar(lrs, ys, errs, label=dataset+label_es[es])
                    axs[row, col].plot(float(lrs[m]), ys[m], "xk")
            if nrhd100:
                axs[row].set_title("Network {}".format(nr_hiddens[nh]))
                axs[row].set_xscale('log')
                axs[row].set_ylim([50., 100.])
            else:
                axs[row, col].set_title("Network {}".format(nr_hiddens[nh]))
                axs[row, col].set_xscale('log')
                axs[row, col].set_ylim([50., 100.])

    # Plotting Specifics
    if train_type == "train_b_w":
        x_label = "bias learning rate"
    else:
        x_label = "learning rate"
    if nrhd100:
        for col in range(len(hidden_neuron_per_layer_nrs)):
            axs[len(hidden_layer_nrs)-1].set_xlabel(x_label)
        for row in range(len(hidden_layer_nrs)):
            axs[row].set_ylabel('Validation Performance [%]')
        handles, labels = axs[0].get_legend_handles_labels()
    else:
        for col in range(len(hidden_neuron_per_layer_nrs)):
            axs[len(hidden_layer_nrs)-1, col].set_xlabel(x_label)
        for row in range(len(hidden_layer_nrs)):
            axs[row, 0].set_ylabel('Validation Performance [%]')
        handles, labels = axs[0, 0].get_legend_handles_labels()
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


def plot_comparison(train_params, plot_params, datasets=None, nr_hiddens=None, saving=False, root_dir="../../",
                    titling=False):
    """

    Parameters
    ----------
    train_params:
    plot_params:
    datasets:
    nr_hiddens:
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """
    if nr_hiddens is None:
        nr_hiddens = [[10], [100], [500], [10, 10], [100, 100], [500, 500]]
    compared_settings = list(train_params.keys())
    load_dir, save_dir = get_dirs(root_dir, plot_params["subdir"])
    hidden_layer_nrs, hidden_neuron_per_layer_nrs = get_hidden_layer_info(nr_hiddens=nr_hiddens)
    nr_rows = len(hidden_layer_nrs)
    nr_cols = len(hidden_neuron_per_layer_nrs)
    fig, axs = plt.subplots(nr_rows, nr_cols, sharex='all', sharey='all')
    nr_bars = len(compared_settings)
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(datasets))

    for r in range(nr_rows):
        for c in range(nr_cols):
            for i in range(len(compared_settings)):
                nr_hidden = hidden_layer_nrs[r] * [hidden_neuron_per_layer_nrs[c]]
                y = []
                y_err = []
                for dataset in datasets:
                    performances = get_performances(
                        prog_name=plot_params["prog_name"], load_info=load_dir, nr_hidden=nr_hidden, dataset=dataset,
                        train_params=train_params[compared_settings[i]], performance_type=plot_params["perf_type"]
                    )
                    y += [torch_mean(performances)]
                    y_err += [torch_std(performances)]
                axs[r, c].bar(x + x_offset[i], y, width, yerr=y_err, label=plot_params["labels"][i], zorder=3)
            axs[r, c].set_title("{} x {} hidden neurons".format(hidden_layer_nrs[r], hidden_neuron_per_layer_nrs[c]))
            axs[r, c].set_xticks(x)
            axs[r, c].grid(axis='y', alpha=0.4, zorder=0)
            axs[r, c].set_ylim(plot_params["ymin"], 105)

        axs[r, 0].set_ylabel(plot_params["y_label"])
    for c in range(len(hidden_layer_nrs)):
        axs[-1, c].set_xticklabels([d.replace("_", "\n") for d in datasets])

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    fig.subplots_adjust(left=0.05, right=plot_params["right_adjust"], top=0.91, wspace=0.05, hspace=0.16)
    fig.set_size_inches(16, 9)
    if titling:
        plt.suptitle(plot_params["title"], fontsize=14)
    if saving:
        plt.savefig("{}plots/scan_params/svg/{}.svg".format(root_dir, plot_params["save_name"]))
        plt.savefig("{}plots/scan_params/{}.png".format(root_dir, plot_params["save_name"]))
        plt.close()
    else:
        plt.show()


def plot_train_test_performances(prog_name, saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    prog_name: Name of the training program of which the performance should be plotted
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """

    datasets = ["MNIST", "QMNIST", "EMNIST_bymerge", "EMNIST_letters", "EMNIST", "KMNIST"]
    # datasets = ["QMNIST", "EMNIST_bymerge", "EMNIST_letters", "KMNIST"]
    nr_hiddens = [[10], [25], [50], [100], [250], [500],
                  [10, 10], [25, 25], [50, 50], [100, 100], [250, 250], [500, 500]]

    if prog_name == "train_b_w_full":
        nr_hiddens = [[10], [25], [50], [100], [250], [500],
                      [10, 10], [25, 25], [50, 50], [100, 100], [250, 250], [500, 500],
                      [10, 10, 10], [25, 25, 25], [50, 50, 50], [100, 100, 100], [250, 250, 250], [500, 500, 500]]
        train_params = get_biaslearner_training_params(highseed=25)
        subdir = "train_full_dataset"
        title = "Bias learning on full datasets"
        ymin = 80

    elif prog_name == "train_sg_full":
        # nr_hiddens = [[25], [100], [500],
        #               [25, 25], [100, 100], [500, 500],
        #               [25, 25, 25], [100, 100, 100], [500, 500, 500]]
        train_params = get_biaslearner_training_params(highseed=25)
        subdir = "train_full_dataset"
        title = "Dedicating a full network to each task"
        ymin = 50

    elif prog_name == "train_mr_full":
        train_params = get_multireadout_training_params(highseed=20)
        subdir = "train_full_dataset"
        title = "Multireadout learning on full datasets"
        ymin = 0

    elif prog_name == "train_bmr_full":
        train_params = get_binarymr_training_params(highseed=20)
        subdir = "train_full_dataset"
        title = "Binary multireadout learning on full datasets"
        ymin = 80

    elif prog_name == "train_b_w_l1o":
        train_params = get_biaslearner_training_params(highseed=3)
        subdir = "leave_1_out"
        title = "Weight and Bias learning on datasets with all classes except one"
        ymin = 80

    elif prog_name == "train_mr_l1o":
        train_params = get_multireadout_training_params(highseed=3)
        subdir = "leave_1_out"
        title = "Multireadout hidden weights learning on datasets with all classes except one"
        ymin = 0

    elif prog_name == "train_bmr_l1o":
        train_params = get_binarymr_training_params(highseed=3)
        subdir = "leave_1_out"
        title = "Binary multireadout hidden weights learning on datasets with all classes except one"
        ymin = 80

    elif prog_name == "transfer_b_l1o_b_w":
        train_params = get_biaslearner_training_params(highseed=3, transfering=True)
        subdir = "leave_1_out"
        title = "Bias learning on class left out during weight training"
        ymin = 50

    elif prog_name == "transfer_mr_l1o":
        train_params = get_multireadout_training_params(highseed=3)
        subdir = "leave_1_out"
        title = "Readout learning on class left out during hidden weight training"
        ymin = 0

    elif prog_name == "transfer_bmr_l1o":
        train_params = get_binarymr_training_params(highseed=3)
        subdir = "leave_1_out"
        title = "Binary readout learning on class left out during hidden weight training"
        ymin = 0

    else:
        raise ValueError(prog_name)
    load_dir, save_dir = get_dirs(root_dir, subdir)

    hidden_layer_nrs, hidden_neuron_per_layer_nrs = get_hidden_layer_info(nr_hiddens=nr_hiddens)
    fig, axs = plt.subplots(2, len(hidden_layer_nrs), sharex='all', sharey='all')
    performance_types = ["train_performance", "test_performance"]
    y_labels = ["Train Performance [%]", "Test Performance [%]"]
    nr_bars = int(round(len(nr_hiddens)/len(hidden_layer_nrs)))
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(datasets))

    for r in range(len(performance_types)):
        for c in range(len(hidden_layer_nrs)):
            for hnl in range(len(hidden_neuron_per_layer_nrs)):
                hnpl = hidden_neuron_per_layer_nrs[hnl]
                hnpl_str = str(hnpl)
                nr_hidden = [hnpl] * hidden_layer_nrs[c]
                y = []
                y_err = []
                for dataset in datasets:
                    performances = get_performances(prog_name=prog_name, load_info=load_dir, nr_hidden=nr_hidden,
                                                    dataset=dataset, train_params=train_params,
                                                    performance_type=performance_types[r])
                    y += [torch_mean(performances)]
                    y_err += [torch_std(performances)]
                axs[r, c].bar(x + x_offset[hnl], y, width, yerr=y_err, label=hnpl_str+" neurons / layer", zorder=3)
            axs[r, c].set_xticks(x)
            axs[r, c].set_ylim(ymin)
            axs[r, c].grid(axis='y', alpha=0.4, zorder=0)

        axs[r, 0].set_ylabel(y_labels[r])
    for c in range(len(hidden_layer_nrs)):
        axs[-1, c].set_xticklabels([d.replace("_", "\n") for d in datasets])
        axs[0, c].set_title("{} Hidden Layers".format(hidden_layer_nrs[c]))

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    fig.subplots_adjust(left=0.05, right=0.875, wspace=0.05, hspace=0.06)
    fig.set_size_inches(16, 9)
    if titling:
        plt.suptitle(title)
    if saving:
        plt.savefig(save_dir + "svg/{}_train_test_performance_plot.svg".format(prog_name))
        plt.savefig(save_dir + "{}_train_test_performance_plot.png".format(prog_name))
        plt.close()
    else:
        plt.show()


def multitask_all_plot_old(saving=False, root_dir="../../", titling=False):
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
                  [10, 10], [25, 25], [50, 50], [100, 100], [250, 250], [500, 500],
                  [10, 10, 10], [25, 25, 25], [50, 50, 50], [100, 100, 100], [250, 250, 250], [500, 500, 500]]
    traintypes = ["train_sg_full", "train_b_w_full", "train_bmr_full"]
    legends = ["Singular approach", "Context modulation", "Multiple readouts"]

    load_dirs = 3 * ["{}results/train_full_dataset/".format(root_dir)]
    train_params = [get_biaslearner_training_params(),
                    get_biaslearner_training_params(),
                    get_binarymr_training_params()]
    title = "Multi-task learning performances"

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
        plt.savefig("{}plots/train_full_dataset/svg/multitask_all.svg".format(root_dir))
        plt.savefig("{}plots/train_full_dataset/multitask_all.png".format(root_dir))
        plt.close()
    else:
        plt.show()


def multitask_all_plot(saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """

    datasets = ["EMNIST_bymerge", "CIFAR100"]
    nr_hiddens = [[25], [100], [500],
                  [25, 25], [100, 100], [500, 500],
                  [25, 25, 25], [100, 100, 100], [500, 500, 500]]
    # traintypes = ["train_b_w_full", "train_g_bw_full", "train_bg_w_full", "train_bmr_full"]
    # legends = ["Task-specific biases", "Task-specific gains", "Task-specific biases and gains",
    #            "Task-specific readouts"]
    #
    # load_dirs = 4 * ["{}results/train_full_dataset/".format(root_dir)]
    # train_params = [get_biaslearner_training_params(),
    #                 get_biaslearner_training_params(),
    #                 get_biaslearner_training_params(),
    #                 get_binarymr_training_params()]
    traintypes = ["train_b_w_full", "train_g_bw_full", "train_bg_w_full"]
    legends = ["Task-specific biases", "Task-specific gains", "Task-specific biases and gains"]

    load_dirs = 3 * ["{}results/train_full_dataset/".format(root_dir)]
    train_params = [get_biaslearner_training_params(),
                    get_biaslearner_training_params(),
                    get_biaslearner_training_params()]
    title = "Multi-task learning performances"

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
    fig.subplots_adjust(left=0.06, right=0.82, top=0.91, bottom=0.06, hspace=0.15)
    fig.set_size_inches(15, 10)
    if saving:
        plt.savefig("{}plots/train_full_dataset/svg/multitask_all.svg".format(root_dir))
        plt.savefig("{}plots/train_full_dataset/multitask_all.png".format(root_dir))
        plt.close()
    else:
        plt.show()


def multitask_emnist_1hl_plot(saving=False, root_dir="../../", titling=False):
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
    traintypes = ["train_sg_full", "train_b_w_full", "train_bmr_full"]
    legends = ["Singular approach", "Context modulation", "Multiple readouts"]

    load_dirs = 3 * ["{}results/train_full_dataset/".format(root_dir)]
    train_params = [get_biaslearner_training_params(),
                    get_biaslearner_training_params(),
                    get_binarymr_training_params()]
    title = "Multi-task learning on EMNIST"

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
        axs.set_xticks(x)
        axs.grid(axis='y', alpha=0.4, zorder=0)
        axs.set_ylim([50, 100])
        axs.set_ylabel("Test Performance [%]")
    horiz_center = 0.44
    fig.text(horiz_center, 0.015, '# Hidden Neurons', ha='center')
    if titling:
        fig.text(horiz_center, 0.94, title, fontsize=14, ha='center')
    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    axs.set_xticklabels([n[0] for n in nr_hiddens])
    fig.subplots_adjust(left=0.06, right=0.84, top=0.88, bottom=0.12, hspace=0.26)
    fig.set_size_inches(13, 4)
    if saving:
        plt.savefig("{}plots/train_full_dataset/svg/multitask_emnist_1hl.svg".format(root_dir))
        plt.savefig("{}plots/train_full_dataset/multitask_emnist_1hl.png".format(root_dir))
        plt.close()
    else:
        plt.show()


def multitask_plot(saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """

    datasets = ["EMNIST_bymerge"]
    # nr_hiddens = [[10], [100], [10, 10], [100, 100], [10, 10, 10], [100, 100, 100]]
    # nr_hiddens = [[25], [250], [25, 25], [250, 250], [25, 25, 25], [250, 250, 250]]
    # nr_hiddens = [[50], [500], [50, 50], [500, 500], [50, 50, 50], [500, 500, 500]]
    # nr_hiddens = [[10], [25], [50], [100], [250], [500],
    #               [10, 10], [25, 25], [50, 50], [100, 100], [250, 250], [500, 500],
    #               [10, 10, 10], [25, 25, 25], [50, 50, 50], [100, 100, 100], [250, 250, 250], [500, 500, 500]]
    nr_hiddens = [[25], [50], [100], [250], [500],
                  [25, 25, 25], [50, 50, 50], [100, 100, 100], [250, 250, 250], [500, 500, 500]]
    traintypes = ["train_sg_full", "train_b_w_full", "train_bmr_full"]
    legends = ["Singular approach", "Context modulation", "Multiple readouts"]

    load_dirs = 3 * ["{}results/train_full_dataset/".format(root_dir)]
    train_params = [get_biaslearner_training_params(),
                    get_biaslearner_training_params(),
                    get_binarymr_training_params()]
    title = "Multi-task learning performances"

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
        axs.set_xticks(x)
        axs.grid(axis='y', alpha=0.4, zorder=0)
        axs.set_ylim([50, 100])
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


def fig_2(saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """

    datasets = ["EMNIST_bymerge"]
    nr_hiddens = [[10], [25], [50], [100], [250], [500],
                  [10, 10, 10], [25, 25, 25], [50, 50, 50], [100, 100, 100], [250, 250, 250], [500, 500, 500]]
    # nr_hiddens = [[25], [250], [25, 25, 25], [250, 250, 250]]
    # nr_hiddens = [[50], [500], [50, 50, 50], [500, 500, 500]]
    # nr_hiddens = [[25], [250], [25, 25, 25], [250, 250, 250]]
    # nr_hiddens = [[25], [500], [25, 25, 25], [500, 500, 500]]

    traintypes = ["train_sg_full", "polish_b_full", "test_train_bmr_full", "transfer_b_l1o_b_w", "transfer_bmr_l1o"]
    legends = ["Task-specific networks", "Multitask biases", "Multitask readouts", "Transfer learn biases",
               "Transfer learn readout"]
    colors = ["mediumpurple", "royalblue", "olivedrab", "cornflowerblue", "yellowgreen"]

    load_dirs = ["{}results/train_full_dataset/".format(root_dir)] + 2 * ["{}results/train_full_dataset/".format(root_dir)] + 2 * ["{}results/leave_1_out/".format(root_dir)]
    train_params = [get_biaslearner_training_params(highseed=3),
                    get_biaslearner_training_params(),
                    get_binarymr_training_params(),
                    get_biaslearner_training_params(highseed=3, transfering=True),
                    get_binarymr_training_params(highseed=3)]
    results = {}
    title = "Multi-task learning performances"
    fig, axs = plt.subplots(len(datasets), 1, sharex='all', sharey='all')
    nr_bars = len(traintypes)
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(nr_hiddens))

    for row in range(len(datasets)):
        for tt in range(len(traintypes)):
            results[traintypes[tt]] = {}
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
                                                train_params=train_params[tt], performance_type="test_performance", bestseed=True)
                y += [torch_mean(performances)]
                y_err += [torch_std(performances)]
                results[traintypes[tt]][str(nr_hiddens[nh])] = {"mean": y[-1], "std": y_err[-1]}
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
        with open("../../plots/figures/Fig2C.pickle", 'wb') as handle:
            pickle_dump(results, handle, protocol=HIGHEST_PROTOCOL)
    else:
        plt.show()
    print(results)


def k49_plot(saving=False, root_dir="../../", titling=False):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    titling: Whether to add a title to the figure or not
    -------
    """

    datasets = ["K49"]
    nr_hiddens = [[10], [25], [50], [100], [250], [500]]
    # nr_hiddens = [[10], [25], [50], [100], [250], [500],
    #               [10, 10, 10], [25, 25, 25], [50, 50, 50], [100, 100, 100], [250, 250, 250], [500, 500, 500]]
    # nr_hiddens = [[25], [250], [25, 25, 25], [250, 250, 250]]
    # nr_hiddens = [[50], [500], [50, 50, 50], [500, 500, 500]]
    # nr_hiddens = [[25], [250], [25, 25, 25], [250, 250, 250]]
    # nr_hiddens = [[25], [500], [25, 25, 25], [500, 500, 500]]

    traintypes = ["train_bmr_full", "transfer_bmr_full"]
    legends = ["Multitask readouts", "Transfer learn readout"]
    colors = ["mediumpurple", "royalblue"]

    load_dirs = ["{}results/train_full_dataset/".format(root_dir),
                 "{}results/train_full_dataset/K49/binarymr_full_dataset_EMNIST_bymerge".format(root_dir)]
    train_params = [get_binarymr_training_params(), get_binarymr_training_params()]
    results = {}
    title = "K49 performances"
    fig, axs = plt.subplots(len(datasets), 1, sharex='all', sharey='all')
    nr_bars = len(traintypes)
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(nr_hiddens))

    for row in range(len(datasets)):
        for tt in range(len(traintypes)):
            results[traintypes[tt]] = {}
            y = []
            y_err = []
            for nh in range(len(nr_hiddens)):

                performances = get_performances(prog_name=traintypes[tt], load_info=load_dirs[tt],
                                                nr_hidden=nr_hiddens[nh], dataset=datasets[row],
                                                train_params=train_params[tt], performance_type="test_performance",
                                                bestseed=False)
                y += [torch_mean(performances)]
                y_err += [torch_std(performances)]
                results[traintypes[tt]][str(nr_hiddens[nh])] = {"mean": y[-1], "std": y_err[-1]}

            axs.bar(x + x_offset[tt], y, width, yerr=y_err, label=legends[tt], zorder=3, color=colors[tt])
        axs.set_xticks(x)
        axs.grid(axis='y', alpha=0.4, zorder=0)
        axs.set_ylim([50, 100])
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
        plt.savefig("{}plots/train_full_dataset/svg/transfer_k49.svg".format(root_dir))
        plt.savefig("{}plots/train_full_dataset/transfer_k49.png".format(root_dir))
        plt.close()
        # with open("../../plots/figures/Fig2C.pickle", 'wb') as handle:
        #     pickle_dump(results, handle, protocol=HIGHEST_PROTOCOL)
    else:
        plt.show()
    print(results)


def main_plots(saving=True, tt=True):

    # Parameter scan plots
    plot_best_2d_lr(prog_name="scan_train_blr_wlr", titling=tt, saving=saving)
    plot_best_2d_lr(prog_name="scan_train_bmr_lr", titling=tt, saving=saving)
    for train_type in ["train_b_w", "train_bw", "train_bw_hardtanh", "train_mr", "train_sg", "transfer_b_l1o",
                       "transfer_mr_l1o", "train_b_w", "train_bw", "train_bw_hardtanh", "train_mr", "train_sg",
                       "transfer_b_l1o", "transfer_mr_l1o"]:
        plot_scan_lr(train_type, early_stopping=True, titling=tt, saving=saving)

    # Training and testing performance plots
    for train_type in ["train_b_w_full", "train_sg_full", "train_mr_full", "train_b_w_l1o", "train_mr_l1o",
                       "transfer_b_l1o_b_w", "transfer_mr_l1o"]:
        plot_train_test_performances(train_type, titling=tt, saving=saving)

    multitask_all_plot(saving=saving, titling=tt)
    multitask_emnist_1hl_plot(saving=saving, titling=tt)
