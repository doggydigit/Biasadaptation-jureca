import sys
from re import search
from os import listdir
from os.path import isfile
from torch import stack as torch_stack
from torch import mean as torch_mean
from pickle import load, dump, HIGHEST_PROTOCOL
from data_helper import get_number_classes, get_all_data, get_singular_data
from train_helper import get_train_eval_model
from leave1out import l1o_get_train_eval
from biaslearning_helper import get_biaslearner_training_params
from multireadout_helper import get_binarymr_training_params
from plot_helper import get_performances


def save_best_param(filepath, dataset, netstring, param):
    if isfile(filepath):
        with open(filepath, 'rb') as file:
            bestparams = load(file)
        if dataset not in bestparams:
            bestparams[dataset] = {}
    else:
        bestparams = {dataset: {netstring: []}}

    bestparams[dataset][netstring] = param

    with open(filepath, 'wb') as handle:
        dump(bestparams, handle, protocol=HIGHEST_PROTOCOL)


def save_best_params(traintype, dataset, network, root_dir="../../", verbose=False):

    # Initialize variables depending on the type of training, network and dataset
    netstring = str(network)
    if traintype == "b_w":
        dirpath = root_dir + "results/scan_train_blr_wlr/"
        l1, l2 = "blr", "wlr"
    elif traintype == "g_bw":
        dirpath = root_dir + "results/scan_train_glr_bwlr/"
        l1, l2 = "glr", "wlr"
    elif traintype == "bg_w":
        dirpath = root_dir + "results/scan_train_bglr_wlr/"
        l1, l2 = "bglr", "wlr"
    elif traintype == "bmr":
        dirpath = root_dir + "results/scan_train_bmr_lr/"
        l1, l2 = "rlr", "lr"
    else:
        raise ValueError(traintype)

    # Load all parameter sets simulated
    files = listdir(dirpath + "individual/")
    files = [f for f in files if netstring in f and dataset in f]
    params = [[float(search("{}_(.*?).pickle".format(l1), f).group(1)),
               float(search("{}_(.*?)_{}".format(l2, l1), f).group(1))] for f in files]

    # Extract all the validation performances
    performances = []
    for file in files:
        with open(dirpath + "individual/" + file, "rb") as f:
            result = load(f)
            performances += [torch_mean(torch_stack(result["validation_performance"])).item()]

    # Chose the parameters yielding the best performance
    bestparam = params[performances.index(max(performances))]

    # Update the current set of all best parameters (if exists)
    filepath = dirpath + "bestparams.pickle"
    save_best_param(filepath, dataset, netstring, bestparam)

    if verbose:
        print(traintype, dataset, network, bestparam, max(performances))


def save_all_best_params(root_dir="../../", verbose=False):
    traintypes = ["b_w", "g_bw", "bg_w", "bmr"]
    datasets = ["CIFAR100", "EMNIST_bymerge"]
    networks = [[25], [100], [500], [25, 25], [100, 100], [500, 500], [25, 25, 25], [100, 100, 100], [500, 500, 500]]
    for tt in traintypes:
        for d in datasets:
            for n in networks:
                save_best_params(tt, d, n, root_dir=root_dir, verbose=verbose)

    # Singular training schemes
    datasets = ["CIFAR100", "EMNIST_bymerge", "K49"]
    # networks = [[25], [100], [500], [25, 25], [100, 100], [500, 500], [25, 25, 25], [100, 100, 100], [500, 500, 500]]
    networks = [[100], [100, 100], [100, 100, 100]]
    dirpath = root_dir + "results/scan_train_sg_lr/"
    train_params = get_biaslearner_training_params(highseed=3)

    # Load all parameter sets simulated
    files = listdir(dirpath + "individual/")
    for d in datasets:
        for n in networks:
            the_files = [f for f in files if str(n) in f and d in f]
            lrs = [float(search("_es_lr_(.*?)_testclass_", f).group(1)) for f in the_files]
            mean_perfs = []
            for lr in lrs:
                train_params["lr"] = lr
                performances = get_performances(prog_name="scan_train_sg_lr", load_info=dirpath, nr_hidden=n,
                                                dataset=d, train_params=train_params,
                                                performance_type="validation_performance")

                mean_perfs += [torch_mean(performances)]

            # Chose the parameters yielding the best performance
            bestlr = lrs[mean_perfs.index(max(mean_perfs))]

            # Update the current set of all best parameters (if exists)
            filepath = dirpath + "bestparams.pickle"
            save_best_param(filepath, d, str(n), bestlr)

            if verbose:
                print("sg", d, n, bestlr, max(mean_perfs))


def save_avbest_params(root_dir="../../", verbose=False):
    traintypes = ["b_w", "g_bw", "bg_w", "bmr"]
    datasets = ["CIFAR100", "EMNIST_bymerge"]
    # networks = [[25], [100], [500], [25, 25], [100, 100], [500, 500], [25, 25, 25], [100, 100, 100], [500, 500, 500]]
    networks = [[100], [100, 100], [100, 100, 100]]

    subdirs = {"b_w": "scan_train_blr_wlr", "g_bw": "scan_train_glr_bwlr", "bg_w": "scan_train_bglr_wlr"}
    for tt in traintypes:
        if tt in ["b_w", "g_bw", "bg_w"]:
            prog_name = subdirs[tt]
            load_dir = root_dir + "results/" + subdirs[tt] + "/"
            train_params = get_biaslearner_training_params(highseed=20)
            lrs1 = [0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0002, 0.0001, 0.00006, 0.00003, 0.00002, 0.00001]
            lrs2 = [0.3, 0.2, 0.1, 0.06, 0.03, 0.02, 0.01, 0.006, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0001, 0.00003]
            lr1_name = "lr"
            lr2_name = "b_lr"
        elif tt == "bmr":
            prog_name = "scan_train_bmr_lr"
            load_dir = root_dir + "results/scan_train_bmr_lr/"
            train_params = get_binarymr_training_params(highseed=20)
            lrs1 = [0.1, 0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0002, 0.0001, 0.00006, 0.00003]
            lrs2 = [0.1, 0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0002, 0.0001, 0.00003]
            lr1_name = "lr"
            lr2_name = "r_lr"
        else:
            raise None
        for d in datasets:
            best_avperf = 0.
            bestlr1 = 0.
            bestlr2 = 0.
            for lr1 in lrs1:
                train_params[lr1_name] = lr1
                for lr2 in lrs2:
                    train_params[lr2_name] = lr2
                    avperf = 0.
                    for n in networks:
                        avperf += torch_mean(get_performances(prog_name=prog_name, load_info=load_dir, nr_hidden=n,
                                                              dataset=d, train_params=train_params,
                                                              performance_type="validation_performance")).item()
                    avperf = avperf / float(len(networks))
                    if avperf > best_avperf:
                        bestlr1 = lr1
                        bestlr2 = lr2
                        best_avperf = avperf
            print(prog_name, d, bestlr1, bestlr2, best_avperf)

            # # Update the current set of all best parameters (if exists)
            # filepath = dirpath + "bestparams.pickle"
            # save_best_param(filepath, d, str(n), bestlr)
            #
            # if verbose:
            #     print("sg", d, n, bestlr, max(mean_perfs))

    # For all architectures:
    """
    scan_train_blr_wlr CIFAR100 0.0001 0.06 76.97202640109592
    scan_train_blr_wlr EMNIST_bymerge 0.001 0.02 96.51343282063802
    scan_train_glr_bwlr CIFAR100 1e-05 0.02 75.87881554497613
    scan_train_glr_bwlr EMNIST_bymerge 3e-05 0.06 96.31387583414714
    scan_train_bglr_wlr CIFAR100 1e-05 0.01 77.06021372477214
    scan_train_bglr_wlr EMNIST_bymerge 0.0003 0.006 96.61985100640192
    scan_train_bmr_lr CIFAR100 6e-05 0.002 77.68179321289062
    scan_train_bmr_lr EMNIST_bymerge 6e-05 0.003 96.68902587890625
    """

    # For 100 archis:
    """
    scan_train_blr_wlr CIFAR100 6e-05 0.02 77.25488535563152
    scan_train_blr_wlr EMNIST_bymerge 0.001 0.02 96.92967987060547
    scan_train_glr_bwlr CIFAR100 1e-05 0.02 76.36254119873047
    scan_train_glr_bwlr EMNIST_bymerge 6e-05 0.06 96.84518432617188
    scan_train_bglr_wlr CIFAR100 2e-05 0.02 77.34906514485677
    scan_train_bglr_wlr EMNIST_bymerge 0.0006 0.006 97.05760955810547
    scan_train_bmr_lr CIFAR100 3e-05 0.002 78.18812815348308
    scan_train_bmr_lr EMNIST_bymerge 6e-05 0.003 97.30079396565755
    """


def save_avbest_paramsk49(root_dir="../../", verbose=False):
    traintypes = ["b_w", "g_bw", "bg_w", "bmr"]
    datasets = ["K49"]
    networks = [[25], [100], [500], [25, 25], [100, 100], [500, 500], [25, 25, 25], [100, 100, 100], [500, 500, 500]]
    # networks = [[100], [100, 100], [100, 100, 100]]

    subdirs = {"b_w": "scan_train_blr_wlr", "g_bw": "scan_train_glr_bwlr", "bg_w": "scan_train_bglr_wlr"}
    for tt in traintypes:
        if tt in ["b_w", "g_bw", "bg_w"]:
            prog_name = subdirs[tt]
            load_dir = root_dir + "results/" + subdirs[tt] + "/"
            train_params = get_biaslearner_training_params(highseed=20)
            # bls = ("0.3" "0.2" "0.1" "0.06" "0.03" "0.01" "0.003" "0.0001" "0.00003")
            # ls = ("0.03" "0.01" "0.003" "0.002" "0.001" "0.0006" "0.0003" "0.0001" "0.00006" "0.00003")
            # lrs1 = [0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0001, 0.00006, 0.00003]
            # lrs2 = [0.3, 0.2, 0.1, 0.06, 0.03, 0.02, 0.01, 0.006, 0.003, 0.0001, 0.00003]
            lrs1 = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
            lrs2 = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
            lr1_name = "lr"
            lr2_name = "b_lr"
        elif tt == "bmr":
            prog_name = "scan_train_bmr_lr"
            load_dir = root_dir + "results/scan_train_bmr_lr/"
            train_params = get_binarymr_training_params(highseed=20)
            # rls = ("0.01" "0.003" "0.002" "0.001" "0.0006" "0.0003" "0.0002" "0.0001")
            # ls = ("0.01" "0.003" "0.002" "0.001" "0.0006" "0.0003" "0.0002" "0.0001")
            # lrs1 = [0.1, 0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0002, 0.0001]
            # lrs2 = [0.1, 0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0002, 0.0001]
            lrs1 = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
            lrs2 = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
            lr1_name = "lr"
            lr2_name = "r_lr"
        else:
            raise None
        for d in datasets:
            best_avperf = 0.
            bestlr1 = 0.
            bestlr2 = 0.
            for lr1 in lrs1:
                train_params[lr1_name] = lr1
                for lr2 in lrs2:
                    train_params[lr2_name] = lr2
                    avperf = 0.
                    for n in networks:
                        avperf += torch_mean(get_performances(prog_name=prog_name, load_info=load_dir, nr_hidden=n,
                                                              dataset=d, train_params=train_params,
                                                              performance_type="validation_performance")).item()
                    avperf = avperf / float(len(networks))
                    if avperf > best_avperf:
                        bestlr1 = lr1
                        bestlr2 = lr2
                        best_avperf = avperf
            print(prog_name, d, bestlr1, bestlr2, best_avperf)

            # # Update the current set of all best parameters (if exists)
            # filepath = dirpath + "bestparams.pickle"
            # save_best_param(filepath, d, str(n), bestlr)
            #
            # if verbose:
            #     print("sg", d, n, bestlr, max(mean_perfs))

    #  For all archis:
    """
    scan_train_blr_wlr K49 0.001 0.01 94.47021569146051
    scan_train_glr_bwlr K49 0.0001 0.1 94.25004069010417
    scan_train_bglr_wlr K49 0.001 0.01 94.59927368164062
    scan_train_bmr_lr K49 0.0001 0.01 94.7098880343967
    """


def save_avbest_paramst2d(root_dir="../../", verbose=False):
    traintypes = ["b_w", "g_bw", "bg_w", "bmr"]
    datasets = ["K49"]
    dataset = "K49"
    networks = [[25], [100], [500], [25, 25], [100, 100], [500, 500], [25, 25, 25], [100, 100, 100], [500, 500, 500]]
    # networks = [[100], [100, 100], [100, 100, 100]]

    subdirs = {"b_w": "scan_train_blr_wlr", "g_bw": "scan_train_glr_bwlr", "bg_w": "scan_train_bglr_wlr"}
    for tt in traintypes:
        if tt in ["b_w", "g_bw", "bg_w"]:
            prog_name = subdirs[tt]
            load_dir = root_dir + "results/" + subdirs[tt] + "/"
            train_params = get_biaslearner_training_params(highseed=20)
            # lrs1 = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
            # lrs2 = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

            netstring = str([25, 25])
            if tt == "b_w":
                dirpath = root_dir + "results/scan_train_blr_wlr/"
                l1, l2 = "blr", "wlr"
            elif tt == "g_bw":
                dirpath = root_dir + "results/scan_train_glr_bwlr/"
                l1, l2 = "glr", "wlr"
            elif tt == "bg_w":
                dirpath = root_dir + "results/scan_train_bglr_wlr/"
                l1, l2 = "bglr", "wlr"
            elif tt == "bmr":
                dirpath = root_dir + "results/scan_train_bmr_lr/"
                l1, l2 = "rlr", "lr"
            else:
                raise ValueError(tt)
            lr1_name = "b_lr"
            lr2_name = "lr"
        elif tt == "bmr":
            prog_name = "scan_train_bmr_lr"
            load_dir = root_dir + "results/scan_train_bmr_lr/"
            dirpath = root_dir + "results/scan_train_bmr_lr/"
            l1, l2 = "rlr", "lr"
            train_params = get_binarymr_training_params(highseed=20)
            # lrs1 = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
            # lrs2 = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
            lr1_name = "r_lr"
            lr2_name = "lr"
        else:
            raise None

        for d in datasets:
            best_avperf = 0.
            bestlr1 = 0.
            bestlr2 = 0.
            # Load all parameter sets simulated
            files = listdir(dirpath + "individual/")
            files = [f for f in files if netstring in f and d in f]
            params = [[float(search("{}_(.*?).pickle".format(l1), f).group(1)),
                       float(search("{}_(.*?)_{}".format(l2, l1), f).group(1))] for f in files]
            for lrs in params:
                train_params[lr1_name] = lrs[0]
                train_params[lr2_name] = lrs[1]
                avperf = 0.
                for n in networks:
                    avperf += torch_mean(get_performances(prog_name=prog_name, load_info=load_dir, nr_hidden=n,
                                                          dataset=d, train_params=train_params,
                                                          performance_type="validation_performance")).item()
                avperf = avperf / float(len(networks))
                if avperf > best_avperf:
                    bestlr1 = lrs[0]
                    bestlr2 = lrs[1]
                    best_avperf = avperf
            print(prog_name, d, bestlr1, bestlr2, best_avperf)

            # # Update the current set of all best parameters (if exists)
            # filepath = dirpath + "bestparams.pickle"
            # save_best_param(filepath, d, str(n), bestlr)
            #
            # if verbose:
            #     print("sg", d, n, bestlr, max(mean_perfs))

    # For all archis:
    """
    scan_train_blr_wlr TASKS2D 0.01 0.01 93.81676419576009
    """


def prepare_scan_simulation(prog_params, early_stopping, save_params, lrs, debug):
    prog_params["validate_performance"] = True
    prog_params["test_performance"] = False
    if debug:
        save_params["save_performance"] = False

    if lrs is None:
        lrs = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]

    if early_stopping:
        spes_add = "_es"
    else:
        spes_add = ""

    return prog_params, lrs, spes_add, save_params


def scan_train_lrs(prog_params, train_params, save_params, lrs=None, verbose=False, debug=False,
                   result_dir="../../results/tmp"):

    # Prepare simulations
    prog_params, lrs, spes_add, save_params = prepare_scan_simulation(
        prog_params, train_params["early_stopping"], save_params, lrs, debug
    )

    # Get training and validation data
    train_data, validation_data = get_all_data(dataset=prog_params["dataset"], model_type=prog_params["model_type"],
                                               debug=debug)

    # Try the different learning rates with the desired training conditions
    for lr in lrs:
        train_params["lr"] = lr
        save_params["save_name"] = "network_{}_{}_{}_{}{}_lr_{}".format(
            prog_params["nr_hidden"], prog_params["dataset"], train_params["loss_function"],
            train_params["readout_function"], spes_add, lr
        )

        get_train_eval_model(result_dir=result_dir, prog_params=prog_params, save_params=save_params,
                             train_params=train_params, train_data=train_data, validation_data=validation_data,
                             verbose=verbose)


def scan_train_bw_lrs(prog_params, train_params, save_params, lrs=None, verbose=False, debug=False, root_dir="../../"):

    # Some initializations
    prog_params["name"] = "scan_train_bw_lrs"
    prog_params["model_type"] = "biaslearner"
    prog_params["training_type"] = "train_bw"
    prog_params["model_getter_type"] = "random_biaslearner"
    prog_params["model_arg"] = {"nr_hidden": prog_params["nr_hidden"],
                                "nr_tasks": get_number_classes(prog_params["dataset"])}

    result_dir = root_dir + "results/scan_trainbw_lr/individual/"

    scan_train_lrs(prog_params=prog_params, train_params=train_params, save_params=save_params, lrs=lrs,
                   verbose=verbose, debug=debug, result_dir=result_dir)


def scan_train_blr_wlr(prog_params, train_params, save_params, b_lr, lrs=None, verbose=False, debug=False,
                       root_dir="../../"):

    # Some initializations
    prog_params["name"] = "scan_train_bw_lrs"
    prog_params["model_type"] = "biaslearner"
    prog_params["training_type"] = "train_b_w"
    prog_params["model_getter_type"] = "random_biaslearner"
    prog_params["model_arg"] = {"nr_hidden": prog_params["nr_hidden"],
                                "nr_tasks": get_number_classes(prog_params["dataset"])}

    result_dir = root_dir + "results/scan_train_blr_wlr/individual/"

    # Prepare simulations
    train_params["b_lr"] = b_lr
    prog_params, lrs, spes_add, save_params = prepare_scan_simulation(
        prog_params, train_params["early_stopping"], save_params, lrs, debug
    )

    # Get training and validation data
    train_data, validation_data = get_all_data(dataset=prog_params["dataset"], model_type=prog_params["model_type"],
                                               debug=debug)

    # Try the different learning rates with the desired training conditions
    for lr in lrs:
        train_params["lr"] = lr
        save_params["save_name"] = "network_{}_{}_{}_{}{}_wlr_{}_blr_{}".format(
            prog_params["nr_hidden"], prog_params["dataset"], train_params["loss_function"],
            train_params["readout_function"], spes_add, lr, train_params["b_lr"]
        )

        get_train_eval_model(result_dir=result_dir, prog_params=prog_params, save_params=save_params,
                             train_params=train_params, train_data=train_data, validation_data=validation_data,
                             verbose=verbose)


def scan_train_glr_xwlr(prog_params, train_params, save_params, g_lr, lrs=None, verbose=False, debug=False,
                        root_dir="../../"):

    # Some initializations
    prog_params["name"] = "scan_train_x_w_lrs"
    prog_params["model_type"] = "xshiftlearner"
    prog_params["training_type"] = "train_x_w"
    prog_params["model_getter_type"] = "random_xshiftlearner"
    prog_params["model_arg"] = {"nr_hidden": prog_params["nr_hidden"],
                                "nr_tasks": get_number_classes(prog_params["dataset"])}

    result_dir = root_dir + "results/scan_train_xlr_wlr/individual/"

    # Prepare simulations
    train_params["g_lr"] = g_lr
    prog_params, lrs, spes_add, save_params = prepare_scan_simulation(
        prog_params, train_params["early_stopping"], save_params, lrs, debug
    )

    # Get training and validation data
    train_data, validation_data = get_all_data(dataset=prog_params["dataset"], model_type=prog_params["model_type"],
                                               debug=debug)

    # Try the different learning rates with the desired training conditions
    for lr in lrs:
        train_params["lr"] = lr
        save_params["save_name"] = "network_{}_{}_{}_{}{}_wlr_{}_glr_{}".format(
            prog_params["nr_hidden"], prog_params["dataset"], train_params["loss_function"],
            train_params["readout_function"], spes_add, lr, train_params["g_lr"]
        )

        get_train_eval_model(result_dir=result_dir, prog_params=prog_params, save_params=save_params,
                             train_params=train_params, train_data=train_data, validation_data=validation_data,
                             verbose=verbose)


def scan_train_glr_bwlr(prog_params, train_params, save_params, g_lr, lrs=None, verbose=False, debug=False,
                        root_dir="../../"):

    # Some initializations
    prog_params["name"] = "scan_train_g_bw_lrs"
    prog_params["model_type"] = "gainlearner"
    prog_params["training_type"] = "train_g_bw"
    prog_params["model_getter_type"] = "random_gainlearner"
    prog_params["model_arg"] = {"nr_hidden": prog_params["nr_hidden"],
                                "nr_tasks": get_number_classes(prog_params["dataset"])}

    result_dir = root_dir + "results/scan_train_glr_bwlr/individual/"

    # Prepare simulations
    train_params["g_lr"] = g_lr
    prog_params, lrs, spes_add, save_params = prepare_scan_simulation(
        prog_params, train_params["early_stopping"], save_params, lrs, debug
    )

    # Get training and validation data
    train_data, validation_data = get_all_data(dataset=prog_params["dataset"], model_type=prog_params["model_type"],
                                               debug=debug)

    # Try the different learning rates with the desired training conditions
    for lr in lrs:
        train_params["lr"] = lr
        save_params["save_name"] = "network_{}_{}_{}_{}{}_wlr_{}_glr_{}".format(
            prog_params["nr_hidden"], prog_params["dataset"], train_params["loss_function"],
            train_params["readout_function"], spes_add, lr, train_params["g_lr"]
        )

        get_train_eval_model(result_dir=result_dir, prog_params=prog_params, save_params=save_params,
                             train_params=train_params, train_data=train_data, validation_data=validation_data,
                             verbose=verbose)


def scan_train_bglr_wlr(prog_params, train_params, save_params, bg_lr, lrs=None, verbose=False, debug=False,
                        root_dir="../../"):

    # Some initializations
    prog_params["name"] = "scan_train_bg_w_lrs"
    prog_params["model_type"] = "bglearner"
    prog_params["training_type"] = "train_bg_w"
    prog_params["model_getter_type"] = "random_bglearner"
    prog_params["model_arg"] = {"nr_hidden": prog_params["nr_hidden"],
                                "nr_tasks": get_number_classes(prog_params["dataset"])}

    result_dir = root_dir + "results/scan_train_bglr_wlr/individual/"

    # Prepare simulations
    train_params["b_lr"] = bg_lr
    train_params["g_lr"] = bg_lr
    prog_params, lrs, spes_add, save_params = prepare_scan_simulation(
        prog_params, train_params["early_stopping"], save_params, lrs, debug
    )

    # Get training and validation data
    train_data, validation_data = get_all_data(dataset=prog_params["dataset"], model_type=prog_params["model_type"],
                                               debug=debug)

    # Try the different learning rates with the desired training conditions
    for lr in lrs:
        train_params["lr"] = lr
        save_params["save_name"] = "network_{}_{}_{}_{}{}_wlr_{}_bglr_{}".format(
            prog_params["nr_hidden"], prog_params["dataset"], train_params["loss_function"],
            train_params["readout_function"], spes_add, lr, train_params["b_lr"]
        )

        get_train_eval_model(result_dir=result_dir, prog_params=prog_params, save_params=save_params,
                             train_params=train_params, train_data=train_data, validation_data=validation_data,
                             verbose=verbose)


def scan_train_sg_lrs(prog_params, train_params, save_params, lrs=None, nrseeds=1, verbose=False, debug=False,
                      root_dir="../../", folder="scan_train_sg_lr"):

    # Some initializations
    prog_params["name"] = "scan_train_sg_lrs"
    prog_params["model_type"] = "biaslearner"
    prog_params["training_type"] = "train_bw"
    prog_params["model_getter_type"] = "random_biaslearner"
    prog_params["model_arg"] = {"nr_hidden": prog_params["nr_hidden"], "nr_tasks": 1}
    result_dir = root_dir + "results/{}/individual/".format(folder)

    # Prepare simulations
    prog_params, lrs, spes_add, save_params = prepare_scan_simulation(prog_params, train_params["early_stopping"],
                                                                      save_params, lrs, debug)

    # Try the different learning rates with the desired training conditions
    nrc = get_number_classes(prog_params["dataset"])
    for lr in lrs:
        train_params["lr"] = lr
        train_params["b_lr"] = lr
        for test_class in prog_params["test_classes"]:
            train_params["seeds"] = [test_class + i * nrc for i in range(nrseeds)]
            save_params["save_name"] = "network_{}_{}_{}_{}{}_lr_{}_testclass_{}".format(
                prog_params["nr_hidden"], prog_params["dataset"], train_params["loss_function"],
                train_params["readout_function"], spes_add, lr, test_class
            )

            # Get training and validation data
            train_data, validation_data = get_singular_data(class_id=test_class, dataset=prog_params["dataset"],
                                                            debug=debug, root_dir=root_dir)

            get_train_eval_model(result_dir=result_dir, prog_params=prog_params, save_params=save_params,
                                 train_params=train_params, train_data=train_data, validation_data=validation_data,
                                 verbose=verbose)


def scan_train_mr_lrs(prog_params, train_params, save_params, lrs=None, verbose=False, debug=False,
                      root_dir="../../"):

    # Some initializations
    prog_params["name"] = "scan_train_mr_lrs"
    prog_params["model_type"] = "multireadout"
    prog_params["training_type"] = "train_multireadout"
    prog_params["model_getter_type"] = "random_multireadout"
    prog_params["model_arg"] = {"nr_hidden": prog_params["nr_hidden"],
                                "nr_readouts": get_number_classes(prog_params["dataset"])}
    result_dir = root_dir + "results/scan_train_mr_lr/individual/"

    # Simulate parameter scan
    scan_train_lrs(prog_params=prog_params, train_params=train_params, save_params=save_params, lrs=lrs,
                   verbose=verbose, debug=debug, result_dir=result_dir)


def scan_train_bmr_lrs(prog_params, train_params, save_params, lrs, r_lr, verbose=False, debug=False,
                       root_dir="../../"):

    # Some initializations
    prog_params["name"] = "scan_train_bmr_lrs"
    prog_params["model_type"] = "binarymr"
    prog_params["training_type"] = "train_binarymr"
    prog_params["model_getter_type"] = "random_multireadout"
    prog_params["model_arg"] = {"nr_hidden": prog_params["nr_hidden"],
                                "nr_readouts": get_number_classes(prog_params["dataset"])}
    result_dir = root_dir + "results/scan_train_bmr_lr/individual/"

    # Prepare simulations
    train_params["r_lr"] = r_lr
    prog_params, lrs, spes_add, save_params = prepare_scan_simulation(
        prog_params, train_params["early_stopping"], save_params, lrs, debug
    )

    # Get training and validation data
    train_data, validation_data = get_all_data(dataset=prog_params["dataset"], model_type=prog_params["model_type"],
                                               debug=debug)

    # Try the different learning rates with the desired training conditions
    for lr in lrs:
        train_params["lr"] = lr
        save_params["save_name"] = "network_{}_{}_{}_{}{}_lr_{}_rlr_{}".format(
            prog_params["nr_hidden"], prog_params["dataset"], train_params["loss_function"],
            train_params["readout_function"], spes_add, lr, train_params["r_lr"]
        )

        get_train_eval_model(result_dir=result_dir, prog_params=prog_params, save_params=save_params,
                             train_params=train_params, train_data=train_data, validation_data=validation_data,
                             verbose=verbose)


def scan_l1o_transfer_lrs(prog_params, train_params, save_params, lrs, verbose, debug, result_dir, root_dir):

    # Prepare simulations
    prog_params, lrs, spes_add, save_params = prepare_scan_simulation(
        prog_params, train_params["early_stopping"], save_params, lrs, debug
    )

    # Define some weight loading parameters
    base_dir = root_dir + "results/leave_1_out/" + prog_params["dataset"] + "/"
    prog_params["load_weights"] = True
    prog_params["load_weight_type"] = "train_l1o"
    prog_params["load_weight_dir"] = base_dir + "train/final_weights/"
    prog_params["validate_performance"] = True
    prog_params["test_performance"] = False

    # Try the different learning rates with the desired training conditions
    for lr in lrs:
        train_params["lr"] = lr
        prog_params["model_type_save_name"] = "{}_{}_lr_{}".format(prog_params["model_type"],
                                                                   prog_params["dataset"],
                                                                   lr)
        l1o_get_train_eval(
            transfering=True, test_classes=prog_params["test_classes"], prog_params=prog_params,
            train_params=train_params, save_params=save_params, result_dir=result_dir, verbose=verbose, debug=debug
        )


def scan_l10_transfer_b_lrs(prog_params, train_params, save_params, lrs=None, verbose=False, debug=False,
                            root_dir="../../"):

    # Some initializations
    prog_params["name"] = "scan_l10_transfer_b_lrs"
    prog_params["model_type"] = "biaslearner"
    prog_params["model_type_load_name"] = prog_params["model_type"]
    prog_params["training_type"] = "transfer_b"
    prog_params["model_getter_type"] = "loaded_w_random_b_biaslearner"
    prog_params["model_arg"] = {"nr_hidden": prog_params["nr_hidden"],
                                "nr_tasks": 1}
    result_dir = root_dir + "results/scan_transfer_b_lr/individual/"

    # Simulate parameter scan
    scan_l1o_transfer_lrs(prog_params=prog_params, train_params=train_params, save_params=save_params, lrs=lrs,
                          verbose=verbose, debug=debug, result_dir=result_dir, root_dir=root_dir)


def scan_l1o_transfer_mr_lrs(prog_params, train_params, save_params, lrs=None, verbose=False, debug=False,
                             root_dir="../../"):

    # Some initializations
    prog_params["name"] = "scan_l10_transfer_mr_lrs"
    prog_params["model_type"] = "multireadout"
    prog_params["model_type_load_name"] = prog_params["model_type"]
    prog_params["training_type"] = "transfer_multireadout"
    prog_params["model_getter_type"] = "load_w_random_readout_multireadout"
    prog_params["model_arg"] = {"nr_hidden": prog_params["nr_hidden"],
                                "tot_nr_tasks": get_number_classes(prog_params["dataset"])}
    result_dir = root_dir + "results/scan_transfer_mr_lr/individual/"

    # Simulate parameter scan
    scan_l1o_transfer_lrs(prog_params=prog_params, train_params=train_params, save_params=save_params, lrs=lrs,
                          verbose=verbose, debug=debug, result_dir=result_dir, root_dir=root_dir)


def scan_l1o_transfer_bmr_lrs(prog_params, train_params, save_params, lrs=None, verbose=False, debug=False,
                              root_dir="../../"):

    # Some initializations
    prog_params["name"] = "scan_l10_transfer_bmr_lrs"
    prog_params["model_type"] = "binarymr"
    prog_params["model_type_load_name"] = prog_params["model_type"]
    prog_params["training_type"] = "transfer_binarymr"
    prog_params["model_getter_type"] = "load_w_random_readout_multireadout"
    prog_params["model_arg"] = {"nr_hidden": prog_params["nr_hidden"],
                                "tot_nr_tasks": get_number_classes(prog_params["dataset"])}
    result_dir = root_dir + "results/scan_transfer_bmr_lr/individual/"

    # Simulate parameter scan
    scan_l1o_transfer_lrs(prog_params=prog_params, train_params=train_params, save_params=save_params, lrs=lrs,
                          verbose=verbose, debug=debug, result_dir=result_dir, root_dir=root_dir)


def scan_params(scantype, nrhiddens, readout_functions=None, datasets=None, lrs=None, nr_epochs=None, batch_sizes=None,
                loss_functions=None, early_stopping=None, seeds=None, highseed=None, recompute=False, saving=True,
                verbose=False, debug=False, root_dir="../../", testclasses=None, b_lr=None, g_lr=None, bg_lr=None,
                r_lr=None):
    assert scantype in ["train_bw", "train_b_w", "train_g_xw", "train_g_bw", "train_bg_w", "train_sg",
                        "train_sg_bigbatch", "train_mr", "train_bmr", "transfer_b_l1o", "transfer_mr_l1o",
                        "transfer_bmr_l1o"]
    if datasets is None:
        datasets = ["MNIST", "QMNIST", "EMNIST", "EMNIST_letters", "EMNIST_bymerge", "K49", "CIFAR100", "TASKS2D"]
    if nr_epochs is None:
        nr_epochs = [50]
    if batch_sizes is None:
        batch_sizes = [200]
    if loss_functions is None:
        loss_functions = ["mse"]
    if early_stopping is None:
        early_stoppings = [True, False]
    else:
        early_stoppings = [early_stopping]
    if highseed is None:
        if scantype in ["train_bw", "train_b_w", "train_g_xw", "train_g_bw", "train_bg_w", "train_mr", "train_bmr"]:
            highseed = 20
        elif scantype in ["transfer_b_l1o", "transfer_mr_l1o", "transfer_bmr_l1o", "train_sg", "train_sg_bigbatch"]:
            highseed = 3

        else:
            raise ValueError(scantype)
    if readout_functions is None:
        if scantype in ["train_bw", "train_b_w", "train_g_xw", "train_g_bw", "train_bg_w", "transfer_b_l1o", "train_sg",
                        "train_sg_bigbatch", "train_bmr", "transfer_bmr_l1o"]:
            readout_functions = ["tanh"]
        elif scantype in ["train_mr", "transfer_mr_l1o"]:
            readout_functions = ["softmax"]
        else:
            raise ValueError(scantype)

    if testclasses is None:
        prog_params = {}
    else:
        prog_params = {"test_classes": testclasses}
    train_params = {"highseed": highseed, "seeds": seeds}
    save_params = {"save_performance": saving,
                   "save_initweights": False,
                   "save_finalweights": False,
                   "recompute": recompute}
    for nrhidden in nrhiddens:
        prog_params["nr_hidden"] = nrhidden
        for readout_function in readout_functions:
            train_params["readout_function"] = readout_function
            for dataset in datasets:
                prog_params["dataset"] = dataset
                for nr_epoch in nr_epochs:
                    train_params["nr_epochs"] = nr_epoch
                    for batch_size in batch_sizes:
                        train_params["batch_size"] = batch_size
                        for loss_function in loss_functions:
                            train_params["loss_function"] = loss_function
                            for ess in early_stoppings:
                                train_params["early_stopping"] = ess
                                if scantype == "train_bw":
                                    scan_train_bw_lrs(prog_params=prog_params, train_params=train_params,
                                                      save_params=save_params, lrs=lrs, verbose=verbose, debug=debug,
                                                      root_dir=root_dir)
                                elif scantype == "train_b_w":
                                    scan_train_blr_wlr(prog_params=prog_params, train_params=train_params,
                                                       save_params=save_params, lrs=lrs, verbose=verbose, debug=debug,
                                                       root_dir=root_dir, b_lr=b_lr)
                                elif scantype == "train_g_xw":
                                    scan_train_glr_xwlr(prog_params=prog_params, train_params=train_params,
                                                        save_params=save_params, lrs=lrs, verbose=verbose, debug=debug,
                                                        root_dir=root_dir, g_lr=g_lr)
                                elif scantype == "train_g_bw":
                                    scan_train_glr_bwlr(prog_params=prog_params, train_params=train_params,
                                                        save_params=save_params, lrs=lrs, verbose=verbose, debug=debug,
                                                        root_dir=root_dir, g_lr=g_lr)
                                elif scantype == "train_bg_w":
                                    scan_train_bglr_wlr(prog_params=prog_params, train_params=train_params,
                                                        save_params=save_params, lrs=lrs, verbose=verbose, debug=debug,
                                                        root_dir=root_dir, bg_lr=bg_lr)
                                elif scantype == "train_sg":
                                    scan_train_sg_lrs(prog_params=prog_params, train_params=train_params,
                                                      save_params=save_params, lrs=lrs, verbose=verbose, debug=debug,
                                                      root_dir=root_dir)
                                elif scantype == "train_sg_bigbatch":
                                    scan_train_sg_lrs(prog_params=prog_params, train_params=train_params,
                                                      save_params=save_params, lrs=lrs, verbose=verbose, debug=debug,
                                                      root_dir=root_dir, folder="scan_train_sg_bigbatch_lr")
                                elif scantype == "train_mr":
                                    scan_train_mr_lrs(prog_params=prog_params, train_params=train_params,
                                                      save_params=save_params, lrs=lrs, verbose=verbose, debug=debug,
                                                      root_dir=root_dir)
                                elif scantype == "train_bmr":
                                    scan_train_bmr_lrs(prog_params=prog_params, train_params=train_params,
                                                       save_params=save_params, lrs=lrs, r_lr=r_lr, verbose=verbose,
                                                       debug=debug, root_dir=root_dir)
                                elif scantype == "transfer_b_l1o":
                                    scan_l10_transfer_b_lrs(prog_params=prog_params, train_params=train_params,
                                                            save_params=save_params, lrs=lrs, verbose=verbose,
                                                            debug=debug, root_dir=root_dir)
                                elif scantype == "transfer_mr_l1o":
                                    scan_l1o_transfer_mr_lrs(prog_params=prog_params, train_params=train_params,
                                                             save_params=save_params, lrs=lrs, verbose=verbose,
                                                             debug=debug, root_dir=root_dir)
                                elif scantype == "transfer_bmr_l1o":
                                    scan_l1o_transfer_bmr_lrs(prog_params=prog_params, train_params=train_params,
                                                              save_params=save_params, lrs=lrs, verbose=verbose,
                                                              debug=debug, root_dir=root_dir)


if __name__ == '__main__':
    usage_error = False

    # Test some scripts
    if len(sys.argv) == 1:
        net = [[100, 25]]
        ds = ["EMNIST_bymerge"]
        ls = [0.001]
        es = True
        rc = True
        tcs = [1]
        bl = 0.03
        # scan_params(scantype="train_g_bw", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
        #             verbose=True, g_lr=bl, saving=False)
        # scan_params(scantype="transfer_mr_l1o", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
        #             verbose=True, testclasses=tcs, saving=False)
        # scan_params(scantype="train_sg", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
        #             verbose=True, testclasses=tcs, saving=False)
        # scan_params(scantype="train_sg_bigbatch", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
        #             verbose=True, testclasses=tcs, saving=False, batch_sizes=[4096], nr_epochs=[1000])
        scan_params(scantype="train_g_xw", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
                    verbose=True, testclasses=tcs, saving=False, g_lr=bl)
        # scan_params(scantype="train_g_bw", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
        #             verbose=True, testclasses=tcs, saving=False, g_lr=bl)
        # scan_params(scantype="train_bg_w", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
        #             verbose=True, testclasses=tcs, saving=False, bg_lr=bl, debug=False)
        # scan_params(scantype="transfer_b_l1o", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
        #             verbose=True, testclasses=tcs, saving=False)
        # scan_params(scantype="train_mr", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
        #             verbose=True, testclasses=tcs, saving=False, readout_functions=["sigmoid"])
        # scan_params(scantype="train_bmr", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
        #             verbose=True, testclasses=tcs, saving=False, r_lr=0.01)
        # scan_params(scantype="transfer_bmr_l1o", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
        #             verbose=True, testclasses=tcs, saving=False)
        # scan_params(scantype="train_sg", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
        #             verbose=True, testclasses=tcs, saving=False)
        # save_all_best_params(verbose=True)
        # save_best_params("b_w", "K49", str([100]), verbose=True)
        # save_avbest_params()
        # save_avbest_paramsk49()
        # save_avbest_paramst2d()
        # traintypes = ["b_w", "g_bw", "bg_w", "bmr"]
        # datasets = ["K49"]
        # networks = [[25], [100], [500], [25, 25], [100, 100], [500, 500], [25, 25, 25], [100, 100, 100],
        #             [500, 500, 500]]
        # for tt in traintypes:
        #     for d in datasets:
        #         for n in networks:
        #             save_best_params(tt, d, n)

    # Call in an iterated manner in a big loop
    elif len(sys.argv) >= 6:
        if str(sys.argv[2]) == "linear":
            net = []
        else:
            net = [list(map(int, sys.argv[2].split(',')))]
        ds = [str(sys.argv[3])]
        es = str(sys.argv[4]) == "early_stopping"
        ls = [float(sys.argv[5])]
        rc = False
        if sys.argv[1] == "train_bw_lr":
            scan_params(scantype="train_bw", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
                        verbose=True)
        elif sys.argv[1] == "train_b_w_lr":
            bl = float(sys.argv[6])
            scan_params(scantype="train_b_w", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
                        verbose=True, b_lr=bl)
        elif sys.argv[1] == "train_g_bw_lr":
            gl = float(sys.argv[6])
            scan_params(scantype="train_g_bw", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
                        verbose=True, g_lr=gl)
        elif sys.argv[1] == "train_bg_w_lr":
            bgl = float(sys.argv[6])
            scan_params(scantype="train_bg_w", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
                        verbose=True, bg_lr=bgl)
        elif sys.argv[1] == "train_bw_hardtanh_lr":
            scan_params(scantype="train_bw", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
                        readout_functions=["hardtanh"], verbose=True)
        elif sys.argv[1] == "train_sg_lr":
            tcs = [int(sys.argv[6])]
            scan_params(scantype="train_sg", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
                        verbose=True, testclasses=tcs)
        elif sys.argv[1] == "train_sg_lr_bigbatch":
            tcs = [int(sys.argv[6])]
            scan_params(scantype="train_sg_bigbatch", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es,
                        recompute=rc, verbose=True, testclasses=tcs, batch_sizes=[4096], nr_epochs=[1000])
        elif sys.argv[1] == "train_mr_lr":
            scan_params(scantype="train_mr", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
                        verbose=True)
        elif sys.argv[1] == "train_bmr_lr":
            rl = float(sys.argv[6])
            scan_params(scantype="train_bmr", nrhiddens=net, datasets=ds, lrs=ls, r_lr=rl, early_stopping=es,
                        recompute=rc, verbose=True)
        elif sys.argv[1] == "train_bmr_lr_2d":
            rl = float(sys.argv[6])
            scan_params(scantype="train_bmr", nrhiddens=net, datasets=ds, lrs=ls, r_lr=rl, early_stopping=es,
                        recompute=rc, verbose=True, batch_sizes=[4096], nr_epochs=[1000])
        elif sys.argv[1] == "transfer_b_l1o_lr":
            tcs = [int(sys.argv[6])]
            scan_params(scantype="transfer_b_l1o", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es, recompute=rc,
                        verbose=True, testclasses=tcs)
        elif sys.argv[1] == "transfer_mr_l1o_lr":
            tcs = [int(sys.argv[6])]
            scan_params(scantype="transfer_mr_l1o", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es,
                        recompute=rc, verbose=True, testclasses=tcs)
        elif sys.argv[1] == "transfer_bmr_l1o_lr":
            tcs = [int(sys.argv[6])]
            scan_params(scantype="transfer_bmr_l1o", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=es,
                        recompute=rc, verbose=True, testclasses=tcs)
        else:
            usage_error = True
    else:
        usage_error = True

    if usage_error:
        print("Error: couldn't resolve desired options\nusage: python find_params [option]")
