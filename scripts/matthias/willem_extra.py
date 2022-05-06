import sys
import matplotlib.pyplot as plt
from numpy import arange as np_arange
from torch import mean as torch_mean
from torch import std as torch_std
from plot_helper import get_performances
from multireadout_helper import get_multireadout_training_params
from biaslearning_helper import get_biaslearner_training_params
from leave1out import run_leave1out
from willem_helper import get_willem_weight_load_path


def transfer_b_willem_l1o(nrhidden, dataset, weight_type, test_classes=None, train_params=None, save_performance=True,
                          save_finalweights=True, recompute=False, verbose=False, debug=False,
                          root_dir="../../", readout="tanh"):

    if readout != "tanh":
        save_add_on = "_{}".format(readout)
    else:
        save_add_on = ""

    # Load path for desired set of weights
    w_load_path = get_willem_weight_load_path(weight_type=weight_type, dataset=dataset,
                                              nr_hidden=nrhidden, root_dir=root_dir)

    # Program features
    prog_params = {
        "nr_hidden": nrhidden,
        "dataset": dataset,
        "model_type": "biaslearner",
        "model_type_save_name": weight_type + save_add_on,
        "model_getter_type": "load_willem_biaslearner",
        "training_type": "transfer_b",
        "model_arg": {"nr_hidden": nrhidden, "nr_tasks": 1, "load_path": w_load_path}
    }

    # Training features
    if train_params is None:
        train_params = {"highseed": 3}
    elif "highseed" not in train_params.keys():
        train_params["highseed"] = 3
    train_params = get_biaslearner_training_params(readout_function=readout, new_train_params=train_params,
                                                   transfering=True)

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": False,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute}

    run_leave1out(transfering=True, prog_params=prog_params, train_params=train_params, save_params=save_params,
                  test_classes=test_classes, verbose=verbose, debug=debug, root_dir=root_dir)


def plot_willem_tansfer(saving=False, root_dir="../../"):
    """

    Parameters
    ----------
    saving: Whether to save the plot to file or just show it
    root_dir: Root directory of Biasadaptation project
    -------
    """

    dataset = "EMNIST_bymerge"
    nr_hiddens = [[10], [25], [50], [100], [250], [500]]
    traintypes = ["train_sg_full", "train_bw_l1o", "transfer_b_l1o", "transfer_b_scd_l1o", "transfer_b_pmdd_l1o",
                  "transfer_b_scd_hardtanh_l1o", "transfer_b_pmdd_hardtanh_l1o", "train_mr_l1o", "transfer_mr_l1o"]
    legends = ["Singular approach", "Train weights + biases", "Transfer learn biases", "Transfer biases on SCD",
               "Transfer biases on PMDD", "Transfer biases on SCD with hardtanh",
               "Transfer biases on PMDD with hardtanh", "Train multireadout", "Transfer learn readout"]

    load_dirs = ["{}results/train_full_dataset/".format(root_dir)] + 8 * ["{}results/leave_1_out/".format(root_dir)]
    train_params = [get_biaslearner_training_params(highseed=3),
                    get_biaslearner_training_params(),
                    get_biaslearner_training_params(transfering=True),
                    get_biaslearner_training_params(transfering=True),
                    get_biaslearner_training_params(transfering=True),
                    get_biaslearner_training_params(transfering=True),
                    get_biaslearner_training_params(transfering=True),
                    get_multireadout_training_params(),
                    get_multireadout_training_params()]

    fig = plt.figure(figsize=(16, 6))
    nr_bars = len(traintypes)
    width = 1. / (nr_bars + 1)
    x_offset = [i * width - 0.5 for i in range(1, nr_bars + 1)]
    x = np_arange(len(nr_hiddens))

    for tt in range(len(traintypes)):
        y = []
        y_err = []
        for nh in range(len(nr_hiddens)):
            performances = get_performances(prog_name=traintypes[tt], load_info=load_dirs[tt],
                                            nr_hidden=nr_hiddens[nh], dataset=dataset,
                                            train_params=train_params[tt], performance_type="test_performance")
            y += [torch_mean(performances)]
            y_err += [torch_std(performances)]
        plt.bar(x + x_offset[tt], y, width, yerr=y_err, label=legends[tt], zorder=3)
    plt.title("Leave one out transfer learning performances on EMNIST bymerge")
    plt.xlabel('Network Architecture')
    plt.xticks(x, nr_hiddens)
    plt.grid(axis='y', alpha=0.4, zorder=0)
    plt.ylim([0, 100])
    plt.ylabel("Test Performance [%]")
    plt.legend(loc="lower right", framealpha=1)
    if saving:
        plt.savefig("{}plots/leave_1_out/svg/willem_transfer_bymerge.svg".format(root_dir))
        plt.savefig("{}plots/leave_1_out/willem_transfer_bymerge.png".format(root_dir))
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':

    # Test some specific script
    if len(sys.argv) == 1:
        plot_willem_tansfer()

    # Call in a big loop to simulate all the desired configurations
    elif len(sys.argv) == 5:
        if str(sys.argv[2]) == "linear":
            net = []
        else:
            net = list(map(int, sys.argv[2].split(',')))
        ds = str(sys.argv[3])
        tcs = [int(sys.argv[4])]
        rc = True
        vb = True
        # if sys.argv[1] == "train_hardtanh_bw":
        #     train_hardtanh_bw_l1o(nrhidden=net, dataset=ds, test_classes=tcs, recompute=rc, verbose=vb)
        # else:
        #     raise ValueError(sys.argv[1])

    elif len(sys.argv) == 6:
        if str(sys.argv[2]) == "linear":
            net = []
        else:
            net = list(map(int, sys.argv[2].split(',')))
        ds = str(sys.argv[3])
        tcs = [int(sys.argv[4])]
        wt = str(sys.argv[5])
        rc = True
        vb = True
        if sys.argv[1] == "transfer_b_willem":
            transfer_b_willem_l1o(nrhidden=net, dataset=ds, weight_type=wt, test_classes=tcs, recompute=rc, verbose=vb)
        elif sys.argv[1] == "transfer_b_willem_hardtanh":
            transfer_b_willem_l1o(nrhidden=net, dataset=ds, weight_type=wt, test_classes=tcs, recompute=rc, verbose=vb,
                                  readout="hardtanh")
        else:
            raise ValueError(sys.argv[1])
