from pickle import load as pickleload
from sklearn.decomposition import SparseCoder
from biasadaptation.utils.utils import differences_numpy
from data_helper import get_dataset
import numpy as np
import copy
import warnings
from tasks_2d_helper import get_all_data_2dtasks
from torch.utils.data import DataLoader


def get_missing_scan_params():
    out = ""
    from biaslearning_helper import get_biaslearner_training_params
    from multireadout_helper import get_binarymr_training_params
    from plot_helper import get_dirs, get_performances
    nr_hiddens = [[25], [100], [500], [25, 25], [100, 100], [500, 500], [25, 25, 25], [100, 100, 100],
                  [500, 500, 500]]
    script_names = {"scan_train_blr_wlr": "train_b_w_lr", "scan_train_glr_bwlr": "train_g_bw_lr",
                    "scan_train_bglr_wlr": "train_bg_w_lr", "scan_train_bmr_lr": "train_bmr_lr"}
    log_pre = {"scan_train_blr_wlr": "scan_train_b_w", "scan_train_glr_bwlr": "scan_train_g_bw",
               "scan_train_bglr_wlr": "scan_train_bg_w", "scan_train_bmr_lr": "scan_train_bmr"}
    netname = {"[25]": "25", "[100]": "100", "[500]": "500", "[25, 25]": "25,25", "[100, 100]": "100,100",
               "[500, 500]": "500,500", "[25, 25, 25]": "25,25,25", "[100, 100, 100]": "100,100,100",
               "[500, 500, 500]": "500,500,500"}
    for dataset in ["EMNIST_bymerge", "CIFAR100"]:
        for prog_name in ["scan_train_blr_wlr", "scan_train_glr_bwlr", "scan_train_bglr_wlr", "scan_train_bmr_lr"]:
            if prog_name in ["scan_train_blr_wlr", "scan_train_glr_bwlr", "scan_train_bglr_wlr"]:
                subdir = prog_name
                lrs2 = [0.3, 0.2, 0.1, 0.06, 0.03, 0.02, 0.01, 0.006, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0001,
                        0.00003]
                lrs1 = [0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0002, 0.0001, 0.00006, 0.00003, 0.00002,
                        0.00001]
                train_params = get_biaslearner_training_params(highseed=20)
                lr1_name = "lr"
                lr2_name = "b_lr"
            elif prog_name == "scan_train_bmr_lr":
                subdir = "scan_train_bmr_lr"
                lrs1 = [0.1, 0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0002, 0.0001, 0.00006, 0.00003]
                lrs2 = [0.1, 0.03, 0.01, 0.003, 0.002, 0.001, 0.0006, 0.0003, 0.0002, 0.0001, 0.00006, 0.00003]
                train_params = get_binarymr_training_params(highseed=20)
                lr1_name = "lr"
                lr2_name = "r_lr"
            load_dir, _ = get_dirs("../../", subdir)
            for nh in nr_hiddens:
                for lr1 in lrs1:
                    train_params[lr1_name] = lr1
                    for lr2 in lrs2:
                        train_params[lr2_name] = lr2
                        try:
                            get_performances(prog_name=prog_name, load_info=load_dir, nr_hidden=nh, dataset=dataset,
                                             train_params=train_params, performance_type="validation_performance")
                        except FileNotFoundError:
                            out += "bash sbatch_8_inner.sh scan_params.py {} {} {} early_stopping {} {} {}_[{}]_{}_{}" \
                                   "_{}\n".format(script_names[prog_name], netname[str(nh)], dataset, lr1, lr2,
                                                  log_pre[prog_name], netname[str(nh)], dataset, lr1, lr2)
    print(out)


# get_missing_scan_params()

# import pickle
# file = open("../../results/scan_train_sg_lr/bestparams.pickle", 'rb')
# object_file = pickle.load(file)
# print(object_file)
# file.close()

train, valid = get_all_data_2dtasks(True, True)
tdtl = DataLoader(train, batch_size=2, shuffle=True)
for x in enumerate(tdtl):
    print(x)
