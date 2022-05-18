import sys
from data_helper import get_all_data, get_number_classes, get_singular_data
from train_helper import get_train_eval_model
from biaslearning_helper import get_biaslearner_training_params
from leave1out import get_l1o_datas
from multireadout_helper import get_multireadout_training_params, get_binarymr_training_params
from willem_helper import get_willem_weight_load_path


def get_full_datas(prog_params, train_params, debug):
    if train_params["early_stopping"]:
        train_data, validation_data = get_all_data(dataset=prog_params["dataset"], model_type=prog_params["model_type"],
                                                   debug=debug)
    else:
        train_data = get_all_data(dataset=prog_params["dataset"], model_type=prog_params["model_type"], splitting=False,
                                  debug=debug)
        validation_data = None
    test_data = get_all_data(dataset=prog_params["dataset"], model_type=prog_params["model_type"], splitting=False,
                             train=False, debug=debug)

    return train_data, validation_data, test_data


def run_full_dataset(result_dir, prog_params, train_params, save_params, verbose=False, veryverbose=False, debug=False):

    prog_params["validate_performance"] = train_params["early_stopping"]
    prog_params["test_performance"] = True

    # Get data
    train_data, validation_data, test_data = get_full_datas(prog_params, train_params, debug)

    # Create model, train it and evaluate performance
    get_train_eval_model(
        result_dir=result_dir, prog_params=prog_params, train_params=train_params, save_params=save_params,
        train_data=train_data, validation_data=validation_data, test_data=test_data, verbose=verbose,
        veryverbose=veryverbose
    )


def train_full_dataset(prog_params, train_params, save_params, verbose=False, veryverbose=False, debug=False,
                       root_dir="../../"):

    prog_params["nr_labels"] = get_number_classes(prog_params["dataset"])
    result_dir = root_dir + "results/train_full_dataset/" + prog_params["dataset"] + "/"
    save_params["save_name"] = "{}_{}".format(prog_params["model_type_save_name"], prog_params["nr_hidden"])

    run_full_dataset(result_dir=result_dir, prog_params=prog_params, train_params=train_params, save_params=save_params,
                     verbose=verbose, veryverbose=veryverbose, debug=debug)


def polish_b_full_dataset(nrhidden, dataset, train_params=None, save_performance=True, save_finalweights=True,
                          recompute=False, verbose=False, debug=False, root_dir="../../", train_classes=None):

    # Program features
    prog_params = {"nr_hidden": nrhidden,
                   "dataset": dataset,
                   "model_type": "biaslearner",
                   "model_type_save_name": "biaslearner_polished",
                   "model_getter_type": "load_path_biaslearner",
                   "training_type": "transfer_b",
                   "model_arg": {"nr_hidden": nrhidden},
                   "load_weights": True,
                   "load_weight_type": "train_full",
                   "load_weight_dir":  "{}results/train_full_dataset/{}/final_weights/".format(root_dir, dataset),
                   "model_type_load_name": "biaslearner"}

    # Training features
    train_params = get_biaslearner_training_params(new_train_params=train_params, blr=0.0006, transfering=True)

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": False,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute}

    result_dir = root_dir + "results/train_full_dataset/" + prog_params["dataset"] + "/"
    # prog_params["nr_labels"] = get_number_classes(prog_params["dataset"])
    if train_classes is None:
        train_classes = range(get_number_classes(dataset))
    for task in train_classes:
        train_params["tid0"] = task
        save_params["save_name"] = "biaslearner_polished_{}_task_{}".format(prog_params["nr_hidden"], task)
        prog_params["validate_performance"] = train_params["early_stopping"]
        prog_params["test_performance"] = True

        # Get data
        train_data, validation_data, test_data = get_l1o_datas(
            transfering=True, test_class=task, model_type=prog_params["model_type"],
            dataset=prog_params["dataset"], early_stopping=train_params["early_stopping"], debug=debug
        )

        # Create model, train it and evaluate performance
        get_train_eval_model(
            result_dir=result_dir, prog_params=prog_params, train_params=train_params, save_params=save_params,
            train_data=train_data, validation_data=validation_data, test_data=test_data, verbose=verbose
        )


def polish_bmr_full_dataset(nrhidden, dataset, train_params=None, save_performance=True, save_finalweights=True,
                            recompute=False, verbose=False, debug=False, root_dir="../../", train_classes=None):

    # Program features
    prog_params = {"nr_hidden": nrhidden,
                   "dataset": dataset,
                   "model_type": "binarymr",
                   "model_type_save_name": "binarymr_polished",
                   "model_getter_type": "load_path_multireadout",
                   "training_type": "transfer_bmr",
                   "model_arg": {"nr_hidden": nrhidden},
                   "load_weights": True,
                   "load_weight_type": "train_full",
                   "load_weight_dir":  "{}results/train_full_dataset/{}/final_weights/".format(root_dir, dataset),
                   "model_type_load_name": "binarymr"}

    # Training features
    train_params = get_binarymr_training_params(new_train_params=train_params)

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": False,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute}

    result_dir = root_dir + "results/train_full_dataset/" + prog_params["dataset"] + "/"
    # prog_params["nr_labels"] = get_number_classes(prog_params["dataset"])
    if train_classes is None:
        train_classes = range(get_number_classes(dataset))
    for task in train_classes:
        train_params["tid0"] = task
        save_params["save_name"] = "binarymr_polished_{}_task_{}".format(prog_params["nr_hidden"], task)
        prog_params["validate_performance"] = train_params["early_stopping"]
        prog_params["test_performance"] = True

        # Get data
        train_data, validation_data, test_data = get_l1o_datas(
            transfering=True, test_class=task, model_type=prog_params["model_type"],
            dataset=prog_params["dataset"], early_stopping=train_params["early_stopping"], debug=debug
        )

        # Create model, train it and evaluate performance
        get_train_eval_model(
            result_dir=result_dir, prog_params=prog_params, train_params=train_params, save_params=save_params,
            train_data=train_data, validation_data=validation_data, test_data=test_data, verbose=verbose
        )


def train_biaslearner_full_dataset(dataset, nrhidden, prog_params, train_type="train_bw", train_params=None,
                                   recompute=False, save_performance=True, save_initweights=True,
                                   save_finalweights=True, verbose=False, veryverbose=False, debug=False,
                                   root_dir="../../"):
    # Program features
    prog_params.update({"nr_hidden": nrhidden,
                        "dataset": dataset,
                        "training_type": train_type,
                        "model_arg": {"nr_hidden": nrhidden, "nr_tasks": get_number_classes(dataset)}})

    # Training features
    train_params = get_biaslearner_training_params(new_train_params=train_params, prog_params=prog_params)

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": save_initweights,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute}

    train_full_dataset(prog_params=prog_params, train_params=train_params, save_params=save_params,
                       verbose=verbose, veryverbose=veryverbose, debug=debug, root_dir=root_dir)


def train_b_w_full_dataset(dataset, nrhidden, train_params=None, recompute=False, save_performance=True,
                           save_initweights=True, save_finalweights=True, verbose=False, veryverbose=False, debug=False,
                           root_dir="../../"):
    prog_params = {"model_type": "biaslearner",
                   "model_type_save_name": "biaslearner",
                   "model_getter_type": "random_biaslearner"}
    train_biaslearner_full_dataset(dataset=dataset, nrhidden=nrhidden, prog_params=prog_params, train_type="train_b_w",
                                   train_params=train_params, recompute=recompute, save_performance=save_performance,
                                   save_initweights=save_initweights, save_finalweights=save_finalweights,
                                   verbose=verbose, veryverbose=veryverbose, debug=debug, root_dir=root_dir)


def train_g_bw_full_dataset(dataset, nrhidden, train_params=None, recompute=False, save_performance=True,
                            save_initweights=True, save_finalweights=True, verbose=False, veryverbose=False,
                            debug=False, root_dir="../../"):
    prog_params = {"model_type": "gainlearner",
                   "model_type_save_name": "gainlearner",
                   "model_getter_type": "random_gainlearner"}
    train_biaslearner_full_dataset(dataset=dataset, nrhidden=nrhidden, prog_params=prog_params, train_type="train_g_bw",
                                   train_params=train_params, recompute=recompute, save_performance=save_performance,
                                   save_initweights=save_initweights, save_finalweights=save_finalweights,
                                   verbose=verbose, veryverbose=veryverbose, debug=debug, root_dir=root_dir)


def train_g_xw_full_dataset(dataset, nrhidden, train_params=None, recompute=False, save_performance=True,
                            save_initweights=True, save_finalweights=True, verbose=False, veryverbose=False,
                            debug=False, root_dir="../../"):
    prog_params = {"model_type": "xshiftlearner",
                   "model_type_save_name": "xshiftlearner",
                   "model_getter_type": "random_xshiftlearner"}
    train_biaslearner_full_dataset(dataset=dataset, nrhidden=nrhidden, prog_params=prog_params, train_type="train_g_xw",
                                   train_params=train_params, recompute=recompute, save_performance=save_performance,
                                   save_initweights=save_initweights, save_finalweights=save_finalweights,
                                   verbose=verbose, veryverbose=veryverbose, debug=debug, root_dir=root_dir)


def train_bg_w_full_dataset(dataset, nrhidden, train_params=None, recompute=False, save_performance=True,
                            save_initweights=True, save_finalweights=True, verbose=False, veryverbose=False,
                            debug=False, root_dir="../../"):
    prog_params = {"model_type": "bglearner",
                   "model_type_save_name": "bglearner",
                   "model_getter_type": "random_bglearner"}
    train_biaslearner_full_dataset(dataset=dataset, nrhidden=nrhidden, prog_params=prog_params, train_type="train_bg_w",
                                   train_params=train_params, recompute=recompute, save_performance=save_performance,
                                   save_initweights=save_initweights, save_finalweights=save_finalweights,
                                   verbose=verbose, veryverbose=veryverbose, debug=debug, root_dir=root_dir)


def train_bw_willem_init_full_dataset(dataset, nrhidden, weight_type, train_params=None, recompute=False,
                                      save_performance=True, save_finalweights=True, verbose=False, debug=False,
                                      root_dir="../../"):

    # Load path for desired set of weights
    w_load_path = get_willem_weight_load_path(weight_type=weight_type, dataset=dataset,
                                              nr_hidden=nrhidden, root_dir=root_dir)

    # Program features
    prog_params = {
        "nr_hidden": nrhidden,
        "dataset": dataset,
        "model_type": "biaslearner",
        "model_type_save_name": "{}_winit_biaslearner".format(weight_type),
        "model_getter_type": "load_willem_biaslearner",
        "training_type": "train_bw",
        "model_arg": {"nr_hidden": nrhidden,
                      "nr_tasks": get_number_classes(dataset),
                      "load_path": w_load_path}
    }

    # Training features
    train_params = get_biaslearner_training_params(new_train_params=train_params, prog_params=prog_params)

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": False,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute}

    train_full_dataset(prog_params=prog_params, train_params=train_params, save_params=save_params,
                       verbose=verbose, debug=debug, root_dir=root_dir)


def train_sg_full_dataset(test_classes, dataset, nrhidden, train_params=None, recompute=False, save_performance=True,
                          save_initweights=True, save_finalweights=True, verbose=False, veryverbose=False, debug=False,
                          root_dir="../../"):
    # Program features
    prog_params = {
        "nr_hidden": nrhidden,
        "dataset": dataset,
        "model_type": "biaslearner",
        "model_getter_type": "random_biaslearner",
        "training_type": "train_bw",
        "model_arg": {"nr_hidden": nrhidden, "nr_tasks": 1}
    }

    # Training features
    train_params = get_biaslearner_training_params(highseed=4, wlr=0.003, blr=0.003, new_train_params=train_params)

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": save_initweights,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute}

    prog_params["validate_performance"] = train_params["early_stopping"]
    prog_params["test_performance"] = True
    result_dir = root_dir + "results/train_full_dataset/" + prog_params["dataset"] + "/"
    for test_class in test_classes:
        prog_params["model_type_save_name"] = "singular_{}".format(test_class)
        save_params["save_name"] = "singular_{}_testclass_{}".format(prog_params["nr_hidden"], test_class)

        # Get data
        train_data, valid_data = get_singular_data(class_id=test_class, dataset=prog_params["dataset"], debug=debug)
        test_data = get_singular_data(class_id=test_class, dataset=prog_params["dataset"], splitting=False, train=False,
                                      debug=debug)

        # Create model, train it and evaluate performance
        get_train_eval_model(
            result_dir=result_dir, prog_params=prog_params, train_params=train_params, save_params=save_params,
            train_data=train_data, validation_data=valid_data, test_data=test_data, verbose=verbose,
            veryverbose=veryverbose
        )


def train_multireadout_full_dataset(dataset, nrhidden, train_params=None, recompute=False, save_performance=True,
                                    save_initweights=True, save_finalweights=True, verbose=False, debug=False,
                                    root_dir="../../"):
    # Program features
    prog_params = {
        "nr_hidden": nrhidden,
        "dataset": dataset,
        "model_type": "multireadout",
        "model_type_save_name": "multireadout",
        "model_getter_type": "random_multireadout",
        "training_type": "train_multireadout",
        "model_arg": {"nr_hidden": nrhidden, "nr_readouts": get_number_classes(dataset)},
    }

    # Training features
    train_params = get_multireadout_training_params(new_train_params=train_params, prog_params=prog_params)

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": save_initweights,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute}

    train_full_dataset(prog_params=prog_params, train_params=train_params, save_params=save_params,
                       verbose=verbose, debug=debug, root_dir=root_dir)


def train_binarymr_full_dataset(dataset, nrhidden, train_params=None, recompute=False, save_performance=True,
                                save_initweights=True, save_finalweights=True, verbose=False, debug=False,
                                root_dir="../../"):
    # Program features
    prog_params = {
        "nr_hidden": nrhidden,
        "dataset": dataset,
        "model_type": "binarymr",
        "model_type_save_name": "binarymr",
        "model_getter_type": "random_multireadout",
        "training_type": "train_binarymr",
        "model_arg": {"nr_hidden": nrhidden, "nr_readouts": get_number_classes(dataset)},
    }

    if dataset == "TASKS2D":
        train_params["lr"] = 0.001
        train_params["r_lr"] = 0.01
        raise NotImplementedError("Include these in the bestparams.pickle files")

    # Training features
    train_params = get_binarymr_training_params(new_train_params=train_params, prog_params=prog_params)

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": save_initweights,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute}

    train_full_dataset(prog_params=prog_params, train_params=train_params, save_params=save_params,
                       verbose=verbose, debug=debug, root_dir=root_dir)


def transfer_full_dataset(prog_params, train_params, save_params, verbose=False, debug=False, root_dir="../../"):

    prog_params["nr_labels"] = get_number_classes(prog_params["dataset"])
    result_dir = root_dir + "results/transfer_full_dataset/" + prog_params["dataset"] + "/"
    save_params["save_name"] = "{}_{}".format(prog_params["model_type_save_name"], prog_params["nr_hidden"])

    run_full_dataset(result_dir=result_dir, prog_params=prog_params, train_params=train_params, save_params=save_params,
                     verbose=verbose, debug=debug)


def transfer_b_full_dataset(nrhidden, transfer_dataset, load_weight_types=None, train_params=None,
                            save_performance=True, save_finalweights=True, recompute=False, verbose=False, debug=False,
                            root_dir="../../"):

    # Program features
    prog_params = {"nr_hidden": nrhidden,
                   "dataset": transfer_dataset,
                   "model_type": "biaslearner",
                   "model_type_save_name": "biaslearner_full_dataset_{}".format(load_weight_types),
                   "model_getter_type": "loaded_w_random_b_biaslearner",
                   "training_type": "transfer_b",
                   "model_arg": {"nr_hidden": nrhidden, "nr_tasks": get_number_classes(transfer_dataset)},
                   "load_weights": True,
                   "load_weight_type": "train_full",
                   "model_type_load_name": "biaslearner",
                   "load_weight_dir": "{}results/train_full_dataset/{}/final_weights/"
                                      "".format(root_dir, load_weight_types)}

    # Training features
    train_params = get_biaslearner_training_params(new_train_params=train_params, transfering=True,
                                                   prog_params=prog_params)

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": False,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute}

    train_full_dataset(prog_params=prog_params, train_params=train_params, save_params=save_params, verbose=verbose,
                       debug=debug, root_dir=root_dir)


def transfer_bmr_full_dataset(nrhidden, transfer_dataset, load_weight_types=None, train_params=None,
                              save_performance=True, save_finalweights=True, recompute=False, verbose=False,
                              debug=False, root_dir="../../"):

    # Program features
    prog_params = {"nr_hidden": nrhidden,
                   "dataset": transfer_dataset,
                   "model_type": "binarymr",
                   "model_type_save_name": "binarymr_full_dataset_{}".format(load_weight_types),
                   "model_getter_type": "load_w_random_readout_multireadout",
                   "training_type": "transfer_binarymr",
                   "model_arg": {"nr_hidden": nrhidden, "nr_readouts": get_number_classes(transfer_dataset)},
                   "load_weights": True,
                   "load_weight_type": "train_full",
                   "model_type_load_name": "binarymr",
                   "load_weight_dir": "{}results/train_full_dataset/{}/final_weights/"
                                      "".format(root_dir, load_weight_types)}

    # Training features
    train_params = get_biaslearner_training_params(new_train_params=train_params, transfering=True,
                                                   prog_params=prog_params)
    train_params["new_task_ids"] = list(range(get_number_classes(transfer_dataset)))

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": False,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute}

    train_full_dataset(prog_params=prog_params, train_params=train_params, save_params=save_params, verbose=verbose,
                       debug=debug, root_dir=root_dir)


if __name__ == '__main__':

    # Test some specific script
    if len(sys.argv) == 1:
        # ds = "CIFAR100"
        # ds = "EMNIST_bymerge"
        ds = "K49"
        # nets = [[1000, 1000], [1000, 1000, 1000]]
        # nets = [[10], [10, 10], [25], [25, 25], [50], [50, 50], [50, 50, 50]]
        # nets = [[25, 25], [50, 50, 50, 50], [50, 50, 50]]
        # nets = [[100, 100, 100, 100], [100, 100, 100]]

        net = [2]
        vb = True
        vvb = True
        rc = True
        # prms = {"highseed": 25, "track_training_dir": "../../results/tracktraining/"}
        # train_b_w_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb, veryverbose=vvb,
        #                        save_performance=False)
        # train_bg_w_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb, veryverbose=vvb,
        #                         save_performance=False)
        train_g_xw_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb, veryverbose=vvb,
                                save_performance=False)
        # train_g_bw_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb, veryverbose=vvb,
        #                         save_performance=False)
        # train_binarymr_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb, save_performance=False)

        # train_biaslearner_full_dataset(nrhidden=net, dataset=ds, recompute=rc, verbose=vb,
        #                                save_performance=False)
        # train_sg_full_dataset(test_classes=[0], nrhidden=[100], dataset="EMNIST_bymerge", train_params=prms,
        #                       recompute=True, verbose=True, save_performance=False)
        # prms = {"highseed": 25}
        # polish_b_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb, train_classes=[1])
        # prms = {"highseed": 25, "track_epoch_dir": "../../results/trackepoch/", "nr_epochs": 1}
        # train_biaslearner_full_dataset(nrhidden=[100], dataset="EMNIST_bymerge", train_params=prms, recompute=True,
        #                                verbose=True, save_performance=False)

        # prms = {"highseed": 25, "track_epoch_dir": "../../results/trackepoch/", "nr_epochs": 1, "seeds": [0]}
        # # train_biaslearner_full_dataset(nrhidden=[100], dataset="EMNIST_bymerge", train_params=prms, recompute=True,
        # #                                verbose=True, save_performance=False)
        # train_sg_full_dataset(test_classes=[0], nrhidden=[100], dataset="EMNIST_bymerge", train_params=prms,
        #                       recompute=True, verbose=True, save_performance=False)

        # from pickle import load
        # with open("../../results/train_full_dataset/{}/binarymr_{}.pickle".format(ds, net), "rb") as f:
        #     results = load(f)
        #     performances = results["test_performance"]
        # polish_bmr_full_dataset(net, ds, save_performance=False, verbose=True)
        # transfer_bmr_full_dataset(nrhidden=[100], transfer_dataset="K49", load_weight_types="EMNIST_bymerge",
        #                           recompute=True, verbose=True, save_performance=False)
        # tcs = list(range(100))
        # prms = {"highseed": 3, "track_training_dir": "../../results/tracktraining/"}
        # train_sg_full_dataset(test_classes=tcs, nrhidden=[100], dataset=ds, train_params=prms,
        #                       recompute=True, verbose=True, save_performance=False)
        # dss = ["CIFAR100"]
        # prms = {"highseed": 3, "track_training_dir": "../../results/tracktraining/",
        #         "track_epoch_dir": "../../results/trackepoch/", "seeds": s}
        # train_sg_full_dataset(test_classes=tcs, nrhidden=[100], dataset="EMNIST_bymerge", train_params=prms,
        #                       recompute=True, verbose=True, save_performance=False)

        # prms = {"highseed": 25, "track_training_dir": "../../results/tracktraining/"}
        # train_b_w_full_dataset(nrhidden=[100], dataset=ds, train_params=prms, recompute=True, verbose=True,
        #                        veryverbose=True, save_performance=False)
        # prms = {"highseed": 3, "track_training_dir": "../../results/tracktraining/"}
        # train_sg_full_dataset(test_classes=[5], nrhidden=[100], dataset=ds, train_params=prms,
        #                       recompute=True, verbose=True, veryverbose=True, save_performance=False)
        # prms = {"highseed": 25, "track_epoch_dir": "../../results/trackepoch/", "nr_epochs": 1, "seeds": [4]}
        # train_b_w_full_dataset(nrhidden=[100], dataset=ds, train_params=prms, recompute=True, verbose=True,
        #                        veryverbose=True, save_performance=False)
        # prms = {"highseed": 3, "track_epoch_dir": "../../results/trackepoch/", "nr_epochs": 1, "seeds": [2]}
        # train_sg_full_dataset(test_classes=[2], nrhidden=[100], dataset=ds, train_params=prms,
        #                       recompute=True, verbose=True, veryverbose=True, save_performance=False)

    # Call in a big loop to simulate all the desired configurations
    elif sys.argv[1] in ["train_b_w", "train_g_bw", "train_bg_w", "polish_b", "train_multireadout", "train_binarymr",
                         "train_bw_willem_init", "train_sg", "full_transfer_full_train"]:
        if len(sys.argv) < 4:
            raise SyntaxError("Lacking arguments in ".format(sys.argv))
        rc = True
        vb = True
        if str(sys.argv[2]) == "linear":
            net = []
        else:
            net = list(map(int, sys.argv[2].split(',')))
        ds = str(sys.argv[3])
        if sys.argv[1] == "train_b_w":
            train_b_w_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb)
        elif sys.argv[1] == "train_g_bw":
            train_g_bw_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb)
        elif sys.argv[1] == "train_bg_w":
            train_bg_w_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb)
        elif sys.argv[1] == "polish_b":
            tcs = None
            if len(sys.argv) == 5:
                tcs = [int(sys.argv[4])]
            polish_b_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb, train_classes=tcs)
        elif sys.argv[1] == "train_multireadout":
            train_multireadout_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb)
        elif sys.argv[1] == "train_binarymr":
            train_binarymr_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb)
        elif sys.argv[1] == "train_bw_willem_init":
            wtype = str(sys.argv[4])
            train_bw_willem_init_full_dataset(dataset=ds, nrhidden=net, weight_type=wtype, recompute=rc, verbose=vb)
        elif sys.argv[1] == "train_sg":
            tcs = [int(sys.argv[4])]
            train_sg_full_dataset(test_classes=tcs, dataset=ds, nrhidden=net, recompute=rc, verbose=vb)
        elif sys.argv[1] == "full_transfer_full_train":
            train_dataset = str(sys.argv[4])
            transfer_b_full_dataset(nrhidden=net, transfer_dataset=ds, load_weight_types=train_dataset, recompute=rc,
                                    verbose=vb)
        else:
            raise ValueError(sys.argv[1])
    else:
        if len(sys.argv) >= 4:
            ds = sys.argv[2]
            s = [int(sys.argv[3])]
        else:
            ds = "EMNIST_bymerge"
            s = list(range(25))

        if sys.argv[1] == "track_training_fullbw":
            prms = {"highseed": 25, "track_training_dir": "../../results/tracktraining/", "seeds": s}
            train_b_w_full_dataset(nrhidden=[100], dataset=ds, train_params=prms, recompute=True, verbose=True,
                                   save_performance=False)
        elif sys.argv[1] == "track_training_sg":
            tcs = [int(sys.argv[4])]
            prms = {"highseed": 3, "track_training_dir": "../../results/tracktraining/", "seeds": s}
            train_sg_full_dataset(test_classes=tcs, nrhidden=[100], dataset=ds, train_params=prms,
                                  recompute=True, verbose=True, save_performance=False)
        elif sys.argv[1] == "track_epoch_fullbw":
            prms = {"highseed": 25, "track_epoch_dir": "../../results/trackepoch/", "nr_epochs": 1, "seeds": s}
            train_b_w_full_dataset(nrhidden=[100], dataset=ds, train_params=prms, recompute=True, verbose=True,
                                   save_performance=False)
        elif sys.argv[1] == "track_epoch_sg":
            tcs = [int(sys.argv[4])]
            prms = {"highseed": 3, "track_epoch_dir": "../../results/trackepoch/", "nr_epochs": 1, "seeds": s}
            train_sg_full_dataset(test_classes=tcs, nrhidden=[100], dataset=ds, train_params=prms,
                                  recompute=True, verbose=True, save_performance=False)
