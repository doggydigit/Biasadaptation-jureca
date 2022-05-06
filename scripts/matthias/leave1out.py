import sys
from data_helper import get_leave1out_traindata, get_leave1out_transferdata, get_number_classes
from train_helper import get_train_eval_model
from biaslearning_helper import get_biaslearner_training_params
from multireadout_helper import get_multireadout_training_params, get_binarymr_training_params


def get_l1o_datas(transfering, test_class, model_type, dataset, early_stopping, debug):
    if transfering:
        if early_stopping:
            train_data, validation_data = get_leave1out_transferdata(testclass=test_class, dataset=dataset,
                                                                     model_type=model_type, debug=debug)
        else:
            train_data = get_leave1out_transferdata(testclass=test_class, dataset=dataset,
                                                    model_type=model_type, splitting=False, debug=debug)
            validation_data = None
        test_data = get_leave1out_transferdata(testclass=test_class, dataset=dataset, debug=debug,
                                               model_type=model_type, splitting=False, train=False)
    else:
        if early_stopping:
            train_data, validation_data = get_leave1out_traindata(testclass=test_class, dataset=dataset,
                                                                  model_type=model_type, debug=debug)
        else:
            train_data = get_leave1out_traindata(testclass=test_class, dataset=dataset,
                                                 model_type=model_type, splitting=False, debug=debug)
            validation_data = None
        test_data = get_leave1out_traindata(testclass=test_class, dataset=dataset, debug=debug,
                                            model_type=model_type, splitting=False, train=False)

    return train_data, validation_data, test_data


def l1o_get_train_eval(transfering, test_classes, prog_params, train_params, save_params, result_dir, verbose, debug):

    for test_class in test_classes:
        if transfering:
            train_params["new_task_ids"] = [test_class]
            prog_params["model_arg"]["new_task_ids"] = [test_class]
            if prog_params["model_type"] == "binarymr":
                train_params["tid0"] = test_class
            else:
                train_params["tid0"] = 0
        save_params["save_name"] = "{}_{}_testclass_{}".format(prog_params["model_type_save_name"],
                                                               prog_params["nr_hidden"], test_class)

        # Get data
        train_data, validation_data, test_data = get_l1o_datas(
            transfering=transfering, test_class=test_class, model_type=prog_params["model_type"],
            dataset=prog_params["dataset"], early_stopping=train_params["early_stopping"], debug=debug
        )

        # Create model, train it and evaluate performance
        get_train_eval_model(
            result_dir=result_dir, prog_params=prog_params, train_params=train_params, save_params=save_params,
            train_data=train_data, validation_data=validation_data, test_data=test_data, verbose=verbose
        )


def run_leave1out(transfering, prog_params, train_params, save_params, verbose, debug, root_dir, test_classes=None):

    prog_params["validate_performance"] = train_params["early_stopping"]
    prog_params["test_performance"] = True
    base_dir = root_dir + "results/leave_1_out/" + prog_params["dataset"] + "/"
    if transfering:
        prog_params["load_weights"] = True
        if "load_weight_type" not in prog_params.keys():
            prog_params["load_weight_type"] = "train_l1o"
        if "load_weight_dir" not in prog_params.keys():
            prog_params["load_weight_dir"] = base_dir + "train/final_weights/"
        if "model_type_load_name" not in prog_params.keys():
            prog_params["model_type_load_name"] = prog_params["model_type_save_name"]
        result_dir = base_dir + "transfer/"
    else:
        result_dir = base_dir + "train/"

    # Leave out all classes (that should be simulated)
    if test_classes is None:
        test_classes = range(get_number_classes(prog_params["dataset"]))
    l1o_get_train_eval(
        transfering=transfering, test_classes=test_classes, prog_params=prog_params, train_params=train_params,
        save_params=save_params, result_dir=result_dir, verbose=verbose, debug=debug
    )


def train_biaslearner_l1o(nrhidden, dataset, train_type="train_b_w", test_classes=None, train_params=None,
                          save_performance=True, save_initweights=True, save_finalweights=True, recompute=False,
                          verbose=False, debug=False, root_dir="../../"):

    # Program features
    prog_params = {
        "nr_hidden": nrhidden,
        "dataset": dataset,
        "model_type": "biaslearner",
        "model_type_save_name": "biaslearner",
        "model_getter_type": "random_biaslearner",
        "training_type": train_type,
        "model_arg": {"nr_hidden": nrhidden, "nr_tasks": get_number_classes(dataset) - 1},
    }

    # Training features
    if train_params is None:
        train_params = {"highseed": 3}
    elif "highseed" not in train_params.keys():
        train_params["highseed"] = 3
    train_params = get_biaslearner_training_params(new_train_params=train_params)

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": save_initweights,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute}

    run_leave1out(transfering=False, prog_params=prog_params, train_params=train_params, save_params=save_params,
                  test_classes=test_classes, verbose=verbose, debug=debug, root_dir=root_dir)


def train_b_w_l1o(nrhidden, dataset, test_classes=None, train_params=None, save_performance=True, save_initweights=True,
                  save_finalweights=True, recompute=False, verbose=False, debug=False, root_dir="../../"):

    train_biaslearner_l1o(nrhidden=nrhidden, dataset=dataset, train_type="train_b_w", test_classes=test_classes,
                          train_params=train_params, save_performance=save_performance,
                          save_initweights=save_initweights, save_finalweights=save_finalweights, recompute=recompute,
                          verbose=verbose, debug=debug, root_dir=root_dir)


def train_mr_l1o(nrhidden, dataset, test_classes=None, train_params=None, save_performance=True, save_initweights=True,
                 save_finalweights=True, recompute=False, verbose=False, debug=False, root_dir="../../"):

    # Program features
    prog_params = {
        "nr_hidden": nrhidden,
        "dataset": dataset,
        "model_type": "multireadout",
        "model_type_save_name": "multireadout",
        "model_getter_type": "random_multireadout",
        "training_type": "train_multireadout",
        "model_arg": {"nr_hidden": nrhidden, "nr_readouts": get_number_classes(dataset) - 1},
    }

    # Training features
    if train_params is None:
        train_params = {"highseed": 3}
    elif "highseed" not in train_params.keys():
        train_params["highseed"] = 3
    train_params = get_multireadout_training_params(new_train_params=train_params)

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": save_initweights,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute}
    run_leave1out(transfering=False, prog_params=prog_params, train_params=train_params, save_params=save_params,
                  test_classes=test_classes, verbose=verbose, debug=debug, root_dir=root_dir)


def train_bmr_l1o(nrhidden, dataset, test_classes=None, train_params=None, save_performance=True, save_initweights=True,
                  save_finalweights=True, recompute=False, verbose=False, debug=False, root_dir="../../"):

    # Program features
    prog_params = {
        "nr_hidden": nrhidden,
        "dataset": dataset,
        "model_type": "binarymr",
        "model_type_save_name": "binarymr",
        "model_getter_type": "random_multireadout",
        "training_type": "train_binarymr",
        "model_arg": {"nr_hidden": nrhidden, "nr_readouts": get_number_classes(dataset) - 1},
    }

    # Training features
    if train_params is None:
        train_params = {"highseed": 3}
    elif "highseed" not in train_params.keys():
        train_params["highseed"] = 3
    train_params = get_binarymr_training_params(new_train_params=train_params)

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": save_initweights,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute}
    run_leave1out(transfering=False, prog_params=prog_params, train_params=train_params, save_params=save_params,
                  test_classes=test_classes, verbose=verbose, debug=debug, root_dir=root_dir)


def transfer_b_l1o(nrhidden, dataset, test_classes=None, train_params=None, save_performance=True,
                   save_finalweights=True, recompute=False, verbose=False, debug=False, root_dir="../../"):

    # Program features
    prog_params = {
        "nr_hidden": nrhidden,
        "dataset": dataset,
        "model_type": "biaslearner",
        "model_type_save_name": "biaslearner",
        "model_getter_type": "loaded_w_random_b_biaslearner",
        "training_type": "transfer_b",
        "model_arg": {"nr_hidden": nrhidden, "nr_tasks": 1},
    }

    # Training features
    if train_params is None:
        train_params = {"highseed": 3}
    elif "highseed" not in train_params.keys():
        train_params["highseed"] = 3
    train_params = get_biaslearner_training_params(new_train_params=train_params, transfering=True)

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": False,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute}

    run_leave1out(transfering=True, prog_params=prog_params, train_params=train_params, save_params=save_params,
                  test_classes=test_classes, verbose=verbose, debug=debug, root_dir=root_dir)


def transfer_mr_l1o(nrhidden, dataset, test_classes=None, train_params=None, save_performance=True,
                    save_finalweights=True, recompute=False, verbose=False, debug=False, root_dir="../../"):

    # Program features
    prog_params = {
        "nr_hidden": nrhidden,
        "dataset": dataset,
        "model_type": "multireadout",
        "model_type_save_name": "multireadout",
        "model_getter_type": "load_w_random_readout_multireadout",
        "training_type": "transfer_multireadout",
        "model_arg": {"nr_hidden": nrhidden, "tot_nr_tasks": get_number_classes(dataset)},
    }

    # Training features
    if train_params is None:
        train_params = {"highseed": 3}
    elif "highseed" not in train_params.keys():
        train_params["highseed"] = 3
    train_params = get_multireadout_training_params(new_train_params=train_params)

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": False,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute}

    run_leave1out(transfering=True, prog_params=prog_params, train_params=train_params, save_params=save_params,
                  test_classes=test_classes, verbose=verbose, debug=debug, root_dir=root_dir)


def transfer_bmr_l1o(nrhidden, dataset, test_classes=None, train_params=None, save_performance=True,
                     save_finalweights=True, recompute=False, verbose=False, debug=False, root_dir="../../"):

    # Program features
    prog_params = {
        "nr_hidden": nrhidden,
        "dataset": dataset,
        "model_type": "binarymr",
        "model_type_save_name": "binarymr",
        "model_getter_type": "load_w_new_readout_multireadout",
        "training_type": "transfer_binarymr",
        "model_arg": {"nr_hidden": nrhidden, "tot_nr_tasks": get_number_classes(dataset)},
    }

    # Training features
    if train_params is None:
        train_params = {"highseed": 3}
    elif "highseed" not in train_params.keys():
        train_params["highseed"] = 3
    train_params = get_binarymr_training_params(new_train_params=train_params)

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": False,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute}

    run_leave1out(transfering=True, prog_params=prog_params, train_params=train_params, save_params=save_params,
                  test_classes=test_classes, verbose=verbose, debug=debug, root_dir=root_dir)


if __name__ == '__main__':

    # Test some specific script
    if len(sys.argv) == 1:
        ds = "CIFAR100"
        net = [2]
        # prms = {"nr_epochs": 2, "highseed": 2}
        prms = None
        vb = True
        rc = True
        sp = False
        tcs = list(range(26))
        # train_bw_l1o(nrhidden=net, dataset=ds, test_classes=tcs, train_params=prms, verbose=vb, recompute=rc,
        #              save_performance=sp)
        # train_mr_l1o(nrhidden=net, dataset=ds, test_classes=tcs, train_params=prms, verbose=vb, recompute=rc,
        #              save_performance=sp)
        # train_bmr_l1o(nrhidden=net, dataset=ds, test_classes=tcs, train_params=prms, verbose=vb, recompute=rc,
        #               save_performance=sp)
        # transfer_b_l1o(nrhidden=net, dataset=ds, test_classes=[9], train_params=prms, verbose=vb, recompute=rc,
        #                save_performance=sp)
        train_b_w_l1o(nrhidden=net, dataset=ds, test_classes=tcs, train_params=prms, verbose=vb, recompute=rc,
                      save_performance=sp)
        transfer_bmr_l1o(nrhidden=net, dataset=ds, test_classes=tcs, train_params=prms, verbose=vb, recompute=rc,
                         save_performance=sp)

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
        if sys.argv[1] == "train_b_w":
            train_b_w_l1o(nrhidden=net, dataset=ds, test_classes=tcs, recompute=rc, verbose=vb)
        elif sys.argv[1] == "train_multireadout":
            train_mr_l1o(nrhidden=net, dataset=ds, test_classes=tcs, recompute=rc, verbose=vb)
        elif sys.argv[1] == "train_bmr":
            train_bmr_l1o(nrhidden=net, dataset=ds, test_classes=tcs, recompute=rc, verbose=vb)
        elif sys.argv[1] == "transfer_b":
            transfer_b_l1o(nrhidden=net, dataset=ds, test_classes=tcs, recompute=rc, verbose=vb)
        elif sys.argv[1] == "transfer_mr":
            transfer_mr_l1o(nrhidden=net, dataset=ds, test_classes=tcs, recompute=rc, verbose=vb)
        elif sys.argv[1] == "transfer_bmr":
            transfer_bmr_l1o(nrhidden=net, dataset=ds, test_classes=tcs, recompute=rc, verbose=vb)
        else:
            raise ValueError(sys.argv[1])
    else:
        raise ValueError(sys.argv)
