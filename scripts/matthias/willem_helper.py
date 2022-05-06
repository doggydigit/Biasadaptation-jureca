import sys
from torch.optim import Adam
from data_helper import get_number_classes, get_leave1out_traindata
from biaslearning_helper import get_biaslearner_training_params


def get_willem_weight_load_path(weight_type, dataset, nr_hidden, root_dir="../../"):
    if len(nr_hidden) == 1:
        return root_dir + "results/willem_weights/{}_{}{}.npy".format(dataset, weight_type, nr_hidden[0])
    elif len(nr_hidden) == 2:
        return root_dir + "results/willem_weights/{}_{}{}.npy".format(dataset, weight_type, nr_hidden[0])
    else:
        raise ValueError(nr_hidden)


def train_b_w_deepen1(model, loss_f, train_params, evaluate_training, data, validation_data=None):
    from train_helper import train_epochs
    optimizer = Adam([{"params": model.ws[1:]}, {"params": model.bs, "lr": train_params["b_lr"]}], train_params["lr"])

    return train_epochs(
        data=data, validation_data=validation_data, model_type="biaslearner", model=model, optimizer=optimizer,
        parameters=None, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training
    )


def train_bw_deepen1(model, loss_f, train_params, evaluate_training, data, validation_data=None):
    from train_helper import train_epochs
    t_params = [w for w in model.ws[1:]] + [b for b in model.bs]
    return train_epochs(
        data=data, validation_data=validation_data, model_type="biaslearner", model=model,
        parameters=t_params, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training
    )


def deepen_networks_l1o(dataset, nrhidden, weight_type, test_class, train_params=None, recompute=False,
                        save_performance=True, save_finalweights=True, verbose=False, debug=False, root_dir="../../"):

    # Load path for desired set of weights
    w_load_path = get_willem_weight_load_path(weight_type=weight_type, dataset=dataset,
                                              nr_hidden=nrhidden, root_dir=root_dir)

    # Program features
    prog_params = {
        "nr_hidden": nrhidden,
        "dataset": dataset,
        "model_type": "biaslearner",
        "model_getter_type": "load_willem_biaslearner",
        "training_type": "train_b_w_deepen1",
        "model_arg": {"nr_hidden": nrhidden,
                      "nr_tasks": get_number_classes(dataset)-1,
                      "load_path": w_load_path},
        "validate_performance": True,
        "test_performance": False,
        "nr_labels": get_number_classes(dataset)-1
    }

    # Training features
    train_params = get_biaslearner_training_params(new_train_params=train_params, highseed=1)

    # Saving features
    save_params = {"save_performance": save_performance,
                   "save_initweights": False,
                   "save_finalweights": save_finalweights,
                   "recompute": recompute,
                   "save_name": "deepened_{}_{}_{}_testclass_{}".format(weight_type, nrhidden, dataset, test_class)}

    result_dir = root_dir + "results/willem_weights/deepen_networks/"

    train_data, validation_data = get_leave1out_traindata(testclass=test_class, dataset=dataset,
                                                          model_type=prog_params["model_type"], debug=debug)

    # Create model, train it and evaluate performance
    from train_helper import get_train_eval_model
    get_train_eval_model(
        result_dir=result_dir, prog_params=prog_params, train_params=train_params, save_params=save_params,
        train_data=train_data, validation_data=validation_data, verbose=verbose
    )


if __name__ == '__main__':

    # Test some specific script
    if len(sys.argv) == 1:
        # ds = "EMNIST_willem"
        ds = "EMNIST_bymerge"
        # ds = "MNIST"
        net = [10, 100]
        # prms = {"nr_epochs": 20, "highseed": 2}
        prms = None
        vb = True
        rc = True
        sp = False
        # trainbw_all_willem_initialized()
        wtype = "pmdd"
        tc = 0
        deepen_networks_l1o(dataset=ds, nrhidden=net, weight_type=wtype, test_class=tc, train_params=prms, recompute=rc,
                            save_performance=sp, verbose=vb, debug=True)

    # Call in a big loop to simulate all the desired configurations
    elif len(sys.argv) == 6:
        if str(sys.argv[2]) == "linear":
            net = []
        else:
            net = list(map(int, sys.argv[2].split(',')))
        ds = str(sys.argv[3])
        wtype = str(sys.argv[4])
        tc = int(sys.argv[5])
        if sys.argv[1] == "deepen_nets":
            deepen_networks_l1o(dataset=ds, nrhidden=net, weight_type=wtype, test_class=tc, recompute=True)
        else:
            raise ValueError(sys.argv[1])
