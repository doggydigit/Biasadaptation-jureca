from pickle import load as pickle_load
from pickle import dump as pickle_dump
from pickle import HIGHEST_PROTOCOL
from numpy import dot as npdot
from numpy import sum as npsum
from numpy import transpose as nptrps
from numpy.linalg import norm as npnorm
from sklearn.decomposition import SparseCoder
from biasadaptation.utils.utils import differences_numpy
from torch.optim import Adam


def get_best_params(traintype, dataset, network=None, architecture_specific=False, root_dir="../../"):
    if architecture_specific:
        if traintype == "train_b_w":
            prog_name = "scan_train_blr_wlr"
        elif traintype == "train_g_xw":
            prog_name = "scan_train_glr_xwlr"
        elif traintype == "train_g_bw":
            prog_name = "scan_train_glr_bwlr"
        elif traintype == "train_bg_w":
            prog_name = "scan_train_bglr_wlr"
        elif traintype == "train_binarymr":
            prog_name = "scan_train_bmr_lr"
        elif traintype == "train_bw":
            prog_name = "scan_train_sg_lr"
        else:
            prog_name = traintype
        with open("{}results/{}/bestparams.pickle".format(root_dir, prog_name), 'rb') as file:
            return pickle_load(file)[dataset][str(network)]
    else:
        if dataset == "EMNIST_bymerge":
            if traintype == "train_b_w":
                return 0.02, 0.001
            elif traintype == "train_g_xw":
                return 0.01, 0.001
            elif traintype == "train_g_bw":
                return 0.06, 0.00006
            elif traintype == "train_bg_w":
                return 0.006, 0.0006
            elif traintype == "train_binarymr":
                return 0.003, 0.00006
            elif traintype == "train_bw":
                return 0.006
            else:
                raise ValueError(traintype)

        elif dataset == "CIFAR100":
            if traintype == "train_b_w":
                return 0.02, 0.00006
            elif traintype == "train_g_xw":
                return 0.1, 0.00003
            elif traintype == "train_g_bw":
                return 0.02, 0.00001
            elif traintype == "train_bg_w":
                return 0.006, 0.000006
            elif traintype == "train_binarymr":
                return 0.002, 0.00003
            elif traintype == "train_bw":
                return 0.000006
            else:
                raise ValueError(traintype)

        elif dataset == "K49":
            if traintype == "train_b_w":
                return 0.03, 0.001
            elif traintype == "train_g_xw":
                return 0.03, 0.0003
            elif traintype == "train_g_bw":
                return 0.1, 0.00006
            elif traintype == "train_bg_w":
                return 0.01, 0.001
            elif traintype == "train_binarymr":
                return 0.006, 0.0002
            elif traintype == "train_bw":
                return 0.006
            else:
                raise ValueError(traintype)
        elif dataset == "TASKS2D":
            if traintype == "train_b_w":
                return 0.01, 0.01
            else:
                raise ValueError(traintype)
        else:
            raise ValueError(dataset)


def get_biaslearner_training_params(readout_function="tanh", loss_function="mse", batchsize=200, nr_epochs=50,
                                    early_stopping=True, wlr=0.0006, blr=0.06, glr=0.06, highseed=25,
                                    new_train_params=None, transfering=False, prog_params=None):

    train_params = {"readout_function": readout_function,
                    "nr_epochs": nr_epochs,
                    "loss_function": loss_function,
                    "batch_size": batchsize,
                    "early_stopping": early_stopping,
                    "highseed": highseed,
                    "b_lr": blr,
                    "g_lr": glr,
                    "lr": wlr}

    if prog_params is not None:
        lrs = get_best_params(prog_params["training_type"], prog_params["dataset"], prog_params["nr_hidden"])
        if prog_params["training_type"] == "train_bw":
            train_params["lr"] = lrs
        else:
            train_params["lr"] = lrs[1]
            if prog_params["training_type"] == "train_b_w":
                train_params["b_lr"] = lrs[0]
            elif prog_params["training_type"] == "train_g_bw":
                train_params["g_lr"] = lrs[0]
            elif prog_params["training_type"] == "train_bg_w":
                train_params["b_lr"] = lrs[0]
                train_params["g_lr"] = lrs[0]
            elif prog_params["training_type"] == "train_g_xw":
                train_params["g_lr"] = lrs[0]
            else:
                raise ValueError(prog_params["training_type"])

    if transfering:
        if prog_params["training_type"] == "train_g_bw":
            train_params["lr"] = train_params["g_lr"]
        elif prog_params["training_type"] == "train_g_xw":
            train_params["lr"] = train_params["g_lr"]
        else:
            train_params["lr"] = train_params["b_lr"]

    if isinstance(new_train_params, dict):
        for k in new_train_params.keys():
            train_params[k] = new_train_params[k]

    return train_params


def save_biaslearner(model, save_path):
    np_ws = [w.detach().numpy() for w in model.ws]
    np_bs = [b.detach().numpy() for b in model.bs]
    with open(save_path, 'wb') as handle:
        pickle_dump((np_ws, np_bs), handle, protocol=HIGHEST_PROTOCOL)


def save_bglearner(model, save_path):
    np_ws = [w.detach().numpy() for w in model.ws]
    np_bs = [b.detach().numpy() for b in model.bs]
    np_gs = [g.detach().numpy() for g in model.gs]
    with open(save_path, 'wb') as handle:
        pickle_dump((np_ws, np_bs, np_gs), handle, protocol=HIGHEST_PROTOCOL)


def get_pmdd_loss(data, weights, nsamples=1000):
    diffdata = differences_numpy(data, nsamples)
    coder = SparseCoder(weights, transform_algorithm='lasso_lars', transform_alpha=0.)
    return npsum(npnorm(diffdata - npdot(coder.transform(diffdata), weights), ord=2, axis=1)) / nsamples


def train_epoch_biaslearner(model, data_loader, optimizer, loss_f, task_id0, track_epoch=None, track_ds=None,
                            dataset=None, verbose=False):
    """
    Train the network model for one epoch.
    :param model: network object
    :param data_loader: dataset batch loader for the tasks to train
    :param loss_f: loss function (pointer to function)
    :param optimizer: PyTorch neural network optimizer
    :param task_id0: network task id to start with
    :param track_epoch:
    :param track_ds:
    :param dataset:
    :param verbose:
    :return: training losses for the epoch
    """

    loss_list = []

    if track_epoch is not None:
        if track_ds == "CIFAR100":
            tbs = list(range(1, 10)) + list(range(10, 100, 10)) + [150, 200]
        elif track_ds == "K49":
            tbs = list(range(1, 10)) + list(range(10, 100, 10)) + list(range(100, 1000, 100))
        elif track_ds == "EMNIST_bymerge":
            tbs = list(range(1, 10)) + list(range(10, 100, 10)) + list(range(100, 1000, 100)) \
                  + [1000, 1500, 2000, 2500, 3000]
        else:
            raise NotImplementedError(track_ds)
        l2error = get_pmdd_loss(data=dataset, weights=nptrps(model.ws[0].detach().numpy()))
        training_tracker = {"train_loss": [], "valid_perf": [], "datadif_weight_span": [l2error]}
        print("Batch 0 had pmdd L2 {}".format(l2error))
    else:
        tbs = []
        training_tracker = {}

    # Train one batch at a time
    for batch_idx, ((data, _), (tasks, targets)) in enumerate(data_loader):
        optimizer.zero_grad()
        out = model.forward(data, tasks + task_id0).reshape(-1)
        loss = loss_f(out, targets)
        loss_list += [loss.detach()]
        loss.backward()
        optimizer.step()
        if track_epoch is not None:
            if (batch_idx+1) in tbs:
                training_tracker["train_loss"] += [loss_list[-1]]
                l2error = get_pmdd_loss(data=dataset, weights=nptrps(model.ws[0].detach().numpy()))
                training_tracker["datadif_weight_span"] += [l2error]
                print("Batch {} had training loss {} and pmdd L2 {}".format(batch_idx + 1, loss, l2error))
        if verbose and batch_idx % 100 == 0:
            print("Batch {} had training loss {}".format(batch_idx + 1, loss))

    if track_epoch is not None:
        with open(track_epoch, 'wb') as handle:
            pickle_dump(training_tracker, handle, protocol=HIGHEST_PROTOCOL)

    return loss_list


def train_bw(model, loss_f, train_params, evaluate_training, data, validation_data=None, verbose=False):
    from train_helper import train_epochs
    optimizer = Adam(model.parameters(), train_params["lr"])
    return train_epochs(
        data=data, validation_data=validation_data, model_type="biaslearner", model=model, optimizer=optimizer,
        parameters=None, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training, verbose=verbose
    )


def train_g_xw(model, loss_f, train_params, evaluate_training, data, validation_data=None, verbose=False):
    from train_helper import train_epochs
    optimizer = Adam([{"params": model.ws}, {"params": model.xs},
                      {"params": model.gs, "lr": train_params["g_lr"]}],
                     train_params["lr"])
    return train_epochs(
        data=data, validation_data=validation_data, model_type="biaslearner", model=model, optimizer=optimizer,
        parameters=None, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training, verbose=verbose
    )


def train_b_w(model, loss_f, train_params, evaluate_training, data, validation_data=None, verbose=False):
    from train_helper import train_epochs
    optimizer = Adam([{"params": model.ws}, {"params": model.bs, "lr": train_params["b_lr"]}], train_params["lr"])
    return train_epochs(
        data=data, validation_data=validation_data, model_type="biaslearner", model=model, optimizer=optimizer,
        parameters=None, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training, verbose=verbose
    )


def train_g_bw(model, loss_f, train_params, evaluate_training, data, validation_data=None, verbose=False):
    from train_helper import train_epochs
    optimizer = Adam([{"params": model.ws}, {"params": model.bs},
                      {"params": model.gs, "lr": train_params["g_lr"]}],
                     train_params["lr"])
    return train_epochs(
        data=data, validation_data=validation_data, model_type="biaslearner", model=model, optimizer=optimizer,
        parameters=None, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training, verbose=verbose
    )


def train_bg_w(model, loss_f, train_params, evaluate_training, data, validation_data=None, verbose=False):
    from train_helper import train_epochs
    optimizer = Adam([{"params": model.ws},
                      {"params": model.bs, "lr": train_params["b_lr"]},
                      {"params": model.gs, "lr": train_params["g_lr"]}],
                     train_params["lr"])
    return train_epochs(
        data=data, validation_data=validation_data, model_type="biaslearner", model=model, optimizer=optimizer,
        parameters=None, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training, verbose=verbose
    )


def transfer_b(model, loss_f, train_params, evaluate_training, data, validation_data=None, verbose=False):
    from train_helper import train_epochs
    return train_epochs(
        data=data, validation_data=validation_data, model_type="biaslearner", model=model,
        parameters=model.bs, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training,
        task_id0=train_params["tid0"], verbose=verbose
    )


def train_b_w_deepen(model, loss_f, train_params, evaluate_training, data, lay=1, validation_data=None):
    from train_helper import train_epochs
    optimizer = Adam([{"params": model.ws[lay:]}, {"params": model.bs, "lr": train_params["b_lr"]}], train_params["lr"])

    return train_epochs(
        data=data, validation_data=validation_data, model_type="biaslearner", model=model, optimizer=optimizer,
        parameters=None, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training
    )
