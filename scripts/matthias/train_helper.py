from copy import deepcopy
from os.path import isfile
from pathlib import Path
from pickle import dump as pickle_dump
from pickle import load as pickle_load
from pickle import HIGHEST_PROTOCOL
from random import seed as rd_seed
from numpy.random import seed as np_seed
from numpy import transpose as nptrps
from torch import manual_seed as torch_seed
from torch.optim import Adam
from torch.utils.data import DataLoader
from biasadaptation.utils.losses import get_loss_function
from data_helper import get_dataset, get_dataloader
from network_builder import get_model, save_model
from network_evaluator import evaluate_performance
from biaslearning_helper import train_epoch_biaslearner, train_bw, train_b_w, train_g_xw, train_g_bxw
from biaslearning_helper import train_g_bw, train_bg_w, transfer_b
from biaslearning_helper import get_pmdd_loss, train_b_w_deepen
from multireadout_helper import train_epoch_multireadout, train_multireadout, transfer_multireadout
from multireadout_helper import train_epoch_binarymr, train_binarymr, transfer_binarymr
from willem_helper import train_bw_deepen1


def all_seeds(seed):
    rd_seed(seed)
    np_seed(seed)
    torch_seed(seed)


def train_epoch(model_type, model, data, batch_size, optimizer, loss_f, task_id0, track_epoch, track_ds, dataset,
                verbose):
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    if model_type == "biaslearner":
        return train_epoch_biaslearner(model, data_loader, optimizer, loss_f, task_id0, track_epoch, track_ds,
                                       dataset=dataset, verbose=verbose)
    elif model_type == "multireadout":
        return train_epoch_multireadout(model, data_loader, optimizer, loss_f)
    elif model_type == "binarymr":
        return train_epoch_binarymr(model, data_loader, optimizer, loss_f, task_id0)
    else:
        raise ValueError(model_type)


def train_network_fixed_epochs(data, model, optimizer, loss_f, epochs, batch_size, task_id0, model_type, verbose=False):

    # Train for a fixed number of epochs
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model_type, model, data, batch_size, optimizer, loss_f, task_id0, dataset=None,
                           track_ds=None, track_epoch=None, verbose=verbose)
        if verbose:
            print("Epoch {} had training loss {}".format(epoch, sum(loss)))

    return model


def train_network_early_stopping(data, validation_data, model, optimizer, loss_f, epochs, batch_size, task_id0,
                                 model_type, patience=1, verbose=False, track_training=None, track_1st_epoch=None,
                                 track_ds=None):

    # Prepare to evaluate best time to stop training
    validation_data_loader, nr_samples = get_dataloader(validation_data)
    best_model = deepcopy(model)
    best_validation_performance = -1.
    stagnation_counter = 0
    training_tracker = None
    if track_training is not None or track_1st_epoch is not None:
        if track_ds == "CIFAR100":
            dataset = get_dataset(track_ds, "../../", False).data.reshape([-1, 3072])
        elif track_ds == "K49":
            dataset = get_dataset(track_ds, "../../", False).data.reshape([-1, 784])
        elif track_ds == "EMNIST_bymerge":
            dataset = get_dataset(track_ds, "../../", False).data.numpy().reshape([-1, 784])
        else:
            raise NotImplementedError(track_ds)

        if track_training is not None:
            training_tracker = {"train_loss": [], "valid_perf": [], "datadif_weight_span": []}
            l2error = get_pmdd_loss(data=dataset, weights=nptrps(model.ws[0].detach().numpy()))
            training_tracker["datadif_weight_span"] += [l2error]
            print("Epoch 0 had pmdd L2 {}".format(l2error))
    else:
        dataset = None

    # Train epochs until performance stops to improve
    for epoch in range(1, epochs + 1):

        # Train for one epoch
        if epoch == 1:
            loss = train_epoch(model_type, model, data, batch_size, optimizer, loss_f, task_id0, track_1st_epoch,
                               track_ds=track_ds, dataset=dataset, verbose=verbose)
        else:
            loss = train_epoch(model_type, model, data, batch_size, optimizer, loss_f, task_id0, None,
                               track_ds=track_ds, dataset=dataset, verbose=verbose)

        # Evaluate model performance on validation dataset
        validation_performance, _ = evaluate_performance(model_type=model_type, model=model,
                                                         data_loader=validation_data_loader, loss_f=loss_f,
                                                         nr_samples=nr_samples, task_id0=task_id0)

        if verbose:
            print("Epoch {} had training loss {} and validation performance {}%"
                  "".format(epoch, sum(loss), validation_performance))

        if track_training is not None:
            training_tracker["train_loss"] += [loss]
            training_tracker["valid_perf"] += [validation_performance]
            l2error = get_pmdd_loss(data=dataset, weights=nptrps(model.ws[0].detach().numpy()))
            training_tracker["datadif_weight_span"] += [l2error]
            print("Epoch {} had training loss {}, validation performance {}% and pmdd L2 {}"
                  "".format(epoch, sum(loss), validation_performance, l2error))

        # Check whether performance is still improving and stop training otherwise
        if validation_performance > best_validation_performance:
            stagnation_counter = 0
            best_validation_performance = validation_performance
            best_model = deepcopy(model)
        else:
            stagnation_counter += 1
            if stagnation_counter >= patience:
                if verbose:
                    print("Early stopping training at epoch {} with validation performance of {}%"
                          "".format(epoch, best_validation_performance))
                break

    if track_training is not None:
        with open(track_training, 'wb') as handle:
            pickle_dump(training_tracker, handle, protocol=HIGHEST_PROTOCOL)

    return best_model


def train_epochs(data, validation_data, model_type, model, parameters, loss_f, train_params, task_id0=0,
                 evaluate_training=False, optimizer=None, verbose=False):

    if optimizer is None:
        optimizer = Adam(parameters, train_params["lr"])

    if train_params["early_stopping"]:
        assert validation_data is not None
        if "track_training_path" not in train_params.keys():
            train_params["track_training_path"] = None
        if "track_epoch_path" not in train_params.keys():
            train_params["track_epoch_path"] = None
        model = train_network_early_stopping(
            data=data, validation_data=validation_data, model_type=model_type, model=model, optimizer=optimizer,
            loss_f=loss_f, epochs=train_params["nr_epochs"], batch_size=train_params["batch_size"], task_id0=task_id0,
            patience=train_params["es_patience"], verbose=verbose, track_training=train_params["track_training_path"],
            track_1st_epoch=train_params["track_epoch_path"], track_ds=train_params["track_dataset"]
        )
    else:
        model = train_network_fixed_epochs(
            data=data, model_type=model_type, model=model, optimizer=optimizer, loss_f=loss_f,
            epochs=train_params["nr_epochs"], batch_size=train_params["batch_size"], task_id0=task_id0,
            verbose=verbose
        )

    # Evaluate training performance
    if evaluate_training:
        data_loader, nr_samples = get_dataloader(data)
        performance, loss = evaluate_performance(model_type=model_type, model=model, data_loader=data_loader,
                                                 loss_f=loss_f, nr_samples=nr_samples, task_id0=task_id0)
    else:
        performance, loss = None, None

    return model, performance, loss


def train_model(training_type, model, loss_f, train_params, train_data, validation_data, evaluate_training, task_id0=0,
                verbose=False):

    # Train weights and biases of the biaslearner network model with same learning rate
    if training_type == "train_bw":
        return train_bw(model=model, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training,
                        data=train_data, validation_data=validation_data, verbose=verbose)

    # Train weights and biases of the biaslearner network model with separate learning rates
    elif training_type == "train_b_w":
        return train_b_w(model=model, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training,
                         data=train_data, validation_data=validation_data, verbose=verbose)

    # Train weights and biases of the biaslearner network model with separate learning rates
    elif training_type == "train_g_xw":
        return train_g_xw(model=model, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training,
                          data=train_data, validation_data=validation_data, verbose=verbose)

    # Train weights and biases of the biaslearner network model with separate learning rates
    elif training_type == "train_g_bxw":
        return train_g_bxw(model=model, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training,
                          data=train_data, validation_data=validation_data, verbose=verbose)

    # Train weights, biases and task specific gains of the biaslearner network model with separate learning rates
    elif training_type == "train_g_bw":
        return train_g_bw(model=model, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training,
                          data=train_data, validation_data=validation_data, verbose=verbose)

    # Train weights, biases and gains of the biaslearner network model with separate learning rates
    elif training_type == "train_bg_w":
        return train_bg_w(model=model, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training,
                          data=train_data, validation_data=validation_data, verbose=verbose)

    # Train all weights and biases of the biaslearner network model except the nth layer of weights
    elif training_type == "train_b_w_deepen":
        return train_b_w_deepen(model=model, loss_f=loss_f, train_params=train_params,
                                evaluate_training=evaluate_training, data=train_data, validation_data=validation_data)

    # Train all weights and biases of the biaslearner network model except 1st layer of weights
    elif training_type == "train_bw_deepen1":
        return train_bw_deepen1(model=model, loss_f=loss_f, train_params=train_params,
                                evaluate_training=evaluate_training, data=train_data, validation_data=validation_data)

    # Train biases only of the biaslearner network model (should be used only for unseen tasks)
    elif training_type == "transfer_b":
        return transfer_b(model=model, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training,
                          data=train_data, validation_data=validation_data, verbose=verbose)

    # Train multiple labels with individual readout neurons each all at the same time
    elif training_type == "train_multireadout":
        return train_multireadout(model=model, loss_f=loss_f, train_params=train_params,
                                  evaluate_training=evaluate_training, data=train_data, validation_data=validation_data)

    # Train multiple labels with individual readout neurons each all at the same time
    elif training_type == "transfer_multireadout":
        return transfer_multireadout(model=model, loss_f=loss_f, train_params=train_params,
                                     evaluate_training=evaluate_training, data=train_data,
                                     validation_data=validation_data)

    # Train multiple labels with individual readout neurons each all at the same time
    elif training_type == "train_binarymr":
        return train_binarymr(model=model, loss_f=loss_f, train_params=train_params,
                              evaluate_training=evaluate_training, data=train_data, validation_data=validation_data)

    # Train multiple labels with individual readout neurons each all at the same time
    elif training_type == "transfer_binarymr":
        return transfer_binarymr(model=model, loss_f=loss_f, train_params=train_params,
                                 evaluate_training=evaluate_training, data=train_data,
                                 validation_data=validation_data, task_id0=task_id0)

    else:
        raise ValueError(training_type)


def get_train_eval_seed(results, seed, loss_f, prog_params, train_params, save_params, train_data,
                        validation_data=None, test_data=None, verbose=False, veryverbose=False):
    all_seeds(seed)

    if "track_training_dir" in train_params.keys():
        train_params["track_training_path"] = "{}{}_{}_{}_seed_{}.pickle".format(train_params["track_training_dir"],
                                                                                 prog_params["model_type_save_name"],
                                                                                 prog_params["nr_hidden"],
                                                                                 train_params["track_dataset"], seed)

    if "track_epoch_dir" in train_params.keys():
        train_params["track_epoch_path"] = "{}{}_{}_{}_seed_{}.pickle".format(train_params["track_epoch_dir"],
                                                                              prog_params["model_type_save_name"],
                                                                              prog_params["nr_hidden"],
                                                                              train_params["track_dataset"], seed)

    if prog_params["model_type_save_name"] == "biaslearner_polished":
        prog_params["model_arg"]["load_path"] = "{}biaslearner_{}_seed_{}.pickle".format(
            prog_params["load_weight_dir"], prog_params["nr_hidden"], seed)

    # Get the model to train
    model = get_model(prog_params, train_params["readout_function"], input_size=train_params["input_size"])

    # Define path to save initial weights
    weight_name = save_params["save_name"] + "_seed_{}.pickle".format(seed)
    if save_params["save_initweights"]:
        save_model(model, save_params["init_weight_save_dir"] + weight_name, prog_params["model_type"])

    # Train model
    model, train_performance, train_loss = train_model(
        training_type=prog_params["training_type"], model=model, loss_f=loss_f, train_params=train_params,
        train_data=train_data, validation_data=validation_data, evaluate_training=True, task_id0=train_params["tid0"],
        verbose=veryverbose
    )

    # Store training outcomes
    results["train_performance"] += [train_performance]
    results["train_loss"] += [train_loss]
    ptype = "no"
    performance = -666

    # Save final model
    if save_params["save_finalweights"]:
        save_model(model=model, save_path=save_params["final_weight_save_dir"] + weight_name,
                   model_type=prog_params["model_type"])

    # Evaluate validation performance of the trained model
    if prog_params["validate_performance"]:
        valid_loader, nr_samples = get_dataloader(validation_data)
        valid_performance, valid_loss = evaluate_performance(model_type=prog_params["model_type"], model=model,
                                                             data_loader=valid_loader, nr_samples=nr_samples,
                                                             loss_f=loss_f, task_id0=train_params["tid0"])
        results["validation_performance"] += [valid_performance]
        results["validation_loss"] += [valid_loss]
        ptype = "validation"
        performance = valid_performance

    # Evaluate Test performance of the trained model
    if prog_params["test_performance"]:
        test_loader, nr_samples = get_dataloader(test_data)
        test_performance, test_loss = evaluate_performance(model_type=prog_params["model_type"], model=model,
                                                           data_loader=test_loader, nr_samples=nr_samples,
                                                           loss_f=loss_f, task_id0=train_params["tid0"])
        results["test_performance"] += [test_performance]
        results["test_loss"] += [test_loss]
        ptype = "test"
        performance = test_performance

    if verbose:
        print("Seed {} yielded training performance {:.1f}% and {} performance {:.1f}%"
              "".format(seed, train_performance, ptype, performance))
    return results


def prepare_simulation(result_dir, prog_params, train_params, save_params, verbose=False):

    # Make sure not to save things you don't want to
    if not save_params["save_performance"]:
        save_params["save_initweights"] = False
        save_params["save_finalweights"] = False

    if "seeds" in train_params.keys():
        save_params["result_path"] = result_dir + "individual_seed/" + save_params["save_name"] + "_" \
                                     + str(train_params["seeds"]) + ".pickle"
    else:
        save_params["result_path"] = result_dir + save_params["save_name"] + ".pickle"
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    if save_params["save_initweights"]:
        save_params["init_weight_save_dir"] = result_dir + "init_weights/"
        Path(save_params["init_weight_save_dir"]).mkdir(parents=True, exist_ok=True)
    if save_params["save_performance"]:
        save_params["final_weight_save_dir"] = result_dir + "final_weights/"
        Path(save_params["final_weight_save_dir"]).mkdir(parents=True, exist_ok=True)

    # Initializations
    if "tid0" not in train_params:
        train_params["tid0"] = 0

    if train_params["early_stopping"]:
        es_print = "with"
    else:
        es_print = "without"

    if prog_params["dataset"] == "TASKS2D":
        train_params["input_size"] = 2
    elif prog_params["dataset"] == "CIFAR100":
        train_params["input_size"] = 3072
    else:
        train_params["input_size"] = 784
    train_params["es_patience"] = 5

    if "model_type_save_name" not in prog_params:
        prog_params["model_type_save_name"] = prog_params["model_type"]

    # Result save dictionary
    results = {"loss_function": train_params["loss_function"],
               "readout_function": train_params["readout_function"],
               "lr": train_params["lr"],
               "batch_size": train_params["batch_size"],
               "nr_epochs": train_params["nr_epochs"],
               "early_stopping": train_params["early_stopping"]}

    # Load results, in case they were already simulated
    if not save_params["recompute"] and isfile(save_params["result_path"]):
        with open(save_params["result_path"], "rb") as f:
            load_results = pickle_load(f)
            for i in results.keys():
                if load_results[i] != results[i]:
                    print("Loaded ", load_results)
                    print("Current: ", results)
                    raise ValueError("The existing results were computed with different training parameters. Please, "
                                     "make sure to set them equal or set recompute = True to overwrite old results.")
            if "test_performance" in load_results.keys():
                print_1 = " with test performances {}%".format(load_results["test_performance"])
            elif "validation_performance" in load_results.keys():
                print_1 = " with validation performances {}%".format(load_results["validation_performance"])
            else:
                print_1 = ""
            print("Training was already simulated with up to seed {}{}".format(load_results["highseed"], print_1))
            if load_results["highseed"] >= train_params["highseed"]:
                return None, True, results, None, None, None
            else:
                results = load_results
                results["highseed"] = train_params["highseed"]
                seeds = range(load_results["highseed"], train_params["highseed"])
    else:
        if "seeds" in train_params.keys():
            seeds = train_params["seeds"]
        else:
            seeds = range(train_params["highseed"])
        results["highseed"] = train_params["highseed"]
        result_types = ["train_performance", "train_loss"]
        if prog_params["validate_performance"]:
            result_types += ["validation_performance", "validation_loss"]
        if prog_params["test_performance"]:
            result_types += ["test_performance", "test_loss"]
        for r in result_types:
            results[r] = []

    loss_f = get_loss_function(train_params["loss_function"])

    if verbose:
        print("Training {} network {} on {} with loss {}, readout {}, learning rate {}, batch size {} over {} epochs {}"
              " early stopping".format(prog_params["model_type"], prog_params["nr_hidden"], prog_params["dataset"],
                                       train_params["loss_function"], train_params["readout_function"],
                                       train_params["lr"], train_params["batch_size"], train_params["nr_epochs"],
                                       es_print))

    if "track_training_dir" in train_params.keys() or "track_epoch_dir" in train_params.keys():
        train_params["track_dataset"] = prog_params["dataset"]
    else:
        train_params["track_dataset"] = None

    return seeds, False, results, loss_f, train_params, save_params


def get_train_eval_model(result_dir, prog_params, save_params, train_params, train_data,
                         validation_data=None, test_data=None, verbose=False, veryverbose=False):

    # Prepare simulation with some initializations and checking if it was already simulated
    seeds, skip_simulations, results, loss_f, train_params, save_params = prepare_simulation(
        result_dir=result_dir, prog_params=prog_params, save_params=save_params, train_params=train_params,
        verbose=verbose
    )

    if skip_simulations:
        return results

    # Repeat simulation for each seed
    for seed in seeds:
        prog_params["seed"] = seed
        results = get_train_eval_seed(
            results=results, seed=seed, loss_f=loss_f, prog_params=prog_params, train_params=train_params,
            save_params=save_params, train_data=train_data, validation_data=validation_data, test_data=test_data,
            verbose=verbose, veryverbose=veryverbose
        )

    # Print average performance
    if prog_params["test_performance"]:
        ptype = "test"
        average_performance = sum(results["test_performance"]) / len(seeds)
    elif prog_params["validate_performance"]:
        ptype = "validation"
        average_performance = sum(results["validation_performance"]) / len(seeds)
    else:
        ptype = "training"
        average_performance = sum(results["train_performance"]) / len(seeds)
    print("{} network {} trained on {} had an average {} performance of {}% over {} seeds".format(
        prog_params["model_type"], prog_params["nr_hidden"], prog_params["dataset"], ptype, average_performance,
        train_params["highseed"])
    )

    # Save outcome of training to pickle file
    if save_params["save_performance"]:
        with open(save_params["result_path"], 'wb') as handle:
            pickle_dump(results, handle, protocol=HIGHEST_PROTOCOL)

    return results
