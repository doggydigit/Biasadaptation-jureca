import pickle
from numpy import load as np_load
from numpy import transpose as np_transpose
from numpy import exp as npexp
from numpy.random import randn as np_randn
from biasadaptation.biasfit import biasfit, bgfit, gfit, xshiftfit
from multireadout_helper import MultiReadout


def create_random_biaslearner(nr_hidden_neurons, nr_tasks, readout="linear", input_size=784):
    nr_neurons = [input_size] + nr_hidden_neurons + [1]
    ws = [np_randn(nr_neurons[i], nr_neurons[i+1]) / nr_neurons[i] for i in range(len(nr_neurons)-1)]
    bs = [np_randn(nr_tasks, nr_neurons[i]) / nr_neurons[i] for i in range(1, len(nr_neurons))]
    return biasfit.ReLuFit(ws, bs, True, readout=readout)


def create_random_xshiftlearner(nr_hidden_neurons, nr_tasks, readout="linear", input_size=784):
    nr_neurons = [input_size] + nr_hidden_neurons + [1]
    ws = [np_randn(nr_neurons[i], nr_neurons[i+1]) / nr_neurons[i] for i in range(len(nr_neurons)-1)]
    xs = [np_randn(nr_neurons[i]) / nr_neurons[i] for i in range(1, len(nr_neurons))]
    gs = [npexp(np_randn(nr_tasks, nr_neurons[i])) for i in range(1, len(nr_neurons))]
    return xshiftfit.ReLuFit(ws, xs, gs, True, readout=readout)


def load_biaslearner_with_random_biases(nr_hidden, readout, weight_path, nr_tasks=1, input_size=784):
    nr_neurons = [input_size] + nr_hidden + [1]
    bs = [np_randn(nr_tasks, nr_neurons[i]) / nr_neurons[i] for i in range(1, len(nr_neurons))]
    with open(weight_path, 'rb') as file:
        weights = pickle.load(file)
        return biasfit.ReLuFit(weights[0], bs, True, readout=readout)


def create_biaslearner_around_weights(nr_hidden, readout, weights, nr_tasks=1, input_size=784):
    nr_neurons = [input_size] + nr_hidden + [1]
    bs = [np_randn(nr_tasks, nr_neurons[i]) / nr_neurons[i] for i in range(1, len(nr_neurons))]
    ws = [np_randn(nr_neurons[i], nr_neurons[i + 1]) / nr_neurons[i] for i in range(len(weights), len(nr_neurons) - 1)]
    return biasfit.ReLuFit(weights + ws, bs, True, readout=readout)


def create_random_gainlearner(nr_hidden_neurons, nr_tasks, readout="linear", input_size=784):
    nr_neurons = [input_size] + nr_hidden_neurons + [1]
    ws = [np_randn(nr_neurons[i], nr_neurons[i+1]) / nr_neurons[i] for i in range(len(nr_neurons)-1)]
    bs = [np_randn(nr_neurons[i]) / nr_neurons[i] for i in range(1, len(nr_neurons))]
    gs = [npexp(np_randn(nr_tasks, nr_neurons[i])) for i in range(1, len(nr_neurons))]
    return gfit.ReLuFit(ws, bs, gs, True, readout=readout)


def create_random_bglearner(nr_hidden_neurons, nr_tasks, readout="linear", input_size=784):
    nr_neurons = [input_size] + nr_hidden_neurons + [1]
    ws = [np_randn(nr_neurons[i], nr_neurons[i+1]) / nr_neurons[i] for i in range(len(nr_neurons)-1)]
    bs = [np_randn(nr_tasks, nr_neurons[i]) / nr_neurons[i] for i in range(1, len(nr_neurons))]
    gs = [npexp(np_randn(nr_tasks, nr_neurons[i])) for i in range(1, len(nr_neurons))]
    return bgfit.ReLuFit(ws, bs, gs, True, readout=readout)


def create_random_multireadout_network(nr_hidden_neurons, nr_readouts, readout, input_size=784):
    nr_neurons = [input_size] + nr_hidden_neurons + [nr_readouts]
    rs = [np_randn(nr_neurons[-2], 1) / nr_neurons[-2] for _ in range(nr_readouts)]
    rbs = [np_randn() / nr_readouts for _ in range(nr_readouts)]
    ws = [np_randn(nr_neurons[i], nr_neurons[i + 1]) / nr_neurons[i] for i in range(len(nr_hidden_neurons))]
    bs = [np_randn(nr_neurons[i]) / nr_neurons[i] for i in range(1, len(nr_neurons))]
    model = MultiReadout(rs, rbs, ws, bs, readout)
    return model


def load_multireadout_with_new_readouts(weight_path, tot_nr_tasks, new_task_ids, readout_function):

    # Load parameters from path
    with open(weight_path, 'rb') as file:
        ps = pickle.load(file)
    if len(ps) == 4:
        ws, bs, rs, rbs = ps
    elif len(ps) == 3:
        ws, biases, rs = ps
        bs = biases[:-1]
        rbs = [biases[-1][i] for i in range(len(biases[-1]))]
    else:
        raise ValueError(ps)
    assert len(rs) == tot_nr_tasks - len(new_task_ids)

    # Expand readout weights and biases with new additions for the desired new readouts
    new_task_ids.sort()
    nr_hidden = ws[-1].shape[1]  # double check that
    for i in new_task_ids:
        rs = rs[:i] + [np_randn(nr_hidden, 1) / nr_hidden] + rs[i:]
        rbs = rbs[:i] + [np_randn() / nr_hidden] + rbs[i:]

    return MultiReadout(readout_weights=rs, readout_biases=rbs, ws=ws, bs=bs, readout=readout_function)


def load_multireadout_with_random_readouts(weight_path, nr_readouts, readout_function):

    # Load parameters from path
    with open(weight_path, 'rb') as file:
        ws, bs, rs, rbs = pickle.load(file)

    # create readouts
    nr_hidden = ws[-1].shape[1]
    rs = [np_randn(nr_hidden, 1) / nr_hidden for _ in range(nr_readouts)]
    rbs = [np_randn() / nr_readouts for _ in range(nr_readouts)]

    return MultiReadout(readout_weights=rs, readout_biases=rbs, ws=ws, bs=bs, readout=readout_function)


def get_weight_load_path(prog_params):
    if prog_params["load_weight_type"] == "train_l1o":
        load_path = "{}{}_{}_testclass_{}_seed_{}.pickle".format(
            prog_params["load_weight_dir"], prog_params["model_type_load_name"], prog_params["nr_hidden"],
            prog_params["model_arg"]["new_task_ids"][0], prog_params["seed"]
        )
    elif prog_params["load_weight_type"] == "train_full":
        load_path = "{}{}_{}_seed_{}.pickle".format(
            prog_params["load_weight_dir"], prog_params["model_type_load_name"], prog_params["nr_hidden"],
            prog_params["seed"]
        )
    else:
        raise ValueError(prog_params["load_weight_type"])

    return load_path


def get_model(prog_params, readout_function, input_size=784):

    arg = prog_params["model_arg"].copy()

    if "load_weights" in prog_params.keys() and "load_path" not in arg.keys():
        if prog_params["load_weights"]:
            arg["load_path"] = get_weight_load_path(prog_params)

    # Completely new bias learner from scratch
    if prog_params["model_getter_type"] == "random_biaslearner":
        model = create_random_biaslearner(nr_hidden_neurons=arg["nr_hidden"], nr_tasks=arg["nr_tasks"],
                                          readout=readout_function, input_size=input_size)

    # Completely new xshift learner from scratch
    elif prog_params["model_getter_type"] == "random_xshiftlearner":
        model = create_random_xshiftlearner(nr_hidden_neurons=arg["nr_hidden"], nr_tasks=arg["nr_tasks"],
                                            readout=readout_function, input_size=input_size)

    # Completely new gain learner from scratch
    elif prog_params["model_getter_type"] == "random_gainlearner":
        model = create_random_gainlearner(nr_hidden_neurons=arg["nr_hidden"], nr_tasks=arg["nr_tasks"],
                                          readout=readout_function, input_size=input_size)

    # Completely new bias adn gain learner from scratch
    elif prog_params["model_getter_type"] == "random_bglearner":
        model = create_random_bglearner(nr_hidden_neurons=arg["nr_hidden"], nr_tasks=arg["nr_tasks"],
                                        readout=readout_function, input_size=input_size)

    # Build bias learner from weights and biases
    elif prog_params["model_getter_type"] == "loaded_bw_biaslearner":
        model = biasfit.ReLuFit(ws=arg["ws"], bs=arg["bs"], opt_w=True, readout=readout_function)

    # Load bias learner from saved file
    elif prog_params["model_getter_type"] == "load_path_biaslearner":
        with open(arg["load_path"], 'rb') as file:
            weights = pickle.load(file)
            model = biasfit.ReLuFit(weights[0], weights[1], True, readout=readout_function)

    # Load weights for bias learner but create random biases for transfer learning
    elif prog_params["model_getter_type"] == "loaded_w_random_b_biaslearner":
        model = load_biaslearner_with_random_biases(nr_hidden=arg["nr_hidden"], readout=readout_function,
                                                    weight_path=arg["load_path"], nr_tasks=arg["nr_tasks"])

    elif prog_params["model_getter_type"] == "load_willem_biaslearner":
        weights = [np_transpose(np_load(arg["load_path"]))]
        model = create_biaslearner_around_weights(nr_hidden=arg["nr_hidden"], readout=readout_function, weights=weights,
                                                  nr_tasks=arg["nr_tasks"])

    # Completely new multireadout network from scratch
    elif prog_params["model_getter_type"] == "random_multireadout":
        model = create_random_multireadout_network(nr_hidden_neurons=arg["nr_hidden"], nr_readouts=arg["nr_readouts"],
                                                   readout=readout_function, input_size=input_size)

    # Load multireadout network from saved file
    elif prog_params["model_getter_type"] == "load_path_multireadout":
        with open(arg["load_path"], 'rb') as file:
            weights = pickle.load(file)
            model = MultiReadout(readout_weights=weights[2], readout_biases=weights[3], ws=weights[0], bs=weights[1],
                                 readout=readout_function)

    # Load multireadout network from saved file but append new readout neuron(s) for transfer learning
    elif prog_params["model_getter_type"] == "load_w_new_readout_multireadout":
        model = load_multireadout_with_new_readouts(weight_path=arg["load_path"], tot_nr_tasks=arg["tot_nr_tasks"],
                                                    new_task_ids=arg["new_task_ids"], readout_function=readout_function)

    # Load multireadout network from saved file but append new readout neuron(s) for transfer learning
    elif prog_params["model_getter_type"] == "load_w_random_readout_multireadout":
        model = load_multireadout_with_random_readouts(weight_path=arg["load_path"], nr_readouts=arg["nr_readouts"],
                                                       readout_function=readout_function)

    else:
        raise ValueError(prog_params["model_getter_type"])

    return model


def save_model(model, save_path, model_type):
    from biaslearning_helper import save_biaslearner, save_bglearner, save_xshiftlearner
    from multireadout_helper import save_multireadout
    if model_type in ["biaslearner"]:
        save_biaslearner(model, save_path)
    elif model_type in ["multireadout", "binarymr"]:
        save_multireadout(model, save_path)
    elif model_type in ["xshiftlearner"]:
        save_xshiftlearner(model, save_path)
    elif model_type in ["gainlearner", "bglearner"]:
        save_bglearner(model, save_path)
    else:
        raise ValueError(model_type)
