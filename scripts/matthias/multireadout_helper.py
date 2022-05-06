from abc import ABC
from pickle import HIGHEST_PROTOCOL
from pickle import dump as pickle_dump
from torch import FloatTensor, as_tensor, transpose, tanh, gather, sigmoid, add
from torch import stack as torch_stack
from torch import mm as torch_mm
from torch import squeeze as torch_squeeze
from torch.optim import Adam
from torch.nn import Module, Parameter, ParameterList
from torch.nn.functional import softmax, relu
from biaslearning_helper import get_best_params


class MultiReadout(Module, ABC):
    def __init__(self, readout_weights, readout_biases, ws, bs, readout="softmax"):
        super(MultiReadout, self).__init__()

        # initialize the weights and biases
        self.nr_hidden_layers = len(ws)
        self.ws = ParameterList([Parameter(FloatTensor(w)) for w in ws])
        self.bs = ParameterList([Parameter(FloatTensor(b)) for b in bs])
        self.rbs = ParameterList([Parameter(as_tensor(rb)) for rb in readout_biases])
        self.rs = ParameterList([Parameter(FloatTensor(r)) for r in readout_weights])
        self.readout = readout

        # initialize the activation functions
        if readout == "linear":
            self.af = lambda x: x
        elif readout == "tanh":
            self.af = tanh
        elif readout == "sigmoid":
            self.af = sigmoid
        elif readout == "softmax":
            self.af = softmax
        else:
            raise NotImplementedError("The available readout activation functions are: linear and softmax")

    def forward(self, x):
        """
        compute the output of the network

        Parameters
        ----------
        x: torch.floatTensor (batch_size, input_dim)
            The input data points
        """
        o = FloatTensor(x)

        # Hidden layer neuron activations
        for w, b in zip(self.ws, self.bs):
            o = relu(torch_mm(o, w) + b)

        # Output neuron activations
        if self.readout == "softmax":
            return self.af(torch_squeeze(torch_stack([torch_mm(o, r)+b for r, b in zip(self.rs, self.rbs)])), dim=1)
        else:
            return self.af(torch_squeeze(torch_stack([torch_mm(o, r)+b for r, b in zip(self.rs, self.rbs)])))


def get_multireadout_training_params(readout_function="softmax", loss_function="mse", batchsize=200, nr_epochs=50,
                                     early_stopping=True, lr=0.001, r_lr=0.01, highseed=25, new_train_params=None,
                                     prog_params=None):
    train_params = {"readout_function": readout_function,
                    "nr_epochs": nr_epochs,
                    "loss_function": loss_function,
                    "batch_size": batchsize,
                    "early_stopping": early_stopping,
                    "highseed": highseed,
                    "lr": lr,
                    "r_lr": r_lr}

    if prog_params is not None:
        lrs = get_best_params(prog_params["training_type"], prog_params["dataset"], prog_params["nr_hidden"])
        train_params["lr"] = lrs[1]
        train_params["r_lr"] = lrs[0]

    if isinstance(new_train_params, dict):
        for k in new_train_params.keys():
            train_params[k] = new_train_params[k]

    return train_params


def get_binarymr_training_params(readout_function="tanh", loss_function="mse", batchsize=200, nr_epochs=50,
                                 early_stopping=True, lr=0.001, r_lr=0.001, highseed=25, new_train_params=None,
                                 prog_params=None):

    train_params = {"readout_function": readout_function,
                    "nr_epochs": nr_epochs,
                    "loss_function": loss_function,
                    "batch_size": batchsize,
                    "early_stopping": early_stopping,
                    "highseed": highseed,
                    "lr": lr,
                    "r_lr": r_lr}

    if prog_params is not None:
        lrs = get_best_params(prog_params["training_type"], prog_params["dataset"], prog_params["nr_hidden"])
        train_params["lr"] = lrs[1]
        train_params["r_lr"] = lrs[0]

    if isinstance(new_train_params, dict):
        for k in new_train_params.keys():
            train_params[k] = new_train_params[k]

    return train_params


def save_multireadout(model, save_path):
    np_ws = [w.detach().numpy() for w in model.ws]
    np_bs = [b.detach().numpy() for b in model.bs]
    np_rs = [r.detach().numpy() for r in model.rs]
    np_rbs = [r.detach().numpy() for r in model.rbs]
    with open(save_path, 'wb') as handle:
        pickle_dump((np_ws, np_bs, np_rs, np_rbs), handle, protocol=HIGHEST_PROTOCOL)


def train_epoch_multireadout(model, data_loader, optimizer, loss_f):
    """
    Train the multireadout network model for one epoch.
    :param model: network object
    :param data_loader: dataset batch loader for the tasks to train
    :param loss_f: loss function (pointer to function)
    :param optimizer: PyTorch neural network optimizer
    :return: training losses for the epoch
    """
    loss_list = []

    for batch_idx, (data, (_, targets)) in enumerate(data_loader):
        optimizer.zero_grad()
        out = model.forward(data)
        loss = loss_f(out.reshape(-1), transpose(targets, 0, 1).reshape(-1))
        loss.backward()
        optimizer.step()
        loss_list += [loss.detach()]
    return loss_list


def train_multireadout(data, model, loss_f, train_params, evaluate_training=False, validation_data=None):
    from train_helper import train_epochs
    return train_epochs(
        data=data, validation_data=validation_data, model_type="multireadout", model=model,
        parameters=model.parameters(), loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training
    )


def transfer_multireadout(model, loss_f, train_params, evaluate_training, data, validation_data=None):
    from train_helper import train_epochs
    parameters = [model.rbs[i] for i in train_params["new_task_ids"]]
    parameters += [model.rs[i] for i in train_params["new_task_ids"]]
    return train_epochs(
        data=data, validation_data=validation_data, model_type="multireadout", model=model,
        parameters=parameters, loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training,
    )


def train_epoch_binarymr(model, data_loader, optimizer, loss_f, task_id0):
    """
    Train the multireadout network model for one epoch.
    :param model: network object
    :param data_loader: dataset batch loader for the tasks to train
    :param loss_f: loss function (pointer to function)
    :param optimizer: PyTorch neural network optimizer
    :param task_id0:
    :return: training losses for the epoch
    """
    loss_list = []

    for batch_idx, ((data, _), (task, targets)) in enumerate(data_loader):
        optimizer.zero_grad()
        out = gather(model.forward(data), 0, add(task.view(1, -1), task_id0))
        loss = loss_f(out.reshape(-1), targets)
        loss.backward()
        optimizer.step()
        loss_list += [loss.detach()]

    return loss_list


def train_binarymr(data, model, loss_f, train_params, evaluate_training=False, validation_data=None):
    from train_helper import train_epochs
    optimizer = Adam([{"params": [w for w in model.ws] + [b for b in model.bs]},
                      {"params": [r for r in model.rs] + [rb for rb in model.rbs], "lr": train_params["r_lr"]}],
                     train_params["lr"])
    return train_epochs(
        data=data, validation_data=validation_data, model_type="binarymr", model=model, optimizer=optimizer,
        parameters=model.parameters(), loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training
    )


def transfer_binarymr(model, loss_f, train_params, evaluate_training, data, validation_data=None, task_id0=0):
    from train_helper import train_epochs
    parameters = [model.rbs[i] for i in train_params["new_task_ids"]]
    parameters += [model.rs[i] for i in train_params["new_task_ids"]]
    return train_epochs(
        data=data, validation_data=validation_data, model_type="binarymr", model=model, parameters=parameters,
        loss_f=loss_f, train_params=train_params, evaluate_training=evaluate_training, task_id0=task_id0
    )
