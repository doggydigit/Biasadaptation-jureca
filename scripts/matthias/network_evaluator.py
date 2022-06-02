import sys
import torch
from pickle import dump as pickle_dump
from pickle import HIGHEST_PROTOCOL
from data_helper import get_number_classes, get_dataloader, get_leave1out_transferdata
from network_builder import get_model
from biasadaptation.utils.losses import get_loss_function


def evaluate_performance_biaslearner(model, data_loader, loss_f, nr_samples, task_id0=0):
    """
    Test the network model on a testing set
    :param model: network object
    :param data_loader: dataset loader for the tasks to evaluate
    :param loss_f: loss function (pointer to function)
    :param nr_samples: total number of data samples
    :param task_id0: task id to start with for the set of biases of the network model
    :return:
    """
    loss = 0.
    performance = 0.
    with torch.no_grad():
        for bidx, ((data, _), (tasks, targets)) in enumerate(data_loader):
            out = model.forward(data, tasks + task_id0).reshape(-1).detach()
            loss += loss_f(out, targets)
            performance += torch.sum((targets * out > 0).int())
        performance = 100 * performance / nr_samples

    return performance, loss


def evaluate_classification_performance_biaslearner(model, data_loader, loss_f, nr_samples, task_id0=0):
    """
    Test the network model on a testing set
    :param model: network object
    :param data_loader: dataset loader for the tasks to evaluate
    :param loss_f: loss function (pointer to function)
    :param nr_samples: total number of data samples
    :param task_id0: task id to start with for the set of biases of the network model
    :return:
    """
    loss = 0.
    performance = 0.
    with torch.no_grad():
        for _, (data, (tasks, targets)) in enumerate(data_loader):
            out = model.forward(data, tasks + task_id0).reshape(-1).detach()
            loss += loss_f(out, targets)
            performance += torch.sum((targets * out > 0).int())

        performance = 100 * performance / nr_samples

    return performance, loss


def evaluate_performance_benchmarks(model, data_loader, loss_f, nr_samples):
    """
    Test the network model on a testing set
    :param model: network object
    :param data_loader: dataset loader for the tasks to evaluate
    :param loss_f: loss function (pointer to function)
    :param nr_samples: total number of data samples
    :return:
    """
    loss = 0.
    performance = 0.
    with torch.no_grad():
        for batch_idx, (data, (_, targets)) in enumerate(data_loader):
            out = model.forward(data).detach()
            loss += loss_f(out.reshape(-1), torch.transpose(targets, 0, 1).reshape(-1))
            performance += torch.eq(torch.argmax(torch.transpose(targets, 0, 1), dim=0), torch.argmax(out, dim=0)).sum()
        performance = 100 * performance / nr_samples

    return performance, loss


def evaluate_performance_binarymr(model, data_loader, loss_f, nr_samples, task_id0=0):
    """
    Test the network model on a testing set
    :param model: network object
    :param data_loader: dataset loader for the tasks to evaluate
    :param loss_f: loss function (pointer to function)
    :param nr_samples: total number of data samples
    :param task_id0: task id to start with for the set of biases of the network model
    :return:
    """
    loss = 0.
    performance = 0.
    with torch.no_grad():
        nr_batches = 0
        for batch_idx, ((data, _), (tasks, targets)) in enumerate(data_loader):
            nr_batches += 1
            out = torch.gather(model.forward(data), 0, torch.add(tasks, task_id0).view(1, -1)).reshape(-1).detach()
            loss += loss_f(out, targets)
            performance += torch.sum((targets * out > 0).int())

        performance = 100 * performance / nr_samples

    return performance, loss


def evaluate_performance(model_type, model, data_loader, loss_f, nr_samples, task_id0=0):
    if model_type in ["biaslearner", "gainlearner", "gainlearnerxb", "bglearner", "xshiftlearner"]:
        return evaluate_performance_biaslearner(model, data_loader, loss_f, nr_samples, task_id0=task_id0)
    elif model_type == "multireadout":
        return evaluate_performance_benchmarks(model, data_loader, loss_f, nr_samples)
    elif model_type == "binarymr":
        return evaluate_performance_binarymr(model, data_loader, loss_f, nr_samples, task_id0)
    else:
        raise ValueError(model_type)


def test_network(nrhidden, dataset, seeds=None, model_type="biaslearner", progname="train_full_dataset", classes=None,
                 saving=True):
    if classes is None:
        classes = range(get_number_classes(dataset))

    prog_params = {"model_arg": {}}
    lossf = get_loss_function("mse")
    if progname == "train_full_dataset":
        weight_path = "../../results/train_full_dataset/{}/final_weights/{}_{}_seed_" \
                      "".format(dataset, model_type, nrhidden) + "{}.pickle"
        save_path = "../../results/train_full_dataset/{}/{}_{}_seed_" \
                      "".format(dataset, model_type, nrhidden) + "{}.pickle"
        if seeds is None:
            seeds = list(range(25))
    else:
        raise NotImplementedError(progname)

    if model_type == "biaslearner":
        prog_params["model_getter_type"] = "load_path_biaslearner"
    elif model_type == "binarymr":
            prog_params["model_getter_type"] = "load_path_multireadout"

    for seed in seeds:
        prog_params["model_arg"]["load_path"] = weight_path.format(seed)
        model = get_model(prog_params, "tanh")
        performances = []
        for tc in classes:
            test_data = get_leave1out_transferdata(testclass=tc, dataset=dataset, debug=False, model_type=model_type,
                                                   splitting=False, train=False)
            dataloader, nrs = get_dataloader(test_data)
            classperf, _ = evaluate_performance(model_type, model, dataloader, lossf, nrs, task_id0=tc)
            performances += [classperf]
            print("Seed {}: test class {} performance is {}".format(seed, tc, classperf))

        if saving:
            with open(save_path.format(seed), 'wb') as handle:
                pickle_dump(performances, handle, protocol=HIGHEST_PROTOCOL)


if __name__ == '__main__':

    if len(sys.argv) == 1:
        mt = "binarymr"
        ds = "EMNIST_bymerge"
        net = [25]
        from pickle import load
        with open("../../results/train_full_dataset/{}/{}_{}.pickle".format(ds, mt, net), "rb") as f:
            results = load(f)
            print(results["test_performance"])
        test_network(net, ds, [0], mt)

    else:
        m = str(sys.argv[1])
        net = list(map(int, sys.argv[2].split(',')))
        ds = str(sys.argv[3])
        s = [int(sys.argv[4])]
        test_network(net, ds, s, m)
