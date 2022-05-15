from pickle import HIGHEST_PROTOCOL
from pickle import dump as pickle_dump
from pickle import load as pickle_load


def bind_job(ns, ds, ls, gls, p0):
    fdir = "../../results/{}/individual/".format(p0)
    nrsubjobs = 10
    for p1 in ns:
        for p2 in ds:
            for p3 in ls[p2]:
                for p4 in gls[p2]:
                    results = None
                    fname = "network_{}_{}_mse_tanh_es_wlr_{}_glr_{}".format(p1, p2, float(p3), float(p4))
                    for p5 in range(nrsubjobs):
                        fpath = fdir + "individual_seed/" + fname + "_{}.pickle".format([2 * p5, 2 * p5 + 1])
                        with open(fpath, "rb") as f:
                            if results is None:
                                results = pickle_load(f)
                            else:
                                result = pickle_load(f)
                                for k in []:
                                    results[k] += result[k]
                    with open(fdir + fname + ".pickle", "wb") as f:
                        pickle_dump(results, f, protocol=HIGHEST_PROTOCOL)


if __name__ == '__main__':
    ns = [[500, 500, 500], [100, 100, 100], [25, 25, 25], [500, 500], [100, 100], [25, 25], [500], [100], [25]]

    ds = ["K49", "EMNIST_bymerge", "CIFAR100"]
    ls = {"K49": ["0.1", "0.01", "0.001", "0.0001", "0.00001", "0.000001"],
          "EMNIST_bymerge": ["0.1", "0.01", "0.001", "0.0001", "0.00001", "0.000001"],
          "CIFAR100": ["0.0002", "0.0001", "0.00006", "0.00003", "0.00002"]}
    gls = {"K49": ["0.6"],
           "EMNIST_bymerge": ["0.6"],
           "CIFAR100": ["0.06", "0.03", "0.02", "0.01", "0.006"]}
    p0 = "scan_train_xlr_wlr"
    bind_job(ns, ds, ls, gls, p0)

    ds = ["K49", "EMNIST_bymerge", "CIFAR100"]
    ls = {"K49": ["0.001"],
          "EMNIST_bymerge": ["0.001"],
          "CIFAR100": ["0.03", "0.01", "0.003", "0.001", "0.0003", "0.0001"]}
    gls = {"K49": ["0.3", "0.2", "0.06", "0.03", "0.02"],
           "EMNIST_bymerge": ["0.3", "0.2", "0.06", "0.03", "0.02"],
           "CIFAR100": ["0.3", "0.1", "0.03", "0.01"]}
    p0 = "scan_train_g_xw"
    bind_job(ns, ds, ls, gls, p0)

    ds = ["K49"]
    ls = {"K49": ["0.003", "0.0003"]}
    gls = {"K49": ["0.00003"]}
    p0 = "scan_train_b_w"
    bind_job(ns, ds, ls, gls, p0)

    ds = ["K49"]
    ls = {"K49": ["0.03"]}
    gls = {"K49": ["0.003", "0.0003", "0.00003"]}
    p0 = "scan_train_b_w"
    bind_job(ns, ds, ls, gls, p0)

    ds = ["K49"]
    ls = {"K49": ["0.00003"]}
    gls = {"K49": ["0.003", "0.0003", "0.00003"]}
    p0 = "scan_train_b_w"
    bind_job(ns, ds, ls, gls, p0)

    ds = ["K49"]
    ls = {"K49": ["0.1", "0.01", "0.001", "0.00001", "0.000001"]}
    gls = {"K49": ["0.3"]}
    p0 = "scan_train_bmr"
    bind_job(ns, ds, ls, gls, p0)

    ds = ["K49"]
    ls = {"K49": ["0.1", "0.01", "0.001", "0.00001", "0.000001"]}
    gls = {"K49": ["0.3"]}
    p0 = "scan_train_g_bw"
    bind_job(ns, ds, ls, gls, p0)

    ds = ["K49", "EMNIST_bymerge", "CIFAR100"]
    ls = {"K49": ["0.003"],
          "EMNIST_bymerge": ["0.003"],
          "CIFAR100": ["0.1", "0.01", "0.001", "0.0001", "0.00001", "0.000001"]}
    gls = {"K49": ["0.3", "0.03"],
           "EMNIST_bymerge": ["0.3", "0.03"],
           "CIFAR100": ["0.6"]}
    p0 = "scan_train_g_xw"
    bind_job(ns, ds, ls, gls, p0)

