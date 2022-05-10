from pickle import load as pickle_load

if __name__ == '__main__':
    ns = [[500, 500, 500], [100, 100, 100], [25, 25, 25], [500, 500], [100, 100], [25, 25], [500], [100], [25]]
    ds = ["K49", "EMNIST_bymerge", "CIFAR100"]
    ls = ["0.1", "0.01", "0.001", "0.0001", "0.00001", "0.000001"]
    gls = ["0.1", "0.01", "0.001", "0.0001", "0.00001", "0.000001"]
    ss = ["0,1", "2,3", "4,5", "6,7", "8,9", "10,11", "12,13", "14,15", "16,17", "18,19"]
    job_text = "scan_train_g_xw"
    fdir = "results/{}/individual/"
    nrsubjobs = 10
    for p1 in ns:
        for p2 in ds:
            for p3 in ls:
                for p4 in gls:
                    fname = "network_{}_{}_mse_tanh_es_wlr_{}_glr_{}".format(p1, p2, p3, p4)
                    for p5 in range(nrsubjobs):
                        fpath = fdir + "individual_seed/" + fname + "_{}.pickle".format([2*p5, 2*p5+1])
                        with open(fpath, "rb") as f:
                            result = pickle_load(f)
                            print(result)
                            exit
