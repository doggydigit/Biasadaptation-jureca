import sys
from mpi4py import MPI
from scan_params import scan_params
from train_full_dataset import train_b_w_full_dataset, train_g_bw_full_dataset, train_g_xw_full_dataset, \
    train_bg_w_full_dataset, train_binarymr_full_dataset


if __name__ == '__main__':

    if len(sys.argv) == 2:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        jobid = int(sys.argv[1])
        sys.stdout = open("jureca_logs/job_{}_proc_{}.log".format(jobid, rank), 'w')
        with open("jureca_jobs/job_{}.txt".format(jobid)) as job_file:
            jobs = job_file.readlines()

        params = jobs[1 + rank].split(";")
        if len(params) == 5:
            net = [list(map(int, params[0].split(',')))]
            ds = [str(params[1])]
            ls = [float(params[2])]
            cl = float(params[3])
            seeds = list(map(int, params[4].split(',')))
            print(seeds)

            if "scan_train_g_xw" in jobs[0]:
                scan_params(scantype="train_g_xw", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=True, recompute=False,
                            verbose=True, saving=True, g_lr=cl, seeds=seeds)
            elif "scan_train_g_bw" in jobs[0]:
                scan_params(scantype="train_g_bw", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=True, recompute=False,
                            verbose=True, saving=True, g_lr=cl, seeds=seeds)
            elif "scan_train_b_w" in jobs[0]:
                scan_params(scantype="train_b_w", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=True, recompute=False,
                            verbose=True, saving=True, b_lr=cl, seeds=seeds)
            elif "scan_train_bg_w" in jobs[0]:
                scan_params(scantype="train_bg_w", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=True, recompute=False,
                            verbose=True, saving=True, bg_lr=cl, seeds=seeds)
            elif "scan_train_bmr" in jobs[0]:
                scan_params(scantype="train_bmr", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=True, recompute=False,
                            verbose=True, saving=True, r_lr=cl, seeds=seeds)
            else:
                raise ValueError(jobs[0])
        elif len(params) == 3 and jobs[0] == "train_full":
            rc = True
            vb = True
            tt = str(params[2])
            net = [list(map(int, params[1].split(',')))]
            ds = [str(params[2])]
            if tt == "train_b_w":
                train_b_w_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb)
            elif tt == "train_g_bw":
                train_g_bw_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb)
            elif tt == "train_g_xw":
                train_g_xw_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb)
            elif tt == "train_bg_w":
                train_bg_w_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb)
            elif tt == "train_bmr":
                train_binarymr_full_dataset(dataset=ds, nrhidden=net, recompute=rc, verbose=vb)
            else:
                raise ValueError(jobs[0])
        else:
            raise ValueError(params)
        sys.stdout.close()
    elif len(sys.argv) == 1:
        print(0)
    else:
        raise ValueError(sys.argv)
