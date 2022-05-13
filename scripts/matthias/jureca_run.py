import sys
from mpi4py import MPI
from scan_params import scan_params


if __name__ == '__main__':

    if len(sys.argv) == 2:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        jobid = int(sys.argv[1])
        sys.stdout = open("jureca_logs/job_{}_proc_{}.log".format(jobid, rank), 'w')
        with open("jureca_jobs/job_{}.txt".format(jobid)) as job_file:
            jobs = job_file.readlines()

        if "scan_train_g_xw" in jobs[0]:
            params = jobs[1+rank].split(";")
            if not len(params) == 5:
                raise ValueError(params)
            net = [list(map(int, params[0].split(',')))]
            ds = [str(params[1])]
            ls = [float(params[2])]
            gl = float(params[3])
            seeds = list(map(int, params[4].split(',')))
            print(seeds)
            scan_params(scantype="train_g_xw", nrhiddens=net, datasets=ds, lrs=ls, early_stopping=True, recompute=False,
                        verbose=True, saving=True, g_lr=gl, seeds=seeds)
        else:
            raise ValueError(jobs[0])
        sys.stdout.close()
    else:
        raise ValueError(sys.argv)
