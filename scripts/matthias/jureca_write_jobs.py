
def write_scan_job(ns, ds, ls, gls, p0, job=0, proc=0):
    ss = ["0,1", "2,3", "4,5", "6,7", "8,9", "10,11", "12,13", "14,15", "16,17", "18,19"]
    job_text = p0
    for p1 in ns:
        for p2 in ds:
            for p3 in ls[p2]:
                for p4 in gls[p2]:
                    for p5 in ss:
                        job_text += "\n" + p1 + ";" + p2 + ";" + p3 + ";" + p4 + ";" + p5
                        if proc == 127:
                            f = open("jureca_jobs/job_{}.txt".format(job), "w")
                            f.write(job_text)
                            f.close()
                            job += 1
                            proc = 0
                            job_text = p0
                        else:
                            proc += 1
    if not proc == 0:
        f = open("jureca_jobs/job_{}.txt".format(job), "w")
        f.write(job_text)
        f.close()
        return job + 1
    else:
        return job


def write_train_job(tts, ns, ds, p0, job=0, proc=0):
    job_text = p0
    for p2 in ds:
        for tt in tts:
            for p1 in ns:
                job_text += "\n" + tt + ";" + p1 + ";" + p2
                if proc == 127:
                    f = open("jureca_jobs/job_{}.txt".format(job), "w")
                    f.write(job_text)
                    f.close()
                    job += 1
                    proc = 0
                    job_text = p0
                else:
                    proc += 1
    if not proc == 0:
        f = open("jureca_jobs/job_{}.txt".format(job), "w")
        f.write(job_text)
        f.close()
        return job + 1
    else:
        return job


if __name__ == '__main__':
    ns = ["500,500,500", "100,100,100", "25,25,25", "500,500", "100,100", "25,25", "500", "100", "25"]
    job = 0

    ds = ["K49", "EMNIST_bymerge", "CIFAR100"]
    tts = ["train_b_w", "train_g_bw", "train_g_xw", "train_bmr", "train_bg_w"]
    p0 = "train_full"
    job = write_train_job(tts=tts, ns=ns, ds=ds, p0=p0, job=job)

    ns = ["500,500,500", "100,100,100", "25,25,25", "500,500", "100,100", "25,25", "500", "100", "25"]
    ds = ["CIFAR100"]
    ls = {"CIFAR100": ["0.000002"]}
    gls = {"CIFAR100": ["0.3", "0.1", "0.03", "0.02", "0.01", "0.006", "0.003", "0.002", "0.001", "0.0001", "0.00003"]}
    p0 = "scan_train_bg_w"
    job = write_scan_job(ns=ns, ds=ds, ls=ls, gls=gls, p0=p0, job=job)

    ds = ["CIFAR100"]
    ls = {"CIFAR100": ["0.00002", "0.00001"],
          "EMNIST_bymerge": ["0.00002"]}
    gls = {"CIFAR100": ["0.1", "0.01", "0.006", "0.003", "0.002", "0.001", "0.0006", "0.0001", "0.00003"],
           "EMNIST_bymerge": ["0.1", "0.01", "0.006", "0.003", "0.002", "0.001", "0.0001", "0.00003"]}
    p0 = "scan_train_bmr"
    job = write_scan_job(ns=ns, ds=ds, ls=ls, gls=gls, p0=p0, job=job)

    # ds = ["K49", "EMNIST_bymerge", "CIFAR100"]
    # ls = {"K49": ["0.001"],
    #       "EMNIST_bymerge": ["0.001"],
    #       "CIFAR100": ["0.03", "0.01", "0.003", "0.001", "0.0003", "0.0001"]}
    # gls = {"K49": ["0.3", "0.2", "0.06", "0.03", "0.02"],
    #        "EMNIST_bymerge": ["0.3", "0.2", "0.06", "0.03", "0.02"],
    #        "CIFAR100": ["0.3", "0.1", "0.03", "0.01"]}
    # p0 = "scan_train_g_xw"
    # job = write_scan_job(ns=ns, ds=ds, ls=ls, gls=gls, p0=p0, job=job)
    # print(job)

    ns = ["500", "100", "25"]

    ds = ["K49"]
    ls = {"K49": ["0.003", "0.0003"]}
    gls = {"K49": ["0.00003"]}
    p0 = "scan_train_b_w"
    job = write_scan_job(ns=ns, ds=ds, ls=ls, gls=gls, p0=p0, job=job)

    ds = ["K49"]
    ls = {"K49": ["0.03"]}
    gls = {"K49": ["0.003", "0.0003", "0.00003"]}
    p0 = "scan_train_b_w"
    job = write_scan_job(ns=ns, ds=ds, ls=ls, gls=gls, p0=p0, job=job)
    print(job)
    ds = ["K49"]
    ls = {"K49": ["0.00003"]}
    gls = {"K49": ["0.003", "0.0003", "0.00003"]}
    p0 = "scan_train_b_w"
    job = write_scan_job(ns=ns, ds=ds, ls=ls, gls=gls, p0=p0, job=job)
