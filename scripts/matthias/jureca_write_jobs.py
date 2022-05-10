
if __name__ == '__main__':
    ns = ["500,500,500", "100,100,100", "25,25,25", "500,500", "100,100", "25,25", "500", "100", "25"]
    ds = ["K49", "EMNIST_bymerge", "CIFAR100"]
    ls = {"K49": ["0.0006", "0.0003", "0.0002", "0.0001", "0.00006", "0.00003", "0.00002"],
          "EMNIST_bymerge": ["0.0006", "0.0003", "0.0002", "0.0001", "0.00006", "0.00003", "0.00002"],
          "CIFAR100": ["0.00006", "0.00003", "0.00002", "0.00001", "0.000006", "0.000003", "0.000002"]}
    gls = {"K49": ["0.6", "0.3", "0.2", "0.1", "0.06", "0.03", "0.02"],
           "EMNIST_bymerge": ["0.6", "0.3", "0.2", "0.1", "0.06", "0.03", "0.02"],
           "CIFAR100": ["0.6", "0.3", "0.2", "0.1", "0.06", "0.03", "0.02"]}
    ss = ["0,1", "2,3", "4,5", "6,7", "8,9", "10,11", "12,13", "14,15", "16,17", "18,19"]
    job = 0
    proc = 0
    job_text = "scan_train_g_xw"
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
                            job_text = "scan_train_g_xw"
                        else:
                            proc += 1
    if not proc == 0:
        f = open("jureca_jobs/job_{}.txt".format(job), "w")
        f.write(job_text)
        f.close()

