
if __name__ == '__main__':
    ns = ["500,500,500", "100,100,100", "25,25,25", "500,500", "100,100", "25,25", "500", "100", "25"]
    ds = ["K49", "EMNIST_bymerge", "CIFAR100"]
    ls = ["0.1", "0.01", "0.001", "0.0001", "0.00001", "0.000001"]
    gls = ["0.1", "0.01", "0.001", "0.0001", "0.00001", "0.000001"]
    ss = ["0,1,2,3", "4,5,6,7", "8,9,10,11", "12,13,14,15", "16,17,18,19"]
    job = 0
    proc = 0
    job_text = "scan_train_g_xw"
    for p1 in ns:
        for p2 in ds:
            for p3 in ls:
                for p4 in gls:
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
