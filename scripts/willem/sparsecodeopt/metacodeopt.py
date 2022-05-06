import numpy as np
import torch

import argparse
import pickle
import copy
import concurrent.futures
import time
import uuid
import os
import shutil
import subprocess

import sys
sys.path.append('..')
sys.path.insert(0, '../biasopt/metaopt')
# sys.path.insert(0,'/home/anand/')

import optim_new, helperfuncs
import codedataset
import metaopt, deapopt, tools, algorithms


# read command line args and kwargs
parser = argparse.ArgumentParser()
parser.add_argument("--nhidden", nargs="*", type=int, help="number of hidden neurons", default=[100])

parser.add_argument("--readout", type=str, help="type of readout: \'linear\', \'hardtanh\', \'tanh\', \'sigmoidX\'", default="tanh")
parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")

parser.add_argument("--nprocesses", type=int, help="number of parallel process", default=2)
parser.add_argument("--noffspring", type=int, help="number of offspring for evolutionary algo", default=2)
parser.add_argument("--ngen", type=int, help="number of generations for evolutionary algo", default=2)
parser.add_argument("--datasetpath", type=str, help="path where dataset is downloaded", default='/Users/wybo/Data/code_matrices')

parser.add_argument("--path", type=str, help="path to which to save the file", default="")

args = parser.parse_args()



"""
example usage

    python3 metacodeopt.py --nprocesses 2 --noffspring 2 --ngen 2 --readout tanh

"""


class CodeEvaluator(metaopt.Evaluator):
    """
    call base class initializer with `algo = 'code'`
    """

    def optimize_net(self, verbose=False):
        t0 = time.time()

        performances = []
        f_pos0, f_pos1 = [], []
        for n_h in self.n_hs:

            # construct the data sampler
            source_train = codedataset.CodeDataset(n_h, algo_w='sc', dataset=self.dataset, algo_c='sc', path=self.datasetpath, train=True)
            source_test = codedataset.CodeDataset(n_h, algo_w='sc', dataset=self.dataset, algo_c='sc', path=self.datasetpath, train=False)

            # sample the possible tasks
            seed = np.random.randint(50000)
            # seed = 16
            tasks = helperfuncs.sample_binary_tasks_(self.n_task,
                            dataset=self.dataset, task_type=self.tasktype, seed=seed
                )

            perfs = []
            fpos0, fpos1 = [], []
            for task in tasks:
                print('\n', task, '\n')

                # initialize weights
                w_out = helperfuncs.get_weight_matrix_out(n_h, bias_opt=False)
                ws = [w_out]

                # initilize biasses
                b_div = self.param_dict['b_div'].value
                b_add = self.param_dict['b_add'].value
                bs = [np.random.randn(1, 1) / b_div + b_add]

                # construct data loaders
                data_loaders = helperfuncs.construct_knm_dataloader_triplet(
                                self.dataset, task, copy.deepcopy(task),
                                source_train, source_test,
                                self.n_perbatch, self.n_perbatch*10, 100,
                    )

                # run the optimization
                ws_opt, bs_opt, gs_opt, perf = optim_new.run_optim(
                        ws, bs, data_loaders,
                        lr=self.param_dict['lr'].value, readout=self.readout, w_idx=[0], n_epoch=self.n_epoch,
                        verbose=verbose, test=False,
                    )

                perfs.append(np.max(perf['train']))

            performances.append(np.mean(perfs))

        if self.save:
            res_dict = {'params': self.param_dict,
                        'perf': performances,
                       }
            f_name = str(uuid.uuid4()) + '.pkl'
            with open(os.path.join(self.path_name, f_name), 'wb') as file:
                pickle.dump(res_dict, file)

        t1 = time.time()
        print("time single opt (s): ", t1 - t0)

        return np.array(performances)


def optimize():
    t0 = time.time()

    evaluator = CodeEvaluator(readout=args.readout,
                              n_task=5, n_hs=args.nhidden,
                              dataset=args.dataset, algo='code', tasktype='1vall',
                              n_epoch=100, n_perbatch=100, n_perepoch=20,
                              save_to_path=args.path, datasetpath=args.datasetpath)
    # evaluator = CodeEvaluator(readout=args.readout,
    #                           n_task=1, n_hs=[100],
    #                           dataset=args.dataset, algo='code', tasktype='1vall',
    #                           n_epoch=4, n_perbatch=100, n_perepoch=4, datasetpath=args.datasetpath)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.nprocesses) as pool:
        optimisation = deapopt.DEAPOptimisation(evaluator=evaluator,
                                                offspring_size=args.noffspring,
                                                map_function=pool.map)
        final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=args.ngen)

        evaluator.set_param_values(hall_of_fame[0])
        perf = evaluator.optimize_net(verbose=True)

        print(logs)

        nh_str = '-'.join([str(n_h) for n_h in args.nhidden])
        with open(args.path + "metaopt_1hl_%s_%s_1vall_nh=%s_ro=%s.p"%(args.dataset, 'code', nh_str, args.readout), 'wb') as file:
            pickle.dump(hall_of_fame, file)
            pickle.dump(logs, file)
            pickle.dump(np.array(perf), file)
            # pickle.dump(, file)

    t1 = time.time()
    print("time passed all opt (s): ", t1 - t0)


if __name__ == "__main__":
    optimize()