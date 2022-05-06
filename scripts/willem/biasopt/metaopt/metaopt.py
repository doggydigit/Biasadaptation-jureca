"""
optimizations for one hidden layer
"""
import numpy as np
import torch
import torchvision.datasets as tdatasets
import torchvision.transforms as ttransforms
import torchvision.transforms.functional as tfunctional

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
sys.path.append('../..')

from datarep import paths

import deapopt
import helperfuncs, optim

"""
example usage

    python3 metaopt.py --nprocesses 2 --noffspring 2 --ngen 2 --readout tanh --dataset EMNIST

"""

torch.set_num_threads(1)

# read command line args and kwargs
parser = argparse.ArgumentParser()
parser.add_argument("--nhidden", nargs="*", type=int, help="number of hidden neurons", default=[100])

parser.add_argument("--readout", type=str, help="type of readout: \'linear\', \'hardtanh\', \'tanh\', \'sigmoidX\'", default="tanh")
parser.add_argument("--algo", type=str, help="methods to be applied to create weight matrix", default='pmdd')
parser.add_argument("--dataset", type=str, help="dataset to load", default="EMNIST")
parser.add_argument("--wgain", type=int, help="optimize the gains as well", default=0)

parser.add_argument("--nprocesses", type=int, help="number of parallel process", default=2)
parser.add_argument("--noffspring", type=int, help="number of offspring for evolutionary algo", default=2)
parser.add_argument("--ngen", type=int, help="number of generations for evolutionary algo", default=2)
parser.add_argument("--datasetpath", type=str, help="path where dataset is downloaded", default='/Users/wybo/Data/')

parser.add_argument("--path", type=str, help="path to which to save the file", default="")

args = parser.parse_args()


class ReshapeTransform:
    """
    Transform class to reshape tensors
    """
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


# def get_dataset(dataset, rotate=True, train=True, x_add=0., x_div=1., path=None):
#     """
#     Return the torch dataset
#     """
#     if path is None:
#         path = paths.data_path

#     transform_list = []
#     if rotate:
#         transform_list.extend([ lambda img: tfunctional.rotate(img, -90),
#                                 lambda img: tfunctional.hflip(img),
#                               ])
#     transform_list.extend([ ttransforms.ToTensor(),
#                             lambda x: x / x_div + x_add,
#                             ReshapeTransform((-1,)),
#                           ])


#     kwargs = dict(download=True, transform=ttransforms.Compose(transform_list))
#     if dataset == 'EMNIST':
#         kwargs['split'] = 'bymerge'
#         # kwargs['split'] = 'byclass'
#     kwargs['train'] = True if train else False
#     tdataset = eval('tdatasets.%s(path, **kwargs)'%dataset)

#     return tdataset


def check_positive_units(ws, bs, xdata, xtarget):
    w_0 = torch.FloatTensor(ws[0])
    b_0 = torch.FloatTensor(bs[0])

    wx_0 = torch.matmul(xdata, w_0)
    a_1 =  wx_0 + b_0

    frac_b = float(torch.mean((torch.sign(b_0) + 1.) / 2.)*100.)
    frac_a1 = float(torch.mean((torch.sign(a_1) + 1.) / 2.)*100.)

    idx_m1 = torch.where(xtarget < 0)
    idx_p1 = torch.where(xtarget > 0)

    frac_a1m1 = float(torch.mean((torch.sign(a_1[idx_m1]) + 1.) / 2.)*100.)
    frac_a1p1 = float(torch.mean((torch.sign(a_1[idx_p1]) + 1.) / 2.)*100.)

    print("\n--- % positive ---")
    print("%(b0 > 0) =", frac_b)
    print("%(a1 > 0) =", frac_a1)
    print("%(a1[-1] > 0) =", frac_a1m1)
    print("%(a1[+1] > 0) =", frac_a1p1)
    print("------------------\n")

    return {'b': frac_b, 'a1': frac_a1, 'a1m1': frac_a1m1, 'a1p1': frac_a1p1}


class Evaluator:
    def __init__(self, n_task=5, n_hs=[10,25,50,100,250,500],
                       w_idx=[], b_idx=[0,1], g_idx=[],
                       dataset='EMNIST', algo='pmdd', tasktype='1vall', readout='tanh',
                       n_epoch=100, n_perbatch=100, n_perepoch=20,
                       save_to_path=None, datasetpath=None, with_g=False):
        self.n_task = n_task
        self.n_hs = n_hs

        self.w_idx = w_idx
        self.b_idx = b_idx
        self.g_idx = g_idx
        suffix = helperfuncs.get_suffix(w_idx, b_idx, g_idx)

        self.dataset = dataset
        self.tasktype = tasktype
        self.readout = readout
        self.algo = algo
        self.datasetpath = datasetpath

        self.n_epoch = n_epoch
        self.n_perbatch = n_perbatch
        self.n_perepoch = n_perepoch

        self.save = False
        self.path_name = ''
        if save_to_path is not None:
            nh_str = '-'.join([str(n_h) for n_h in n_hs])
            path_name = os.path.join(save_to_path, "metaopt_interm_1hl_%s_%s_%s_%s_nh=%s_ro=%s/"%(suffix, dataset, algo, tasktype, nh_str, readout))

            # delete old files if they exist
            if os.path.exists(path_name):
                shutil.rmtree(path_name)

            os.mkdir(path_name)
            self.path_name = path_name
            self.save = True

        self.param_dict = {
            # learning rate
            'lr': deapopt.Param(0.03, bounds=[0.0001, 0.1]),
            # bias shift and rescale
            'b_add': deapopt.Param(0., bounds=[-5., 5.]),
            'b_div': deapopt.Param(1., bounds=[0.01, 10.]),
            'x_div': deapopt.Param(1., bounds=[0.01, 100.]),
        }

        if with_g:
            self.param_dict.update({
                'g_add': deapopt.Param(1., bounds=[-5., 5.]),
                'g_div': deapopt.Param(1., bounds=[0.05, 50.]),
                })

        self.param_names = ['lr', 'b_add', 'b_div', 'x_div']
        if with_g:
            self.param_names.extend(['g_add', 'g_div'])
        self.params = [self.param_dict[p_name] for p_name in self.param_names]

        self.objectives = self.n_hs

    def set_param_values(self, param_values):
        for param, param_value in zip(self.params, param_values):
            param.value = param_value

    def optimize_net(self, verbose=False):
        t0 = time.time()
        # train and test datasets
        source_train = helperfuncs.get_dataset(self.dataset, train=True,  rotate=False,
                                    x_div=self.param_dict['x_div'].value, path=self.datasetpath)
        source_test  = helperfuncs.get_dataset(self.dataset, train=False, rotate=False,
                                    x_div=self.param_dict['x_div'].value, path=self.datasetpath)

        performances = []
        f_pos0, f_pos1 = [], []
        for n_h in self.n_hs:

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
                w_in  = helperfuncs.get_weight_matrix_in(n_h, self.algo, dataset=self.dataset, task=task)
                w_out = helperfuncs.get_weight_matrix_out(n_h, algo=self.algo, dataset=self.dataset, task=task, bias_opt=len(self.w_idx)==0)
                ws = [w_in, w_out]

                # initiliaze biasses
                b_div = self.param_dict['b_div'].value
                b_add = self.param_dict['b_add'].value
                bs = [np.random.randn(1, n_h) / (n_h), np.random.randn(1, 1)]
                bs = [bs[0] / b_div + b_add,
                      bs[1] / b_div + b_add]

                # initiliaze gains
                if 'g_add' in self.param_dict:
                    g_div = self.param_dict['g_div'].value
                    g_add = self.param_dict['g_add'].value
                    gs = [np.random.randn(1, n_h) / (n_h), np.random.randn(1, 1)]
                    gs = [gs[0] / g_div + g_add,
                          gs[1] / g_div + g_add]
                else:
                    gs = None

                # construct data loaders
                data_loaders = helperfuncs.construct_knm_dataloader_triplet(
                                self.dataset, task, copy.deepcopy(task),
                                source_train, source_test,
                                self.n_perbatch, self.n_perbatch*10, 100,
                    )

                (xdata, _), (_, xtarget) = next(iter(data_loaders[1]))
                frac_pos0 = check_positive_units(ws, bs, xdata, xtarget)

                # run the optimization
                ws_opt, bs_opt, gs_opt, perf = optim.run_optim(
                        ws, bs, data_loaders, gs=gs,
                        lr=self.param_dict['lr'].value, readout=self.readout, w_idx=self.w_idx, b_idx=self.b_idx, g_idx=self.g_idx, n_epoch=self.n_epoch,
                        verbose=verbose, return_g=True, test_perf=False
                    )

                frac_pos1 = check_positive_units(ws_opt, bs_opt, xdata, xtarget)

                perfs.append(np.max(perf['train']))
                fpos0.append(frac_pos0)
                fpos1.append(frac_pos1)

            performances.append(np.mean(perfs))
            f_pos0.append({key: np.mean([fp0[key] for fp0 in fpos0]) for key in frac_pos0.keys()})
            f_pos1.append({key: np.mean([fp1[key] for fp1 in fpos1]) for key in frac_pos1.keys()})

        if self.save:
            res_dict = {'params': self.param_dict,
                        'perf': performances,
                        'frac_pos0': f_pos0,
                        'frac_pos1': f_pos1,
                       }
            f_name = str(uuid.uuid4()) + '.pkl'
            with open(os.path.join(self.path_name, f_name), 'wb') as file:
                pickle.dump(res_dict, file)

        t1 = time.time()
        print("time single opt (s): ", t1 - t0)

        return np.array(performances)

    def evaluate_with_lists(self, param_values):
        self.set_param_values(param_values)
        # print([str(key) + ": " + str(param.value) for key, param in self.param_dict.items()])
        performances = self.optimize_net()
        return np.abs(performances - 100.).tolist()

def optimize():
    t0 = time.time()

    w_idx = [0,1] if args.algo == 'rpw' else []
    if args.algo == 'rpw':
        w_idx = [0,1]
        b_idx = [0,1]
    elif args.algo == 'mr' or args.algo == 'br':
        w_idx = [1]
        b_idx = [1]
    else:
        w_idx = []
        b_idx = [0,1]

    if args.wgain:
        g_idx = [0,1]
    else:
        g_idx = []

    suffix = helperfuncs.get_suffix(w_idx, b_idx, g_idx)
    print(suffix)


    evaluator = Evaluator(readout=args.readout,
                          w_idx=w_idx, b_idx=b_idx, g_idx=g_idx,
                          n_task=10, n_hs=args.nhidden,
                          dataset=args.dataset, algo=args.algo, tasktype='1vall',
                          n_epoch=100, n_perbatch=100, n_perepoch=20,
                          save_to_path=args.path, datasetpath=args.datasetpath)
    # evaluator = Evaluator(readout=args.readout,
    #                       n_task=1, n_hs=[100],
    #                       dataset=args.dataset, algo=args.algo, tasktype='1vall',
    #                       n_epoch=4, n_perbatch=100, n_perepoch=4)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.nprocesses) as pool:
        optimisation = deapopt.DEAPOptimisation(evaluator=evaluator,
                                                offspring_size=args.noffspring,
                                                map_function=pool.map)
        final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=args.ngen)

        evaluator.set_param_values(hall_of_fame[0])
        perf = evaluator.optimize_net()

        print(logs)

        nh_str = '-'.join([str(n_h) for n_h in args.nhidden])
        with open(args.path + "metaopt_1hl_%s_%s_%s_1vall_nh=%s_ro=%s.p"%(suffix, args.dataset, args.algo, nh_str, args.readout), 'wb') as file:
            pickle.dump(hall_of_fame, file)
            pickle.dump(logs, file)
            pickle.dump(np.array(perf), file)
            # pickle.dump(, file)

    t1 = time.time()
    print("time passed all opt (s): ", t1 - t0)
    # optimisation = deapopt.DEAPOptimisation(evaluator=Evaluator(),
    #                                         offspring_size=args.noffspring,
    #                                         map_function=map)
    # final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=args.ngen)


def test_hof_params():

    evaluator = Evaluator(readout=args.readout,
                          n_task=1, n_hs=[100],
                          dataset=args.dataset, algo=args.algo, tasktype='1vall',
                          n_epoch=100, n_perbatch=100, n_perepoch=20,
                          save_to_path='')

    with open(args.path + "metaopt_1hl_%s_%s_ro=%s.p"%(args.dataset, args.algo, args.readout), 'rb') as file:
                hall_of_fame = pickle.load(file)
                logs = pickle.load(file)
                perf = pickle.load(file)
                print(">\n> Perf: %.4f\n>"%(-perf))

    # for hof in hall_of_fame:
    # hof = hall_of_fame[0]
    print(hall_of_fame[0])
    hof = hall_of_fame[0][0:3] + hall_of_fame[0][6:7]
    # ['lr', 'b_add', 'b_div', 'w_add', 'w_div', 'x_add', 'x_div']
    # hof = [0.05, -0.5, 10., 0., 1., 0., 0.5] #<-- good
    # hof = [0.05, -0.5, 10., 0., 1., 0., .4]
    # hof = [0.05, 0., 100., 0., 1., 0., 1.]
    # hof = [0.05, -0.5, 10., 0.5] #<-- good
    # hof = [0.005, -0.5, 10., 0.5] #<-- good
    evaluator.set_param_values(hof)

    print('>>>')
    for key, param in evaluator.param_dict.items():
        print(key, ':', param.value)

    perfs = evaluator.optimize_net(verbose=True)

    print("avg perf:", np.mean(perfs))


def trial_hof_params():

    evaluator = Evaluator(readout=args.readout,
                          n_task=20, n_hs=[50],
                          dataset=args.dataset, algo=args.algo, tasktype='1vall',
                          n_epoch=100, n_perbatch=100, n_perepoch=20,
                          save_to_path=None)

    hof = [0.05, -0.5, 10., 0., 1., 0., 0.5] #<-- good
    # hof = [0.05, -0.5, 10., 0., 1., 0., .4]
    # hof = [0.05, 0., 100., 0., 1., 0., 1.]
    hof = [0.05, -0.5, 10., 0.5] #<-- good
    # hof = [0.005, -0.5, 10., 0.5] #<-- good
    evaluator.set_param_values(hof)

    print('>>>')
    for key, param in evaluator.param_dict.items():
        print(key, ':', param.value)

    perfs = evaluator.optimize_net(verbose=True)

    print("avg perf:", np.mean(perfs))

if __name__ == "__main__":
    optimize()
    # test_hof_params()








