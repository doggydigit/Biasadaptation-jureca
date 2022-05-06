import numpy as np
import scipy.linalg as sla
import torch
import torchvision.datasets as tdatasets
import torchvision.transforms as ttransforms
import torch.utils.data as tdata
from tqdm import tqdm
from sklearn.decomposition import MiniBatchDictionaryLearning, dict_learning_online

import copy
import pickle

from datarep import paths
from datarep.matplotlibsettings import *

from biasadaptation.weightmatrices import pmd, bmd
from biasadaptation.utils import samplers, losses, utils
import optim



def sample_binary_task(dataset='EMNIST', task_type='1v1'):
    if task_type == '1v1':
        classes = np.random.choice(np.arange(samplers.N_CLASSES[dataset]), 2, replace=False)
        return [(cc,) for cc in classes]
    elif task_type == '1vall':
        class_idx = np.arange(samplers.N_CLASSES[dataset]),
        c1 = np.random.choice(class_idx, 1, replace=False)
        return [(c1,)] + [tuple([cc for cc in class_idx if cc != c1])]
    else:
        raise Exception('Invalid `task_type`')


def get_data_matrix(n_data, dataset='EMNIST'):
    transforms = ttransforms.Compose([ttransforms.ToTensor(), lambda x: x/samplers.DATA_NORMS[dataset]])
    # transforms = ttransforms.Compose([ttransforms.ToTensor(), ttransforms.Normalize((0.,), (1.,))])
    # transforms = ttransforms.Compose([ttransforms.ToTensor(), ttransforms.Normalize((0.,), (.1,))])
    data_set = tdatasets.EMNIST(paths.tool_path, train=True, download=True, split="bymerge", transform=transforms)
    data_loader = tdata.DataLoader(data_set, batch_size=n_data)


    d, t = next(iter(data_loader))
    s = d.shape
    x = d.numpy().reshape(s[0], s[-1]*s[-2])

    data_mat = utils.differences_numpy(x, n_data)

    return data_mat


def construct_matrices_pmd(n_hs, dataset='EMNIST', save=True):
    """
    Parameters
    ----------
    n_units: list of int
        [number of neurons in first layer, number of neurons in second layer,
         ..., number of neurons in second to last hidden layer]
    """

    Ws = []
    Cs = []
    C = get_data_matrix(100000, dataset=dataset)
    C_orig = copy.deepcopy(C)

    for nh in n_hs:

        n, p = C.shape

        pmd_ = pmd.PenalizedMatrixDecomposition(C, c1=0.5*np.sqrt(n), c2=0.3*np.sqrt(p))
        D, U, V = pmd_(nh)

        # svd for reference
        pmd.calc_svd(pmd_.X_copy, nh)

        C = np.dot(U, D)

        Cs.append(copy.deepcopy(C))
        Ws.append(V)

    if save:
        namestring = '_'.join([str(nh) for nh in n_hs])
        with open(paths.data_path + 'weight_matrices/%s_weight_mats_pmd_nh=%s.p'%(dataset, namestring), 'wb') as f:
            pickle.dump(Ws, f)
            pickle.dump(Cs, f)
            pickle.dump(C_orig, f)

    return Ws, Cs, C_orig


def construct_matrices_scd(n_hs, dataset='EMNIST', save=True):
    """
    Parameters
    ----------
    n_units: list of int
        [number of neurons in first layer, number of neurons in second layer,
         ..., number of neurons in second to last hidden layer]
    """

    Ws = []
    Cs = []
    C = get_data_matrix(100000, dataset=dataset)
    C_orig = copy.deepcopy(C)

    for nh in n_hs:

        n, p = C.shape

        U_, V_, n_iter = dict_learning_online(C, n_components=nh, alpha=0.1, n_iter=100,
                            return_code=True, dict_init=None, callback=None,
                            batch_size=3, verbose=True, shuffle=True,
                            n_jobs=None, method='cd', iter_offset=0,
                            random_state=None, return_inner_stats=False,
                            inner_stats=None, return_n_iter=True,
                            positive_dict=False, positive_code=True,
                            method_max_iter=1000)

        print('--', n_iter)

        mbdl = MiniBatchDictionaryLearning(n_components=nh, alpha=0.1, n_iter=100,
                            batch_size=3, verbose=True, shuffle=True,
                            transform_algorithm='lasso_cd', fit_algorithm='cd',
                            n_jobs=None,  positive_code=True, transform_max_iter=1000)
        mbdl.fit(C)
        # V = mbdl.components_
        # U = mbdl.transform(C)

        C = U_

        Cs.append(C)
        Ws.append(V_)

    if save:
        namestring = '_'.join([str(nh) for nh in n_hs])
        with open(paths.data_path + 'weight_matrices/%s_weight_mats_scd_nh=%s.p'%(dataset, namestring), 'wb') as f:
            pickle.dump(Ws, f)
            pickle.dump(Cs, f)

    return Ws, Cs, C_orig


def run_optimizations_pmd(n_h, n_l, n_task=15, dataset='EMNIST',
                          opt_g=False,
                          n_per_batch=100, n_per_epoch=20, n_epoch=200):
    """

    `n_h`: int
        number of hidden units in each layer (same for each layer)
    `n_l`: int
        number of hidden layers
    """
    namestring = '_'.join([str(n_h) for _ in range(4)])
    with open(paths.data_path + 'weight_matrices/%s_weight_mats_pmd_nh=%s.p'%(dataset, namestring), 'rb') as f:
        Ws = pickle.load(f)
        Cs = pickle.load(f)
        DX = pickle.load(f)

    namestring = '_'.join(['25', '25'])
    with open(paths.data_path + 'weight_matrices/%s_weight_mats_pbmd_nh=%s.p'%(dataset, namestring), 'rb') as f:
        Ws = pickle.load(f)

    # construct the data sampler
    sampler_pair = samplers.SamplerPair('EMNIST', nb_factor_test=10,
                        n_per_batch=n_per_batch, n_per_epoch=n_per_epoch)
    so = ''
    n_inp = sampler_pair.get_input_dim()

    bias_storage = []
    perfs = []
    for nt in range(n_task):
        task = sample_binary_task(dataset=dataset)
        print('optimizing task ', task)
        sampler_pair.set_tasks(task, target_type='perceptron')

        # construct the initial weights and biases
        ws = Ws[:n_l] + [np.ones((1, n_h)) / np.sqrt(n_h)]
        bs = [np.random.randn(n_h, 1) / (10.*n_h) for _ in range(n_l)] + [np.random.randn(1, 1) / 10.]

        print([w.shape for w in ws])
        print([b.shape for b in bs])

        w_final, b_final, perf = optim.run_optim(ws, bs, sampler_pair, n_epoch=n_epoch)

        bias_storage.append({'b_final': b_final, 'perf': perf, 'task': task})

        perfs.append(np.array(perf))

    print('\n>>> average performance per epoch:')
    print(np.mean(perfs, 0))

    with open(paths.data_path + 'bias_storage_pmd_nh=%d_nl=%d_%s.p'%(n_h, n_l, so), 'wb') as f:
        pickle.dump(bias_storage, f)


def test_single_layer(nh, dataset='EMNIST'):
    Ws, Cs, DX = construct_matrices_pmd([nh, nh, nh, nh], dataset=dataset, save=True)
    # Ws, Cs, DX = construct_matrices_scd([nh], dataset=dataset, save=True)
    W, C = Ws[0], Cs[0]

    print('W shape =', W.shape)
    print('C shape =', C.shape)

    print('Frac C nonzero = ', np.sum(np.abs(C) > 1e-8).astype(int) / C.size)

    pl.figure('w_i', figsize=(5,5))
    gs = GridSpec(5,5)
    gs.update(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.05, wspace=0.05)

    for kk in range(nh):
        ii = kk//5
        jj = kk%5

        ax = pl.subplot(gs[ii,jj])
        ax.imshow(utils.to_image_mnist(W[kk,:]))

        ax.set_xticks([])
        ax.set_yticks([])


    pl.figure('DX', figsize=(10,5))
    gs1 = GridSpec(5,5)
    gs1.update(top=0.95, bottom=0.05, left=0.05, right=0.475, hspace=0.05, wspace=0.05)
    gs2 = GridSpec(5,5)
    gs2.update(top=0.95, bottom=0.05, left=0.525, right=0.95, hspace=0.05, wspace=0.05)

    inds = np.random.choice(np.arange(DX.shape[0]), size=25, replace=False)

    for kk, ind in enumerate(inds):
        ii = kk//5
        jj = kk%5

        ax1 = pl.subplot(gs1[ii,jj])
        ax1.imshow(utils.to_image_mnist(DX[ind,:]))

        ax1.set_xticks([])
        ax1.set_yticks([])

        dx_ = np.dot(C[ind:ind+1,:], W)

        ax2 = pl.subplot(gs2[ii,jj])
        ax2.imshow(utils.to_image_mnist(dx_))

        ax2.set_xticks([])
        ax2.set_yticks([])

    pl.show()


def normnorm(v1, v2):
    return np.linalg.norm(v1 / np.linalg.norm(v1) - v2 / np.linalg.norm(v2))


def pl_im(ax, arr):
    ax.imshow(utils.to_image_mnist(arr))
    ax.set_xticks([])
    ax.set_yticks([])


def difference_vector_representations(dataset='EMNIST'):
    n_hs = [25, 25, 25, 25]

    namestring = '_'.join([str(nh) for nh in n_hs])
    with open(paths.data_path + 'weight_matrices/%s_weight_mats_pmd_nh=%s.p'%(dataset, namestring), 'rb') as f:
        Ws = pickle.load(f)
        Cs = pickle.load(f)
        DX = pickle.load(f)

    Bs = [((np.abs(C) > 1e-6)).astype(float) for C in Cs]
    Bs_ = [B / np.linalg.norm(B, axis=1)[:, None] * np.linalg.norm(C, axis=1)[:, None] for B, C in zip(Bs, Cs)]

    DX_1L = np.dot(Bs_[0], Ws[0])
    DX_2L = np.dot(np.dot(Bs_[1], Ws[1])*Bs[0], Ws[0])
    DX_3L = np.dot(np.dot(np.dot(Bs_[2], Ws[2])*Bs[1], Ws[1])*Bs[0], Ws[0])
    DX_4L = np.dot(np.dot(np.dot(np.dot(Bs_[3], Ws[3])*Bs[2], Ws[2])*Bs[1], Ws[1])*Bs[0], Ws[0])

    # DX_1L_ = np.dot(Bs_[0], Ws[0])
    # DX_2L_ = np.dot(np.dot(Bs_[1], Ws[1]), Ws[0])
    # DX_3L_ = np.dot(np.dot(np.dot(Bs_[2], Ws[2]), Ws[1]), Ws[0])
    # DX_4L_ = np.dot(np.dot(np.dot(np.dot(Bs_[3], Ws[3]), Ws[2]), Ws[1]), Ws[0])

    DX_1L_ = np.dot(Cs[0], Ws[0])
    DX_2L_ = np.dot(np.dot(Cs[1], Ws[1]), Ws[0])
    DX_3L_ = np.dot(np.dot(np.dot(Cs[2], Ws[2]), Ws[1]), Ws[0])
    DX_4L_ = np.dot(np.dot(np.dot(np.dot(Cs[3], Ws[3]), Ws[2]), Ws[1]), Ws[0])

    # DX_1L /= np.linalg.norm(DX_1L, axis=1)[:,None]
    # DX_2L /= np.linalg.norm(DX_2L, axis=1)[:,None]
    # DX_3L /= np.linalg.norm(DX_3L, axis=1)[:,None]
    # DX_4L /= np.linalg.norm(DX_4L, axis=1)[:,None]

    # DX_1L_ /= np.linalg.norm(DX_1L_, axis=1)[:,None]
    # DX_2L_ /= np.linalg.norm(DX_2L_, axis=1)[:,None]
    # DX_3L_ /= np.linalg.norm(DX_3L_, axis=1)[:,None]
    # DX_4L_ /= np.linalg.norm(DX_4L_, axis=1)[:,None]

    pl.figure('DX', figsize=(10,5))
    gs0 = GridSpec(5,5)
    gs0.update(top=0.95, bottom=0.05, left=0.05, right=0.475, hspace=0.05, wspace=0.05)
    gs1 = GridSpec(5,5)
    gs1.update(top=0.95, bottom=0.05, left=0.525, right=0.95, hspace=0.05, wspace=0.05)

    inds = np.random.choice(np.arange(DX.shape[0]), size=5, replace=False)

    for kk, ind in enumerate(inds):

        ax0 = pl.subplot(gs0[kk,0])
        ax1 = pl.subplot(gs0[kk,1])
        ax2 = pl.subplot(gs0[kk,2])
        ax3 = pl.subplot(gs0[kk,3])
        ax4 = pl.subplot(gs0[kk,4])

        pl_im(ax0, DX[ind])
        pl_im(ax1, DX_1L[ind])
        pl_im(ax2, DX_2L[ind])
        pl_im(ax3, DX_3L[ind])
        pl_im(ax4, DX_4L[ind])

        dxs = (normnorm(DX[ind], DX_1L[ind]),
               normnorm(DX[ind], DX_2L[ind]),
               normnorm(DX[ind], DX_3L[ind]),
               normnorm(DX[ind], DX_4L[ind]),
              )

        print("\n|Dx - Dx_RL| --> 1L = %.4f, 2L = %.4f, 3L = %.4f, 4L = %.4f"%dxs)


        ax0 = pl.subplot(gs1[kk,0])
        ax1 = pl.subplot(gs1[kk,1])
        ax2 = pl.subplot(gs1[kk,2])
        ax3 = pl.subplot(gs1[kk,3])
        ax4 = pl.subplot(gs1[kk,4])

        pl_im(ax0, DX[ind])
        pl_im(ax1, DX_1L_[ind])
        pl_im(ax2, DX_2L_[ind])
        pl_im(ax3, DX_3L_[ind])
        pl_im(ax4, DX_4L_[ind])

        dxs = (normnorm(DX[ind], DX_1L_[ind]),
               normnorm(DX[ind], DX_2L_[ind]),
               normnorm(DX[ind], DX_3L_[ind]),
               normnorm(DX[ind], DX_4L_[ind]),
              )
        print("|Dx - Dx_DC| --> 1L = %.4f, 2L = %.4f, 3L = %.4f, 4L = %.4f"%dxs)

    pl.show()


def binary_match(xx, W, verbose=True):
    """
    xx: np.array of float
        vector of `k` elements
    W: np.ndarray of float
        matrix of `(p,k)` elements
    """

    v0 = np.zeros(W.shape[1])
    i0s = list(range(W.shape[0]))
    i1s = []
    s0, s1 = -0.1, 0.
    kk = 0

    while s1 > s0 and kk < W.shape[0]:
        if verbose: print('\n--> iter %d, current score = %.5f'%(kk, s0))
        kk += 1

        W__ = np.array([W[ii] for ii in i0s])
        W_ = v0[None,:] + W__

        scores = np.dot(W_, xx) / (np.linalg.norm(W_, axis=1) * np.linalg.norm(xx))
        imax = np.argmax(scores)

        s0 = s1
        s1 = scores[imax]

        if s1 > s0:
            i0 = i0s.pop(imax)
            i1s.append(i0)

            v0 = np.sum(np.array([W[ii] for ii in i1s]), axis=0)

        sdot = np.dot(v0/np.linalg.norm(v0), xx/np.linalg.norm(xx))
        sdiff = normnorm(v0, xx)

        if verbose: print("new score = %.5f, score diff = %.5f"%(sdot, sdiff))

    return v0, i1s, sdot


def matching_pursuit(dataset='EMNIST'):
    # n_hs = [25, 25, 25, 25]
    n_hs = [100, 25]
    # n_hs = [50, 25]

    NN = 10000

    namestring = '_'.join([str(nh) for nh in n_hs])
    with open(paths.data_path + 'weight_matrices/%s_weight_mats_pmd_nh=%s.p'%(dataset, namestring), 'rb') as f:
        Ws = pickle.load(f)
        Cs = pickle.load(f)
        DX = pickle.load(f)

    nh = n_hs[-1]
    Bs = [((np.abs(C) > 1e-6)).astype(float) for C in Cs]

    print('\n-------\nBegin Testing BMD')
    # U, D, V = sla.svd(Cs[0][:100,:], full_matrices=False)
    # U = U[:,:nh]
    # D = D[:nh]
    # V = V[:nh,:]
    # U = U @ np.diag(D)
    print('i')
    C_ = np.linalg.lstsq(Ws[0].T, DX.T, rcond=None)[0].T
    print('ii')

    VV = np.random.randn(n_hs[1],n_hs[0])
    # VV = np.eye(nh)
    O0 = bmd.minimize_o_cost(DX[:NN], Ws[0])

    O, W = bmd.find_weight_and_mask(np.ones((NN, nh)), VV, C_[:NN,:], nh, n_iter=100)
    # O, W = bmd.find_weight_and_mask(np.ones((NN, nh)), VV, Cs[0][:NN,:], 25, n_iter=100)
    # O, W = bmd.find_weight_and_mask(np.ones((NN, nh)), VV, O0, 25, n_iter=100)

    # namestring = '_'.join(['100', '50'])
    with  open(paths.data_path + 'weight_matrices/%s_weight_mats_pbmd_nh=%s.p'%(dataset, namestring), 'wb') as f:
        Ws_ = [Ws[0], W]
        pickle.dump(Ws_, f)

    V = np.dot(np.dot(O, W) * Bs[0][:NN,:], Ws[0])


    V_ = ((np.ones((NN, nh)) @ VV) * O0) @ Ws[0]
    # V_ = ((np.ones((NN, nh)) @ VV) * Bs[0][:NN,:]) @ Ws[0]
    print('End Testing BMD\n-------\n')

    # DX_1L_ = np.dot(Cs[0], Ws[0])
    DX_1L_ = np.dot(C_, Ws[0])
    DX_2L_ = np.dot(np.dot(Cs[1], Ws[1]), Ws[0])
    DX_3L_ = np.dot(np.dot(np.dot(Cs[2], Ws[2]), Ws[1]), Ws[0])
    DX_4L_ = np.dot(np.dot(np.dot(np.dot(Cs[3], Ws[3]), Ws[2]), Ws[1]), Ws[0])

    Bs = [B / np.linalg.norm(B, axis=1)[:, None] * np.linalg.norm(C, axis=1)[:, None] for B, C in zip(Bs, Cs)]

    DX_1L = np.dot(Bs[0], Ws[0])

    # inds = np.random.choice(np.arange(DX.shape[0]), size=5, replace=False)
    # inds = [1,3,5,7,9]
    inds = [0,2,4,6,8]

    pl.figure('DX', figsize=(10,5))
    gs0 = GridSpec(5,5)
    gs0.update(top=0.95, bottom=0.05, left=0.05, right=0.475, hspace=0.05, wspace=0.05)
    gs1 = GridSpec(5,5)
    gs1.update(top=0.95, bottom=0.05, left=0.525, right=0.95, hspace=0.05, wspace=0.05)

    for kk, ind in enumerate(inds):
        print("\n--> Image %d <--"%ind)

        # original
        axa = pl.subplot(gs0[kk,0])
        axb = pl.subplot(gs1[kk,0])

        pl_im(axa, DX[ind])
        pl_im(axb, DX[ind])

        # 1st layer
        print("-- layer 1 --")
        axa = pl.subplot(gs0[kk,1])
        axb = pl.subplot(gs1[kk,1])

        v1, idx, score = binary_match(DX[ind], Ws[0], verbose=False)
        vn = v1

        pl_im(axa, vn)
        pl_im(axb, DX_1L_[ind])

        print('components         : %s'%(', '.join([str(ii) for ii in np.sort(idx)])))
        print('coordinates (c > 0): %s'%(', '.join([str(ii) for ii in np.where(Cs[0][ind] > 0.)[0]])))

        print('diff components : %.6f'%normnorm(vn, DX[ind]))
        print('diff coordinates: %.6f'%normnorm(DX_1L_[ind], DX[ind]))

        # 2nd layer
        print("\n-- layer 2 --")
        axa = pl.subplot(gs0[kk,2])
        axb = pl.subplot(gs1[kk,2])

        v1, idx, score = binary_match(Cs[0][ind], Ws[1], verbose=False)

        idx0 = np.where(Cs[0][ind] > 0.)[0]
        vn = np.dot(v1[idx0], Ws[0][idx0,:])

        pl_im(axa, V[ind])
        pl_im(axb, DX_2L_[ind])

        print('components         : %s'%(', '.join([str(ii) for ii in np.sort(idx)])))
        print('coordinates (c > 0): %s'%(', '.join([str(ii) for ii in np.where(Cs[1][ind] > 0.)[0]])))

        print('diff components : %.6f'%normnorm(V[ind], DX[ind]))
        print('diff components2: %.6f'%normnorm(vn, DX[ind]))
        # print('diff components2: %.6f'%normnorm(V_[ind], DX[ind]))
        print('diff coordinates: %.6f'%normnorm(DX_2L_[ind], DX[ind]))

        print("----------------")

    pl.show()





if __name__ == "__main__":
    # construct_matrices_pmd([100, 25])
    # construct_matrices_scd([100, 25])

    # test_single_layer(25)
    # difference_vector_representations()

    # run_optimizations_pmd(25, 1)
    # run_optimizations_pmd(25, 2)
    # run_optimizations_pmd(25, 3)
    # run_optimizations_pmd(25, 4)

    matching_pursuit()
