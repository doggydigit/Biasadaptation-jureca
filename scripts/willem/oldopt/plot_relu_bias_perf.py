import numpy as np

import pickle
import glob
import copy

from datarep.matplotlibsettings import *
import datarep.paths as paths


def calc_perf_avg_std(trees):
    tree = copy.deepcopy(trees[0])

    for node in tree:
        if 'perf' in node.content:
            perf = []
            for tree_ in trees:
                node_ = tree_[node.index]
                perf.append(np.array(node_.content['perf']))

                if node.index == 1:
                    print(np.array(node_.content['perf']))

            node.content['perf_avg'] = np.mean(perf, 0)
            node.content['perf_std'] = np.std(perf, 0)
            node.content['perf_med'] = np.median(perf, 0)

            if node.index == 1:
                print('!!!')
                print(len(node.content['w_final']))
                print(node.content['perf_avg'])


            # node.content['perf'] = np.median(perf, 0)
            # node.content['perf_std'] = np.std(perf, 0)
            # node.content['perf'] = np.max(perf, 0)
            # node.content['perf_std'] = np.std(perf, 0)

    return tree


def get_glob_name(algo, opt_g, dataset, n_h):
    sg = '_g' if opt_g else ''
    glob_name = paths.data_path + \
                'optim_tree_w_%s%s_%s_NH=%d_*'%(algo, sg, dataset, n_h)

    return glob_name


def calc_performance(algo, opt_g, dataset, n_hs, node_idx=1, n_epochs=-20):
    """
    Compute average (over last 20 epochs) and maximal performance of a given
    optimization node (default bias adaptation), for all algorithms in `algos`

    Parameters
    ----------
    algos: lstr
        the algorithm name, one of 'pca', 'ica', 'rp', 'rg', 'sc', 'scd', 'sm'
    opt_g: bool
        if ``True``, also with gain optimization
    dataset: str
        the dataset
    n_hs: list of ints
        the numbers of hidden units
    node_idx: int
        node index in the optimization tree. Default ``1`` for bias optimization
    n_epochs: int (< 0)
        the number of last epochs over which to compute performance average

    Returns
    -------
    np.array, np.array
        the average performance over last `n_epochs` epochs resp. the maximal
        performance
    """

    perfs_avg, perfs_max = [], []
    for n_h in n_hs:
        print('\n>>>> loading %s for nh = %d'%(algo, n_h))
        glob_name = get_glob_name(algo, opt_g, dataset, n_h)
        print(glob_name)

        trees = []
        for f_name in glob.glob(glob_name):
            print('   > loading file %s'%f_name)
            with open(f_name, 'rb') as f:
                tree = pickle.load(f)
                trees.append(tree)

        if len(trees) > 0:
            perf_tree = calc_perf_avg_std(trees)

            perf_avg = np.mean(perf_tree[node_idx].content['perf_avg'][n_epochs:])
            perf_max = np.max(perf_tree[node_idx].content['perf_avg'][n_epochs:])

            print('\n!!! avg')
            print(perf_tree[node_idx].content['perf_avg'])

            perfs_avg.append(perf_avg)
            perfs_max.append(perf_max)

        else:
            perfs_avg.append(np.nan)
            perfs_max.append(np.nan)

    return perfs_avg, perfs_max


def plot_performances(algos=['random', 'pca', 'ica', 'rp', 'rg', 'sc', 'scd', 'sm'],
                      dataset='EMNIST',
                      n_hs=[10, 25, 50, 100, 250, 500, 1000]):
    """
    Plot the performances of the bias optimizations.

    When `algo = 'random'`, takes the reference of full weight+bias optimization
    from ra
    """

    perfsdict_avg_g, perfsdict_max_g = {}, {}
    perfsdict_avg  , perfsdict_max   = {}, {}

    for algo in algos:
        node_idx = 2 if algo == 'random' else 1

        perfsdict_avg_g[algo], perfsdict_max_g[algo] = calc_performance(algo, True , dataset, n_hs, node_idx=node_idx)
        perfsdict_avg[algo]  , perfsdict_max[algo]   = calc_performance(algo, False, dataset, n_hs, node_idx=node_idx)

    all_res = [perfsdict_avg_g, perfsdict_max_g,
               perfsdict_avg,   perfsdict_max]

    all_titles = [r'AVG(% correct) Bias + Gain', r'MAX(% correct) Bias + Gain',
                  r'AVG(% correct) Bias'       , r'MAX(% correct) Bias']

    pl.figure('perfs', figsize=(14,7))
    axes = [pl.subplot(221),
            pl.subplot(223),
            pl.subplot(222),
            pl.subplot(224)]

    xvals = np.arange(len(n_hs))
    xwidth = 1. / (len(algos)+3)

    for ax, res, title in zip(axes, all_res, all_titles):
        ax = myAx(ax)
        ax.set_title(title)

        for ii, algo in enumerate(algos):
            label = 'ref' if algo == 'random' else algo

            ax.bar(xvals+ii*xwidth, res[algo], width=xwidth,
                   color=colours[ii%len(colours)], edgecolor='k', align='edge', label=label)

        myLegend(ax, loc=0)

        ax.set_xticks(xvals + len(algos)*xwidth / 2.)
        ax.set_xticklabels([str(n_h) for n_h in n_hs])

        ax.set_ylim((50.,100.))

    pl.tight_layout()
    pl.show()


if __name__ == "__main__":
    plot_performances(algos=['sm'],
                      n_hs=[25])
    # plot_performances()









