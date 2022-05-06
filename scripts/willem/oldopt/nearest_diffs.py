import torch
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
# own package
from biasadaptation.utils import utils

from datarep import paths
from datarep.matplotlibsettings import *



# data import
print("loading data")
data_loader = utils.load_data(dataset='EMNIST', data_path=paths.tool_path, batch_size=10000)
# data_matrix = utils.get_big_data_matrix(data_loader)
data_batch, label_batch = next(iter(data_loader))
# print("computing differences")
# diff_matrix = utils.differences_numpy(data_matrix, 100)
s = data_batch.shape
data_matrix = data_batch.numpy().reshape(data_loader.batch_size, s[-1]*s[-2])
print(data_matrix.shape)


print("constructing ball tree")
neigh = NearestNeighbors(n_neighbors=20, algorithm='ball_tree')
print(neigh.n_neighbors)
neigh.fit(data_matrix)
print("computing neighbors")
nn = neigh.kneighbors(data_matrix, return_distance=False)

pl.figure(figsize=(10,8))
gs = GridSpec(4,5)

for jj in range(data_matrix.shape[0]):
    for ii in range(neigh.n_neighbors):
        ax = pl.subplot(gs[ii//5, ii%5])
        ax.imshow(utils.to_image_mnist(data_matrix[nn[jj,ii]]))

    # pl.tight_layout()
    pl.show()