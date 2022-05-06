# Principal Component Analysis

import time
from tqdm import tqdm

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize

def get_pca_trafo_matrix(data_loader, n_h):
    transformer = IncrementalPCA(n_components=n_h)
    for (d, t) in tqdm(data_loader):
        s = d.shape
        transformer.partial_fit(d.numpy().reshape(data_loader.batch_size, s[-1]*s[-2]))

    return transformer.components_

def get_weightmatrix_pca(data_loader, n_h):
    print("creating weigth matrix for PCA for "+str(n_h)+" hidden neurons...")
    s = data_loader.dataset.data.shape
    n_in_features = s[-1]*s[-2]
    assert n_h <= n_in_features, "Number of requested principal components higher than input dimensionality!"
    W = get_pca_trafo_matrix(data_loader, n_h)
    W = normalize(W)
    return W
