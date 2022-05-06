# Independent Component Analysis

from sklearn.decomposition import FastICA
from sklearn.preprocessing import normalize


def get_ica_trafo_matrix(data_matrix, n_h):
    transformer = FastICA(n_components=n_h, tol=0.001)
    data_transformed = transformer.fit_transform(data_matrix)
    return transformer.components_

def get_weightmatrix_ica(data_matrix, n_h):
    print("creating weigth matrix for ICA for "+str(n_h)+" hidden neurons...")
    n_in_features = data_matrix.shape[1]
    assert n_h <= n_in_features, "Number of requested independent components higher than input dimensionality!"
    W = get_ica_trafo_matrix(data_matrix, n_h)
    W = normalize(W)
    return W
