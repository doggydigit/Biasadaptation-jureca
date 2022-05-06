# Random Projections

import numpy as np
from sklearn.preprocessing import normalize

def get_weightmatrix_rp(data_matrix, n_h):
    print("creating weigth matrix for RP for "+str(n_h)+" hidden neurons...")
    N = data_matrix.shape[1]
    W = np.random.randn(n_h, N)
    W = normalize(W)
    return W
