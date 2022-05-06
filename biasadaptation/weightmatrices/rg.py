# Random Gabors

import numpy as np
import random
from sklearn.preprocessing import normalize

def get_gabor_kernel(lbda, theta, psi, sigma, gamma, N):
    w = np.zeros((N, N))
    rm = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
    for x in range(N):
        for y in range(N):
            r = np.matmul(rm, np.array([x, y])-(N/2))
            w[x, y] = np.exp(-(r[0]**2 + gamma**2*r[1]**2)/(2*sigma**2)) * np.cos(2*np.pi*r[0]/lbda + psi)

    return w

# these are heuristic boundaries for random Gabors
def get_random_gabor(N):
    return get_gabor_kernel((2*N-N/4)*random.random()+N/4,
                            2*np.pi*random.random(), 2*np.pi*random.random(),
                            (N-N/8)*random.random()+N/8, random.random(), N)


def get_weightmatrix_rg(data_matrix, n_h):
    print("creating weigth matrix for RG for "+str(n_h)+" hidden neurons...")
    N = int(np.sqrt(data_matrix.shape[1]))
    W = np.zeros((n_h, N**2))
    for i in range(n_h):
        W[i, :] = get_random_gabor(N).flatten()
    W = normalize(W)
    return W
