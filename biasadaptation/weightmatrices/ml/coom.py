import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from biasadaptation.utils import utils, countfuncs

import copy


class CoordinateMatch(nn.Module):
    def __init__(self, w2, w1, b1):
        super().__init__()
        self.w = nn.Parameter(torch.FloatTensor(w2))

        self.w1 = torch.FloatTensor(w1)
        self.b1 = torch.FloatTensor(b1)
        # self.norm_w()

    def norm_w(self):
        """
        Ensure columns of `w` are vectors of unit norm
        """
        self.wn = self.w / torch.norm(self.w, dim=0, keepdim=True)

    def get_residuals(self, x):
        print('---')
        print('i')
        # compute difference vectors
        x_diff, x_midp = utils.differences_and_midpoints_torch(x, x.shape[0])

        # # dimensionalities
        # k = x_diff.shape[0]
        # nt = self.b1.shape[0]
        # nh1 = self.w1.shape[0]

        print(x_midp.shape)
        print(self.w1.T.shape)
        print(self.b1.shape)

        print('ii')
        # activation profile
        ap = countfuncs.get_activation_profile(x_midp, self.w1.T, self.b1).float() # (k x nh1 x nt)
        print(ap.shape)
        # idx = np.random.randint(ap.shape[-1], size=ap.shape[0])
        # ap = ap[:,:,idx]
        # print(ap.shape)
        # ap (k x nh1 x nt) -> (k x nt x nh1)
        ap = torch.transpose(ap, 1, 2)
        # ap (k x nt x nh1) -> diag ap (k x nt x nh1 x nh1)
        ap = torch.diag_embed(ap)

        print('iii')
        # ap x w1 (k x nt x nh1 x n)
        ap_x_w1 = ap @ self.w1

        print('iv')
        # w2 x [ap x w1] (k x nt x nh2 x n)
        w2_x_ap_x_w1 = self.w @ ap_x_w1
        # (k x nt x nh2 x n) -> (k x nt x n x nh2)
        w2_x_ap_x_w1 = torch.transpose(w2_x_ap_x_w1, 2, 3)

        print(w2_x_ap_x_w1.shape)
        print(x_diff[:,None,:,None].shape)
        print(torch.linalg.pinv(w2_x_ap_x_w1).shape)

        print('v')
        # least squares solution for coordinates with gradient (k x nt x nh2)
        c2 = torch.linalg.pinv(w2_x_ap_x_w1) @ x_diff[:,None,:,None]

        print('vi')
        # residual
        res = torch.linalg.norm(x_diff[:,None,:,None] - w2_x_ap_x_w1 @ c2, dim=2)

        print('vii')
        return torch.sum(res)



def find_matrix(data_loader, n_h2, w1, b1,
                  n_max_iter=2000):
    """
    Match the ``n_h``-dimensional subspace to the distribution of difference
    vectors of the EMNIST dataset

    Parameters
    ----------
    data_loader: torch.dataloader
        the data loader object
    n_h2: int
        the number of hidden neurons in the second layer
    w1: torch.FloatTensor (nh1 x n)
        the input weight matrix, nh1 number of hidden units in first layer,
        n the number of input features
    b1: torch.FloatTensor (nt x nh1)
        The biases of the first layer neurons, nt the number of tasks, nh1
        the number of hidden units
    n_max_iter: int
        maximum number of iterations

    Returns
    -------
    torch.FloatTensor (n_h2, n_h1)
        The weight matrix to the second layer
    """
    # get input dimension
    batch = next(iter(data_loader))[0]
    batch_size = batch.shape[0]
    input_dim = batch.view(batch_size, -1).shape[1]

    # subspace optimizer
    w_init = torch.randn(n_h2, w1.shape[0])
    coom = CoordinateMatch(w_init, w1, b1)

    # stochastic gradien descent with MSEloss
    optimizer = optim.Adam(coom.parameters(), lr=0.005, betas=(0.9, 0.999))
    mse = nn.MSELoss()

    kk = 0
    while kk < n_max_iter:
        for ii, (x_data, _) in enumerate(data_loader):
            print('>>', kk, ii)
            w_orig = copy.deepcopy(coom.w).detach()
            optimizer.zero_grad()

            # compute projection, loss, gradient
            loss = coom.get_residuals(x_data)
            loss.backward()
            # perform gradient step
            optimizer.step()

            print('\n>>> iter %d --> loss = %.8f <<<'%(kk+ii, loss))
            print('L%d -> |w_init - w_final| = %.5f'%(ii,torch.linalg.norm(w_orig - coom.w)))

            if kk+ii >= n_max_iter:
                break
        kk += ii

    return coom.w.detach().numpy().T

