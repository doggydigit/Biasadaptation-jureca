import numpy as np
import scipy.optimize as so


def relu(x):
    x[x<0] = 0
    return x


class ZeroFinder:
    def __init__(self, Ws, bs, gs=None):
        """
        Defines the weight matrices `Ws`
        """
        if gs is None:
            self.gs = [np.ones_like(b) for b in bs]

        self.Ws = Ws
        self.bs = bs
        self.funcs = [relu for _ in self.Ws[:-1]] + [lambda x: x]

    def tfunc(self, t, x0, x1):
        x = x0 * (1.-t) + x1 * t

        for f, W, b, g in zip(self.funcs, self.Ws, self.bs, self.gs):
            x = f(g * (W @ x) + b)

        return x[0]

    def find_zero(self, x0, x1, verbose=False):
        """
        Finds a zero of the neural network on the line between `x0` and `x1`
        """
        if verbose:
            print('f(0) = %.8f, f(1) = %.8f'%(self.tfunc(0., x0, x1), self.tfunc(1., x0, x1)))

        sol = so.root_scalar(self.tfunc, bracket=[0., 1.], args=(x0, x1))
        t_ = sol.root

        if verbose:
            if sol.converged:
                print('Converged succesfully, root = %.4f'%t_)
            else:
                print(sol.flag)

        return x0 * (1. - t_) + x1 * t_

    def find_affine_transform(self, x):
        """
        Finds the affine transformation wx + b weight vector that the neural networks implements
        at x
        """

        wn = np.eye(self.Ws[0].shape[1])
        idx0 = np.ones(self.Ws[0].shape[1], dtype=bool)

        for ii, (f, W, g, b) in enumerate(zip(self.funcs, self.Ws, self.gs, self.bs)):
            x = f(g * (W @ x) + b)

            if ii < len(self.Ws) - 1:
                idx1 = x > 0
            else:
                idx1 = np.array([True])

            wn = W[idx1,:][:,idx0] @ wn
            idx0 = idx1

        return wn


def test_zerofinder():
    Ws = [np.array([[1., 1.], [-1., -1.]]), np.array([[1.,1.]])]
    bs = [np.array([2.,2.]), np.array([-8.])]

    x0 = np.array([9.,9.])
    # x0 = np.array([0.,0.])
    x1 = np.array([10.,10.])

    zf = ZeroFinder(Ws, bs)
    x_ = zf.find_zero(x0, x1)

    assert np.allclose(x_, np.array([3.,3.]))

    wn = zf.find_affine_transform(x_)

    assert np.allclose(wn, np.array([1.,1.]))


if __name__ == "__main__":
    test_zerofinder()




