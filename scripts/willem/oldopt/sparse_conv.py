import numpy as np
import scipy.linalg as sla
import torch
import torchvision.datasets as tdatasets
import torchvision.transforms as ttransforms
import torch.utils.data as tdata


from datarep import paths
from datarep.matplotlibsettings import *

import penalized_matrix_decomposition as pmd
import binary_matrix_decomposition as bmd
from biasadaptation.utils import samplers, losses, utils
import optim


from sporco.dictlrn import cbpdndl
from sporco import util
from sporco import plot




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


Xdat = get_data_matrix(1000).T

print(Xdat.shape)
s = Xdat.shape
Xdat = Xdat.reshape(28, 28, s[-1])
print(Xdat.shape)

np.random.seed(12345)
D0 = np.random.randn(14, 14, 64)


lmbda = 0.2
opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': 100,
                            'CBPDN': {'rho': 50.0*lmbda + 0.5},
                            'CCMOD': {'rho': 10.0, 'ZeroMean': True}},
                            dmethod='cns')

d = cbpdndl.ConvBPDNDictLearn(D0, Xdat, lmbda, opt, dmethod='cns')
D1 = d.solve()
print("ConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))

D1 = D1.squeeze()
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(util.tiledict(D0), title='D0', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(util.tiledict(D1), title='D1', fig=fig)
# fig.show()

pl.show()


