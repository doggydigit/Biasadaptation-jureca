import glob
import numpy as np

for fname in glob.glob("*scdnfs*"):
    if fname[-4:] == ".npy":
        W = np.load(fname)
        print(fname, " --> W0:", W.shape, ", W1:", W.T.shape)
        np.save(fname, W.T)