import numpy as np

import matplotlib.pyplot as pl
import matplotlib.colors as mcolors

aa = np.array([[1.,2.,3.],
               [4.,5.,6.]])

bb = np.array([1.,2.,3.,4.,5.,6.])

norm = mcolors.Normalize(vmin=1., vmax=6.)

print(norm(aa))
print(norm(bb))