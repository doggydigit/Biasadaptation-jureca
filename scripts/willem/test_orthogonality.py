import numpy as np

import helperfuncs

w_pmd = helperfuncs.get_weight_matrix_in(100, 'pmdd')
print(w_pmd.shape)

print(np.dot(w_pmd[:,85], w_pmd[:,85]))