import timeit

import numpy as np

from sklearn import linear_model

from glmnet import linear

from biasadaptation.weightmatrices.ml import _least_angle

W = np.random.randn(1000, 800) # (input dim, nh)

# 1 target (datapoint)
X = np.random.randn(1000)


reg_lars = linear_model.Lars(n_nonzero_coefs=100)
reg_lasso = linear_model.LassoLars(max_iter=100, alpha=1e-4)


reg_en = linear.ElasticNet(n_lambda=100)

print('\n1 target -->')

print("\n> sklearn->Lars:")
print(timeit.timeit("reg_lars.fit(W, X)", number=10, globals=globals()))

print("\n> sklearn->LassoLars:")
print(timeit.timeit("reg_lasso.fit(W, X)", number=10, globals=globals()))

print("\n> sklearn->lars_path:")
print(timeit.timeit("linear_model.lars_path(W, X, max_iter=100)", number=10, globals=globals()))

print("\n> biasadpation->least_angle:")
print(timeit.timeit("_least_angle.sparse_encode_featureselect(X, W.T, n_nonzero_coefs=100)", number=10, globals=globals()))

print("\n> sklearn->ElasticNet:")
print(timeit.timeit("linear_model.enet_path(W, X, n_alphas=1, l1_ratio=1)", number=10, globals=globals()))

print("\n> glmnet->ElasticNet:")
print(timeit.timeit("reg_en.fit(W, X)", number=10, globals=globals()))


# reg_lars = linear_model.Lars(n_nonzero_coefs=10, fit_path=False)
# reg_lasso = linear_model.LassoLars(max_iter=10, fit_path=False)

# print('\nwithout path -->\n')

# print("\n> sklearn->Lars:")
# print(timeit.timeit("reg_lars.fit(W, X)", number=10, globals=globals()))

# print("\n> sklearn->LassoLars:")
# print(timeit.timeit("reg_lasso.fit(W, X)", number=10, globals=globals()))


print('\n50 targets -->\n')

# 50 target (datapoints)
X_ = np.random.randn(1000, 50)
print(X_.shape)


print("\n> sklearn->Lars:")
print(timeit.timeit("reg_lars.fit(W, X_)", number=10, globals=globals()))

print("\n> sklearn->LassoLars:")
print(timeit.timeit("reg_lasso.fit(W, X_)", number=10, globals=globals()))

print("\n> biasadpation->least_angle:")
print(timeit.timeit("_least_angle.sparse_encode_featureselect(X_.T, W.T, n_nonzero_coefs=100)", number=10, globals=globals()))
