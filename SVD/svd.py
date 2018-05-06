# -*- coding: utf-8 -*-

from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
import ipdb
ipdb.set_trace()

X = sparse_random_matrix(100, 100, density=0.01, random_state=42)
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
svd.fit(X)
print(svd.explained_variance_ratio_)
print(svd.explained_variance_ratio_.sum())
print(svd.singular_values_)
