# -*- coding: utf-8 -*-

#import numpy as np
#
#X = [[2, 0, -1.4],
#     [2.2, 0.2, -1.5],
#     [2.4, 0.1, -1],
#     [1.9, 0, -1.2]]
#
##print (np.mean(X, axis=0))
##print (np.cov(np.array(X).T))
#
#w, v = np.linalg.eig(np.array([[1, -2],
#                               [2, -3]]))
##print('特征值：{}\n特征向量：{}'.format(w, v))
#
#x = np.mat([[0.9, 2.4, 1.2, 0.5, 0.3, 1.8, 0.5, 0.3, 2.5, 1.3],
#            [1, 2.6, 1.7, 0.7, 0.7, 1.4, 0.6, 0.6, 2.6, 1.1]])
#
#x = x.T
#T = x - x.mean(axis=0)
## print T
#C = np.cov(x.T)
##print C
#w, v = np.linalg.eig(C)
#v_ = np.mat(v[:, 0])
#v_ = v_.T
#y = T * v_
##print y

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
y = data.target
X = data.data
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)

print reduced_X
red_x, red_y = [],[]
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced_X)):
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(green_x, green_y, c='g', marker='D')
plt.scatter(blue_x, blue_y, c='b', marker='.')
plt.show()
