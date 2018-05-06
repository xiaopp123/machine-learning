#encoding=utf-8

from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from PIL import Image

#print(__doc__)

#Display progress logs on stdout
#logging.basicConfig(level=logging.INFO, format='%(asctime)s % (message)s')

#download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people()

#返回图像的个数和高度宽带
n_samples, h, w = lfw_people.images.shape

#print(n_samples, h, w)

X = lfw_people.data
n_features = X.shape[1]

y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

##print(target_names)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
##print(X_train)
#
#n_components = 150
#print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
#t0 = time()
#pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
#print("done in %0.3fs" % (time() - t0))
#
#eigenfaces = pca.components_.reshape((n_components, h, w))
#
#print("Projecting the input data on the eigenfaces orthonnormal basis")
#t0 = time()
#X_train_pca = pca.transform(X_train)
#X_test_pca = pca.transform(X_test)
#print("done in %0.3fs"% (time() - t0))
#
##Train a SVM classification model
#
#print("Fitting the classifier to the training set")
##t0 = time()
##param_grid = {'C':[1e3, 5e3, 1e4, 5e4, 1e5],
##        'gamma':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
##clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
##
##print(y_train)
##clf = clf.fit(X_train_pca, y_train)
#t0 = time()
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
#clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
#clf = clf.fit(X_train_pca, y_train)
#print("done in %0.3fs" % (time() - t0))
#print("best estimator found by grid search:")
#print(clf.best_estimator_)
