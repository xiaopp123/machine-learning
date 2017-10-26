#encoding=utf-8
import numpy as np
import pylab as pl
from sklearn import svm

#we create 40 separable points
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

#fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X,Y)

#get the separating hyperoplane
#$w0x + w1y + w3 = 0可以写成y=-(w0/w1)x-w3/w1
#得到w向量
w = clf.coef_[0]
a = -w[0] / w[1]
#随机得到50个[-5,5]之间的数
xx = np.linspace(-5, 5)
#根据上面的y的表达式,求每个x对应的y值,w3值为clf.intercept_[0]
yy = a * xx - (clf.intercept_[0] / w[1])

#print(w[0], w[1])
#print(type(xx))
#print(yy)

#support vectors
#得到支持向量的两条直线,与分割线平行
#第一个支持向量
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
#最后一个支持向量
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

#print "w: ", w
#print "a: ", a

pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

#print(type(clf.support_vectors_))
#print(clf.support_vectors_[:,0])
pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s = 80, facecolors='none')
pl.scatter(X[:, 0], X[:, 1], c = Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()


