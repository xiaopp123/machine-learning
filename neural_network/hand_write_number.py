#encoding=utf-8

#每个图片 8x8, 识别数字:0,1,2,3,4,5,6,7,8,9

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from neural_network import NeuralNetwork
from sklearn.cross_validation import train_test_split

digits = load_digits()
X = digits.data
y = digits.target
#归一化,先减去最小,平移到以0作为基准点,然后在除以最大的,此时X的范围就是[0,1]
X -= X.min()
X /= X.max()

nn = NeuralNetwork([64, 100, 100], 'logistic')
X_train, X_test, y_train, y_test = train_test_split(X, y)

#print("X_train:")
#print(X_train)
#print("x_test:")
#print(X_test)
#
#print("y_train")
#print(y_train)
#
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
print ("start fitting")
nn.fit(X_train, labels_train, epochs=3000)
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

