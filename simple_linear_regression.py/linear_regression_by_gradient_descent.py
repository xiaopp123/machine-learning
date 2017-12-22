# -*- coding: utf-8 -*-

import pylab
import numpy as np


def compute_error(b, m, data):

    totalError = 0
    x = data[:, 0]
    y = data[:, 1]

    totalError = (y - m * x - b)**2
    # print(totalError)
    totalEror = np.sum(totalError, axis=0)

    return totalEror / float(len(data))


def compute_gradient(b_cuurent, m_current, data, learning_rate):
    b_gradient = 0
    m_gradient = 0

    N = float(len(data))

    x = data[:, 0]
    y = data[:, 1]
    b_gradient = 2 / N * (m_current * x + b_cuurent - y)
    b_gradient = np.sum(b_gradient, axis=0)
    m_gradient = 2 / N * x * (m_current * x + b_cuurent - y)
    m_gradient = np.sum(m_gradient, axis=0)

    new_b = b_cuurent - (learning_rate * b_gradient)
    new_w = m_current - (learning_rate * m_gradient)

    return [new_b, new_w]


def optimizer(data, starting_b, starting_m, learning_rate, num_iter):
    b = starting_b
    m = starting_m

    # gradient descent
    for i in range(num_iter):
        b, m = compute_gradient(b, m, data, learning_rate)
        if i % 100 == 0:
            print("iter {0}: error = {1}".format(i, compute_error(b, m, data)))

    return [b, m]


def plot_data(data, b, m):

    x = data[:, 0]
    y = data[:, 1]
    y_predict = m * x + b
    pylab.plot(x, y, 'o')
    pylab.plot(x, y_predict, 'k-')
    pylab.show()


def Linear_regression():
    # get train data
    data = np.loadtxt("data", delimiter=",")
    # print(data)

    learning_rate = 0.001
    initial_b = 0.0
    initial_m = 0.0
    num_iter = 1000
    # train model
    # print b,m,error
    print("initial variables:\ninitial_b = {0}\nintial_m = {1}\nerror of begin = {2}\\n".format(initial_b, initial_m, compute_error(initial_b, initial_m, data)))

    # optimizing b and m
    [b, m] = optimizer(data, initial_b, initial_m, learning_rate, num_iter)

    print("final formula parmaters:\n b = {1}\n m = {2} error of end = {3}\n".format(num_iter, b, m, compute_error(b, m, data)))

    plot_data(data, b, m)


if __name__ == "__main__":
    Linear_regression()
