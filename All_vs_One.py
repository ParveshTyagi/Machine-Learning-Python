import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def Sigmoid(X, theta):
    t = np.dot(X, theta)
    return 1 / (1 + np.exp(-t))


def Cost(X, y, theta):
    t = Sigmoid(X, theta)
    return -1 * sum((y * np.log(t) + (1 - y) * np.log(1 - t)) / len(y))


def Gradient(X, y, theta, alpha, num):
    J = np.empty(shape=(num))
    for i in range(num):
        theta = theta - alpha * (np.dot(X.T, (Sigmoid(X, theta) - y))) / len(y)
        J[i] = Cost(X, y, theta)
    plt.plot(J)
    plt.show()
    return theta


def Conv(y, x):
    result = np.zeros(y.shape)
    for i in range(len(y)):
        if y[i, 0] == x:
            result[i, 0] = 1
    return result


if __name__ == '__main__':
    temp = pd.read_csv("iris.data")
    X = np.array(temp[['x1', 'x2', 'x3', 'x4']])
    X = X / 10
    y = np.array(temp['y']).reshape(-1, 1)
    c = np.array(temp['y'].unique())
    X = np.concatenate((X, X ** 2), axis=1)
    X = np.concatenate((np.ones((len(y), 1)), X), axis=1)
    m = len(np.unique(y))
    theta = np.zeros((X.shape[1], m))
    y1 = np.zeros((len(y), 1))
    print(c[2])
    for j in range(len(y)):
        if y[j, 0] == c[1]:
            y1[j, 0] = 1
        elif y[j, 0] == c[2]:
            y1[j, 0] = 2
    for i in range(m):
        y2 = Conv(y1, i)
        theta[:, i] = Gradient(X, y2, np.zeros((X.shape[1], 1)), 4, 1000).flatten()
    print(theta)
    x = Sigmoid(X, theta)
    for i in range(x.shape[0]):
        mx = np.max(x[i, :])
        ind = np.where(x[i, :] == mx)
        print(c[ind], y[i, 0])
