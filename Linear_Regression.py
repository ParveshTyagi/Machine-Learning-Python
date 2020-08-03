import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Sigmoid(X, theta):
    t = np.dot(X, theta)
    return 1/(1 + np.exp(-t))

def Cost(X, y, theta):
    return np.sum((((np.dot(X, theta)) - y)**2)/len(y))

def Gradient(X, y, theta, alpha, num):
    plt.figure()
    J = np.empty(shape=(num))
    for i in range(num):
        theta = theta - alpha*(np.dot(X.T, (np.dot(X, theta) - y)))/len(y)
        J[i] = Cost(X, y, theta)
    plt.plot(J)
    plt.show()
    return theta

if __name__ == '__main__':
    temp = pd.read_csv("data2.txt")
    X = np.array(temp['X']).reshape(-1, 1)
    X = X / 25
    y = np.array(temp['y']).reshape(-1, 1)
    X = np.concatenate((np.ones((len(y), 1)), X), axis=1)
    theta = Gradient(X, y, np.zeros((X.shape[1], 1)), 1, 200)
    plt.figure()
    plt.plot(X[:, 1], y, 'X')
    plt.plot(X[:, 1], np.dot(X, theta))
    plt.show()
