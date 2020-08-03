import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def Sigmoid(X, theta):
    t = np.dot(X, theta)
    return 1/(1 + np.exp(-t))

def Cost(X, y, theta):
    t = Sigmoid(X, theta)
    return -1*sum((y*np.log(t) + (1 - y)*np.log(1 - t))/len(y))

def Gradient(X, y, theta, alpha, num):
    plt.figure()
    J = np.empty(shape=(num))
    for i in range(num):
        theta = theta - alpha*(np.dot(X.T, (Sigmoid(X, theta) - y)))/len(y)
        J[i] = Cost(X, y, theta)
    plt.plot(J)
    #plt.show()
    return theta

if __name__ == '__main__':
    temp = pd.read_csv("data1.txt")
    X = np.array(temp[['x1', 'x2']])
    X = X / 100
    y = np.array(temp['y']).reshape(-1, 1)
    X = np.concatenate((np.ones((len(y), 1)), X), axis=1)
    theta = Gradient(X, y, np.zeros((X.shape[1], 1)), 4, 800)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 1], X[:, 2], zs=y)
    t = Sigmoid(X, theta)
    ax.scatter(X[:, 1], X[:, 2], zs=t)
    print(theta)
    plt.show()
