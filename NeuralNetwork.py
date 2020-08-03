import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def Sigmoid(x, w):
    t = np.dot(x, w)
    return 1 / (1 + np.exp(-t))


class NeuralNetwork:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w1 = np.random.uniform(-1, 1, (X.shape[1], 2))
        self.w2 = np.random.uniform(-1, 1, (2, 1))

    def Cost(self):
        return np.sum((self.output - self.y) ** 2) / (2 * len(y))

    def feedForward(self):
        self.hl = Sigmoid(self.X, self.w1)
        self.output = Sigmoid(self.hl, self.w2)
        return self.output

    def backpropagation(self, alpha):
        cw2 = self.output*(1-self.output)*(y - self.output)
        cw1 = self.hl*(1 - self.hl)*cw2.dot(self.w2.T)
        self.w1 += self.X.T.dot(cw1)*alpha
        self.w2 += alpha*self.hl.T.dot(cw2)

    def Gradient(self, alpha, iteration):
        J = np.empty(iteration)
        for i in range(iteration):
            self.feedForward()
            self.backpropagation(alpha)
            J[i] = self.Cost()
        plt.plot(J)
        plt.show()


if __name__ == "__main__":
    temp = pd.read_csv('data1.txt')
    X = np.array(temp[['x1', 'x2']])
    y = np.array(temp['y']).reshape(-1, 1)
    X = X/100
    nu = NeuralNetwork(X, y)
    nu.Gradient(0.01, 40000)
    z = np.round(nu.output)
    print("Cost:", nu.Cost())
    print("Accuracy Score: ", accuracy_score(y, z))





