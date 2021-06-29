import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path =  'ex1data1.txt'

def prepare_data(path):
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

    data.insert(0, 'ones', 1)

    cols = data.shape[1]
    X = data.iloc[:, :-1]  # X是data里的除最后列
    y = data.iloc[:, cols - 1:cols]  # y是data最后一列

    X = X.values
    y = y.values
    theta = np.matrix(np.array([0, 0]))
    return X ,y ,theta

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)

    return np.sum(inner) / (2 * len(X))


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)



    return theta, cost

X, y, theta=prepare_data(path)
# erro1=computeCost(X, y, theta)
# print(erro1)

# alpha = 0.01
# iters = 1500

alpha = 0.2
iters = 15

g, cost = gradientDescent(X, y, theta, alpha, iters)
print(g)