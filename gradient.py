import numpy as np
import random
from numpy import genfromtxt


# 迭代阀值，当两次迭代损失函数之差小于该阀值时停止迭代
epsilon = 0.0000001


def getData(dataSet):
    m, n = np.shape(dataSet)
    trainData = np.ones((m, n))
    trainData[:, :-1] = dataSet[:, :-1]
    trainLabel = dataSet[:, -1]
    return trainData, trainLabel


def batchGradientDescent(x, y, theta, alpha, m, maxIterations):
    xTrains = x.transpose()
    cnt = 0
    #print('train:', xTrains)
    while True:
        cnt += 1
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        gradient = np.dot(xTrains, loss) / m
        theta = theta - alpha * gradient
        squared_err = (np.dot(x, theta)-y)**2
        err = np.sqrt(np.mean(squared_err))
        if err < epsilon:
            break
    print('theta:',theta,'cnt:',cnt)
    return theta


def predict(x, theta):
    m, n = np.shape(x)
    xTest = np.ones((m, n+1))
    xTest[:, :-1] = x
    yP = np.dot(xTest, theta)
    return yP


dataSet = genfromtxt('house.csv', delimiter=',')
trainData, trainLabel = getData(dataSet)
m, n = np.shape(trainData)
theta = np.ones(n)

alpha = 0.05
maxIteration = 5000

theta = batchGradientDescent(trainData, trainLabel, theta, alpha, m, 15000)
# x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
