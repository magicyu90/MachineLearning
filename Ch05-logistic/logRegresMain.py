import logRegres as logR
from numpy import *

dataMatrix, classLabels = logR.loadDataSet()

#weights1 = logR.gradAscent(dataMatrix, classLabels)
weights2 = logR.stocGradAscent0(dataMatrix, classLabels)
weights3 = logR.stocGradAscent1(dataMatrix, classLabels)
logR.plotBestFit(weights2)


def f_prime(x_old):
    '''f(x) =-x²+3x+1
       f'(x)= -2x+3
    '''
    return -2 * x_old + 3


def cal():
    '''使用迭代求函数的极大值'''
    x_old = 0
    x_new = 2
    eps = 0.01
    precision = 0.00001
    while abs(x_new - x_old) > precision:
        x_old = x_new
        x_new = x_old + eps * f_prime(x_old)

    print('x_new:', x_new)
    return x_new


# cal()
