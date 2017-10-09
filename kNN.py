'''k-近邻算法'''
from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    print('diffMat:', diffMat)
    sqlDiffMat = diffMat ** 2
    print('sqlDiffMat:', sqlDiffMat)
    sqlDistance = sqlDiffMat.sum(axis=1)
    print('sqlDistances:', sqlDistance)
    distances = sqlDistance**0.5
    sortedDistIndices = distances.argsort()

    print('sortedDistIndices:', sortedDistIndices)

    classCount = {}

    for i in range(k):
        label = labels[sortedDistIndices[i]]
        classCount[label] = classCount.get(label, 0) + 1

    print('classCount:', classCount)
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)

    print('sortedClassCount:', sortedClassCount)

    res = sortedClassCount[0, 0]
    print('result:', res)


group, labels = createDataSet()

classify0([0.0, 0cxvcxxxxxxxxxxxxxxxxxxxx.0], group, labels, 3)
