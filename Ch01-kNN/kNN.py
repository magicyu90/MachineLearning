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
    sqlDiffMat = diffMat ** 2
    sqlDistance = sqlDiffMat.sum(axis=1)
    distances = sqlDistance**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}

    for i in range(k):
        label = labels[sortedDistIndices[i]]
        classCount[label] = classCount.get(label, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)

    res = sortedClassCount[0, 0]
    print('result:', res)


# group, labels = createDataSet()

# classify0([0, 0], group, labels, 3)


def file2matrix(fileName):
    fr = open(fileName)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMatrix = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0

    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMatrix[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1


    print('returnMatrix:', returnMatrix, 'classLabelVector:', classLabelVector)
    return returnMatrix, classLabelVector


file2matrix('datingTestSet2.txt')
