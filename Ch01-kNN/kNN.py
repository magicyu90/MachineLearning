'''k-近邻算法'''
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 使用inx填充同样大小的矩阵并和原矩阵求差值
    sqlDiffMat = diffMat ** 2  # 差值平方
    sqlDistance = sqlDiffMat.sum(axis=1)  # 差值求和
    distances = sqlDistance**0.5  # 差值开平方
    sortedDistIndices = distances.argsort()
    classCount = {}

    for i in range(k):
        label = labels[sortedDistIndices[i]]
        classCount[label] = classCount.get(label, 0) + 1
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0, 0]


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
        returnMatrix[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    # print('returnMatrix:', returnMatrix, 'classLabelVector:', classLabelVector)
    return returnMatrix, classLabelVector


def autoNorm(dataSet):
    '''归一化特征值,newValue = (oldValue-min)/(max-min)'''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))  # 得到value是0的矩阵
    rows = dataSet.shape[0]  # 获得行数
    normDataSet = dataSet - tile(minVals, (rows, 1))  # 矩阵中的值和最小值的差值
    rangesDataSet = tile(ranges, (rows, 1))  # 获取最大值最小值差的矩阵
    normDataSet = normDataSet / rangesDataSet  # 归一处理
    return normDataSet, ranges, minVals


def datingClassTest():
    '''测试算法'''
    hoRatio = 0.1  # 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 读取文件
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    print('normMat[numTestVecs:m, :]:', normMat[numTestVecs:m, :])
    for i in range(numTestVecs):
        classifierResult = classify0(
            normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('classfierResult:', classifierResult)
        print("the classifier came back with: %d, the real answer is: %d" %
              (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print("errorCount:", errorCount)


def image2Vector(fileName):
    '''图像中的像素转换为矩阵'''
    returnVector = zeros((1, 1024))
    fr = open(fileName)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0, i * 32 + j] = int(lineStr[j])
    #print('returnVector:', returnVector)
    return returnVector
