from numpy import *


def loadDataSet():
    '''获取测试数据'''
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))  # 添加类别
    return dataMat, labelMat


def sigmoid(inX):
    '''Sigmoid函数'''
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # 标签矩阵（100*3）
    labelMat = mat(classLabels).transpose()  # 类别向量(矩阵倒置:[100*1]=>[1*100])
    m, n = shape(dataMatrix)  # 得到dataMatrix行、列分别为100、3
    alpha = 0.001  # 步长
    maxCycles = 500  # 迭代次数
    weights = ones((n, 1))  # （3*1）
    print('weights:', weights)
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 【100*3】*【3*1】=>[100*1]
        error = (labelMat - h)  # 类别(正确值)和预测值的差
        weights = weights + alpha * dataMatrix .transpose() * \
            error  # [3*1]+([3*100] * [100*1])=>[3*1]+[3*1]=>[3*1]
    print('weights:', weights)
    return weights


def stocGradAscent0(dataMatrix, classLabels):
    dataMatrix = mat(dataMatrix)
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones((1, 3))
    for i in range(m):
        a = dataMatrix[i].copy()
        a = a.transpose()
        h = sigmoid(weights * a)
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    print('stocGradAscent0 weights:', weights)
    return weights.transpose()


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    dataMatrix = mat(dataMatrix)          # dataMatIn格式：(1.0，第一特征值，第二特征值)
    # labelMat = mat(classLabels).transpose()  # 标签向量转置为列矩阵
    m, n = shape(dataMatrix)
    alpha = 0.001
    weights = ones((n, 1))
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * \
                dataMatrix[randIndex].transpose()
    return weights


def plotBestFit(weight):
    """画出分类线"""
    import matplotlib.pyplot as plt
    weights = weight.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def coliTest():
    horseTrain = open('horseColicTraining.txt')
    horseTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in horseTrain.readlines():
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currentLine[21]))
    trainWeights = stocGradAscent1(trainingSet, trainingLabels, 500)
    errorCount = 0.0
    numTestVec = 0.0
    for line in horseTest.readlines():
        numTestVec += 1.0
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currentLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("The error rate of this test is:%f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += coliTest()
    print('After %d times test,the average error rate is:%f' %
          (numTests, errorSum / float(numTests)))
