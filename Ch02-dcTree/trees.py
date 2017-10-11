from math import log


def createDataSet():
    '''创建测试数据集'''
    dataSet = [[1, 1, 'y'], [1, 1, 'y'], [1, 0, 'n'], [0, 1, 'n'], [0, 1, 'n']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    '''计算数据集的香农火商(entropy)'''
    numEntries = len(dataSet)
    labelCounts = {}
    for item in dataSet:
        currentLabel = item[-1]  # 最后一列
        print('currentLabel:', currentLabel)
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    print('labelCounts:', labelCounts)

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 出现概率
        shannonEnt -= prob * log(prob, 2)  # 计算火商(entropy)
    print('shannonEnt:', shannonEnt)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    '''按照给定特征划分数据集'''
    retDataSet = []
    for item in dataSet:
        if item[axis] == value:
            reducedFeatVec = item[:axis]  # 得到空的[]
            reducedFeatVec.extend(item[axis + 1:])
            retDataSet.append(reducedFeatVec)
    print('retDataSet:', retDataSet)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''选择最好的数据集划分方式'''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1

    for i in range(numFeatures):  # i遍历每一个特征
        featList = [example[i] for example in dataSet]
        print('featList:', featList)
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
