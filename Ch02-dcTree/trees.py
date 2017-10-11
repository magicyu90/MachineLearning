from math import log


def createDataSet():
    '''创建测试数据集'''
    dataSet = [[1, 1, 'y'], [1, 1, 'y'], [1, 0, 'n'], [0, 1, 'n'], [0, 1, 'n']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShanoonEnt(dataSet):
    '''计算数据集的香农火商(entropy)'''
    numEntries = len(dataSet)
    labelCounts = {}
    for item in dataSet:
        currentLabel = item[-1]
        print('currentLabel:', currentLabel)
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    print('labelCounts:', labelCounts)

    shanoonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 出现概率
        shanoonEnt -= prob * log(prob, 2)  # 计算火商(entropy)
    print('shanoonEnt:', shanoonEnt)


def splitDataSet(dataSet, axis, value):
    '''按照给定特征划分数据集'''
    retDataSet = []
    for item in dataSet:
        if item[axis] == value:
            reducedFeatVec = item[:axis]  # 得到空的[]
            reducedFeatVec.extend(item[axis + 1:])
            retDataSet.append(reducedFeatVec)

    print('retDataSet:', retDataSet)
