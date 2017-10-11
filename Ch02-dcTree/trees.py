from math import log


def calcShanoonEnt(dataSet):
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
        shanoonEnt -= prob * log(prob, 2)  # 计算火商
    print('shanoonEnt:', shanoonEnt)


def createDataSet():
    dataSet = [[1, 1, 'y'], [1, 0, 'n'], [0, 1, 'n'], [0, 1, 'n']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
