from numpy import *


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak',
                       'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字，0代表言论正常
    return postingList, classVec


def createVocabList(dataSet):
    """创建单词表List"""
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)

    vocabList = list(vocabSet)
    sortedList = sorted(vocabList, key=lambda x: x.lower())
    print('vocabList sorted:', sortedList)
    return sortedList


def setOfWords2Vec(vocabList, inputSet):
    """词汇表中的单词在输入的文档中是否出现"""
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1

    print('returnVec:', returnVec)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p1Denom

    print('p1Num:', p1Num, 'p1Denom:', p1Denom)
    print('p0Num:', p0Num, 'p0Denom:', p0Denom)
    print('pAbusive:', pAbusive)
    print('p0Vect:', p0Vect)
    print('p1Vect:', p1Vect)
    return p0Vect, p1Vect, pAbusive
