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
