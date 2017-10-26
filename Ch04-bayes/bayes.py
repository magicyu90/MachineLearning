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
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)

    vocabList = list(vocabSet)
    sortedList = sorted(vocabList, key=lambda x: x.lower())
    return sortedList


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1

    print('returnVec:', returnVec)
    return returnVec


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p1Denom)

    print('p1Num:', p1Num, 'p1Denom:', p1Denom)
    print('p0Num:', p0Num, 'p0Denom:', p0Denom)
    print('pAbusive:', pAbusive)
    print('p0Vect:', p0Vect)
    print('p1Vect:', p1Vect)
    return p0Vect, p1Vect, pAbusive


def textParse(bigStr):
    import re
    listOfTokens = re.split(r'\W*', bigStr)
    words = [tok.lower() for tok in listOfTokens if len(tok) > 2]
    # print('words:', words)
    return words


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    trainingSet = list(range(50))
    testSet = []  # create test set

    print('docList:', docList)

    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]

    print('testSet:', testSet)
    print('trainingSet:', trainingSet)

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        item = bagOfWords2VecMN(vocabList, docList[docIndex])
        trainMat.append(item)
        trainClasses.append(classList[docIndex])


def calMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for word in vocabList:
        freqDict[word] = fullText.count(word)
    sortedFreq = sorted(
        freqDict.items(), key=operator.itemgetter(1), reverse=True)

   # print('sortedFreq[:30]:', sortedFreq[:30])
    return sortedFreq[:30]


def localWords(feed1, feed0):
    """
    Function:   RSS源分类器

    Args:   feed1:RSS源
            feed0:RSS源

    Returns:    vocablist:词汇表
                p0V:类别概率向量
                p1V:类别概率向量
    """
    import feedparser
    docList = []
    classList = []
    fullText = []

    minLen = min(len(feed1['entries']), len(feed0['entries']))
    print('minLen:', minLen)
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calMostFreq(vocabList, fullText)

    for pairN in top30Words:
        if pairN[0] in vocabList:
            vocabList.remove(pairN[0])

    trainingSet = list(range(2 * minLen))  # 使用此数组进行训练
    testSet = []  # 使用此数组进行测试

    for i in range(20):
        randomIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randomIndex])
        del trainingSet[randomIndex]

    trainingMatrix = []
    trainingClass = []
    # 训练
    for docIndex in trainingSet:
        trainingMatrix.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainingClass.append(classList[docIndex])
    p0v, p1v, pSapm = trainNB0(array(trainingMatrix), array(trainingClass))

    errorCount = 0
    # 测试
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(wordVector, p0v, p1v, pSapm) != classList[docIndex]:
            errorCount += 1

    print('the error rate is:', float(errorCount) / len(testSet))
