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


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOfPosts, listClass = loadDataSet()
    myVocablist = createVocabList(listOfPosts)
    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(myVocablist, postinDoc))
    p0v, p1v, pAb = trainNB0(array(trainMat), array(listClass))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocablist, testEntry))
    print('thisDoc:', thisDoc)
    print(testEntry, 'classfied as:', classifyNB(thisDoc, p0v, p1v, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocablist, testEntry))
    print('thisDoc:', thisDoc)
    print(testEntry, 'classfied as:', classifyNB(thisDoc, p0v, p1v, pAb))


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
        print('i:', i)
        wordList = textParse(open('email/spam/%d.txt' %
                                  i, "r", encoding='utf-8', errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' %
                                  i, "r", encoding='utf-8', errors='ignore').read())
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
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0

    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if(classifyNB(array(wordVector), p0V, p1V, pSpam)) != classList[docIndex]:
            errorCount += 1

    print('the error rate is:', float(errorCount / len(testSet)))
