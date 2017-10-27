import bayes
import numpy as np
import feedparser

listOfPosts, listOfLabels = bayes.loadDataSet()

vocabList = bayes.createVocabList(listOfPosts)


trainMatrix = []
for postinDoc in listOfPosts:
    trainMatrix.append(bayes.setOfWords2Vec(vocabList, postinDoc))

print('trainMatrix:', trainMatrix)
bayes.trainNB0(trainMatrix, listOfLabels)

# bayes.testingNB()
# myString = 'This book is the best book that I have laid eyes upon.'
# bayes.textParse(myString)
# bayes.spamTest()

# nyFeed = feedparser.parse('http://newyork.craiglist.org/stp/index.rss')
# sfFeed = feedparser.parse('http://sfbay.craiglist.org/stp/index.rss')

# bayes.localWords(nyFeed, sfFeed)

# bayes.getTopWords(nyFeed, sfFeed)


def createTrainingMatrix():

    classList = ['A', 'B', 'C']
    featList = {'F1': ['a', 'b', 'c'], 'F2': ['d', 'e'], 'F3': ['t', 'f']}

    print('featList length:', len(featList))
    trainingMatrix = []
    trainingClasses = []

    for i in range(100):
        item = [0] * len(featList)
        item[0] = featList['F1'][int(
            np.random.uniform(0, len(featList['F1'])))]
        item[1] = featList['F2'][int(
            np.random.uniform(0, len(featList['F2'])))]
        item[2] = featList['F3'][int(
            np.random.uniform(0, len(featList['F3'])))]
        trainingMatrix.append(item)
        trainingClasses.extend(
            classList[int(np.random.uniform(0, len(classList)))])
    # print('trainingMatrix:', trainingMatrix)
    print('trainingClasses:', trainingClasses)

    pClassA = trainingClasses.count('A') / len(trainingClasses)
    pClassB = trainingClasses.count('B') / len(trainingClasses)
    pClassC = trainingClasses.count('C') / len(trainingClasses)

    print('pClassA:', pClassA, 'pClassB:', pClassB, 'pClassC:', pClassC)

    classAList = []
    classBList = []
    classCList = []
    for i in range(len(trainingClasses)):
        if(trainingClasses[i] == 'A'):
            classAList.append(trainingMatrix[i])
        if(trainingClasses[i] == 'B'):
            classBList.append(trainingMatrix[i])
        if(trainingClasses[i] == 'C'):
            classCList.append(trainingMatrix[i])

    classF1ListStatistic = {'A': {'a': 0, 'b': 0, 'c': 0}, 'B': {
        'a': 0, 'b': 0, 'c': 0}, 'C': {'a': 0, 'b': 0, 'c': 0}}
    classF2ListStatistic = {'A': {'d': 0, 'e': 0},
                            'B': {'d': 0, 'e': 0}, 'C': {'d': 0, 'e': 0}}
    classF3ListStatistic = {'A': {'t': 0, 'f': 0},
                            'B': {'t': 0, 'f': 0}, 'C': {'t': 0, 'f': 0}}
    currentList = []
    for i in range(0, 3):
        keys = list(classF1ListStatistic.keys())
        currentKey = keys[i]
        if i == 0:
            currentList = classAList
        if i == 1:
            currentList = classBList
        if i == 2:
            currentList = classCList
            
        key1Count = 0
        key2Count = 0
        key3Count = 0
        key4Count = 0
        key5Count = 0
        key6Count = 0
        key7Count = 0
        for item in currentList:
            if item[0] == 'a':
                key1Count += 1
            if item[0] == 'b':
                key2Count += 1
            if item[0] == 'c':
                key3Count += 1
            if item[1] == 'd':
                key4Count += 1
            if item[1] == 'e':
                key5Count += 1
            if item[2] == 't':
                key6Count += 1
            if item[2] == 'f':
                key7Count += 1
        classF1ListStatistic[currentKey]['a'] = key1Count
        classF1ListStatistic[currentKey]['b'] = key2Count
        classF1ListStatistic[currentKey]['c'] = key3Count
        classF2ListStatistic[currentKey]['d'] = key4Count
        classF2ListStatistic[currentKey]['e'] = key5Count
        classF3ListStatistic[currentKey]['t'] = key6Count
        classF3ListStatistic[currentKey]['f'] = key7Count

    print('classF1ListStatistic:', classF1ListStatistic)
    print('classF2ListStatistic:', classF2ListStatistic)
    print('classF3ListStatistic:', classF3ListStatistic)


createTrainingMatrix()
