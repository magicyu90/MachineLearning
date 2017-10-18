import bayes

listOfPosts, listOfLabels = bayes.loadDataSet()

vocabList = bayes.createVocabList(listOfPosts)


trainMatrix = []
for postinDoc in listOfPosts:
    trainMatrix.append(bayes.setOfWords2Vec(vocabList, postinDoc))

print('trainMatrix:', trainMatrix)
bayes.trainNB0(trainMatrix, listOfLabels)
