import bayes

listOfPosts, listOfLabels = bayes.loadDataSet()

vocabList = bayes.createVocabList(listOfPosts)

vocabList

bayes.setOfWords2Vec(vocabList, listOfPosts[0])
