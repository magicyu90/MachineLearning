import bayes
import numpy as np

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

