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

nyFeed = feedparser.parse('http://newyork.craiglist.org/stp/index.rss')
sfFeed = feedparser.parse('http://sfbay.craiglist.org/stp/index.rss')

# bayes.localWords(nyFeed, sfFeed)

bayes.getTopWords(nyFeed, sfFeed)
