import logRegres as logR

dataMatrix, classLabels = logR.loadDataSet()

weights = logR.gradAscent(dataMatrix, classLabels)
logR.plotBestFit(weights)
