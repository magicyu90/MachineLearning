import trees as treesClass

dataSet, labels = treesClass.createDataSet()

# treesClass.calcShanoonEnt(dataSet)
treesClass.splitDataSet(dataSet, 0, 1)
