import trees as treesClass
import treePlotter


dataSet, labels = treesClass.createDataSet()

# treesClass.calcShannonEnt(dataSet)
# treesClass.splitDataSet(dataSet, 0, 1)
# treesClass.chooseBestFeatureToSplit(dataSet)
# treesClass.createTree(dataSet, labels)

treePlotter.createPlot()
