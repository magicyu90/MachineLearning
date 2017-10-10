import operator
import matplotlib
import matplotlib.pyplot as plt
import kNN

# datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
# plt.show()


myMatrix, myLabels = kNN.file2matrix('myMatrix.txt')
kNN.autoNorm(myMatrix)
