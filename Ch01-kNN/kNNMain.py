import operator
import matplotlib
import matplotlib.pyplot as plt
import kNN

# datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
# plt.show()


# myMatrix, myLabels = kNN.file2matrix('myMatrix.txt')
# kNN.autoNorm(myMatrix)


# kNN.datingClassTest()
testVector = kNN.image2Vector('testDigits/0_3.txt')
print('testVector 0-31:', testVector[0, 0:31])
print('testVector 32-63:', testVector[0, 32:63])
