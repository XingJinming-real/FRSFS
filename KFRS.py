# KFRS，对于单label问题，并以分类质量为criteria
import time
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

Data = []
dataTrain = []
dataTest = []
digits = load_wine()


#  外部导入数据，直接在address更改地址即可
def getData(address):
    global dataTrain, dataTest
    f = open(address, 'r')
    for perLine in f:
        Data.append((perLine.strip('\n')).split('\t'))
    for i in range(len(Data)):
        for j in range(len(Data[0])):
            Data[i][j] = eval(Data[i][j])
    DataMat = np.array(Data)
    dataTrain = DataMat[0:int(len(Data) * 0.7), :]
    dataTest = np.array(Data)[int(len(Data) * 0.7):, :]


#  找到最近不同类，此处没有考虑噪声的影响，如果考虑，则是返回前k个最近的样本
def findTheNearestDifferentSample(x, featureId):
    tempMin = 999
    nearestSampleId = -1
    for i in range(len(dataTrain)):
        if tempMin == 0:
            break
        if dataTrain[i][-1] == dataTrain[x][-1]:
            continue
        if tempMin > abs(dataTrain[i][featureId] - dataTrain[x][featureId]):
            tempMin = abs(dataTrain[i][featureId] - dataTrain[x][featureId])
            nearestSampleId = i
    return nearestSampleId


#  算出下近似
def Kernel(x, y, featureId, sigma=0.1):
    return 1 - np.exp(-abs(dataTrain[x][featureId] - dataTrain[y][featureId]) / (2 * (sigma ** 2)))


#  算出分类质量又称依赖性
def classificationQuality(featureId):
    tempSum = 0
    for x in range(len(dataTrain)):
        y = findTheNearestDifferentSample(x, featureId)
        tempSum += Kernel(x, y, featureId)
    return tempSum / len(dataTrain)


#  获得每个特征的得分
def getFeatureScore(attr):
    featureScore = []
    for featureId in range(len(attr[0])):
        featureScore.append(classificationQuality(featureId))
    return featureScore


#  获得以得分降序的featureID列表
def FRS():
    global dataTrain, dataTest
    # getData('testData_2.txt')
    # dataTrain = np.array(dataTrain)
    # attrTrain, labelTrain = dataTrain[:, 0:-1], dataTrain[:, -1]
    """如果要自己导入数据则去掉下面三行，加上上面三行"""
    attrTrain, attrTest, labelTrain, labelTest = train_test_split(digits.data, digits.target, test_size=.3)
    dataTrain = np.hstack((attrTrain, labelTrain.reshape(len(labelTrain), 1)))
    dataTest = np.hstack((attrTest, labelTest.reshape(len(labelTest), 1)))
    # 此处使用dataTrain和dataTest
    featureScore = getFeatureScore(attrTrain)
    featureScoreId = np.argsort(featureScore)
    featureScoreId = list(reversed(featureScoreId))
    return featureScoreId


#  前向贪婪搜索，没有考虑特征的冗余，如果考虑则 使用带两个threshold的那个算法
def main():
    beginTime = time.perf_counter()
    featureScoreId = FRS()
    labelTrain = dataTrain[:, -1]
    labelTest = dataTest[:, -1]
    score = []
    for k in range(1, len(dataTrain[0])):
        attrTestRED = []
        attrTrainRED = []
        for i in range(0, k):
            attrTrainRED.append(list(dataTrain[:, featureScoreId[i]]))
            attrTestRED.append(list(dataTest[:, featureScoreId[i]]))
        attrTrainRED = np.array(attrTrainRED).T
        attrTestRED = np.array(attrTestRED).T
        kNN = KNeighborsClassifier(3)
        kNN.fit(attrTrainRED, labelTrain)
        score.append(kNN.score(attrTestRED, labelTest))
    endTime = time.perf_counter()
    plt.plot(list(range(1, len(dataTrain[0]))), score)
    plt.show()
    print(endTime - beginTime)


if __name__ == "__main__":
    main()
