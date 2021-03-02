import random
import time
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

"""

"""
"""

   绿色注释为  改进内容
   白色注释为  代码说明

"""

# Kernelized Fuzzy Rough Sets，对于单label问题
"""

    对于多label问题，只需改变如下
   注意考虑多label在label空间的重叠度
   现有算法只考虑了特征空间信息，没有利用标签空间信息
   即：
        在现有criteria公式下，减去两个样本的标签重叠度,即算出来的quality或certainty 变为quality-标签重叠度
        例:  a的标签为【1，2，4，5】
             b的标签为【1，2，3，5】
             重叠度为  (1+1+0+1)/4
             
   """

Data = []
dataTrain = []
dataTest = []
lastFeatureSet = []
tempDiv = 1
kNeighbours = 3
digits = load_digits()
attrTrain, attrTest, labelTrain, labelTest = train_test_split(digits.data, digits.target, test_size=.3)


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
"""

此处进行2处更改，
1.是找到所有不同类中最近的，期望与所有不同类都远
2.选择前k个，增强鲁棒性

"""


def findTheNearestDifferentSample(x, featureId):
    global kNeighbours
    diffMapLabelToId = {}
    diffNearestSampleId = []
    for i in range(len(dataTrain)):
        if dataTrain[i][-1] == dataTrain[x][-1]:
            continue
        if diffMapLabelToId.get(dataTrain[i][-1]) is None:
            diffMapLabelToId[dataTrain[i][-1]] = [[i, abs(dataTrain[i][featureId] - dataTrain[x][featureId])]]
            continue
        diffMapLabelToId[dataTrain[i][-1]].append([i, abs(dataTrain[i][featureId] - dataTrain[x][featureId])])
    for i in diffMapLabelToId.keys():
        diffMapLabelToId[i] = sorted(diffMapLabelToId[i], key=lambda tempX: tempX[1])
    for i in diffMapLabelToId.keys():
        tempList = []
        for per in range(kNeighbours):
            tempList.append(diffMapLabelToId[i][per][0])
        diffNearestSampleId.append(tempList)
    return diffNearestSampleId


def findTheNearestSameClassKSamples(x, featureId):
    global kNeighbours
    sameMapLabelToId = {}
    sameNearestSampleId = []
    for i in range(len(dataTrain)):
        if dataTrain[i][-1] != dataTrain[x][-1]:
            continue
        if sameMapLabelToId.get(dataTrain[i][-1]) is None:
            sameMapLabelToId[dataTrain[i][-1]] = [[i, abs(dataTrain[i][featureId] - dataTrain[x][featureId])]]
            continue
        sameMapLabelToId[dataTrain[i][-1]].append([i, abs(dataTrain[i][featureId] - dataTrain[x][featureId])])
    for i in sameMapLabelToId.keys():
        sameMapLabelToId[i] = sorted(sameMapLabelToId[i], key=lambda tempX: tempX[1])
    for i in sameMapLabelToId.keys():
        tempList = []
        for per in range(kNeighbours):
            tempList.append(sameMapLabelToId[i][per][0])
        sameNearestSampleId.append(tempList)
    return sameNearestSampleId


#  算出下近似
def Kernel_lower(x, y, featureId, sigma=0.5):
    return 1 - np.exp(-(dataTrain[x][featureId] - dataTrain[y][featureId]) ** 2 / (2 * (sigma ** 2)))


def Kernel_lower_improved(x, y, featureId, sigma=0.5):
    return np.sqrt(1 - (np.exp(-(dataTrain[x][featureId] - dataTrain[y][featureId]) ** 2 / (2 * (sigma ** 2)))) ** 2)


#  计算出上近似，注意是同类的
def Kernel_upper(x, y, featureId, sigma=0.5):
    return np.exp(-(dataTrain[x][featureId] - dataTrain[y][featureId]) ** 2 / (2 * (sigma ** 2)))


def Kernel_upper_improved(x, y, featureId, sigma=0.5):
    return 1 - np.sqrt(
        1 - (np.exp(-(dataTrain[x][featureId] - dataTrain[y][featureId]) ** 2 / (2 * (sigma ** 2)))) ** 2)


#  算出分类质量又称依赖性
def classificationQuality(featureId):
    global tempDiv
    tempSum = 0
    for x in range(len(dataTrain)):
        Y = findTheNearestDifferentSample(x, featureId)
        for y in Y:
            for sub_y in y:
                tempSum += Kernel_lower_improved(x, sub_y, featureId)
    return tempSum / tempDiv


def upperApproximationSum(featureId):
    global kNeighbours
    tempSum = 0
    for x in range(len(dataTrain)):
        Y = findTheNearestSameClassKSamples(x, featureId)
        for y in Y:
            for sub_y in y:
                tempSum += Kernel_upper_improved(x, sub_y, featureId)
    return tempSum / (len(dataTrain) * kNeighbours)


"""

此处更改评价指标，改为分类确定度，利用了上近似的信息

"""


def getSeparationDegree(featureId):
    scoreMatrix = np.zeros((len(dataTrain), len(dataTrain)))
    for i in range(len(dataTrain)):
        for j in range(len(dataTrain)):
            # 返回样本i，j的相似度
            scoreMatrix[i][j] = 1 - Kernel_lower_improved(i, j, featureId)
    Sum = 0
    for perSample in range(len(dataTrain)):
        tempSum = 0
        for perSample_sub in range(len(dataTrain)):
            tempSum += scoreMatrix[perSample][perSample_sub]
        Sum += np.log(tempSum / len(dataTrain))
    return -Sum / len(dataTrain)


def classificationCertainty(featureId):
    #  下近似-所有上近似和
    return classificationQuality(featureId) - upperApproximationSum(featureId)


#  获得每个特征的得分
def getFeatureScore(attr, criteria):
    featureScore = []
    for featureId in range(len(attr[0])):
        if criteria == 'quality':
            #  下面的classificationQuality可改为classificationCertainty
            featureScore.append(classificationQuality(featureId))
        elif criteria == 'certainty':
            featureScore.append(classificationCertainty(featureId))
        elif criteria == 'separationDegree':
            featureScore.append(getSeparationDegree(featureId))
    return featureScore


#  获得以得分降序的featureID列表
def FRS(criteria):
    global dataTrain, dataTest, tempDiv
    # getData('testData_2.txt')
    # dataTrain = np.array(dataTrain)
    # attrTrain, labelTrain = dataTrain[:, 0:-1], dataTrain[:, -1]
    """如果要自己导入数据则去掉下面三行，加上上面三行"""
    dataTrain = np.hstack((attrTrain, labelTrain.reshape(len(labelTrain), 1)))
    dataTest = np.hstack((attrTest, labelTest.reshape(len(labelTest), 1)))
    tempDiv = len(dataTrain) * kNeighbours * len(set(labelTrain))
    # 此处只是使用dataTrain和dataTest
    featureScore = getFeatureScore(attrTrain, criteria)
    featureScoreId = np.argsort(featureScore)
    featureScoreId = list(reversed(featureScoreId))
    return featureScoreId, featureScore


#  前向贪婪搜索，考虑了特征的冗余，改进有使用带两个threshold的那个算法
"""

此处进行更改，使用上述算法去掉高度相关的特征，此处使用相关系数衡量线性关系，也可以使用信息增益使用衡量非线性关系

"""


def getSimilarityLiner(predominantFeatureId, perFeatureId):
    x_bar = np.average(dataTrain[:, predominantFeatureId])
    y_bar = np.average(dataTrain[:, perFeatureId])
    tempSum = 0
    tempSumSquareX = 0
    tempSumSquareY = 0
    for i in range(len(dataTrain)):
        tempSum += (dataTrain[i][predominantFeatureId] - x_bar) * (dataTrain[i][perFeatureId] - y_bar)
        tempSumSquareX += (dataTrain[i][predominantFeatureId] - x_bar) ** 2
        tempSumSquareY += (dataTrain[i][perFeatureId] - y_bar) ** 2
    return abs(tempSum / (np.sqrt(tempSumSquareY * tempSumSquareX)))


def removeRedundancy(predominantFeatureId, featureScoreId, threshold_2=0.5):
    for perFeatureId in featureScoreId:
        if getSimilarityLiner(predominantFeatureId, perFeatureId) > threshold_2:
            featureScoreId.remove(perFeatureId)
    return featureScoreId


"""
特征提取可看为最优子集的求解
基于退火/遗传算法的局部优化算法
"""
"""此处采用一半优优结合，一半随机结合"""

"""此处有两种，一是取集合中单个元素的得分的min作为集合的得分，二是所有单个得分相乘作为集合得分，此处选择后者"""


def getCandidateScore(perCandidate):
    tempSum = 1
    if not any(perCandidate):
        return 0
    for perFeatureId in range(len(perCandidate)):
        if perCandidate[perFeatureId] != 0:
            tempSum *= classificationQuality(perFeatureId)
    return tempSum


"""共获得10个后代,5个优优结合，5个随机结合"""


#  sigma为变异率,sigma*特征数=需要变异的特证数
def vary(newCandidate, sigma=0.05):
    varyNum = sigma * len(newCandidate)
    for i in range(int(varyNum)):
        varyId = random.randint(0, len(newCandidate) - 1)
        if newCandidate[varyId] == 1:
            newCandidate[varyId] = 0
        else:
            newCandidate[varyId] = 1
    return newCandidate


def getNewCandidate(candidates, candidateScoreId, kind, k=5):
    if kind == 'U':
        furtherCandidateId = candidateScoreId[0:k]
    else:
        furtherCandidateId = candidateScoreId
    newCandidateList = []
    for i in range(k):
        x = np.array(candidates[random.choice(furtherCandidateId)])
        y = np.array(candidates[random.choice(furtherCandidateId)])
        newCandidate = (np.hstack((x[0:int(len(attrTrain) / 2)], y[int(len(attrTrain) / 2):])))
        newCandidate = vary(newCandidate)
        newCandidateList.append(newCandidate)
    return newCandidateList


def eliminate(candidateList):
    newCandidate = (sorted(candidateList, key=lambda x: getCandidateScore(x), reverse=True))[:20]
    return newCandidate


def chooseCrossVary(candidates):
    candidateScore = []
    for perCandidate in candidates:
        candidateScore.append(getCandidateScore(perCandidate))
    candidateScoreId = list(reversed(np.argsort(candidateScore)))
    candidates.extend(getNewCandidate(candidates, candidateScoreId, kind='U'))
    candidates.extend(getNewCandidate(candidates, candidateScoreId, kind='L'))
    candidates = eliminate(candidates)
    return candidates


def initialize(featureNum, k=20):
    candidates = [[0 for i in range(featureNum)] for j in range(k)]
    for i in range(k):
        for j in range(featureNum):
            if np.random.rand() > 0.85:
                candidates[i][j] = 1
    return candidates


def memeticAlgorithm(featureNum, iterations=300, goalScore=0.9):
    global dataTrain, dataTest, tempDiv
    dataTrain = np.hstack((attrTrain, labelTrain.reshape(len(labelTrain), 1)))
    dataTest = np.hstack((attrTest, labelTest.reshape(len(labelTest), 1)))
    tempDiv = len(dataTrain) * kNeighbours * len(set(labelTrain))
    candidates = initialize(featureNum)
    loop = 0
    while loop < iterations:
        maxScore = 0
        maxScoreFeatureList = []
        loop += 1
        candidates = chooseCrossVary(candidates)
        for perCandidate in candidates:
            if maxScore > goalScore:
                return maxScoreFeatureList, maxScore
            attrTestRED = []
            attrTrainRED = []
            noneZeroNum = np.nonzero(perCandidate)[0]
            for i in noneZeroNum:
                attrTrainRED.append(list(dataTrain[:, i]))
                attrTestRED.append(list(dataTest[:, i]))
            attrTrainRED = np.array(attrTrainRED).T
            attrTestRED = np.array(attrTestRED).T
            kNN = KNeighborsClassifier(3)
            kNN.fit(attrTrainRED, labelTrain)
            fitScore = kNN.score(attrTestRED, labelTest)
            if maxScore < fitScore:
                maxScore = fitScore
                maxScoreFeatureList = perCandidate
    return None, None


def getSubsequentList(featureScoreId, k):
    global lastFeatureSet
    attrTrainRED, attrTestRED = [], []
    maxScore = 0
    maxFeature = []
    if k == 1:
        lastFeatureSet = [featureScoreId[0]]
        kNN = KNeighborsClassifier(3)
        kNN.fit((np.array(dataTrain[:, featureScoreId[0]])).reshape(-1, 1), labelTrain)
        curScore = kNN.score((np.array(dataTest[:, featureScoreId[0]])).reshape(-1, 1), labelTest)
        return lastFeatureSet.copy(), curScore
    remainFeatureSet = featureScoreId.copy()
    for i in lastFeatureSet:
        remainFeatureSet.remove(i)
    for i in remainFeatureSet:
        tempList = lastFeatureSet.copy()
        tempList.append(i)
        for i_sub in tempList:
            attrTrainRED.append(list(dataTrain[:, featureScoreId[i_sub]]))
            attrTestRED.append(list(dataTest[:, featureScoreId[i_sub]]))
        attrTrainREDArray = np.array(attrTrainRED).T
        attrTestREDArray = np.array(attrTestRED).T
        kNN = KNeighborsClassifier(3)
        kNN.fit(attrTrainREDArray, labelTrain)
        curScore = kNN.score(attrTestREDArray, labelTest)
        if maxScore < curScore:
            maxScore = curScore
            maxFeature = tempList.copy()
        tempList.remove(i)
        attrTrainRED.clear()
        attrTestRED.clear()
    lastFeatureSet = maxFeature.copy()
    return maxFeature, maxScore


def main(criteria, threshold_1=0.1):
    beginTime = time.perf_counter()
    global tempDiv, labelTest, labelTrain
    featureScoreId, featureScore = FRS(criteria)
    labelTrain = dataTrain[:, -1]
    labelTest = dataTest[:, -1]
    score = []
    # """
    #
    # 下面即为两个threshold算法
    # """
    # tempFeatureList = []
    # while True:
    #     predominantFeature = featureScoreId[0]
    #     featureScoreId.remove(predominantFeature)
    #     tempFeatureList.append(predominantFeature)
    #     if featureScore[predominantFeature] < threshold_1 or len(featureScoreId) == 0:
    #         break
    #     featureScoreId = removeRedundancy(predominantFeature, featureScoreId)
    # """
    #     然后只需将    for k in range(1, len(featureScoreId) + 1):
    #                     subsequentList, maxScore = getSubsequentList(featureScoreId, k)
    #                     score.append(maxScore)
    #             去掉
    #       加上，并featureScoreId改为tempFeatureList即可
    #     for k in range(1, len(dataTrain[0])):
    #         attrTestRED = []
    #         attrTrainRED = []
    #         for i in range(0, k):
    #             attrTrainRED.append(list(dataTrain[:, featureScoreId[i]]))
    #             attrTestRED.append(list(dataTest[:, featureScoreId[i]]))
    #         attrTrainRED = np.array(attrTrainRED).T
    #         attrTestRED = np.array(attrTestRED).T
    #         kNN = KNeighborsClassifier(3)
    #         kNN.fit(attrTrainRED, labelTrain)
    #         score.append(kNN.score(attrTestRED, labelTest))
    #
    # """

    for k in range(1, len(featureScoreId) + 1):
        subsequentList, maxScore = getSubsequentList(featureScoreId, k)  # 为前向贪婪，N^2时间复杂度，有间接考虑冗余
        score.append(maxScore)
    endTime = time.perf_counter()
    print("{}方法\n{}数据集\n所选特征为{}\n对应得分为{}\n用时{}s\n".format(criteria, 'digits', lastFeatureSet, score, endTime - beginTime))
    print('\n')
    plt.plot(list(range(1, len(lastFeatureSet) + 1)), score)
    plt.show()
    print(endTime - beginTime)

    """
    
    如果使用MA,则上面代码全删除并如下
    featureSubset, maxScore = memeticAlgorithm(len(attrTrain[0]))
    if featureSubset is not None:
        print(featureSubset, "\n其得分为{}".format(maxScore))
    kNN_test = KNeighborsClassifier(3)
    kNN_test.fit(attrTest, labelTest)
    print("原始数据训练得分为{}".format(kNN_test.score(attrTest, labelTest)))
            
    """


"""
    2021/2/23 加入新的评价指标 separationDegree
    基于矩阵的分开度(自己命名)  ⬆表示值越大越好，⬇表示越小越好
"""

if __name__ == '__main__':
    main('quality')
    """
    
    实验时发现，对于digits数据集，criteria的改变对结果效果不明显
    
    """
    main('certainty')
    main('separationDegree')
