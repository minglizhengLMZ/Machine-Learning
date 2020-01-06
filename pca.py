# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:57:18 2019

@author: lenovo
"""

'''---------PCA算法，主成分分析的降维方法--，作用嵌套到其他机器学习模型里，数据简化的一种工具----------------'''
'''

简化数据的方法，更加通俗的解释见书

主成分分析(Principal Component Analysis, PCA)
通俗理解：就是找出一个最主要的特征，然后进行分析。
例如： 考察一个人的智力情况，就直接看数学成绩就行(存在：数学、语文、英语成绩)

因子分析(Factor Analysis)
通俗理解：将多个实测变量转换为少数几个综合指标。它反映一种降维的思想，通过降维将相关性高的变量聚在一起,从而减少需要分析的变量的数量,而减少问题分析的复杂性
例如： 考察一个人的整体情况，就直接组合3样成绩(隐变量)，看平均成绩就行(存在：数学、语文、英语成绩)
应用的领域：社会科学、金融和其他领域
在因子分析中，我们 
假设观察数据的成分中有一些观察不到的隐变量(latent variable)。
假设观察数据是这些隐变量和某些噪音的线性组合。
那么隐变量的数据可能比观察数据的数目少，也就说通过找到隐变量就可以实现数据的降维。

独立成分分析(Independ Component Analysis, ICA)
通俗理解：ICA 认为观测信号是若干个独立信号的线性组合，ICA 要做的是一个解混过程。
例如：我们去ktv唱歌，想辨别唱的是什么歌曲？ICA 是观察发现是原唱唱的一首歌【2个独立的声音（原唱／主唱）】。
ICA 是假设数据是从 N 个数据源混合组成的，这一点和因子分析有些类似，这些数据源之间在统计上是相互独立的，而在 PCA 中只假设数据是不 相关（线性关系）的。
同因子分析一样，如果数据源的数目少于观察数据的数目，则可以实现降维过程。

优点：降低数据的复杂性，识别最重要的多个特征
缺点：不一定需要，且可能损失有用的信息
适用数据类型：数值型数据

利用原理特征值分析：AV=lamda*V，原始数据矩阵乘特征向量实现数据的降维过程，
A为数据协方差矩阵，方差越大的方向（数据差异性越大，数据信息越多）


将数据转换成前N个主成分的伪代码：

数据减去平均值
计算协方差矩阵
计算协方差矩阵的特征值和特征向量
将特征值从小到大进行排序
保留最上面的N个特征向量
将数据转换到上述N个特征向量构建的新空间
'''

from numpy import *
from __future__ import print_function
import matplotlib.pyplot as plt

#解析数据的函数
def loadDataSet(fileName,delim='\t'):
    fr=open(fileName)
    stringArr=[line.strip().split(delim) for line in fr.readlines()]   #按行读取数据，存成列表形式
    datArr=[list(map(float,line)) for line in stringArr]     #将数据转换成浮点数
    return mat(datArr)         #返回矩阵格式的数据

#定义PCA算法,得到PCA降维后的数据
def pca(dataMat, topNfeat=9999999):
    """pca
    Args:
        dataMat   原数据集矩阵
        topNfeat  应用的N个特征
    Returns:
        lowDDataMat  降维后数据集
        reconMat     新的数据集空间
    """
    # 计算每一列的均值
    meanVals = mean(dataMat, axis=0)
    # print('meanVal', meanVals)
    # 每个向量同时都减去 均值
    meanRemoved = dataMat - meanVals
    # print('meanRemoved=', meanRemoved)
    # cov协方差=[(x1-x均值)*(y1-y均值)+(x2-x均值)*(y2-y均值)+...+(xn-x均值)*(yn-y均值)+]/(n-1)
    '''
    方差：一维）度量两个随机变量关系的统计量
    协方差： （二维）度量各个维度偏离其均值的程度
    协方差矩阵：（多维）度量各个维度偏离其均值的程度
    当 cov(X, Y)>0时，表明X与Y正相关；(X越大，Y也越大；X越小Y，也越小。这种情况，我们称为“正相关”。)
    当 cov(X, Y)<0时，表明X与Y负相关；
    当 cov(X, Y)=0时，表明X与Y不相关。
    '''
    covMat = cov(meanRemoved, rowvar=0)
    # eigVals为特征值， eigVects为特征向量
    eigVals, eigVects = linalg.eig(mat(covMat))
    # print('eigVals=', eigVals)
    # print('eigVects=', eigVects)
    # 对特征值，进行从小到大的排序，返回从小到大的index序号
    # 特征值的逆序就可以得到topNfeat个最大的特征向量
    '''
    >>> x = np.array([3, 1, 2])
    >>> np.argsort(x)
    array([1, 2, 0])  # index,1 = 1; index,2 = 2; index,0 = 3
    >>> y = np.argsort(x)
    >>> y[::-1]
    array([0, 2, 1])
    >>> y[:-3:-1]
    array([0, 2])  # 取出 -1, -2
    >>> y[:-6:-1]
    array([0, 2, 1])
    '''
    eigValInd = argsort(eigVals)
    # print('eigValInd1=', eigValInd)
    # -1表示倒序，返回topN的特征值[-1 到 -(topNfeat+1) 但是不包括-(topNfeat+1)本身的倒叙]
    eigValInd = eigValInd[:-(topNfeat+1):-1]   #列表切片的三个参数
    # print('eigValInd2=', eigValInd)
    # 重组 eigVects 最大到最小
    redEigVects = eigVects[:, eigValInd]
    # print('redEigVects=', redEigVects.T)
    # 将数据转换到新空间
    # print("---", shape(meanRemoved), shape(redEigVects))
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    # print('lowDDataMat=', lowDDataMat)
    # print('reconMat=', reconMat)
    return lowDDataMat, reconMat


    
dataMat=loadDataSet(r'E:\python_study\Python_work\data\13\testSet.txt')
lowDMat,reconMat=pca(dataMat,1)
shape(lowDMat) 
shape(reconMat)

#图像分析降维后的数据于降维前的数据

def show_picture(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #降维前数据用^表示
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    #降维后数据用o表示
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()    
show_picture(dataMat,reconMat)


''' --------PCA的应用，利用PCA对半导体制造数据进行降维--------'''

#数据预处理，数据读取，解析，缺失值处理(用均值代替缺失值)
def replaceNanWithMean():
    datMat = loadDataSet(r'E:\python_study\Python_work\data\13\secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        # 对value不为NaN的求均值
        # .A 返回矩阵基于的数组
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        # 将value为NaN的值赋值为均值
        datMat[nonzero(isnan(datMat[:, i].A))[0],i] = meanVal
    return datMat

dataMat=replaceNanWithMean()
shape(dataMat)   #有1567个数据，数据为590维度的数据
lowDDataMat, reconMat=pca(dataMat,20)

#图形分析原始数据于降维后数据
def show_picture(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()
    
show_picture(dataMat,reconMat)

#计算数据，查看数据的累计方差百分比

def analyse_data(dataMat):
    meanVals = mean(dataMat, axis=0)       #按列计算均值  
    meanRemoved = dataMat-meanVals         #去除均值的数据
    covMat = cov(meanRemoved, rowvar=0)    #计算协方差
    eigvals, eigVects = linalg.eig(mat(covMat))    #计算特征值和特征矩阵
    eigValInd = argsort(eigvals)                  #对特征值进行排序
    topNfeat = 20                            
    eigValInd = eigValInd[:-(topNfeat+1):-1]     #获取最大的20个特征值
    cov_all_score = float(sum(eigvals))          #特征值直接求和
    sum_cov_score = 0                            
    for i in range(0, len(eigValInd)):
        line_cov_score = float(eigvals[eigValInd[i]])
        sum_cov_score += line_cov_score
        '''
        我们发现其中有超过20%的特征值都是0。
        这就意味着这些特征都是其他特征的副本，也就是说，它们可以通过其他特征来表示，而本身并没有提供额外的信息。
        最前面15个值的数量级大于10^5，实际上那以后的值都变得非常小。
        这就相当于告诉我们只有部分重要特征，重要特征的数目也很快就会下降。
        最后，我们可能会注意到有一些小的负值，他们主要源自数值误差应该四舍五入成0.
        '''
        print('主成分：%s, 方差占比：%s%%, 累积方差占比：%s%%' % (format(i+1, '2.0f'), format(line_cov_score/cov_all_score*100, '4.2f'), format(sum_cov_score/cov_all_score*100, '4.1f')))

analyse_data(dataMat)




'''
注意：数据与信息之间存在很大的差别，
数据指的是接受的原始材料，其中可能包含噪声和不相关信息，信息指的数据中的相关部分

错误处理
unsupported operand type(s) for /: 'map' and 'int'
在map前加list(map())
'''
































