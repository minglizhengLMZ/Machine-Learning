# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 18:36:22 2019

@author: Administrator
"""
''''---------------k-近邻算法-------------'''
from numpy import *   #科学计算包
import operator   #运算符模块
#定义数据集
def createDataSet():          #定义数据集函数
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])   #创建数据集
    labels=['A','A','B','B']    #创建标签
    return group,labels 

group,labels =createDataSet()
group
labels   
#定义knn算法
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]     #确实测试样本集的数据量
    diff_mat = tile(inX,(dataSetSize,1))-dataSet  #tile重复代表给定的次数来构造数组
    #将输入向量 inX 按数据集大小重复，并计算输入向量与训练样本集个分量计算离差
    sqDiffMat=diff_mat**2          #计算输入向量与训练样本集的离差平方
    sqDistances = sqDiffMat.sum(axis=1)    #计算离差平方和
    distances = sqDistances**0.5      #得出欧式距离
    sortedDistIndicies = distances.argsort()   ##将计算的距离按从小到大的顺序进行排序
    classCount={}            #创造字典存储距离及其索引
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]   #sortedDisIndicies第i各元素所对应的标签传递给投票标签
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1  #对该标签下的数据个数进行统计
    sortedClassCount = sorted(classCount.items(),key=lambda x:x[1],reverse=True)  
    #按字典的每个元组的第二列元素进行排序，x相当于字典集合中历遍出来的一个元组，按倒序排列
    return sortedClassCount[0][0] #返回发生频率最高的元素标签    
#使用knn算法分类
classify0([0,0], group, labels, 3)  



'''kNN的应用'''
'''
数据说明：
已有的约会数据：三个特征 ：飞行里程，玩游戏时间比，每周小号冰激凌数，
约会对象分为三类：不喜欢的人，魅力一般的人，极具魅力的人



数据文件处理，必须将数据格式转变成分类器可以接受的格式
分三步进行1：得到文件行数，2.创建矩阵，3.解析文件数据到列表'''
import re  #正则表达式块,用于按指定格式进行文本切割
import numpy as np   #科学计算
def file2matrix(filename):
    fr=open(filename)           #打开数据文件
    arrayOlines=fr.readlines()   
    #readlines()每次按行读取整个文件内容，将读取到的内容放到一个列表中，返回list类型。  
    fr.close()
    numberOfLines=len(arrayOlines)#得到文件行数
    returnMat=np.zeros((numberOfLines,3)) #创建和数据集行数相同，3列的矩阵
    classLabelVector =[]           #创建空列表
    index =0
    for line in arrayOlines:       #解析文件数据到列表
        line=line.strip()    #删除空白符（包括'\n', '\r', '\t', ' ')
        listFromLine=re.split('\t',line)  #使用\t将整行元素分解成一个元素列表
        returnMat[index,:]=listFromLine[0:3]  #选取前三各元素存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1])) #将listFromLine最后一列的元素赋值给，classLabelVector
        index+=1   #索引加1
    return returnMat,classLabelVector
datingDataMat,datingLabels=file2matrix(r'E:\Anaconda3\Python_work\datingTestSet2.txt') #读取数据，作为训练集
datingDataMat
datingLabels[0:20]
len(datingLabels)
len(datingDataMat)

'''------------------图像化展示数据-----------------------'''
#初步分析各类别的数据特点
import matplotlib
import matplotlib.pyplot as plt   #绘制图片的函数
fig=plt.figure()   #matplotlib图像都位于figure中，plt.figure()可创建一个新的figure
ax=fig.add_subplot(111)   #不能通过空figure绘图，此处表名图像是1X1的当前为第一个
ax.scatter(datingDataMat[:,1],datingDataMat[:,2])  
plt.show()

#使用彩色或者其他标记来标记不同样本，以便更好得理解数据信息，未标记图像看不出信息
fig=plt.figure()   #每化一次图必须声明一次
ax=fig.add_subplot(111)  
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))  
plt.show()
#利用一二列绘制三个类别的散点图

fig=plt.figure()   #每化一次图必须声明一次
ax=fig.add_subplot(111)  
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels)) 
#数据有三个特征，展示两个特性，从而区分三个特征 
plt.show()

'''准备数据:归一化数据
不同的量化单位对距离的计算会产生较大的影响
将数值归一化到0-1之间，或-1-1
newValue=(oldValue-min)/(max-min)'''
datingDataMat=datingDataMat.astype(np.float64)
datingDataMat
def autoNorm(dataset):     
    minVals=dataset.min(0)    #得到最小值 min(0)从当前列中获取最小值
    maxVals=dataset.max(0)    
    ranges=maxVals-minVals   #获取该列的极差
    normDataset=zeros(shape(dataset))#创建于dataset形状相同的数据集
    m=dataset.shape[0]   #获取第一列的行数
    normDataset=dataset-tile(minVals,(m,1))   #将原来的minVals扩充成m行1列的数组，与数据集做差，得到分子（1000X3）
    normDataset=normDataset/tile(ranges,(m,1))#将极差扩充为m行一列的数组，计算最终的(1000X3)
    return normDataset,ranges,minVals
normDataset,ranges,minVals=autoNorm(datingDataMat)  
normDataset    
ranges
minVals
#输出数据极值和和最小值可以用于对以后新数据进行归一化处理
'''------------------------测试算法------------------------'''
#选择90%的数据作为训练集，剩下10%的数据作为测试集
def datingClassTest():
    hoRatio=0.10             #测试集所占比例
    datingDataMat,datingLabels=file2matrix(r'E:\Anaconda3\Python_work\datingTestSet2.txt') 
    #导入整理的数据集
    normDataset,ranges,minVals=autoNorm(datingDataMat)     #数据集的归一化处理
    m=normDataset.shape[0]             #确定数据集的行数
    numTestVecs=int(m*hoRatio)   #计算测试样本数
    errorCount=0.0       #分类错误数赋初值为0
    for i in range(numTestVecs):
        classifierResult = classify0(normDataset[i,:],normDataset[numTestVecs:m,:],    
        #前面10%用做测试集，后面90%作为训练集，训练集标记，k=3
                                      datingLabels[numTestVecs:m],3)
        print ("the classifier came back with:%d,the real answer is:%d"
                                       %(classifierResult,datingLabels[i]))
                                      #标准化，显示真正的分类和预测分类
        if (classifierResult !=datingLabels[i]):
            errorCount+=1.0
    print ("the tolal error rate is:%f" % (errorCount/float(numTestVecs)))#标准化为浮点数
#调用函数 
datingClassTest()        
        
    
'''-----------------------约会网站的预测函数----------------'''
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats=float(input(
            "percentage of time spent playing video games?"))
    ffMiles = float(input(
            "frequent flier miles earned per year?"))
    iceCream= float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels=file2matrix(r'E:\Anaconda3\Python_work\datingTestSet2.txt')
    #导入数据集，对数据集进行整理
    normDataset,ranges,minVals=autoNorm(datingDataMat)     #数据集的归一化处理
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normDataset,datingLabels,3)
    print("You will probably like this person:",resultList[classifierResult-1])
    #分类结果1对应列表第0个元素

classifyPerson()
    

'''knn算法用于识别系统
目前只能识别0-9数字，图像已处理为相同的颜色大小：32X32像素的黑白图片
编写img2vector(),将图像格式转换成分类器使用的向量格式
1.把32*32的二进制图像数据转换成1*1024的向量
创建1X1024的数组，打开文件，循环读取前32行，并将每行的头32个字符存储在数组中，并返回'''
def img2vector(fliename):
    returnVect = zeros((1,1024)) #创建空的矩阵
    fr=open(fliename)
    for i in range(32):
        lineStr=fr.readline()   #按行读取文件前32行
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j]) #将每行的头32个字符存储在数组中
        #每32个元素为一行循环32次，将矩阵累加为一个向量
    return returnVect
testVector=img2vector(r'E:\Anaconda3\Python_work\0_13.txt')
testVector[0,0:31]
            
    







