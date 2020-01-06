# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 19:24:02 2019

@author: Administrator
"""
'''-------------------------logistic回归（监督学习）----------------------'''
#sigmoid函数和logistic回归分类器
#最优化理论初步
#梯度下降最优化算法
#数据中的缺失处理
'''logisti回归一般内容
（1）搜集数据：采用任意方法搜集数据
（2）准备数据：由于需要计算距离，数据类型应为数值型数据，结构化数据格式最佳
（3）分析数据：采用任意方法对数据进行分析
（4）训练算法：大部分时间将用于训练，训练的目的找到最佳的分类回归系数
（5）测试算法：一旦训练步骤完成，分类将会很快
（6）使用算法：
    1）首先输入数据，将其转换成结构化数值
    2）基于训练好的回归系数对数值进行简单的回归计算判断他们属于哪类
    3）在输出的类别做一些其他分析工作
    
梯度上升算法：求解函数最大值的最优算法   w=w+α∇_wf(w)
梯度下降算法：求解函数最小值的最优算法

梯度上升算法的伪代码：
所有回归系数初始化为1
重复R次：#迭代次数
    计算整个数据集的梯度
    使用alpha*gradient更新回归系数
返回回归系数值
'''


'''-----------------梯度上升算法代码实现------------'''

from numpy import *    #导入
'''------------数据读取与预处理--------------'''
def loadDataSet():      
    dataMat = [];labelMat = []
    fr=open(r'E:\Anaconda3\Python_work\data\testSet.txt')
    for line in fr.readlines():       #逐行读取数据
        lineArr = line.strip().split()   #将数据进行分割传入lineArr
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])]) 
        #将x0的值定义为1.0,每行前两个值为x1,x2
        labelMat.append(int(lineArr[2])) 
    return dataMat,labelMat      

def sigmoid(inX):                 #构造sigmod函数
    return 1.0/(1+exp(-inX))
#使用梯度上升算法求解回归系数
def gradAscent(dataMatIn,classLabels):    
    #dataMatIn二维numpy数组，每列表示不同特征，每行表示不同样本，dataMatIn存放100*3的矩阵
    dataMatrix = mat(dataMatIn)  #获得输入数据，将其转换成numpy矩阵         
    labelMat=mat(classLabels).transpose()  #将标签行向量转置成1*100的列向量
    m,n = shape(dataMatrix)    #得到矩阵的大小
    alpha = 0.001    #α系数一般都取比较小，目标移动的步长
    maxCycles = 500  #设置迭代次数
    weights=ones((n,1))  #权重赋初值1，个数等于变量数，即dataMat的列数
    for k in range(maxCycles):   #迭代训练回归系数
        h=sigmoid(dataMatrix*weights)  #用sigmod函数进行判别
        error=(labelMat-h)   #计算误差
        weights=weights+alpha*dataMatrix.transpose()*error  #梯度上升优化算法迭代回归系数
        
    return weights


dataArr,labelMat=loadDataSet()     #将loadDataSet函数的返回值定义为dataArr,labelMat
gradAscent(dataArr,labelMat)
     
'''--------------分析数据，画出决策边界-绘制分割线-------------------'''
'''上文中已经确定了回归系数，构造了决策边界，确定了数据之间的分割线'''
def plotBestFit(weights):
    import matplotlib.pyplot as plt   #调用motpotlib，可视化程序包的pylib方法
    dataMat,labelMat=loadDataSet() 
    dataArr=array(dataMat)     #将数据转为数组形式
    n=shape(dataArr)[0]       #n为data的行数
    xcord1=[];ycord1=[]     
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:   #判为第一类的数据
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:                     #属于第0类的数据
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()     #创建空图
    ax=fig.add_subplot(111)   #添加sobplot，ax的所有内容都画在空图上
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')  #第一类为红色散点
    ax.scatter(xcord2,ycord2,s=30,c='green',marker='o')    #第二类为绿色散点
    x=arange(-3.0,3.0,0.1)    #设置x取值，arange支持浮点型#拟定拟合线的三个关键点，绘制分界线                   
    y=(-weights[0]-weights[1]*x)/weights[2]  #求出y取值 
    #最佳拟合直线，此处设置了sigmod函数值为0，分界处，0=w0x0+w1x1+w2x2
    ax.plot(x,y)      #绘制分界线
    plt.xlabel('X1');plt.ylabel('X2');
    plt.show()
    
from numpy import*
weights=gradAscent(dataArr,labelMat)
plotBestFit(weights.getA())
    
'''---------算法的改进：训练算法：随机梯度上升------'''
'''梯度上升算法的缺点：
每次更新回归系数时需要将所有数据历遍，不适用大数据情况计算复杂度太高
改进方法：一次仅用一个样本点来更新回归系数，------随机梯度上升算法————-一种在线学习算法
与在线学习对应----一次处理的所有数据被称为批数据
随机梯度上升算法的伪代码：
所有回归系数初始化为1
对数据集中的每个数据:
    计算该样本梯度
    使用alpha*gradient更新回归系数
返回回归系数值
'''
'''------------随机梯度上升算法代码实现------------------'''
def stocGradAscent0(dataMatrix,classLabels):
    m,n=shape(dataMatrix)  #m,n分别为矩阵的行列数
    alpha = 0.01      #设定迭代系数α为0.01
    weights=ones(n)    #产生1维数据，长度为n
    for i in range(m):    #迭代次数同样本数
        h=sigmoid(sum(dataMatrix[i]*weights))  #根据logists回归进行预测
        error=classLabels[i]-h        #计算错误率
        weights= weights + alpha * error * dataMatrix[i]  #每次迭代使用第一个数据进行优化系数值
    return weights

'''随机梯度上升算法与梯度上升算法的区别：1.后者的h和error为向量，整个数据值的预测值，和误差
而前者为数值，仅为当前样本的预测值和误差
2.前者没有矩阵的转换过程，所有数据类型都为numpy数组
'''
from numpy import*
dataArr,labelMat=loadDataSet()
weights=stocGradAscent0(array(dataArr),labelMat)
plotBestFit(weights)

#使用随机梯度算法迭代结果并不完美，错误率很高

'''--------改进的随机梯度上升算法-----(分类结果随机性时好时坏)----'''
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n=shape(dataMatrix)
    weights = ones(n)
    dataIndex = range(m)
    for j in range(numIter):        
        #numIter设置了迭代次数
        for i in range(m):
            #每次用当前样本进行优化，将此过程重复numIter次，（增加了迭代次数，提高拟合度）
            alpha = 4/(1.0+j+i)+0.01    
            #改进1：α每次都改进，会缓解数据波动或者高频波动，a不断减小但不会减小到0，因为有常数项
            #要处理的问题是动态变化是，应适当增大常数值，确保新的值获得更大的回归系数
            randIndex = int(random.uniform(0,len(dataIndex))) 
            #改进二：采用随机选取样本来跟新回归系数
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex]-h
            weights= weights + alpha * error * dataMatrix[randIndex]
        return weights
        
dataArr,labelMat=loadDataSet()
weights=stocGradAscent1(array(dataArr),labelMat)
plotBestFit(weights)


'''1.改进1：α每次都改进，会缓解数据波动或者高频波动，a不断减小但不会减小到0，因为有常数项
#要处理的问题是动态变化是，应适当增大常数值，确保新的值获得更大的回归系数
#2.α每次降低1/（j+i）,i是样本的下标，当j<<max(i)时α不是严格下降的
（避免严格下降也常见于退火算法等优化算法中）'''

weights=stocGradAscent1(array(dataArr),labelMat,500)
plotBestFit(weights)

'''-------从疝气病预测病马的死亡率--------'''
'''准备数据：处理数据的缺失值
（1）使用可用特征的均值来处理缺失值
（2）使用特初值填补缺失值
（3）忽略有缺失值的样本
（4）使用相似样本的均值填补缺失值
（5）使用另外的机器学习算法预测缺失值
'''
'''对于logistic回归
1.所有缺失值用一个实数值替代，因为numpy数据类型不允许缺失值的存在，此处选择0替代缺失值
原因：（1）weights= weights + alpha * error * dataMatrix[randIndex]
缺失值为0时不影响系数更新，此时：weights=weights
（2）sigmoid(0)=0.5 不影响分类结果
2.数据的类别标签缺失处理：将该数据丢弃'''

def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob >0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
	"""
	Function：	训练和测试函数
	Input：		训练集和测试集文本文档
	Output：	分类错误率
	"""	
	#打开训练集
	frTrain = open(r'E:\Anaconda3\Python_work\data\horseColicTraining.txt')
	#打开测试集
	frTest = open(r'E:\Anaconda3\Python_work\data\horseColicTest.txt')
	#初始化训练集数据列表
	trainingSet = []
	#初始化训练集标签列表
	trainingLabels = []
	#遍历训练集数据
	for line in frTrain.readlines():   #按行读取数据
		#切分数据集
		currLine = line.strip().split('\t')  #按tab键切割，将空格等删除
		#初始化临时列表
		lineArr = []
		#遍历21项数据重新生成列表，因为后面格式要求，这里必须重新生成一下。
		for i in range(21):
			lineArr.append(float(currLine[i])) 
		#添加数据列表
		trainingSet.append(lineArr)
		#添加分类标签
		trainingLabels.append(float(currLine[21]))
	#获得权重参数矩阵
	trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
	#初始化错误分类计数
	errorCount = 0 

	numTestVec = 0.0
	#遍历测试集数据
	for line in frTest.readlines():
		#
		numTestVec += 1.0
		#切分数据集
		currLine =line.strip().split('\t')
		#初始化临时列表
		lineArr = []
		#遍历21项数据重新生成列表，因为后面格式要求，这里必须重新生成一下。
		for i in range(21):
			lineArr.append(float(currLine[i]))
		#如果分类结果和分类标签不符，则错误计数+1
		if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
			errorCount += 1
	#计算分类错误率
	errorRate = (float(errorCount)/numTestVec)
	#打印分类错误率
	print("the error rate of this test is: %f" % errorRate)
	#返回分类错误率
	return errorRate


def multiTest():
    numTests = 10; errorSum=0.0     #计算10次，求平均错误率
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is:%f' %(numTests,errorSum/float(numTests)))

multiTest()           
    

















