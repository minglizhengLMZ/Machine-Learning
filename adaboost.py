# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 10:12:36 2019

@author: Administrator
"""

'''
一、集成学习的多种形式：
1.不同算法的集成，可以将多种分类器集成起来
2.同一算法在不同设置的集成
3.数据集不同部分分配给不同分类器之后的集成
Adaboost算法：
优点：泛化错误率低，易编码，可以应用在大部分分类器上，无参数调整
缺点：对离群点敏感
适用数据类型：数值型和标称型数据
bagging:基于数据随机重抽样的分类器构建方法（在原数据集选择s个新数据集的一种技术）
如随机森林：
不同分类器通过并行得到
分类器的类型也是一致
抽取S个样本，使用同一学习算法分别建立s个分类器，进行分类，
选择分类器投票结果最多的类别作为最后的分类结果，各分类器权重相同


boosting：多个分类器的类型也是一致的
不同：不同分类器通过串行训练获得
每个分类器都根据已经训练好的分类器性能来进行训练
分类结果通过各分类器加权求和的结果

boosting最流行版本------AdaBoost（adaptive boosting，自适应boosting）
实质：多个分类器，分类结果的加权平均
AdaBoost一般流程：
（1）搜集数据
（2）准备数据:依赖所使用的的弱分类器类型，本章使用单层决策树，
这种分类器可以处理任何类型数据，也可以使用任何分类器作为弱分类器
（3）分析数据：任何方法
（4）训练算法：AdaBoost的大部分时间都在训练上，分类器将多次在同一数据集上训练弱分类器
（5）测试算法：使用分类的错误率
（6）使用算法：同SVM，AdaBoost预测两个类别中的一个，如果把他们应用到个类别场合，
那么就要像多类SVM中做法一样，对AdaBoost进行修改
过程：
1.训练数据集的每个样本，并为个样本赋予一个权重向量D，初始权重均为1
2.在训练数据上训练一个弱学习器，并计算分类错误率
3.使用同一数据集，再次训练弱学习器根据上一步计算的是错误率调整样本所占比重调整权重向量D，上次分对的比重调低，分错的比重调高
正确分类的权重：D_i^((t+1))=(D_i^((t)) e^(-α))/(sum(D))
错误分类的权重：D_i^((t+1))=(D_i^((t)) e^α)/(sum(D))
4.从所有若学习器得到最终的分类结果，AdaBoost为每个分类器分配一个权重（取决于与分类器的分类错误率），
e=未正确分类的样本数/分类正确的样本数
alpha=1/2ln((1-e)/e)

单层决策树：仅根据单个特征来做决策
'''
from numpy import *
#构建数据集
def loadSimpData():
    datMat=matrix([[1.,2.1],
                   [2.,1.1],
                   [1.3,1.],
                   [1.,1.],
                   [2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels


datMat,classLabels=loadSimpData()

'''通过多个函数建立多层决策树
第一个函数测试是否有某个值小于或者大于正在测试的阙值
第二个函数在加权数据集中循环，找到具有最低错误率的单层决策树
伪代码：
将最小错误率minError设为+无穷
for数据集的每个特征（第一层循环)：
    对每个步长（第二层循环）:
        对每个不等号（第三层循环）:
            建立一颗单层决策树并利用加权数据集进行测试
            如果错误率低于minError，则将当前单层决策树设置为最佳单层决策树
返回最佳单层决策树
'''
'''单层决策树的生成函数'''

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
	"""
	Function：	通过阈值比较对数据进行分类
	Input：		dataMatrix：数据集
				dimen：数据集列数
				threshVal：阈值
				threshIneq：比较方式：lt，gt
	Output：	retArray：分类结果
	"""	
	#新建一个数组用于存放分类结果，初始化都为1
	retArray = ones((shape(dataMatrix)[0],1))  
	#lt：小于，gt；大于；根据阈值进行分类，并将分类结果存储到retArray
	if threshIneq == 'lt':
		retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:, dimen] > threshVal] = -1.0
	#返回分类结果
	return retArray

'''----------------弱分类器的构建--------------'''
def buildStump(dataArr, classLabels, D):
	"""
	Function：	找到最低错误率的单层决策树
	Input：		dataArr：数据集
				classLabels：数据标签
				D：权重向量
	Output：	bestStump：分类结果
				minError：最小错误率
				bestClasEst：最佳单层决策树
	"""	
	#初始化数据集和数据标签
	dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
	#获取行列值
	m,n = shape(dataMatrix)
	#初始化步数，用于在特征的所有可能值上进行遍历
	numSteps = 10.0
	#初始化字典，用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
	bestStump = {}
	#初始化类别估计值
	bestClasEst = mat(zeros((m,1)))
	#将最小错误率设无穷大，之后用于寻找可能的最小错误率
	minError = inf
	#遍历数据集中每一个特征
	for i in range(n):
		#获取数据集的最大最小值
		rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
		#根据步数求得步长
		stepSize = (rangeMax - rangeMin) / numSteps
		#遍历每个步长
		for j in range(-1, int(numSteps) + 1):
			#遍历每个不等号 <=或者>
			for inequal in ['lt', 'gt']:
				#设定阈值
				threshVal = (rangeMin + float(j) * stepSize)
				#通过阈值比较对数据进行分类
				predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
				#初始化错误计数向量
				errArr = mat(ones((m,1)))
				#如果预测结果和标签相同，则相应位置0
				errArr[predictedVals == labelMat] = 0
				#计算权值误差，这就是AdaBoost和分类器交互的地方
				weightedError = D.T * errArr
				#打印输出所有的值
				#print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
				#如果错误率低于minError，则将当前单层决策树设为最佳单层决策树，更新各项值
				if weightedError < minError:
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	#返回最佳单层决策树，最小错误率，类别估计值
	return bestStump, minError, bestClasEst

#D=mat(ones((5,1))/5)
#buildStump(datMat,classLabels,D)

'''---------基于单层决策树的AdaBoost训练过程-----多个弱分类器构建Adaboost'''

'''伪代码：
对于每次迭代:
    利用buildStump()函数找到最佳的单层决策树
    将最佳的单层决策树加入到单层决策树数组
    计算alpha
    计算新的权重向量D
    更新累计类别估计值
    如果错误率等于0.0，退出循环
    
AdaBoost仅一个需要用户指定参数——迭代次数
'''
def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
	"""
	Function：	找到最低错误率的单层决策树
	Input：		dataArr：数据集
				classLabels：数据标签
				numIt：迭代次数
	Output：	weakClassArr：单层决策树列表
				aggClassEst：类别估计值
	"""	
	#初始化列表，用来存放单层决策树的信息
	weakClassArr = []
	#获取数据集行数
	m = shape(dataArr)[0]
	#初始化向量D每个值均为1/m，D包含每个数据点的权重
	D = mat(ones((m,1))/m)
	#初始化列向量，记录每个数据点的类别估计累计值
	aggClassEst = mat(zeros((m,1)))
	#开始迭代
	for i in range(numIt):
		#利用buildStump()函数找到最佳的单层决策树，同一数据集，不同样本权重得到基学习器
		bestStump, error, classEst = buildStump(dataArr, classLabels, D)
		print("D: ", D.T)
		#根据公式计算分类器的权重alpha的值，max(error, 1e-16)用来确保在没有错误时不会发生除零溢出
		alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
		#保存alpha的值
		bestStump['alpha'] = alpha
		#填入数据到列表，将得到的基学习器加入到基学习器列表中
		weakClassArr.append(bestStump)
		print("classEst: ", classEst.T)
		#为下一次迭代计算D，更新样本的权重系数D
		expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
		D = multiply(D, exp(expon))
		D = D / D.sum()
		#累加类别估计值
		aggClassEst += alpha * classEst
		print("aggClassEst: ", aggClassEst.T)
		#计算错误率，aggClassEst本身是浮点数，需要通过sign来得到二分类结果
		aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))
		errorRate = aggErrors.sum() / m
		print("total error: ", errorRate)
		#如果总错误率为0则跳出循环
		if errorRate == 0.0: break
	#返回单层决策树列表和累计错误率
	#return weakClassArr
	return weakClassArr, aggClassEst

#classifierArray=adaBoostTrainDS(datMat, classLabels, 9)

'''测试算法，基于AdaBoost的分类'''
#AdaBoost分类函数,区别于adaBoostTrainDS即将弱分类器抽离出来
def adaClassify(datToClass, classifierArr):
	"""
	Function：	AdaBoost分类函数
	Input：		datToClass：待分类样例
				classifierArr：多个弱分类器组成的数组
	Output：	sign(aggClassEst)：分类结果
	"""	
	#初始化数据集
	dataMatrix = mat(datToClass)
	#获得待分类样例个数
	m = shape(dataMatrix)[0]
	#构建一个初始化为0的列向量，记录每个数据点的类别估计累计值
	aggClassEst = mat(zeros((m,1)))
	#遍历每个弱分类器
	for i in range(len(classifierArr)):
		#基于stumpClassify得到类别估计值
		classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
		#累加类别估计值
		aggClassEst += classifierArr[i]['alpha']*classEst
		#打印aggClassEst，以便我们了解类别估计值其变化情况
		print(aggClassEst)
	#返回分类结果，aggClassEst大于0则返回+1，否则返回-1
	return sign(aggClassEst)
'''
#获得数据集
datArr,labelArr = loadSimpData()
#获得弱分类
classifierArr, aggClassEst= adaBoostTrainDS(datArr, labelArr,30)
adaClassify([0,0], classifierArr)
adaClassify([[5,5],[0,0]], classifierArr)
'''
'''在一个难数据集上应用AdaBoost'''
'''
（1）搜集数据
（2）准备数据:确保数据标签为1和-1，而不是1和0
（3）分析数据：手工检查数据
（4）训练算法：在数据上，利用adaBoostTrainDS()函数训出一系列
（5）测试算法：拥有两个数据集，在不采用随机抽样的方法下，
就会对AdaBoost和logistic回归的结果进行完全对等的比较
（6）使用算法：观察训练误差和测试误差
'''

'''自适应数据加载函数的作用：能自动检测出特征的数目,（但此函数也假设类别标签在最后一列）'''
def loadDataSet(fileName):
	"""
	Function：	自适应数据加载函数
	Input：		fileName：文件名称
	Output：	dataMat：数据集
				labelMat：类别标签
	"""	
	#自动获取特征个数，这是和之前不一样的地方
	numFeat = len(open(fileName).readline().split('\t'))
	#初始化数据集和标签列表
	dataMat = []; labelMat = []
	#打开文件
	fr = open(fileName)
	#遍历每一行
	for line in fr.readlines():
		#初始化列表，用来存储每一行的数据
		lineArr = []
		#切分文本
		curLine = line.strip().split('\t')
		#遍历每一个特征，某人最后一列为标签
		for i in range(numFeat-1):
			#将切分的文本全部加入行列表中
			lineArr.append(float(curLine[i]))
		#将每个行列表加入到数据集中
		dataMat.append(lineArr)
		#将每个标签加入标签列表中
		labelMat.append(float(curLine[-1]))
	#返回数据集和标签列表
	return dataMat, labelMat

datArr,labelArr=loadDataSet(r'E:\Anaconda3\Python_work\data\horseColicTraining2.txt')
#训练AdaBoost,得到弱分类器及训练误差
classifierArr, aggClassEst= adaBoostTrainDS(datArr, labelArr,10)


#测试数据，得到分类结果
testArr,testLabelArr=loadDataSet(r'E:\Anaconda3\Python_work\data\horseColicTest2.txt')
prediction10=adaClassify(testArr,classifierArr)
#得到测试误差
errArr=mat(ones((shape(testArr)[0],1)))
errArr[prediction10 != mat(testLabelArr).T].sum()

'''--------非均衡分类问题---------------'''
'''之前假设所有类别的分类代价相同，单现实中很多分类的代价不同
引入度量指标：正确率，召回率，及ROC曲线，混淆矩阵，F1值，
'''
'''---ROC曲线的绘制，及AUC计算函数-------'''
	#导入pyplot
import matplotlib.pyplot as plt
def plotROC(predStrengths, classLabels):
	"""
	Function：	ROC曲线的绘制及AUC计算函数
	Input：		predStrengths：分类器的预测强度
				classLabels：类别标签
	Output：	ySum * xStep：AUC计算结果
	"""	
	#创建一个浮点数二元组并初始化为(1.0, 1.0),该元祖保留的是绘制光标的位置
	cur = (1.0, 1.0)
	#初始化ySum，用于计算AUC
	ySum = 0.0
	#通过数组过滤计算正例的数目，即y轴的步进数目
	numPosClas = sum(array(classLabels) == 1.0)
	#初始化x轴和y轴的步长
	yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels) - numPosClas)
	#得到排序索引，从<1.0,1.0>开始绘制，一直到<0,0>
	sortedIndicies = predStrengths.argsort()
	#用于构建画笔，熟悉的matlab绘图方式
	fig = plt.figure()
	fig.clf()
	ax = plt.subplot()
	#在所有排序值上遍历，因为python的迭代需要列表形式，所以调用tolist()方法
	for index in sortedIndicies.tolist()[0]:
		#每得到一个标签为1.0的类，则沿y轴下降一个步长
		if classLabels[index] == 1.0:
			delX = 0; delY = yStep
		#否则，则沿x轴下降一个步长
		else:
			delX = xStep; delY = 0
			#计数，用来计算所有高度的和，从而计算AUC
			ySum += cur[1]
		#下降步长的绘制，蓝色
		ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'b')
		#绘制光标坐标值更新
		cur = (cur[0] - delX, cur[1] - delY)
	#蓝色双划线
	ax.plot([0,1], [0,1], 'b--')
	#坐标轴标签
	plt.xlabel('False positive rate'); plt.ylabel('Ture positive rate')
	#表头
	plt.title('ROC curve for AdaBoost horse colic detection system')
	#设定坐标范围
	ax.axis([0,1,0,1])
	#显示绘制结果
	plt.show()
	#打印AUC计算结果
	print("the Area Under the Curve is: ", ySum * xStep)

datArr,labelArr=loadDataSet(r'E:\Anaconda3\Python_work\data\horseColicTraining2.txt')
classifierArr, aggClassEst= adaBoostTrainDS(datArr, labelArr,50)
plotROC(aggClassEst.T,labelArr)
