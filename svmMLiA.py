# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:12:55 2019

@author: Administrator
"""

'''---------支持向量机------------------'''
'''
SVM原理----------找到超平面对数据进行分类
（1）：找到离分割平面最近的的点，支持向量
（2）最大化的间隔（支持向量与超平面间隔尽可能大）
使用拉格朗日乘子法，smo算法进行求解

SVM一般流程
（1）搜集数据：采用任意方法搜集数据
（2）准备数据：数据类型应为数值型数据
（3）分析数据：有助于可视化分割超平面
（4）训练算法：大部分时间将用于训练，该过程主要实现两个参数的调优
（5）测试算法：十分简单的计算过程即可实现
（6）使用算法：几乎所有分类问题都可以使用SVM，svm本身是二类分类器，
多分类问题用svm仅需要对代码做一些修改

SMO算法：将大优化问题分解为多个小优化问题，将这些问题作为顺序问题求解的结果，与作为整体求解的结果一直
目标：求解一系列alpha和b
工作原理：
每次循环两个alpha进行优化处理，一旦找到对合适的alpha就增大其中一个，同减小另一个，
（为什么对其alpha对进行增大和缩小进行优化）
合适的条件：
（1）：两个alpha要在间隔边界之外，
（2）：两个alpha还没有进行过区间化处理，或者不在边界上
同时改变两个alpha的原因：
sum(αi*label（i）)=0,进改变一个alpha会使等式不满足，因此成对优化
构造辅助函数，用于在某个区间范围内随机选择一个整数，另一个辅助函数，用于数值太大时对其进行调整
'''
'''-----------------SMO算法的辅助函数----------------'''
from numpy import *
from time import sleep
#导入数据，数据预处理
def loadDataSet(fileName):
	"""
	Function：	加载数据集
	Input：		fileName：数据集
	Output：	dataMat：数据矩阵
				labelMat：类别标签矩阵
	"""	
	#初始化数据列表
	dataMat = []
	#初始化标签列表
	labelMat = []
	#打开文件
	fr = open(fileName)
	#遍历每一行数据，读取数据和标签
	for line in fr.readlines():
		#strip删除头尾空白字符，然后再进行分割
		lineArr = line.strip().split('\t')
		#填充数据集
		dataMat.append([float(lineArr[0]), float(lineArr[1])])
		#填充类别标签
		labelMat.append(float(lineArr[2]))
	#返回数据集和标签列表
	return dataMat, labelMat
#选择alpha对
def selectJrand(i, m):
	"""
	Function：	随机选择
	Input：		i：alpha下标
				m：alpha数目
	Output：	j：随机选择的alpha下标
	"""	
	#初始化下标j
	j = i
	#随机化产生j，直到不等于i
	while (j == i):
		j = int(random.uniform(0,m))
	#返回j的值
	return j
#设定alpha的阙值，避免过大或者过小
def clipAlpha(aj, H, L):
	"""
	Function：	设定alpha阈值
	Input：		aj：alpha的值
				H：alpha的最大值
				L：alpha的最小值
	Output：	aj：处理后的alpha的值
	"""	
	#如果输入alpha的值大于最大值，则令aj=H
	if aj > H:
		aj = H
	#如果输入alpha的值小于最小值，则令aj=L
	if L > aj:
		aj = L
	#返回处理后的alpha的值
	return aj
#测试           
#dataArr,labelArr=loadDataSet(r'E:\python_study\Python_work\data\Ch06\testSet.txt')
#labelArr

'''SMO函数的伪代码:
创建一个alpha向量，并将其初始化为0向量
当迭代次数小于最大迭代次数时（外循环）:
    对数据集中的每个数据向量（内循环）:
        如果该数据可以被优化:
            随机选择另外一个数据向量
            同时优化这两个向量
            如果这两个向量都不能被优化，退出内循环
    如果所有向量都还没有被优化，增加迭代数目，继续下一次循环'''

'''------------简化版smo算法---------------------'''
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
	"""
	Function：	简化版SMO算法
	Input：		dataMatIn：数据集
				classLabels：类别标签
				C：常数C
				toler：容错率
				maxIter：最大的循环次数
	Output：	b：常数项
				alphas：数据向量
	"""	
	#将输入的数据集和类别标签转换为NumPy矩阵
	dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
	#初始化常数项b，初始化行列数m、n
	b = 0; m,n = shape(dataMatrix)
	#初始化alphas数据向量为0向量
	alphas = mat(zeros((m,1)))
	#初始化iter变量，存储的是在没有任何alpha改变的情况下遍历数据集的次数
	iter = 0
	#外循环，当迭代次数小于maxIter时执行
	while (iter < maxIter):
		#alpha改变标志位每次都要初始化为0
		alphaPairsChanged = 0
		#内循环，遍历所有数据集
		for i in range(m):
			#multiply将alphas和labelMat进行矩阵相乘，求出法向量w(m,1),w`(1,m)
			#dataMatrix * dataMatrix[i,:].T，求出输入向量x(m,1)
			#整个对应输出公式f(x)=w`x + b
			fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b
			#计算误差
			Ei = fXi - float(labelMat[i])
			#如果标签与误差相乘之后在容错范围之外，且超过各自对应的常数值，则进行优化
			if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
				#随机化选择另一个数据向量
				j = selectJrand(i, m)
				#对此向量进行同样的计算
				fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j,:].T)) + b
				#计算误差
				Ej = fXj - float(labelMat[j])
				#利用copy存储刚才的计算值，便于后期比较
				alphaIold = alphas[i].copy(); alpahJold = alphas[j].copy()
				#保证alpha在0和C之间
				if (labelMat[i] != labelMat[j]):
					L = max(0, alphas[j] - alphas[i])
					H = min(C, C + alphas[j] - alphas[i])
				else:
					L = max(0, alphas[j] + alphas[i] - C)
					H = min(C, alphas[j] + alphas[i])
				#如果界限值相同，则不做处理直接跳出本次循环
				if L == H: print("L == H"); continue
				#最优修改量，求两个向量的内积（核函数）
				eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T - dataMatrix[j,:] * dataMatrix[j,:].T
				#如果最优修改量大于0，则不做处理直接跳出本次循环，这里对真实SMO做了简化处理
				if eta >= 0: print("eta >= 0"); continue
				#计算新的alphas[j]的值  
				alphas[j] -= labelMat[j] * (Ei - Ej) / eta
				#对新的alphas[j]进行阈值处理
				alphas[j] = clipAlpha(alphas[j], H, L)
				#如果新旧值差很小，则不做处理跳出本次循环
				if (abs(alphas[j] - alpahJold) < 0.00001): print("j not moving enough"); continue
				#对i进行修改，修改量相同，但是方向相反
				alphas[i] += labelMat[j] * labelMat[i] * (alpahJold - alphas[j])
				#新的常数项
				b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - labelMat[i] * (alphas[j] - alpahJold) * dataMatrix[i,:] * dataMatrix[j,:].T
				b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - labelMat[j] * (alphas[j] - alpahJold) * dataMatrix[j,:] * dataMatrix[j,:].T
				#谁在0到C之间，就听谁的，否则就取平均值
				if (0 < alphas[i]) and (C > alphas[i]): b = b1
				elif (0 < alphas[j]) and (C > alphas[j]): b = b2
				else: b = (b1 + b2) / 2.0
				#标志位加1
				alphaPairsChanged += 1
				#输出迭代次数，alphas的标号以及标志位的值
				print("iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
		#如果标志位没变，即没有进行优化，那么迭代值加1
		if (alphaPairsChanged == 0): iter += 1
		#否则迭代值为0
		else: iter = 0
		#打印迭代次数
		print("iteration number: %d" % iter)
	#返回常数项和alphas的数据向量
	return b, alphas

#检验输出的结果
'''
b,alphas=smoSimple(dataArr,labelArr,0.6,0.001,40)
b
alphas[alphas>0]
shape(alphas[alphas>0])

#了解哪些点是支持向量
for i in range(100):
    if alphas[i]>0.0:
        print(dataArr[i],labelArr[i])
'''

'''----------------利用完整的platt SMO算法加速优化------------'''
'''
简化的SMO算法通过外循环选择第一个alpha,内循环随机选择第二个alpha
完整的SMO根据最大步长选择alpha
'''
'''
class optStruct:
    def _init_(self,dataMatIn,classLabels,C,toler):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]   #m为输入数据的行数
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        self.eCache=mat(zero((self.m,2)))

def calcEk(oS,k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k,:].T)) + oS.b
    Ek=fXk-float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei):
    maxK=-1;maxDeltaE=0;Ej=0
    oS.eCache[i]=[1,Ei]
    validEcacheList=nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i:
                continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if (deltaE>maxDeltaE):
                maxK=k;maxDeltaE=deltaE;Ej=Ek
        return maxK,Ej
    else:
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j ,Ej

def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]
'''   
        
    
'''--------------SMO寻找决策边界的优化历程-----------------'''
'''区别于smoSimple(),
(1)使用自己的数据结构，该数据结构在参数os传递
（2）使用函数selectJ进行参数alpha选择，而不是使用selectJrand
(3)在alpha改变时更新eCache       
'''
'''
def innerL(i,oS):
    Ei=calcEk(oS,i)
    if ((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0)):
        j,Ej=selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L= max(0,oS.alphas[j]-oS.alphas[i])
            H= min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H :
            print('L==H')
            return 0
        eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T
				#如果最优修改量大于0，则不做处理直接跳出本次循环，这里对真实SMO做了简化处理
        if eta >= 0:
            print("eta >= 0")
            return 0
				#计算新的alphas[j]的值
        oS.alphas[j]-= oS.labelMat[j] * (Ei-Ej)/eta #对新的alphas[j]进行阈值处理
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
				#如果新旧值差很小，则不做处理跳出本次循环
        if (abs(oS.alphas[j] - oS.alpahJold) < 0.00001):
            print("j not moving enough")
            return 0
				#对是方向相反i进行修改，修改量相同，但
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alpahJold - oS.alphas[j])
				#新的常数项
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i,:] * oS.X[i,:].T - oS.labelMat[i] * (oS.alphas[j] - alpahJold) * oS.X[i,:] * oS.X[j,:].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i,:] * oS.X[j,:].T - oS.labelMat[j] * (oS.alphas[j] - alpahJold) * oS.X[j,:] * oS.X[j,:].T
				#谁在0到C之间，就听谁的，否则就取平均值
        if (0 < oS.alphas[i]) and (C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (C > oS.alphas[j]): oS.b = b2
        else: 
            oS.b = (b1 + b2) / 2.0
            return  1
    else:
        return 0
'''       


'''----------完整的SMO外循环代码------------'''
'''
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):  
    #此处使用线性核，kTup用来存储核函数信息
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler) 
    #通过类将数据进行结构化
    iter=0
    entireSet = True
    alphaPairsChanged = 0
    while (iter< maxIter) and ((alphaPairsChanged>0) or (entireSet)):
        alphaPairsChanged =0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print('fullSet,iter:%d i:%d,pairs changed %d' %(iter,i,alphaPairsChanged))
            iter +=1
        else:
            nonBoundIs = nonZero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print('non-bound,iter:%d i:%d,pairs changed %d' %(iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print('iteration number: %d' %iter)
    return oS.b,oS.alphas

dataArr,labelArr = loadDataSet(r'E:\python_study\Python_work\data\Ch06\testSet.txt')
b,alphas = smoP(dataArr,labelArr,0.6,0.001,40)
'''       


'''---------核函数的定义，并引入了径向基核函数（基于正态）----------'''

def kernelTrans(X, A, kTup):
	"""
	Function：	核转换函数
	Input：		X：数据集
				A：某一行数据
				kTup：存储核函数信息，三个数据，字符串存储核函数类型，其他两个参数为核函数可选参数
	Output：	K：计算出的核向量
	"""	
	#获取数据集行列数
	m, n = shape(X)
	#初始化列向量,m行1列，元素赋值为0
	K = mat(zeros((m, 1)))
	#根据键值选择相应核函数
	#lin表示的是线性核函数
	if kTup[0] == 'lin': K = X * A.T
	#rbf表示径向基核函数
	elif kTup[0] == 'rbf':
		for j in range(m):
			deltaRow = X[j,:] - A
			K[j] = deltaRow * deltaRow.T
		#对矩阵元素展开计算，而不像在MATLAB中一样计算矩阵的逆
		K =  exp(K/(-1*kTup[1]**2))
	#如果无法识别，就报错
	else: raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
	#返回计算出的核向量
	return K

class optStructK:
	"""
	Function：	存放运算中重要的值
	Input：		dataMatIn：数据集
				classLabels：类别标签
				C：常数C
				toler：容错率
				kTup：速度参数
	Output：	X：数据集
				labelMat：类别标签
				C：常数C
				tol：容错率
				m：数据集行数
				b：常数项
				alphas：alphas矩阵
				eCache：误差缓存
				K：核函数矩阵
	"""	
	def __init__(self, dataMatIn, classLabels, C, toler, kTup):
		self.X = dataMatIn
		self.labelMat = classLabels
		self.C = C
		self.tol = toler
		self.m = shape(dataMatIn)[0]   
		self.alphas = mat(zeros((self.m, 1)))   #存储模型系数alpha
		self.b = 0     #存储偏移值b
		self.eCache = mat(zeros((self.m, 2)))
		

		""" 主要区分 """
		self.K = mat(zeros((self.m, self.m)))    #k为m*m的矩阵，暂时赋值为0
		for i in range(self.m):
			self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup) 
		""" 主要区分 """

def calcEkK(oS, k):
	"""
	Function：	计算误差值E
	Input：		oS：数据结构
				k：下标
	Output：	Ek：计算的E值
	"""	

	""" 主要区分 """
	#计算fXk，整个对应输出公式f(x)=w`x + b
	#fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k,:].T) + oS.b)
	fXk = float(multiply(oS.alphas, oS.labelMat).T*oS.K[:, k] + oS.b) 
    #此处改进用核函数代替内积运算
	""" 主要区分 """

	#计算E值
	Ek = fXk - float(oS.labelMat[k])
	#返回计算的误差值E
	return Ek

def selectJK(i, oS, Ei):
	"""
	Function：	选择第二个alpha的值
	Input：		i：第一个alpha的下标
				oS：数据结构
				Ei：计算出的第一个alpha的误差值
	Output：	j：第二个alpha的下标
				Ej：计算出的第二个alpha的误差值
	"""	
	#初始化参数值
	maxK = -1; maxDeltaE = 0; Ej = 0
	#构建误差缓存
	oS.eCache[i] = [1, Ei]
	#构建一个非零列表，返回值是第一个非零E所对应的alpha值，而不是E本身
	validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
	#如果列表长度大于1，说明不是第一次循环
	if (len(validEcacheList)) > 1:
		#遍历列表中所有元素
		for k in validEcacheList:
			#如果是第一个alpha的下标，就跳出本次循环
			if k == i: continue
			#计算k下标对应的误差值
			Ek = calcEkK(oS, k)
			#取两个alpha误差值的差值的绝对值
			deltaE = abs(Ei - Ek)
			#最大值更新
			if (deltaE > maxDeltaE):
				maxK = k; maxDeltaE = deltaE; Ej = Ek
		#返回最大差值的下标maxK和误差值Ej
		return maxK, Ej
	#如果是第一次循环，则随机选择alpha，然后计算误差
	else:
		j = selectJrand(i, oS.m)
		Ej = calcEkK(oS, j)
	#返回下标j和其对应的误差Ej
	return j, Ej

def updateEkK(oS, k):
	"""
	Function：	更新误差缓存
	Input：		oS：数据结构
				j：alpha的下标
	Output：	无
	"""	
	#计算下表为k的参数的误差
	Ek = calcEkK(oS, k)
	#将误差放入缓存
	oS.eCache[k] = [1, Ek]
'''----------选择第二个alpha的方法，由随机改进为间隔最大-------'''
def innerLK(i, oS):
	"""
	Function：	完整SMO算法中的优化例程
	Input：		oS：数据结构
				i：alpha的下标
	Output：	无
	"""	
	#计算误差
	Ei = calcEkK(oS, i)
	#如果标签与误差相乘之后在容错范围之外，且超过各自对应的常数值，则进行优化
	if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
		#启发式选择第二个alpha值
		j, Ej = selectJK(i, oS, Ei)
		#利用copy存储刚才的计算值，便于后期比较
		alphaIold = oS.alphas[i].copy(); alpahJold = oS.alphas[j].copy();
		#保证alpha在0和C之间
		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS. alphas[i])
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])
		#如果界限值相同，则不做处理直接跳出本次循环
		if L == H: print("L==H"); return 0

		""" 主要区分 """
		#最优修改量，求两个向量的内积（核函数）
		#eta = 2.0 * oS.X[i, :]*oS.X[j, :].T - oS.X[i, :]*oS.X[i, :].T - oS.X[j, :]*oS.X[j, :].T
		eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
		""" 主要区分 """

		#如果最优修改量大于0，则不做处理直接跳出本次循环，这里对真实SMO做了简化处理
		if eta >= 0: print("eta>=0"); return 0
		#计算新的alphas[j]的值
		oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
		#对新的alphas[j]进行阈值处理
		oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
		#更新误差缓存
		updateEkK(oS, j)
		#如果新旧值差很小，则不做处理跳出本次循环
		if (abs(oS.alphas[j] - alpahJold) < 0.00001): print("j not moving enough"); return 0
		#对i进行修改，修改量相同，但是方向相反
		oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alpahJold - oS.alphas[j])
		#更新误差缓存
		updateEkK(oS, i)

		""" 主要区分 """
		#更新常数项
		#b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :]*oS.X[i, :].T - oS.labelMat[j] * (oS.alphas[j] - alpahJold) * oS.X[i, :]*oS.X[j, :].T
		#b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :]*oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j] - alpahJold) * oS.X[j, :]*oS.X[j, :].T
		b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alpahJold) * oS.K[i, j]
		b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alpahJold) * oS.K[j, j]
		""" 主要区分 """

		#谁在0到C之间，就听谁的，否则就取平均值
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[i]): oS.b = b2
		else: oS.b = (b1 + b2) / 2.0
		#成功返回1
		return 1
	#失败返回0
	else: return 0
'''----定义SMO算法，其中包括数据集，数据标记，阙值常数c,最大迭代次数，核函数'''
def smoPK(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
	"""
	Function：	完整SMO算法
	Input：		dataMatIn：数据集
				classLabels：类别标签
				C：常数C
				toler：容错率
				maxIter：最大的循环次数
				kTup：速度参数
	Output：	b：常数项
				alphas：数据向量
	"""	
	#新建数据结构对象
	oS = optStructK(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
	#初始化迭代次数
	iter = 0
	#初始化标志位
	entireSet = True; alphaPairsChanged = 0
	#终止条件：迭代次数超限、遍历整个集合都未对alpha进行修改
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
		alphaPairsChanged = 0
		#根据标志位选择不同的遍历方式
		if entireSet:
			#遍历任意可能的alpha值
			for i in range(oS.m):
				#选择第二个alpha值，并在可能时对其进行优化处理
				alphaPairsChanged += innerLK(i, oS)
				print("fullSet, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
			#迭代次数累加
			iter += 1
		else:
			#得出所有的非边界alpha值
			nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
			#遍历所有的非边界alpha值
			for i in nonBoundIs:
				#选择第二个alpha值，并在可能时对其进行优化处理
				alphaPairsChanged += innerLK(i, oS)
				print("non-bound, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
			#迭代次数累加
			iter += 1
		#在非边界循环和完整遍历之间进行切换
		if entireSet: entireSet = False
		elif (alphaPairsChanged == 0): entireSet =True
		print("iteration number: %d" % iter)
	#返回常数项和数据向量
	return oS.b, oS.alphas

def testRbf(k1 = 1.3):
	"""
	Function：	利用核函数进行分类的径向基测试函数
	Input：		k1：径向基函数的速度参数
	Output：	输出打印信息
	"""	
	#导入数据集
	dataArr, labelArr = loadDataSet(r'E:\python_study\Python_work\data\Ch06\testSetRBF.txt')
	#调用Platt SMO算法
	b, alphas = smoPK(dataArr, labelArr, 200, 0.00001, 10000, ('rbf', k1))
	#初始化数据矩阵和标签向量
	datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
	#记录支持向量序号,.A读出该行数据，如果>0,则记录第一列的数据，即序号
	svInd = nonzero(alphas.A > 0)[0]
	#读取支持向量（支持向量）
	sVs = datMat[svInd]
	#读取支持向量对应标签
	labelSV = labelMat[svInd]
	#输出打印信息
	print("there are %d Support Vectors" % shape(sVs)[0])
	#获取数据集行列值
	m, n = shape(datMat)
	#初始化误差计数
	errorCount = 0
	#遍历每一行，利用核函数对训练集进行分类
	for i in range(m):
		#利用核函数转换数据
		kernelEval = kernelTrans(sVs, datMat[i,:], ('rbf', k1))
		#仅用支持向量预测分类
		predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
		#预测分类结果与标签不符则错误计数加一
		if sign(predict) != sign(labelArr[i]): errorCount += 1
	#打印输出分类错误率
	print("the training error rate is: %f" % (float(errorCount)/m))
	#导入测试数据集
	dataArr, labelArr = loadDataSet(r'E:\python_study\Python_work\data\Ch06\testSetRBF2.txt')
	#初始化误差计数
	errorCount = 0
	#初始化数据矩阵和标签向量
	datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
	#获取数据集行列值
	m, n = shape(datMat)
	#遍历每一行，利用核函数对测试集进行分类
	for i in range(m):
		#利用核函数转换数据
		kernelEval = kernelTrans(sVs, datMat[i,:], ('rbf', k1))
		#仅用支持向量预测分类
		predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
		#预测分类结果与标签不符则错误计数加一
		if sign(predict) != sign(labelArr[i]): errorCount += 1
	#打印输出分类错误率
	print("the test error rate is: %f" % (float(errorCount)/m))

#testRbf(k1 = 1.3)
    
    
    
'''--------SVM在手写识别问题的应用-------------'''
def img2vector(filename):
	"""
	Function：	32*32图像转换为1*1024向量
	Input：		filename：文件名称字符串
	Output：	returnVect：转换之后的1*1024向量
	"""	
#初始化要返回的1*1024向量,将矩阵拉伸成向量
	returnVect = zeros((1, 1024))
	#打开文件
	fr = open(filename)
	#读取文件信息
	for i in range(32):
		#循环读取文件的前32行
		lineStr = fr.readline()
		for j in range(32):
			#将每行的头32个字符存储到要返回的向量中
			returnVect[0, 32*i+j] = int(lineStr[j])
	#返回要输出的1*1024向量
	return returnVect

def loadImages(dirName):
	"""
	Function：	加载图片
	Input：		dirName：文件路径
	Output：	trainingMat：训练数据集
				hwLabels：数据标签
	"""	
	from os import listdir
	#初始化数据标签
	hwLabels = []
	#读取文件列表
	trainingFileList = listdir(dirName)
	#读取文件个数
	m = len(trainingFileList)
	#初始化训练数据集
	trainingMat = zeros((m,1024))
	#填充数据集
	for i in range(m):
		#遍历所有文件
		fileNameStr = trainingFileList[i]
		#提取文件名称
		fileStr = fileNameStr.split('.')[0]
		#提取数字标识
		classNumStr = int(fileStr.split('_')[0])
		#数字9记为-1类
		if classNumStr == 9: hwLabels.append(-1)
		#其他数字记为+1类
		else: hwLabels.append(1)
		#提取图像向量，填充数据集
		trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
	#返回数据集和数据标签
	return trainingMat, hwLabels
    
def testDigits(kTup = ('rbf',10)):
	"""
	Function：	手写数字分类函数
	Input：		kTup：核函数采用径向基函数
	Output：	输出打印信息
	"""	
	#导入数据集
	dataArr, labelArr = loadImages(r'E:\python_study\Python_work\data\Ch06\digits\trainingDigits')
	#调用Platt SMO算法
	b, alphas = smoPK(dataArr, labelArr, 200, 0.0001, 10000, kTup)
	#初始化数据矩阵和标签向量
	datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
	#记录支持向量序号
	svInd = nonzero(alphas.A > 0)[0]
	#读取支持向量
	sVs = datMat[svInd]
	#读取支持向量对应标签
	labelSV = labelMat[svInd]
	#输出打印信息
	print("there are %d Support Vectors" % shape(sVs)[0])
	#获取数据集行列值
	m, n = shape(datMat)
	#初始化误差计数
	errorCount = 0
	#遍历每一行，利用核函数对训练集进行分类
	for i in range(m):
		#利用核函数转换数据
		kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
		#仅用支持向量预测分类
		predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
		#预测分类结果与标签不符则错误计数加一
		if sign(predict)!=sign(labelArr[i]): errorCount += 1
	#打印输出分类错误率
	print("the training error rate is: %f" % (float(errorCount)/m))
	#导入测试数据集
	dataArr,labelArr = loadImages(r'E:\python_study\Python_work\data\Ch06\digits\testDigits')
	#初始化误差计数
	errorCount = 0
	#初始化数据矩阵和标签向量
	datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
	#获取数据集行列值
	m,n = shape(datMat)
	#遍历每一行，利用核函数对测试集进行分类
	for i in range(m):
		#利用核函数转换数据
		kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
		#仅用支持向量预测分类
		predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
		#预测分类结果与标签不符则错误计数加一
		if sign(predict)!=sign(labelArr[i]): errorCount += 1
	#打印输出分类错误率
	print("the test error rate is: %f" % (float(errorCount)/m))

testDigits(('rbf',10))
#######********************************
'''Non-Kernel VErsions below'''
#######********************************
'''
class optStruct:
	"""
	Function：	存放运算中重要的值
	Input：		dataMatIn：数据集
				classLabels：类别标签
				C：常数C
				toler：容错率
	Output：	X：数据集
				labelMat：类别标签
				C：常数C
				tol：容错率
				m：数据集行数
				b：常数项
				alphas：alphas矩阵
				eCache：误差缓存
	"""	
	def __init__(self, dataMatIn, classLabels, C, toler):
		self.X = dataMatIn
		self.labelMat = classLabels
		self.C = C
		self.tol = toler
		self.m = shape(dataMatIn)[0]
		self.alphas = mat(zeros((self.m, 1)))
		self.b = 0
		self.eCache = mat(zeros((self.m, 2)))

def calcEk(oS, k):
	"""
	Function：	计算误差值E
	Input：		oS：数据结构
				k：下标
	Output：	Ek：计算的E值
	"""	
	#计算fXk，整个对应输出公式f(x)=w`x + b
	fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k,:].T)) + oS.b	
	#计算E值
	Ek = fXk - float(oS.labelMat[k])
	#返回计算的误差值E
	return Ek

def selectJ(i, oS, Ei):
	"""
	Function：	选择第二个alpha的值
	Input：		i：第一个alpha的下标
				oS：数据结构
				Ei：计算出的第一个alpha的误差值
	Output：	j：第二个alpha的下标
				Ej：计算出的第二个alpha的误差值
	"""	
	#初始化参数值
	maxK = -1; maxDeltaE = 0; Ej = 0
	#构建误差缓存
	oS.eCache[i] = [1, Ei]
	#构建一个非零列表，返回值是第一个非零E所对应的alpha值，而不是E本身
	validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
	#如果列表长度大于1，说明不是第一次循环
	if (len(validEcacheList)) > 1:
		#遍历列表中所有元素
		for k in validEcacheList:
			#如果是第一个alpha的下标，就跳出本次循环
			if k == i: continue
			#计算k下标对应的误差值
			Ek = calcEk(oS, k)
			#取两个alpha误差值的差值的绝对值
			deltaE = abs(Ei - Ek)
			#最大值更新
			if (deltaE > maxDeltaE):
				maxK = k; maxDeltaE = deltaE; Ej = Ek
		#返回最大差值的下标maxK和误差值Ej
		return maxK, Ej
	#如果是第一次循环，则随机选择alpha，然后计算误差
	else:
		j = selectJrand(i, oS.m)
		Ej = calcEk(oS, j)
	#返回下标j和其对应的误差Ej
	return j, Ej

def updateEk(oS, k):
	"""
	Function：	更新误差缓存
	Input：		oS：数据结构
				j：alpha的下标
	Output：	无
	"""	
	#计算下表为k的参数的误差
	Ek = calcEk(oS, k)
	#将误差放入缓存
	oS.eCache[k] = [1, Ek]

def innerL(i, oS):
	"""
	Function：	完整SMO算法中的优化例程
	Input：		oS：数据结构
				i：alpha的下标
	Output：	无
	"""	
	#计算误差
	Ei = calcEk(oS, i)
	#如果标签与误差相乘之后在容错范围之外，且超过各自对应的常数值，则进行优化
	if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
		#启发式选择第二个alpha值
		j, Ej = selectJ(i, oS, Ei)
		#利用copy存储刚才的计算值，便于后期比较
		alphaIold = oS.alphas[i].copy(); alpahJold = oS.alphas[j].copy();
		#保证alpha在0和C之间
		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS. alphas[i])
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])
		#如果界限值相同，则不做处理直接跳出本次循环
		if L == H: print("L==H"); return 0
		#最优修改量，求两个向量的内积（核函数）
		eta = 2.0 * oS.X[i, :]*oS.X[j, :].T - oS.X[i, :]*oS.X[i, :].T - oS.X[j, :]*oS.X[j, :].T
		#如果最优修改量大于0，则不做处理直接跳出本次循环，这里对真实SMO做了简化处理
		if eta >= 0: print("eta>=0"); return 0
		#计算新的alphas[j]的值
		oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
		#对新的alphas[j]进行阈值处理
		oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
		#更新误差缓存
		updateEk(oS, j)
		#如果新旧值差很小，则不做处理跳出本次循环
		if (abs(oS.alphas[j] - alpahJold) < 0.00001): print("j not moving enough"); return 0
		#对i进行修改，修改量相同，但是方向相反
		oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alpahJold - oS.alphas[j])
		#更新误差缓存
		updateEk(oS, i)
		#更新常数项
		b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :]*oS.X[i, :].T - oS.labelMat[j] * (oS.alphas[j] - alpahJold) * oS.X[i, :]*oS.X[j, :].T
		b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :]*oS.X[j, :].T - oS.labelMat[j] * (oS.alphas[j] - alpahJold) * oS.X[j, :]*oS.X[j, :].T
		#谁在0到C之间，就听谁的，否则就取平均值
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[i]): oS.b = b2
		else: oS.b = (b1 + b2) / 2.0
		#成功返回1
		return 1
	#失败返回0
	else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter):
	"""
	Function：	完整SMO算法
	Input：		dataMatIn：数据集
				classLabels：类别标签
				C：常数C
				toler：容错率
				maxIter：最大的循环次数
	Output：	b：常数项
				alphas：数据向量
	"""	
	#新建数据结构对象
	oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
	#初始化迭代次数
	iter = 0
	#初始化标志位
	entireSet = True; alphaPairsChanged = 0
	#终止条件：迭代次数超限、遍历整个集合都未对alpha进行修改
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
		alphaPairsChanged = 0
		#根据标志位选择不同的遍历方式
		if entireSet:
			#遍历任意可能的alpha值
			for i in range(oS.m):
				#选择第二个alpha值，并在可能时对其进行优化处理
				alphaPairsChanged += innerL(i, oS)
				print("fullSet, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
			#迭代次数累加
			iter += 1
		else:
			#得出所有的非边界alpha值
			nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
			#遍历所有的非边界alpha值
			for i in nonBoundIs:
				#选择第二个alpha值，并在可能时对其进行优化处理
				alphaPairsChanged += innerL(i, oS)
				print("non-bound, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
			#迭代次数累加
			iter += 1
		#在非边界循环和完整遍历之间进行切换
		if entireSet: entireSet = False
		elif (alphaPairsChanged == 0): entireSet =True
		print("iteration number: %d" % iter)
	#返回常数项和数据向量
	return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):
	"""
	Function：	计算W
	Input：		alphas：数据向量
				dataArr：数据集
				classLabels：类别标签
	Output：	w：w*x+b中的w
	"""	
	#初始化参数
	X = mat(dataArr); labelMat = mat(classLabels).transpose()
	#获取数据行列值
	m,n = shape(X)
	#初始化w
	w = zeros((n,1))
	#遍历alpha，更新w
	for i in range(m):
		w += multiply(alphas[i]*labelMat[i],X[i,:].T)
	#返回w值
	return w
'''         
            
                
            
            
            
            
            
            
        
 












