# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 19:34:10 2019

@author: Administrator
"""
'''树回归，CART——分类回归树，即可用于分类，也可用于回归
适用范围：数据拥有众多特征，并且特征之间关系复杂，难以全局线性回归
思路：数据切分成多份易建模数据，然后利用线性回归技术建模
CART算法
回归与模型树
树剪枝算法
python中的GUI的使用

区别于决策树：
决策树不断将数据切分，直到所有目标量完全相同，或者不能再继续切分
决策树是贪心算法，要在给定时间内做出最佳选择，但不关系能否达到全局最优


树回归：即可用于回归，也可用于分类
优点：可以对复杂和非线性的数据建模
缺点：结果不易理解
使用数据：数值型和标称型数据

树回归的一般方法：
（1）搜集数据
（2）准备数据：需要数值型数据，标称型数据映射成二值型数据
（3）分析数据：绘出数据的二维可视化显示结果，以字典方式生成树
（4）训练算法：大部分时间发费在在叶节点数据模型的构建上
（5）测试算法：使用测试数据的R方值来分析模型的效果
（6）使用算法：使用训练出的树做预测，预测结果来可以用来做很多事情


树的构建过程，使用字典来存储树的数据结构，如下：
1.待切分的特征，
2.待切分的数据特征值，
3.右子树（不再需要切分时也可以是单个子集）
4.左子树(同上)

'''

'''-----树回归的重要参数ops作用预剪枝，调参的关键，决定树的构建好坏----'''
'''----面向对象的程序设计来实现树结构'''
import numpy as np
class treeNode() :
    def _init_(self,fead,val,right,left):
        featureToSplitOn=feat
        valueOfSplit=val
        rightBranch=right
        leftBranch=left
        
'''创建两种树：
回归树：每个叶节点包含单个值
模型树：每个叶节点包含一个线性方程
两种树共用伪代码：
找到最佳的待切分特征:
    如果该节点不能再切分，将该节点存为叶节点
    执行二元切分
    在右子树调用createTree()方法
    在左子树调用createTree()方法
'''
'''------CART算法实现代码-----'''
from numpy import *
#数据的调取与处理
#数据存储在一个集合中
def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        #将每行数据映射成浮点数
        fltLine=list(map(float,curLine))
        dataMat.append(fltLine)
    #源代码的纠正将数据集转换成矩阵形式并输出
    return dataMat
'''---------------对数据集进行二切分----------'''
#根据特征值切分数据集为两部分
#函数三个特征数据集，待切分的特征，该特征的某个值
def binSplitDataSet(dataSet, feature, value):
    """binSplitDataSet(将数据集，按照feature列的value进行 二元切分)
        Description：在给定特征和特征值的情况下，该函数通过数组过滤方式将上述数据集合切分得到两个子集并返回。
    Args:
        dataMat 数据集
        feature 待切分的特征列
        value 特征列要比较的值
    Returns:
        mat0 小于等于 value 的数据集在左边
        mat1 大于 value 的数据集在右边
    Raises:
    """
    # # 测试案例
    #print ('dataSet[:, feature]=', dataSet[:, feature])
    #print ('nonzero(dataSet[:, feature] > value)[0]=', nonzero(dataSet[:, feature] > value)[0])
    #print ('nonzero(dataSet[:, feature] <= value)[0]=', nonzero(dataSet[:, feature] <= value)[0])

    # dataSet[:, feature] 取去每一行中，第feature列的值(从0开始算)
    # nonzero(dataSet[:, feature] > value)  返回结果为true行的index下标
    mat0 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    return mat0, mat1


'''树构建函数，
数据集合，三个可选参数（决定树的类型）
leafType建立叶节点的函数，
errType代表误差计算函数，
ops包含树构建所需的其他参数的元组
此函数为递归函数，尝试将数据集分成两部分d

'''


#创建单位矩阵
testMat=mat(eye(4))
testMat
mat0,mat1=binSplitDataSet(testMat,1,0.5)
mat0
mat1

'''chooseBestSplit函数
1.找到函数切分的最佳位置，历遍所有特征，及其所有取值，找到误差最小化的切分阙值
给定误差的计量方法：平方误差的总值，找到数据集上的最佳二元切分方法，
确定停止切分的条件
(1)leafType:创建叶节点函数的引用
(2)errType：总方差计算函数的引用
(3)ops：用户定义的参数构成的元组
伪代码：
对每个特征:
    对每个特征的值:
        将数据切分成两份
        计算切分的误差
        如果当前误差小于当前最小误差，那么当前误差设置为最佳切分并更新最小误差
    返回切分的最佳特征和阙值
'''
# 1.用最佳方式切分数据集
# 2.生成相应的叶节点
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1])*shape(dataSet)[0]


'''-----基于预剪枝设定一定的条件-------'''
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """chooseBestSplit(用最佳方式切分数据集 和 生成相应的叶节点)

    Args:
        dataSet   加载的原始数据集
        leafType  建立叶子点的函数
        errType   误差计算函数(求总方差)
        ops       [容许误差下降值，切分的最少样本数]。
    Returns:
        bestIndex feature的index坐标
        bestValue 切分的最优值
    Raises:
    """

    # ops=(1,4)，非常重要，因为它决定了决策树划分停止的threshold值，被称为预剪枝（prepruning），其实也就是用于控制函数的停止时机。
    # 之所以这样说，是因为它防止决策树的过拟合，所以当误差的下降值小于tolS，即ops[0]，或划分后的集合size小于tolN,ops[1]时，选择停止继续划分。
    # tolS,容许的误差下降值，，划分后的误差减小小于这个差值，就不用继续划分
    tolS = ops[0]
    # 划分的最小样本数，最小 size 小于，就不继续划分了
    tolN = ops[1]
    #将数据转换成矩阵形式
    dataSet=mat(dataSet)
    # 如果结果集(最后一列为1个变量)，就返回退出
    # .T 对数据集进行转置
    # .tolist()[0] 转化为数组并取第0列
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1: # 如果集合size为1,所有元素相同，不用继续划分。
        #  exit cond 1
        return None, leafType(dataSet)
    # 计算行列值
    m, n = shape(dataSet)
    # 无分类误差的总方差和
    # the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    # inf 正无穷大
    bestS, bestIndex, bestValue = inf, 0, 0  #初始化最小误差，最佳划分属性，最佳阙值
    # 循环处理每一列对应的feature值
    for featIndex in range(n-1): # 对于每个特征
        # [0]表示这一列的[所有行]，不要[0]就是一个array[[所有行]]
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            # 对该列进行分组，然后组内的成员的val值进行 二元切分
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 判断二元切分的方式的元素数量是否符合预期
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            # 如果二元切分，算出来的误差在可接受范围内，那么就记录切分点，并记录最小误差
            # 如果划分后误差小于 bestS，则说明找到了新的bestS
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 判断二元切分的方式的元素误差是否符合预期，如果误差减小不大则退出，不再切分，直接创建叶节点
    # if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 对整体的成员进行判断，是否符合预期,如果切分的数据很小则退出
    # 如果集合的 size 小于 tolN 
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): # 当最佳划分后，集合过小，也不划分，产生叶节点
        return None, leafType(dataSet)
    return bestIndex, bestValue

# assume dataSet is NumPy Mat so we can array filtering
# 假设 dataSet 是 NumPy Mat 类型的，那么我们可以进行 array 过滤
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """createTree(获取回归树)
        Description：递归函数：如果构建的是回归树，该模型是一个常数，如果是模型树，其模型师一个线性方程。
    Args:
        dataSet      加载的原始数据集
        leafType     建立叶子点的函数
        errType      误差计算函数
        ops=(1, 4)   [容许误差下降值，切分的最少样本数]
    Returns:
        retTree    决策树最后的结果
    """
    # 选择最好的切分方式： feature索引值，最优切分值
    # choose the best split
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # if the splitting hit a stop condition return val
    # 如果 splitting 达到一个停止条件，那么返回 val
    if feat is None:
        return val
    retTree = {}
    #确定树的切分点，切分值
    retTree['spInd'] = feat
    retTree['spVal'] = val
    # 小于在左边，大于在右边，分为2个数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 递归的进行调用，在左右子树中继续递归生成树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree
                     
'''----------运行上述代码----------'''
'''首先指定参数，使构建的树足够大，，足够复杂，便于剪枝，
从上往下剪叶节点，用测试集来判断这些叶节点合并能否降低预测误差，如果能则合并
伪代码：
基于已有的树切分测试数据集:
    如果存在任一子集是一棵树，则在该子集递归剪枝过程
    计算当前叶节点合并的误差
    就算不合并的误差
    如果合并会降低误差，则将叶节点合并
'''    
from numpy import *
myDat=loadDataSet(r'E:\python_study\Python_work\data\09\ex00.txt')
myMat=mat(myDat)
createTree(myMat)

myDat1 = loadDataSet(r'E:\python_study\Python_work\data\09\ex0.txt')
myMat1=mat(myDat1)
createTree(myMat1)

myDat2 = loadDataSet(r'E:\python_study\Python_work\data\09\ex2.txt')
myMat2=mat(myDat2)
createTree(myMat2)

#图形展示
#从数据中生成一棵回归树
import matplotlib.pyplot as plt
myDat=loadDataSet(r'E:\python_study\Python_work\data\09\ex00.txt')
myMat=mat(myDat)
createTree(myMat)
plt.figure()
plt.subplot(121)  # 总共一排两列，当前是第一个
plt.plot(myMat[:,0],myMat[:,1],'ro')  
myDat=loadDataSet(r'E:\python_study\Python_work\data\09\ex0.txt')
myMat=mat(myDat)
createTree(myMat)
plt.subplot(122)  # 当前是第二个
plt.plot(myMat[:,1],myMat[:,2],'ro')
plt.show()

'''-------基于后剪枝决策树的构建，'''
#判断输入是否为一棵树
#判断当前处理的节点是否是叶节点
def isTree(obj):
    return (type(obj).__name__=='dict') #判断为字典类型返回true
#返回树的平均值
#该函数为递归函数，从上往下历遍树直到叶节点为止，如果找到两个叶节点则计算他们平均值，
#对树进行塌陷处理，即返回树均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0


#树的后剪枝
def prune(tree, testData):#待剪枝的树和剪枝所需的测试数据
    if shape(testData)[0] == 0: return getMean(tree)  
    # 确认数据集非空，一旦非空反复递归调用函数prune对测试数据进行切分
    #假设发生过拟合，采用测试数据对树进行剪枝
    if (isTree(tree['right']) or isTree(tree['left'])): #左右子树非空
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        #判断分支是子树还是节点,如果是子树则用prune进行剪枝
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    #剪枝后判断是否还是有子树，如果没有子树，则可进行合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        #判断是否merge
        #numpy.power()用于数组元素求n次方
        #numpy.power(x1, x2) ：x2可以是数字，也可以是数组，但是x1和x2的列数要相同
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        #如果合并后误差变小
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree
'''
myTree=createTree(myMat2,ops=(0,1))
myDatTest=loadDataSet(r'E:\python_study\Python_work\data\09\ex2test.txt')
myMat2Test=mat(myDatTest)
prune(myTree,myMat2Test)
'''

'''------模型树，用树来对数据建模之把叶节点设定为分段线性函数-----'''
'''
优点：
1.具有更好大的可解释性，
2.更高的预测准确度

采用模型树需要考虑两个问题，一个是切分点，另一个是怎么计算误差
切分点：利用树生成算法对数据进行切分，每份切分数据都能很容易被线性模型所表示
误差：先用线性的模型来对它进行拟合，然后计算真实的目标值与模型预测值间的差值。
最后将这些差值平方就得到了所需的误差'''
from numpy import *
def linearSolve(dataSet):
    #数据格式初始化
    m,n=shape(dataSet)
    X=mat(ones((m,n)));Y=mat(ones((m,1)))
    #X的第一列为常数项，赋值为1，第二列到最后一列存储dataset数据的前n-1列，Y内数据为数据集最后一列
    X[:,1:n]=dataSet[:,0:n-1];Y=dataSet[:,-1]
    xTx=X.T*X
    # linalg.det计算方阵行列式值，==0，不可逆
    if linalg.det(xTx)==0.0:
        raise NameError('This metrix is singular,cannot do inverse,\n\
                        try increasing the second value of ops')
    #定义权重系数.I求逆,.T转置
    ws=xTx.I*(X.T*Y)
    return ws ,X,Y

#返回回归系数ws
def modelLeaf(dataSet):
    ws,X,Y=linearSolve(dataSet)
    return ws
#计算预测误差平方和,会被choosebestSplited调用来产生最佳切分
def modelErr(dataSet):
    ws,X,Y=linearSolve(dataSet)
    yHat=X*ws
    return sum(power(Y-yHat,2))

myMat2=mat(loadDataSet(r'E:\python_study\Python_work\data\09\exp2.txt'))
mytree=createTree(myMat2,modelLeaf,modelErr,(1,10))
plt.plot(myMat2[:,0],myMat2[:,1],'ro')
plt.show()
print(mytree)

'''
{'spInd': 0, 'spVal': 0.285477, 'left': matrix([[3.46877936],
         [1.18521743]]), 'right': matrix([[1.69855694e-03],
         [1.19647739e+01]])}
结果解读，代码以x=0.285477为界,创建两个模型，y=3.46877936+1.18521743x和y=0.0016985+11.9x
'''
#计算R方分析模型优劣
#corrcoef(yHat,y,rowvar=0)



'''------------比较模型树，回归树，一般回归方法---------'''
# 回归树测试案例
# 为了和 modelTreeEval() 保持一致，保留两个输入参数
def regTreeEval(model, inDat):
    """
    Desc:
        对 回归树 进行预测
    Args:
        model -- 指定模型，可选值为 回归树模型 或者 模型树模型，这里为回归树
        inDat -- 输入的测试数据
    Returns:
        float(model) -- 将输入的模型数据转换为 浮点数 返回
    """
    return float(model)


# 模型树测试案例
# 对输入数据进行格式化处理，在原数据矩阵上增加第0列，元素的值都是1，
# 也就是增加偏移值，和我们之前的简单线性回归是一个套路，增加一个偏移量
def modelTreeEval(model, inDat):
    """
    Desc:
        对 模型树 进行预测
    Args:
        model -- 输入模型，可选值为 回归树模型 或者 模型树模型，这里为模型树模型
        inDat -- 输入的测试数据
    Returns:
        float(X * model) -- 将测试数据乘以 回归系数 得到一个预测值 ，转化为 浮点数 返回
    """
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1: n+1] = inDat
    # print X, model
    return float(X * model)
# 计算预测的结果
# 在给定树结构的情况下，对于单个数据点，该函数会给出一个预测值。
# modelEval是对叶节点进行预测的函数引用，指定树的类型，以便在叶节点上调用合适的模型。
# 此函数自顶向下遍历整棵树，直到命中叶节点为止，一旦到达叶节点，它就会在输入数据上
# 调用modelEval()函数，该函数的默认值为regTreeEval()
def treeForeCast(tree, inData, modelEval=regTreeEval):
    """
    Desc:
        对特定模型的树进行预测，可以是 回归树 也可以是 模型树
    Args:
        tree -- 已经训练好的树的模型
        inData -- 输入的测试数据
        modelEval -- 预测的树的模型类型，可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树），默认为回归树
    Returns:
        返回预测值
    """
    #判断模型是否是树结构
    if not isTree(tree):
        return modelEval(tree, inData)
    #切分特征对应的值小于切分值，分析其是否为左子树
    if inData[tree['spInd']] <= tree['spVal']:
        #如果是子树，则继续进行迭代，知道为叶节点为止
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            #如果不是子树，即叶节点，则计算预测值
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


# 预测结果，默认使用回归树求预测值的函数
def createForeCast(tree, testData, modelEval=regTreeEval):
    """
    Desc:
        调用 treeForeCast ，对特定模型的树进行预测，可以是 回归树 也可以是 模型树
    Args:
        tree -- 已经训练好的树的模型
        inData -- 输入的测试数据
        modelEval -- 预测的树的模型类型，可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树），默认为回归树
    Returns:
        返回预测值矩阵
    """
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    # print yHat
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
        # print "yHat==>", yHat[i, 0]
    return yHat

#创建不同的模型进行比较
#创建回归树
trainMat=mat(loadDataSet(r'E:\python_study\Python_work\data\09\bikeSpeedVsIq_train.txt'))
testMat=mat(loadDataSet(r'E:\python_study\Python_work\data\09\bikeSpeedVsIq_test.txt'))
myTree=createTree(trainMat,ops=(1,20))
yHat=createForeCast(myTree,testMat[:,0])
corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]

#创建模型树
trainMat=mat(loadDataSet(r'E:\python_study\Python_work\data\09\bikeSpeedVsIq_train.txt'))
testMat=mat(loadDataSet(r'E:\python_study\Python_work\data\09\bikeSpeedVsIq_test.txt'))
myTree=createTree(trainMat,modelLeaf,modelErr,(1,20))
yHat=createForeCast(myTree,testMat[:,0],modelTreeEval)
corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]

#使用线性回归模型进行预测
ws,X,Y=linearSolve(trainMat)
ws
for i in range(shape(testMat)[0]):
    yHat[i]=testMat[i,0]*ws[1,0]+ws[0,0]
corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]

'''
R^2 判定系数就是拟合优度判定系数，它体现了回归模型中自变量的变异在因变量的变异中
所占的比例。如 R^2=0.99999 表示在因变量 y 的变异中有 99.999% 是由于变量 x 引起。
当 R^2=1 时表示，所有观测点都落在拟合的直线或曲线上；当 R^2=0 时，表示
自变量与因变量不存在直线或曲线关系。
所以我们看出， R^2 的值越接近 1.0 越好。'''

'''---------python 的Tkinter库创建图形用户界面GUI---------'''
'''利用GUI对回归树进行调优
（1）搜集数据
（2）准备数据:使用python解析数据文件，得到数值型数据
（3）分析数据：用tkinter构建一个GUI来展示模型数据
（4）训练算法：训练一颗回归树，和一颗模型树，并与数据集一起展示出来
（5）测试算法：此处不需要测试过程
（6）使用算法：GUI使得人们可以在预剪枝时测试不同参数的影响，还可以帮助我们选择模型的类型    
'''
from numpy import *
from tkinter import *
root=Tk()
myLabel=Label(root,text='Hello World')
myLabel.grid()
root.mainloop()