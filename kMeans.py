# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:37:57 2019

@author: lenovo
"""
'''
分类目标事先已知，而聚类为无监督学习，根据相似度进行无监督聚类，目标事先未知
 -----------------------------均值聚类
优点:
属于无监督学习，无须准备训练集
原理简单，实现起来较为容易
结果可解释性较好
缺点:
需手动设置k值。 在算法开始预测之前，我们需要手动设置k值，即估计数据大概的类别个数，不合理的k值会使结果缺乏解释性
可能收敛到局部最小值, 在大规模数据集上收敛较慢
对于异常点、离群点敏感

适用数据类型:数值型数据
通过聚类可以找到需求的类别群里，进行分析，应用


k均值聚类，用户给定要聚类的簇数，簇用质心表述

k均值聚类的伪代码
创建k个点作为起始质心（通常随机选择）
当任意一个点的簇分配结果发生改变时:
    对数据集的每个数据点:
        对每个质心:
            计算质心与数据点之间的距离
        将数据点分配到距其最近的簇
    对每个簇计算簇中所有点的均值并将均值作为质心
    

k均值算法的一般流程
（1）搜集数据：使用任意方法
（2）准备数据:需要数值型数据计算距离，也可以将标称型数据映射为二值型数据再用于距离计算
（3）分析数据：使用任意方法
（4）训练算法：不适用无监督学习，即无监督学习没有训练过程
（5）测试算法：应用聚类算法，观察结果，可以使用量化的误差指标如误差平方和
（6）使用算法：可以用于所希望的任何应用，通常簇质心可以代表整个簇的数据来做出决策
'''
#unsupported operand type(s) for -: 'map' and 'map'错误
#发现fltLine = map(float, curLine)在python2中返回的是一个list类型数据，而在python3中该语句返回的是一个map类型的数据。
#因此，我们只需要将该语句改为fltLine = list(map(float, curLine)),错误就解决啦。


from numpy import *
#读取数据
'''------------------------------k均值聚类的支持函数---------'''
#导入数据，将文本文件导入到一个列表中，每个列表添加到dataMat中，并返回一个dataMat
def loadData(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        
        fltLine=list(map(float,curLine))  
        dataMat.append(fltLine)
    return dataMat

#计算两向量的欧式距离,此处选择的相似度的度量标准
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))


# 为给定数据集构建一个包含 k 个随机质心的集合。随机质心必须要在整个数据集的边界之内
#这可以通过找到数据集每一维的最小和最大值来完成。然后生成 0~1.0 之间的随机数并通过
#取值范围和最小值，以便确保随机点在数据的边界之内。
    
#创建随机质心的集合，k均值聚类的
def randCent(dataSet, k):
    n = shape(dataSet)[1] # 列的数量，即数据的特征个数
    centroids = mat(zeros((k,n))) # 创建k个质心矩阵
    for j in range(n): # 创建随机簇质心，并且在每一维的边界内
        minJ = min(dataSet[:,j])    # 最小值
        maxJ = max(dataSet[:,j])    #最大值
        rangeJ = float(maxJ - minJ)    # 范围 = 最大值 - 最小值
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)    # 随机生成，mat为numpy函数，需要在最开始写上 from numpy import *
    return centroids


'''--------------------k-means 聚类算法-----------'''
# 该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
# 这个过程重复数次，直到数据点的簇分配结果不再改变位置。
# 运行结果（多次运行结果可能会不一样，可以试试，原因为随机质心的影响，但总的结果是对的， 
#因为数据足够相似，也可能会陷入局部最小值）

#输入参数：数据集，聚类簇数，相似度度量方法，初始类别中心函数
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]    # 行数，即数据个数
    # 创建一个与 dataSet 行数一样，但是有两列的矩阵，用来保存簇分配结果    
    clusterAssment = mat(zeros((m, 2)))   
    centroids = createCent(dataSet, k)    # 创建质心，随机k个质心
    #判定迭代的条件，初始值设为true
    clusterChanged = True
    while clusterChanged:          
        clusterChanged = False
        # 循环每一个数据点并分配到最近的质心中去
        for i in range(m):    
            minDist = inf; minIndex = -1  #初始化样本i到类的距离
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])    # 计算数据点到质心的距离
                #找到距离最小的类别，判为该样本的所属类别
                if distJI < minDist:    # 如果距离比 minDist（最小距离）还小，更新 minDist（最小距离）和最小质心的 index（索引）
                    minDist = distJI; minIndex = j
            if clusterAssment[i, 0] != minIndex:    # 簇分配结果改变
                clusterChanged = True    # 簇改变
                # 更新簇分配结果为最小质心的 index（索引），minDist（最小距离）的平方
                clusterAssment[i, :] = minIndex,minDist**2    
        print (centroids)
        for cent in range(k): # 更新质心
            # 获取该簇中的所有点
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A==cent)[0]] 
            # 将质心修改为簇中所有点的平均值，mean 就是求平均值的，axis为按行求
            centroids[cent,:] = mean(ptsInClust, axis=0) 
    #返回簇中心，（及个数据所属类别，及到该类的距离的平方）
    return centroids, clusterAssment

datMat=mat(loadData(r'E:\python_study\Python_work\data\10\testSet.txt'))
myCentroids,clustAssing=kMeans(datMat,4)
sum(clustAssing[1])

#引入绘图的包
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   #3d图的包

def Show(data,k,cp,cluster):
    num,dim = data.shape
    color = ['r','g','b','c','y','m','k']
    ##二维图
    if dim==2:
        for i in range(num):
            mark = int(cluster[i,0])
            plt.plot(data[i,0],data[i,1],color[mark]+'o')
            
        for i in range(k):
            plt.plot(cp[i,0],cp[i,1],color[i]+'x')
    ##三维图
    elif dim==3:
        ax = plt.subplot(111,projection ='3d')
        for i in range(num):
            mark = int(cluster[i,0])
            ax.scatter(data[i,0],data[i,1],data[i,2],c=color[mark])
            
        for i in range(k):
            ax.scatter(cp[i,0],cp[i,1],cp[i,2],c=color[i],marker='x')
        
    plt.show()

Show(datMat,4,myCentroids,clustAssing)
    
''' -------二分 k均值聚类算法（函数可以运行多次，聚类会收敛到全局最小值）-----------------------'''
'''
目的：为了克服k-均值算法收敛于局部最小值的问题

该算法首先将所有点作为一个簇，然后将该簇一分为二。
之后选择其中一个簇继续进行划分，选择哪一个簇进行划分取决于对其划分时候可以
最大程度降低 SSE（平方和误差）的值。
上述基于 SSE 的划分过程不断重复，直到得到用户指定的簇数目为止。 


二分k-均值算法的伪代码：
将所有点看成一簇
while当簇数目小于k时:
    for 每一个簇:
        计算总误差
        在给定的簇上面进行k-均值聚类（k=2)
        计算将该簇一分为二之后的总误差
    选择使得误差最小的那个簇进行划分操作
'''    
# 二分 KMeans 聚类算法, 基于 kMeans 基础之上的优化，以避免陷入局部最小值
def biKMeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    # 创建二维矩阵保存每个数据点的簇分配结果和平方误差
    clusterAssment = mat(zeros((m,2))) 
     # 计算整个数据集的质心，质心初始化为所有数据点的均值
    centroid0 = mean(dataSet, axis=0).tolist()[0] 
    # 初始化只有 1 个质心的 list
    centList =[centroid0]   #存储质心结果
    # 计算所有数据点到初始质心的距离平方误差
    for j in range(m): 
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    #while循环是函数的主体，该循环会不停地对簇进行划分，知道达到所要的簇数目
    while (len(centList) < k): # 当质心数量小于 k 时
        lowestSSE = inf   #初始化最小SSE为无穷
        for i in range(len(centList)): # 对每一个质心
             # 获取当前簇 i 下的所有数据点
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
             # 将当前簇 i 进行二分 kMeans 处理
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1]) # 将二分 kMeans 结果中的平方和的距离进行求和
            # 将未参与二分 kMeans 分配结果中的平方和的距离进行求和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) 
            print ("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE: # 总的（未拆分和已拆分）误差和越小，越相似，效果越优化，划分的结果更好（注意：这里的理解很重要，不明白的地方可以和我们一起讨论）
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 找出最好的簇分配结果    
        # 调用二分 kMeans 的结果，默认簇是 0,1. 当然也可以改成其它的数字?????
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) 
        # 更新为最佳质心
        print ('the bestCentToSplit is: ',bestCentToSplit)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit 
        print ('the len of bestClustAss is: ', len(bestClustAss))
        # 更新质心列表
        # 更新原质心 list 中的第 i 个质心为使用二分 kMeans 后 bestNewCents 的第一个质心
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] 
        # 添加 bestNewCents 的第二个质心
        centList.append(bestNewCents[1,:].tolist()[0])
        # 重新分配最好簇下的数据（质心）以及SSE
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss 
    return mat(centList), clusterAssment


datMat=mat(loadData(r'E:\python_study\Python_work\data\10\testSet2.txt'))
centList, myNewAssments=biKMeans(datMat,4)
centList   #查看质心结果  ，四个质心
sum(myNewAssments[1])
Show(datMat,4,centList, myNewAssments)

#优点：二分法聚类可以达到全局最小值，而原始聚类只能达到局部最小值



'''---------------应用：对地图上点进行聚类-------------------'''
'''示例：对地理数据应用二分K-means算法
（1）搜集数据：使用Yahoo!PlaceFinder API搜集数据
（2）准备数据:保留经纬度信息
（3）分析数据：用Matplotlib来构建一个二维数据图，其中包括簇与地图
（4）训练算法：不适用无监督学习
（5）测试算法：使用biKmeans()函数
（6）使用算法：最后输出包含簇与簇中心的地图
雅虎提供了将地址准换成经纬度的方法
Yahoo！PlaceFinder API的使用方法


'''
import urllib    
import json
#此函数从雅虎返回一个字典
def geoGrab(stAddress, city): 
    apiStem = "http://where.yahooapis.com/geocode?"   #指定网页地址
    params = {}           #创建字典，可以为字典创建不同的值
    params['flags'] = 'J'  # 返回JSON格式的结果（一种用于序列化数组和字典的文件格式）
    params['appid'] = 'aaa0VN6k'   
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.parse.urlencode(params)  # 将params字典转换为可以通过URL进行传递的字符串格式
    yahooApi = apiStem + url_params      
    print(yahooApi)  # 输出URL
    c=urllib.request.urlopen(yahooApi)  #读取返回值
    return json.loads(c.read())  # 返回一个字典，此时意味着对地址进行了地理编码

from time import sleep

#将所有信息封装起来并且将相关信息保存到文件中
#该函数打开tab分割的文本文件，获取第二列第三列的数据，将值输入到geoGrab中
def massPlaceFind(fileName):
    fw = open(r'E:\python_study\Python_work\data\10\place.txt', 'w') #打开地址文件
    for line in open(fileName).readlines():   #按行读取数据
        line = line.strip()          
        lineArr = line.split('\t')  # 是以tab分隔的文本文件，按照tab分隔符进行分割
        retDict = geoGrab(lineArr[1], lineArr[2]) # 读取2列和第3列
        if retDict['ResultSet']['Error'] == 0: # 检查输出字典，判断有没有出错
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])  # 读取经纬度
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))  # 添加到对应的行上
        else: print ('error fetching') # 有错误时不需要抽取经纬度
        sleep(1)  # 避免频繁调用API，过于频繁的话请求会被封掉
    fw.close()

geoGrab('1 VA Center', 'Augusta,ME')
